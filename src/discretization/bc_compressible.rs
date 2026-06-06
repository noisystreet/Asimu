//! 可压缩流边界条件 ghost 单元施加。
//!
//! 来流 ghost 经 [`FreestreamContext`](crate::physics::FreestreamContext) 构造；理论见
//! [`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §4。

use crate::boundary::{
    BcHandler, BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, WallHeat,
};
use crate::core::{Real, Vector3};
use crate::discretization::wall_thermal::wall_ghost_temperature;
use crate::error::Result;
use crate::field::ConservedFields;
use crate::mesh::{BoundaryMesh3d, FaceGeometry3d};
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
    ViscousPhysicsConfig,
};

/// 单面边界外侧守恒状态。
///
/// 名称保留 `GhostCellState` 以兼容现有调用；对 farfield/inlet/outlet，
/// 该状态由法向特征关系构造，不再只是简单拷贝/镜像 ghost。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GhostCellState {
    pub conserved: ConservedState,
}

/// 边界面 ghost 缓冲（按 `FaceId` 索引）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundaryGhostBuffer {
    states: Vec<Option<GhostCellState>>,
}

impl BoundaryGhostBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_face_capacity(num_faces: usize) -> Self {
        Self {
            states: vec![None; num_faces],
        }
    }

    pub fn ensure_face_capacity(&mut self, num_faces: usize) {
        if self.states.len() < num_faces {
            self.states.resize(num_faces, None);
        }
    }

    pub fn insert_face(&mut self, face: crate::core::FaceId, state: GhostCellState) {
        let index = face.index() as usize;
        if index >= self.states.len() {
            self.states.resize(index + 1, None);
        }
        self.states[index] = Some(state);
    }

    pub fn get_face(&self, face: crate::core::FaceId) -> Option<GhostCellState> {
        self.states.get(face.index() as usize).copied().flatten()
    }
}

/// 等距 ghost 镜像速度：\(u_{n,g}=-u_{n,o}\) 使面心 \(u_n=0\)；滑移保留切向，无滑移 \(\mathbf{u}_g=-\mathbf{u}_o\)。
fn wall_ghost_velocity(owner: [Real; 3], normal: Vector3, no_slip: bool) -> [Real; 3] {
    let un = owner[0] * normal.x + owner[1] * normal.y + owner[2] * normal.z;
    let u_t = [
        owner[0] - un * normal.x,
        owner[1] - un * normal.y,
        owner[2] - un * normal.z,
    ];
    let un_g = -un;
    if no_slip {
        [
            -u_t[0] + un_g * normal.x,
            -u_t[1] + un_g * normal.y,
            -u_t[2] + un_g * normal.z,
        ]
    } else {
        [
            u_t[0] + un_g * normal.x,
            u_t[1] + un_g * normal.y,
            u_t[2] + un_g * normal.z,
        ]
    }
}

/// 壁面 ghost：滑移/无滑移均用等距镜像 ghost；无滑移 \(\mathbf{u}_g=-\mathbf{u}_o\)，滑移 \(u_{n,g}=-u_{n,o}\) 且 \(\mathbf{u}_{t,g}=\mathbf{u}_{t,o}\)。
pub fn wall_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    no_slip: bool,
    heat: WallHeat,
    fs_ctx: &FreestreamContext<'_>,
    min_pressure: Real,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<GhostCellState> {
    let eos = fs_ctx.eos;
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, min_pressure)?;
    let velocity = wall_ghost_velocity(prim.velocity, geom.normal, no_slip);
    let t_owner = viscous
        .map(|v| v.static_temperature(prim.pressure, prim.density, eos))
        .unwrap_or(prim.pressure / (prim.density.max(1.0e-30) * eos.gas_constant));
    let t_ghost = wall_ghost_temperature(t_owner, heat, geom.spacing, viscous, eos)?;
    let ghost_prim = PrimitiveState {
        density: fs_ctx.density_from_pressure_temperature(prim.pressure, t_ghost),
        velocity,
        pressure: prim.pressure,
        temperature: t_ghost,
    };
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(eos, &ghost_prim)?,
    })
}

/// 远场外侧状态：亚声速用法向 Riemann 不变量混合内侧出射波与远场入射波。
pub fn farfield_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    params: &FreestreamParams,
    fs_ctx: &FreestreamContext<'_>,
) -> Result<GhostCellState> {
    let eos = fs_ctx.eos;
    let owner_prim = crate::field::primitive_from_conserved_relaxed(eos, owner, 1.0e-12)?;
    let farfield = fs_ctx.primitive(params)?;
    let prim = characteristic_farfield_primitive(&owner_prim, &farfield, geom.normal, eos)?;
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(eos, &prim)?,
    })
}

/// 入口 ghost 参数（合并多字段以满足复杂度门禁）。
pub struct InletGhostParams<'a> {
    pub supersonic: bool,
    pub velocity_direction: [Real; 3],
    pub freestream: &'a FreestreamParams,
    pub fs_ctx: &'a FreestreamContext<'a>,
    pub total_pressure: Real,
    pub total_temperature: Real,
}

/// 入口外侧状态：超声速入口直接使用静来流；亚声速入口用总压/总温和内侧出射特征恢复。
pub fn inlet_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    params: &InletGhostParams<'_>,
) -> Result<GhostCellState> {
    if params.supersonic {
        let fs = FreestreamParams {
            mach: params.freestream.mach,
            pressure: params.freestream.pressure,
            temperature: params.freestream.temperature,
            alpha: params.freestream.alpha,
            beta: params.freestream.beta,
            velocity_direction: params.velocity_direction,
        };
        return farfield_ghost(owner, geom, &fs, params.fs_ctx);
    }
    subsonic_inlet_ghost(
        owner,
        geom.normal,
        params.total_pressure,
        params.total_temperature,
        params.velocity_direction,
        params.fs_ctx.eos,
    )
}

/// 亚声速入口：总压/总温 + 方向 + 内侧出射特征。
fn subsonic_inlet_ghost(
    owner: &ConservedState,
    normal: Vector3,
    total_pressure: Real,
    total_temperature: Real,
    velocity_direction: [Real; 3],
    eos: &IdealGasEoS,
) -> Result<GhostCellState> {
    let owner_prim = crate::field::primitive_from_conserved_relaxed(eos, owner, 1.0e-12)?;
    let owner_sound = sound_speed(&owner_prim, eos);
    let outgoing =
        normal_velocity(owner_prim.velocity, normal) + 2.0 * owner_sound / (eos.gamma - 1.0);
    let dir = orient_inlet_direction(normalize(velocity_direction)?, normal);
    let normal_projection = dot_array(dir, normal);
    let mach = inlet_mach_from_total_state(outgoing, normal_projection, total_temperature, eos);
    let temp_ratio = 1.0 + 0.5 * (eos.gamma - 1.0) * mach * mach;
    let static_temperature = (total_temperature / temp_ratio).max(1.0e-30);
    let static_pressure = total_pressure
        * (static_temperature / total_temperature).powf(eos.gamma / (eos.gamma - 1.0));
    let density = eos.density(static_pressure, static_temperature)?;
    let speed = mach * (eos.gamma * eos.gas_constant * static_temperature).sqrt();
    let prim = PrimitiveState {
        density,
        velocity: [dir[0] * speed, dir[1] * speed, dir[2] * speed],
        pressure: static_pressure,
        temperature: static_temperature,
    };
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(eos, &prim)?,
    })
}

/// 出口外侧状态：超声速出口零梯度；亚声速出口指定静压并保持内侧出射特征。
pub fn outlet_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    static_pressure: Real,
    supersonic: bool,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<GhostCellState> {
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, min_pressure)?;
    let ghost_prim = if supersonic {
        prim
    } else {
        characteristic_outlet_primitive(&prim, geom.normal, static_pressure, eos)?
    };
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(eos, &ghost_prim)?,
    })
}

/// 对称 ghost：法向速度翻转。
pub fn symmetry_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    fs_ctx: &FreestreamContext<'_>,
    min_pressure: Real,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<GhostCellState> {
    wall_ghost(
        owner,
        geom,
        false,
        WallHeat::Adiabatic,
        fs_ctx,
        min_pressure,
        viscous,
    )
}

fn normalize(v: [Real; 3]) -> Result<[Real; 3]> {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if mag < Real::EPSILON {
        return Err(crate::error::AsimuError::Boundary(
            "速度方向不能为零".to_string(),
        ));
    }
    Ok([v[0] / mag, v[1] / mag, v[2] / mag])
}

fn dot_array(v: [Real; 3], normal: Vector3) -> Real {
    v[0] * normal.x + v[1] * normal.y + v[2] * normal.z
}

fn normal_velocity(velocity: [Real; 3], normal: Vector3) -> Real {
    dot_array(velocity, normal)
}

fn tangential_velocity(velocity: [Real; 3], normal: Vector3) -> [Real; 3] {
    let un = normal_velocity(velocity, normal);
    [
        velocity[0] - un * normal.x,
        velocity[1] - un * normal.y,
        velocity[2] - un * normal.z,
    ]
}

fn velocity_from_normal_tangent(un: Real, tangent: [Real; 3], normal: Vector3) -> [Real; 3] {
    [
        tangent[0] + un * normal.x,
        tangent[1] + un * normal.y,
        tangent[2] + un * normal.z,
    ]
}

fn sound_speed(prim: &PrimitiveState, eos: &IdealGasEoS) -> Real {
    (eos.gamma * prim.pressure / prim.density).sqrt()
}

fn entropy_constant(prim: &PrimitiveState, gamma: Real) -> Real {
    prim.pressure / prim.density.powf(gamma)
}

fn primitive_from_sound_entropy_velocity(
    sound: Real,
    entropy: Real,
    velocity: [Real; 3],
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let density = (sound * sound / (eos.gamma * entropy)).powf(1.0 / (eos.gamma - 1.0));
    let pressure = entropy * density.powf(eos.gamma);
    Ok(PrimitiveState {
        density,
        velocity,
        pressure,
        temperature: pressure / (density * eos.gas_constant),
    })
}

fn primitive_from_pressure_entropy_velocity(
    pressure: Real,
    entropy: Real,
    velocity: [Real; 3],
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let density = (pressure / entropy).powf(1.0 / eos.gamma);
    Ok(PrimitiveState {
        density,
        velocity,
        pressure,
        temperature: pressure / (density * eos.gas_constant),
    })
}

fn characteristic_farfield_primitive(
    owner: &PrimitiveState,
    farfield: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let a_owner = sound_speed(owner, eos);
    let a_farfield = sound_speed(farfield, eos);
    let un_owner = normal_velocity(owner.velocity, normal);
    let un_farfield = normal_velocity(farfield.velocity, normal);
    if un_farfield <= -a_farfield {
        return Ok(*farfield);
    }
    if un_owner >= a_owner {
        return Ok(*owner);
    }
    let r_plus = un_owner + 2.0 * a_owner / (eos.gamma - 1.0);
    let r_minus = un_farfield - 2.0 * a_farfield / (eos.gamma - 1.0);
    let un = 0.5 * (r_plus + r_minus);
    let sound = (0.25 * (eos.gamma - 1.0) * (r_plus - r_minus)).max(1.0e-30);
    let use_farfield_entropy = un < 0.0;
    let entropy_source = if use_farfield_entropy {
        farfield
    } else {
        owner
    };
    let velocity_source = if use_farfield_entropy {
        farfield
    } else {
        owner
    };
    let entropy = entropy_constant(entropy_source, eos.gamma);
    let tangent = tangential_velocity(velocity_source.velocity, normal);
    primitive_from_sound_entropy_velocity(
        sound,
        entropy,
        velocity_from_normal_tangent(un, tangent, normal),
        eos,
    )
}

fn orient_inlet_direction(mut direction: [Real; 3], normal: Vector3) -> [Real; 3] {
    if dot_array(direction, normal) > 0.0 {
        direction = [-direction[0], -direction[1], -direction[2]];
    }
    direction
}

fn inlet_mach_from_total_state(
    outgoing: Real,
    normal_projection: Real,
    total_temperature: Real,
    eos: &IdealGasEoS,
) -> Real {
    let residual = |mach: Real| {
        let temp_ratio = 1.0 + 0.5 * (eos.gamma - 1.0) * mach * mach;
        let sound = (eos.gamma * eos.gas_constant * total_temperature / temp_ratio).sqrt();
        sound * (2.0 / (eos.gamma - 1.0) + normal_projection * mach) - outgoing
    };
    let mut lo = 0.0;
    let mut hi = 0.999;
    let mut f_lo = residual(lo);
    let f_hi = residual(hi);
    if f_lo * f_hi > 0.0 {
        return if f_lo.abs() < f_hi.abs() { lo } else { hi };
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let f_mid = residual(mid);
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    0.5 * (lo + hi)
}

fn characteristic_outlet_primitive(
    owner: &PrimitiveState,
    normal: Vector3,
    static_pressure: Real,
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let owner_sound = sound_speed(owner, eos);
    let outgoing = normal_velocity(owner.velocity, normal) + 2.0 * owner_sound / (eos.gamma - 1.0);
    let entropy = entropy_constant(owner, eos.gamma);
    let density = (static_pressure / entropy).powf(1.0 / eos.gamma);
    let sound = (eos.gamma * static_pressure / density).sqrt();
    let un = outgoing - 2.0 * sound / (eos.gamma - 1.0);
    let tangent = tangential_velocity(owner.velocity, normal);
    primitive_from_pressure_entropy_velocity(
        static_pressure,
        entropy,
        velocity_from_normal_tangent(un, tangent, normal),
        eos,
    )
}

fn apply_patch_compressible(
    mesh: &dyn BoundaryMesh3d,
    patch: &BoundaryPatch,
    fields: &ConservedFields,
    ghosts: &mut BoundaryGhostBuffer,
    fs_ctx: &FreestreamContext<'_>,
    freestream: &FreestreamParams,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<()> {
    let p_floor = crate::field::positivity_pressure_floor(freestream.pressure);
    let handler = BoundaryRegistry::handler_for(&patch.kind);
    for &face in &patch.face_ids {
        if matches!(handler, BcHandler::Periodic) && ghosts.get_face(face).is_some() {
            continue;
        }
        let owner_id = mesh.face_owner(face)?;
        let owner = fields.cell_state(owner_id.index() as usize)?;
        let geom = mesh.face_geometry_3d(face)?;
        let ghost = match (&handler, &patch.kind) {
            (BcHandler::Wall, BoundaryKind::Wall { no_slip, heat }) => {
                wall_ghost(&owner, &geom, *no_slip, *heat, fs_ctx, p_floor, viscous)?
            }
            (
                BcHandler::Farfield,
                BoundaryKind::Farfield {
                    mach,
                    pressure,
                    temperature,
                    alpha,
                    beta,
                },
            ) => farfield_ghost(
                &owner,
                &geom,
                &FreestreamParams {
                    mach: *mach,
                    pressure: *pressure,
                    temperature: *temperature,
                    alpha: *alpha,
                    beta: *beta,
                    velocity_direction: [1.0, 0.0, 0.0],
                },
                fs_ctx,
            )?,
            (
                BcHandler::Inlet,
                BoundaryKind::Inlet {
                    total_pressure,
                    total_temperature,
                    velocity_direction,
                    supersonic,
                    ..
                },
            ) => inlet_ghost(
                &owner,
                &geom,
                &InletGhostParams {
                    supersonic: *supersonic,
                    velocity_direction: *velocity_direction,
                    freestream,
                    fs_ctx,
                    total_pressure: *total_pressure,
                    total_temperature: *total_temperature,
                },
            )?,
            (
                BcHandler::TurbulentInlet,
                BoundaryKind::TurbulentInlet {
                    total_pressure,
                    total_temperature,
                    velocity_direction,
                    ..
                },
            ) => inlet_ghost(
                &owner,
                &geom,
                &InletGhostParams {
                    supersonic: false,
                    velocity_direction: *velocity_direction,
                    freestream,
                    fs_ctx,
                    total_pressure: *total_pressure,
                    total_temperature: *total_temperature,
                },
            )?,
            (
                BcHandler::Outlet,
                BoundaryKind::Outlet {
                    static_pressure,
                    supersonic,
                    ..
                },
            ) => outlet_ghost(
                &owner,
                &geom,
                *static_pressure,
                *supersonic,
                fs_ctx.eos,
                p_floor,
            )?,
            (BcHandler::Symmetry, BoundaryKind::Symmetry) => {
                symmetry_ghost(&owner, &geom, fs_ctx, p_floor, viscous)?
            }
            (BcHandler::Periodic, BoundaryKind::Periodic { .. }) => {
                GhostCellState { conserved: owner }
            }
            _ => farfield_ghost(&owner, &geom, freestream, fs_ctx)?,
        };
        ghosts.insert_face(face, ghost);
    }
    Ok(())
}

/// 可压缩 NS 边界 ghost 施加（类比 CFL3D `bc.F`）。
#[tracing::instrument(skip_all, level = "info", fields(patches = patches.patches().len()))]
pub fn apply_compressible_boundary_conditions(
    mesh: &dyn BoundaryMesh3d,
    patches: &BoundarySet,
    fields: &ConservedFields,
    ghosts: &mut BoundaryGhostBuffer,
    fs_ctx: &FreestreamContext<'_>,
    freestream: &FreestreamParams,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<()> {
    BoundaryRegistry::validate_patches(patches.patches())?;
    for patch in patches.patches() {
        apply_patch_compressible(mesh, patch, fields, ghosts, fs_ctx, freestream, viscous)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "bc_compressible_tests.rs"]
mod tests;
