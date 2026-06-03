//! 可压缩流边界条件 ghost 单元施加。
//!
//! 来流 ghost 经 [`FreestreamContext`](crate::physics::FreestreamContext) 构造；理论见
//! [`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §4。

use crate::boundary::{
    BcHandler, BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, WallHeat,
};
use crate::core::Real;
use crate::discretization::wall_thermal::wall_ghost_temperature;
use crate::error::Result;
use crate::field::ConservedFields;
use crate::mesh::{BoundaryMesh3d, FaceGeometry3d};
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
    ViscousPhysicsConfig,
};

/// 单面 ghost 守恒状态。
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
fn wall_ghost_velocity(owner: [Real; 3], normal: crate::core::Vector3, no_slip: bool) -> [Real; 3] {
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

/// 远场 ghost：基于来流状态与法向简单外推。
pub fn farfield_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    params: &FreestreamParams,
    fs_ctx: &FreestreamContext<'_>,
) -> Result<GhostCellState> {
    let _ = (owner, geom);
    let prim = fs_ctx.primitive(params)?;
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(fs_ctx.eos, &prim)?,
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

/// 入口 ghost：超声速入口（`supersonic`）直接使用 `[freestream]` 静参数；亚声速用总压/总温简化模型。
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
        params.total_pressure,
        params.total_temperature,
        params.velocity_direction,
        params.fs_ctx.eos,
    )
}

/// 亚声速入口：总压/总温 + 方向（简化静参数恢复）。
fn subsonic_inlet_ghost(
    total_pressure: Real,
    total_temperature: Real,
    velocity_direction: [Real; 3],
    eos: &IdealGasEoS,
) -> Result<GhostCellState> {
    let cp = eos.gamma * eos.gas_constant / (eos.gamma - 1.0);
    let static_temperature = total_temperature * 0.95;
    let static_pressure = total_pressure * 0.95;
    let density = eos.density(static_pressure, static_temperature)?;
    let speed = (2.0 * cp * (total_temperature - static_temperature))
        .max(0.0)
        .sqrt();
    let dir = normalize(velocity_direction)?;
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

/// 出口 ghost：超声速出口（`supersonic`）零梯度外推 owner 全部变量；
/// 亚声速出口替换 ghost 压力为 `static_pressure`，其余零梯度（复制 owner）。
pub fn outlet_ghost(
    owner: &ConservedState,
    static_pressure: Real,
    supersonic: bool,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<GhostCellState> {
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, min_pressure)?;
    let ghost_prim = if supersonic {
        prim
    } else {
        PrimitiveState {
            pressure: static_pressure,
            ..prim
        }
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
            ) => outlet_ghost(&owner, *static_pressure, *supersonic, fs_ctx.eos, p_floor)?,
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
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::core::Vector3;
    use crate::mesh::{BoundaryMesh, StructuredMesh3d};

    #[test]
    fn isothermal_wall_ghost_temperature_uses_wall_value() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = crate::physics::ViscousPhysicsConfig::default();
        let t_owner = 400.0;
        let t_wall = 300.0;
        let spacing = 0.25;
        let t_ghost = crate::discretization::wall_ghost_temperature(
            t_owner,
            WallHeat::Isothermal {
                temperature: t_wall,
            },
            spacing,
            Some(&viscous),
            &eos,
        )
        .expect("t_ghost");
        assert!((t_ghost - t_wall).abs() < 1.0e-10);
    }

    #[test]
    fn wall_no_slip_ghost_velocity_negates_owner() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let params = FreestreamParams {
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(1, &eos, &params).expect("fields");
        let owner = fields.cell_state(0).expect("cell");
        let owner_prim = crate::field::primitive_from_conserved(&eos, &owner).expect("owner prim");
        let geom = FaceGeometry3d {
            normal: Vector3::new(-1.0, 0.0, 0.0),
            spacing: 0.5,
            area: 1.0,
            center: Vector3::new(0.0, 0.0, 0.0),
        };
        let p_floor = crate::field::positivity_pressure_floor(params.pressure);
        let fs_ctx = FreestreamContext::dimensional(&eos);
        let ghost = wall_ghost(
            &owner,
            &geom,
            true,
            WallHeat::Adiabatic,
            &fs_ctx,
            p_floor,
            None,
        )
        .expect("ghost");
        let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
        for (g, o) in prim.velocity.iter().zip(owner_prim.velocity.iter()) {
            assert!(
                (g + o).abs() < 1.0e-10,
                "u_g should be -u_o, got {g} vs {o}"
            );
        }
        let u_face = [
            0.5 * (owner_prim.velocity[0] + prim.velocity[0]),
            0.5 * (owner_prim.velocity[1] + prim.velocity[1]),
            0.5 * (owner_prim.velocity[2] + prim.velocity[2]),
        ];
        assert!(u_face.iter().all(|&v| v.abs() < 1.0e-10));
    }

    #[test]
    fn wall_slip_ghost_mirrors_normal_preserves_tangential_at_face() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let p = 101_325.0;
        let t = 300.0;
        let rho = eos.density(p, t).expect("rho");
        let u_owner = [120.0, 45.0, 10.0];
        let prim = PrimitiveState {
            density: rho,
            velocity: u_owner,
            pressure: p,
            temperature: t,
        };
        let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
        let normal = Vector3::new(-1.0, 0.0, 0.0);
        let geom = FaceGeometry3d {
            normal,
            spacing: 0.5,
            area: 1.0,
            center: Vector3::new(0.0, 0.0, 0.0),
        };
        let fs_ctx = FreestreamContext::dimensional(&eos);
        let ghost = wall_ghost(
            &owner,
            &geom,
            false,
            WallHeat::Adiabatic,
            &fs_ctx,
            1.0e-6,
            None,
        )
        .expect("ghost");
        let u_g = crate::field::primitive_from_conserved(&eos, &ghost.conserved)
            .expect("ghost prim")
            .velocity;
        let u_f = [
            0.5 * (u_owner[0] + u_g[0]),
            0.5 * (u_owner[1] + u_g[1]),
            0.5 * (u_owner[2] + u_g[2]),
        ];
        let un_face = u_f[0] * normal.x + u_f[1] * normal.y + u_f[2] * normal.z;
        assert!(
            un_face.abs() < 1.0e-10,
            "slip wall face normal velocity should be 0"
        );
        let un_o = u_owner[0] * normal.x + u_owner[1] * normal.y + u_owner[2] * normal.z;
        let u_t_o = [
            u_owner[0] - un_o * normal.x,
            u_owner[1] - un_o * normal.y,
            u_owner[2] - un_o * normal.z,
        ];
        let u_t_f = [
            u_f[0] - un_face * normal.x,
            u_f[1] - un_face * normal.y,
            u_f[2] - un_face * normal.z,
        ];
        for i in 0..3 {
            assert!(
                (u_t_f[i] - u_t_o[i]).abs() < 1.0e-10,
                "tangential at face should match owner, component {i}"
            );
        }
    }

    #[test]
    fn supersonic_inlet_ghost_uses_freestream_static_state() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 8.0,
            pressure: 714.0,
            temperature: 139.0,
            ..FreestreamParams::default()
        };
        let owner = ConservedFields::from_freestream(1, &eos, &fs)
            .expect("fields")
            .cell_state(0)
            .expect("cell");
        let geom = FaceGeometry3d {
            normal: Vector3::new(1.0, 0.0, 0.0),
            spacing: 0.5,
            area: 1.0,
            center: Vector3::new(0.0, 0.0, 0.0),
        };
        let fs_ctx = FreestreamContext::dimensional(&eos);
        let ghost = inlet_ghost(
            &owner,
            &geom,
            &InletGhostParams {
                supersonic: true,
                velocity_direction: [1.0, 0.0, 0.0],
                freestream: &fs,
                fs_ctx: &fs_ctx,
                total_pressure: 1.0e9,
                total_temperature: 1.0e4,
            },
        )
        .expect("ghost");
        let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
        let ref_prim = fs_ctx.primitive(&fs).expect("ref");
        assert!((prim.density - ref_prim.density).abs() / ref_prim.density < 1.0e-6);
    }

    #[test]
    fn subsonic_inlet_ghost_ignores_high_mach_freestream() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 8.0,
            pressure: 714.0,
            temperature: 139.0,
            ..FreestreamParams::default()
        };
        let owner = ConservedFields::from_freestream(1, &eos, &fs)
            .expect("fields")
            .cell_state(0)
            .expect("cell");
        let geom = FaceGeometry3d {
            normal: Vector3::new(1.0, 0.0, 0.0),
            spacing: 0.5,
            area: 1.0,
            center: Vector3::new(0.0, 0.0, 0.0),
        };
        let fs_ctx = FreestreamContext::dimensional(&eos);
        let ghost = inlet_ghost(
            &owner,
            &geom,
            &InletGhostParams {
                supersonic: false,
                velocity_direction: [1.0, 0.0, 0.0],
                freestream: &fs,
                fs_ctx: &fs_ctx,
                total_pressure: 200_000.0,
                total_temperature: 300.0,
            },
        )
        .expect("ghost");
        let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
        let ref_prim = eos
            .freestream_primitive(fs.mach, fs.pressure, fs.temperature, [1.0, 0.0, 0.0])
            .expect("ref");
        assert!(prim.density > ref_prim.density * 10.0);
    }

    #[test]
    fn supersonic_outlet_ghost_extrapolates_owner_state() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(3.0, 25_000.0, 280.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
        let ghost = outlet_ghost(&owner, 101_325.0, true, &eos, 1.0e-6).expect("ghost");
        let ghost_prim =
            crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("ghost prim");
        assert!((ghost_prim.pressure - prim.pressure).abs() < 1.0e-8);
        assert!((ghost_prim.density - prim.density).abs() < 1.0e-10);
        for i in 0..3 {
            assert!((ghost_prim.velocity[i] - prim.velocity[i]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn subsonic_outlet_ghost_sets_static_pressure() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(0.3, 90_000.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
        let ghost = outlet_ghost(&owner, 101_325.0, false, &eos, 1.0e-6).expect("ghost");
        let ghost_prim =
            crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("ghost prim");
        assert!((ghost_prim.pressure - 101_325.0).abs() < 1.0e-8);
        assert!((ghost_prim.velocity[0] - prim.velocity[0]).abs() < 1.0e-10);
    }

    #[test]
    fn apply_farfield_patch() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let faces = mesh.resolve_logical_boundary("i_max").expect("faces");
        let first_face = faces[0];
        let patches = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: 0.3,
                pressure: 101_325.0,
                temperature: 288.15,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let mut ghosts = BoundaryGhostBuffer::new();
        let fs_ctx = FreestreamContext::dimensional(&eos);
        apply_compressible_boundary_conditions(
            &mesh,
            &patches,
            &fields,
            &mut ghosts,
            &fs_ctx,
            &fs,
            None,
        )
        .expect("bc");
        assert!(ghosts.get_face(first_face).is_some());
    }
}
