//! 可压缩流边界条件 ghost 单元施加。

use crate::boundary::{
    BcHandler, BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, WallHeat,
};
use crate::core::Real;
use crate::discretization::wall_thermal::wall_ghost_temperature;
use crate::error::Result;
use crate::field::ConservedFields;
use crate::mesh::{BoundaryMesh3d, FaceGeometry3d};
use crate::physics::{
    ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState, ViscousPhysicsConfig,
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

/// 壁面 ghost：反射法向动量，无滑移置零切向速度；热 BC 设置 ghost 温度（NS）。
pub fn wall_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    no_slip: bool,
    heat: WallHeat,
    eos: &IdealGasEoS,
    min_pressure: Real,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<GhostCellState> {
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, min_pressure)?;
    let n = geom.normal;
    let un = prim.velocity[0] * n.x + prim.velocity[1] * n.y + prim.velocity[2] * n.z;
    let mut velocity = prim.velocity;
    velocity[0] -= 2.0 * un * n.x;
    velocity[1] -= 2.0 * un * n.y;
    velocity[2] -= 2.0 * un * n.z;
    if no_slip {
        velocity = [0.0, 0.0, 0.0];
    }
    let t_owner = prim.pressure / (prim.density.max(1.0e-30) * eos.gas_constant);
    let t_ghost = wall_ghost_temperature(t_owner, heat, geom.spacing, viscous, eos)?;
    let ghost_prim = PrimitiveState {
        density: prim.pressure / (eos.gas_constant * t_ghost.max(1.0e-30)),
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
    eos: &IdealGasEoS,
) -> Result<GhostCellState> {
    let _ = (owner, geom);
    let prim = eos.freestream_primitive(
        params.mach,
        params.pressure,
        params.temperature,
        params.effective_direction(),
    )?;
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(eos, &prim)?,
    })
}

/// 入口 ghost：超声速入口（`supersonic`）直接使用 `[freestream]` 静参数；亚声速用总压/总温简化模型。
#[allow(clippy::too_many_arguments)]
pub fn inlet_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    supersonic: bool,
    velocity_direction: [Real; 3],
    case_freestream: &FreestreamParams,
    eos: &IdealGasEoS,
    total_pressure: Real,
    total_temperature: Real,
) -> Result<GhostCellState> {
    if supersonic {
        let fs = FreestreamParams {
            mach: case_freestream.mach,
            pressure: case_freestream.pressure,
            temperature: case_freestream.temperature,
            alpha: case_freestream.alpha,
            beta: case_freestream.beta,
            velocity_direction,
        };
        return farfield_ghost(owner, geom, &fs, eos);
    }
    subsonic_inlet_ghost(total_pressure, total_temperature, velocity_direction, eos)
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

/// 出口 ghost：外推 owner 压力，速度零梯度（简化为复制 owner）。
pub fn outlet_ghost(
    owner: &ConservedState,
    static_pressure: Real,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<GhostCellState> {
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, min_pressure)?;
    let ghost_prim = PrimitiveState {
        pressure: static_pressure,
        ..prim
    };
    Ok(GhostCellState {
        conserved: ConservedState::from_primitive(eos, &ghost_prim)?,
    })
}

/// 对称 ghost：法向速度翻转。
pub fn symmetry_ghost(
    owner: &ConservedState,
    geom: &FaceGeometry3d,
    eos: &IdealGasEoS,
    min_pressure: Real,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<GhostCellState> {
    wall_ghost(
        owner,
        geom,
        false,
        WallHeat::Adiabatic,
        eos,
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
    eos: &IdealGasEoS,
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
                wall_ghost(&owner, &geom, *no_slip, *heat, eos, p_floor, viscous)?
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
                eos,
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
                *supersonic,
                *velocity_direction,
                freestream,
                eos,
                *total_pressure,
                *total_temperature,
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
                false,
                *velocity_direction,
                freestream,
                eos,
                *total_pressure,
                *total_temperature,
            )?,
            (
                BcHandler::Outlet,
                BoundaryKind::Outlet {
                    static_pressure, ..
                },
            ) => outlet_ghost(&owner, *static_pressure, eos, p_floor)?,
            (BcHandler::Symmetry, BoundaryKind::Symmetry) => {
                symmetry_ghost(&owner, &geom, eos, p_floor, viscous)?
            }
            (BcHandler::Periodic, BoundaryKind::Periodic { .. }) => {
                GhostCellState { conserved: owner }
            }
            _ => farfield_ghost(&owner, &geom, freestream, eos)?,
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
    eos: &IdealGasEoS,
    freestream: &FreestreamParams,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<()> {
    BoundaryRegistry::validate_patches(patches.patches())?;
    for patch in patches.patches() {
        apply_patch_compressible(mesh, patch, fields, ghosts, eos, freestream, viscous)?;
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
    fn isothermal_wall_ghost_temperature_matches_face_value() {
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
        let t_face = 0.5 * (t_owner + t_ghost);
        assert!((t_face - t_wall).abs() < 1.0e-10);
    }

    #[test]
    fn wall_no_slip_zeros_velocity() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let params = FreestreamParams {
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(1, &eos, &params).expect("fields");
        let owner = fields.cell_state(0).expect("cell");
        let geom = FaceGeometry3d {
            normal: Vector3::new(-1.0, 0.0, 0.0),
            spacing: 0.5,
            area: 1.0,
        };
        let p_floor = crate::field::positivity_pressure_floor(params.pressure);
        let ghost = wall_ghost(
            &owner,
            &geom,
            true,
            WallHeat::Adiabatic,
            &eos,
            p_floor,
            None,
        )
        .expect("ghost");
        let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
        assert!(prim.velocity.iter().all(|&v| v.abs() < 1.0e-12));
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
        };
        let ghost = inlet_ghost(
            &owner,
            &geom,
            true,
            [1.0, 0.0, 0.0],
            &fs,
            &eos,
            1.0e9,
            1.0e4,
        )
        .expect("ghost");
        let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
        let ref_prim = eos
            .freestream_primitive(fs.mach, fs.pressure, fs.temperature, [1.0, 0.0, 0.0])
            .expect("ref");
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
        };
        let ghost = inlet_ghost(
            &owner,
            &geom,
            false,
            [1.0, 0.0, 0.0],
            &fs,
            &eos,
            200_000.0,
            300.0,
        )
        .expect("ghost");
        let prim = crate::field::primitive_from_conserved(&eos, &ghost.conserved).expect("prim");
        let ref_prim = eos
            .freestream_primitive(fs.mach, fs.pressure, fs.temperature, [1.0, 0.0, 0.0])
            .expect("ref");
        assert!(prim.density > ref_prim.density * 10.0);
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
        apply_compressible_boundary_conditions(
            &mesh,
            &patches,
            &fields,
            &mut ghosts,
            &eos,
            &fs,
            None,
        )
        .expect("bc");
        assert!(ghosts.get_face(first_face).is_some());
    }
}
