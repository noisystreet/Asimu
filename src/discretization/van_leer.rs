//! Van Leer / Hanel–Van Leer 通量矢量分裂（FVS）。
//!
//! - Van Leer (1982): \(\hat{\mathbf{F}} = \mathbf{F}^+(\mathbf{U}_L) + \mathbf{F}^-(\mathbf{U}_R)\)
//! - Hanel: 质量/动量分裂同 Van Leer，亚音速能量取 \(F_E^+ = F_m^+ \cdot h\)

use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::physics::{ConservedState, IdealGasEoS};

use super::flux_common::{face_tangent_basis, normalize_face_normal};
use super::inviscid::InviscidFlux;

/// Van Leer FVS 数值通量 \(\hat{\mathbf{F}} \cdot \mathbf{n}\)（理想气体 Euler）。
pub fn van_leer_flux(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<InviscidFlux> {
    fvs_flux(left, right, normal, eos, EnergyFluxSplit::VanLeer)
}

/// Hanel 修正 Van Leer：亚音速 \(F_E^+ = F_m^+ \cdot h\)（定常 Euler 总焓更守恒）。
pub fn hanel_van_leer_flux(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<InviscidFlux> {
    fvs_flux(left, right, normal, eos, EnergyFluxSplit::Hanel)
}

#[derive(Clone, Copy)]
enum EnergyFluxSplit {
    VanLeer,
    Hanel,
}

fn fvs_flux(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    energy_split: EnergyFluxSplit,
) -> Result<InviscidFlux> {
    let n = normalize_face_normal(normal)?;
    let (t1, t2) = face_tangent_basis(n);
    let frame_l = face_frame_from_conserved(left, eos.gamma, n, t1, t2)?;
    let frame_r = face_frame_from_conserved(right, eos.gamma, n, t1, t2)?;
    validate_face_state(&frame_l)?;
    validate_face_state(&frame_r)?;
    let flux_l_plus = fvs_positive_flux(&frame_l, eos.gamma, energy_split);
    let flux_r_minus = fvs_negative_flux(&frame_r, eos.gamma, energy_split);
    let face_flux = add_face_fluxes(flux_l_plus, flux_r_minus);
    Ok(to_global_flux(face_flux, n, t1, t2))
}

#[derive(Clone, Copy)]
struct FaceFrameState {
    rho: Real,
    un: Real,
    ut: [Real; 2],
    p: Real,
    rho_e: Real,
}

#[derive(Clone, Copy)]
struct FaceFrameFlux {
    mass: Real,
    normal_momentum: Real,
    tangential_momentum: [Real; 2],
    energy: Real,
}

fn face_frame_from_conserved(
    cons: &ConservedState,
    gamma: Real,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Result<FaceFrameState> {
    let rho = cons.density;
    if rho <= Real::EPSILON {
        return Err(AsimuError::Field("Van Leer 状态须为正密度".to_string()));
    }
    let inv_rho = 1.0 / rho;
    let ux = cons.momentum[0] * inv_rho;
    let uy = cons.momentum[1] * inv_rho;
    let uz = cons.momentum[2] * inv_rho;
    let ke = 0.5 * rho * (ux * ux + uy * uy + uz * uz);
    // RK 中间态可能略非物理解：压力下限后同步总能，保证通量用的 (p, ρE) 自洽。
    let pressure = ((gamma - 1.0) * (cons.total_energy - ke)).max(1.0e-6);
    let internal = pressure / (gamma - 1.0);
    let rho_e = ke + internal;
    Ok(FaceFrameState {
        rho,
        un: ux * normal.x + uy * normal.y + uz * normal.z,
        ut: [
            ux * t1.x + uy * t1.y + uz * t1.z,
            ux * t2.x + uy * t2.y + uz * t2.z,
        ],
        p: pressure,
        rho_e,
    })
}

fn validate_face_state(state: &FaceFrameState) -> Result<()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(AsimuError::Field(
            "Van Leer 状态须为正密度与压力".to_string(),
        ));
    }
    Ok(())
}

fn sound_speed(rho: Real, pressure: Real, gamma: Real) -> Real {
    (gamma * pressure / rho).sqrt()
}

fn specific_enthalpy(state: &FaceFrameState, gamma: Real) -> Real {
    let a = sound_speed(state.rho, state.p, gamma);
    a * a / (gamma - 1.0)
        + 0.5 * (state.un * state.un + state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1])
}

fn physical_face_flux(state: &FaceFrameState) -> FaceFrameFlux {
    FaceFrameFlux {
        mass: state.rho * state.un,
        normal_momentum: state.rho * state.un * state.un + state.p,
        tangential_momentum: [
            state.rho * state.un * state.ut[0],
            state.rho * state.un * state.ut[1],
        ],
        energy: (state.rho_e + state.p) * state.un,
    }
}

/// Van Leer / Hanel \(\mathbf{F}^+\)（Blazek §4.3.1；Hanel 仅改亚音速能量分裂）。
fn fvs_positive_flux(
    state: &FaceFrameState,
    gamma: Real,
    energy_split: EnergyFluxSplit,
) -> FaceFrameFlux {
    let full = physical_face_flux(state);
    let a = sound_speed(state.rho, state.p, gamma);
    let mach = state.un / a;
    if mach <= -1.0 {
        return FaceFrameFlux {
            mass: 0.0,
            normal_momentum: 0.0,
            tangential_momentum: [0.0, 0.0],
            energy: 0.0,
        };
    }
    if mach >= 1.0 {
        return full;
    }
    let mach_plus = mach + 1.0;
    let mass_plus = 0.25 * state.rho * a * mach_plus * mach_plus;
    let normal_velocity_plus = ((gamma - 1.0) * state.un + 2.0 * a) / gamma;
    let tangential_ke = 0.5 * (state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1]);
    let energy = match energy_split {
        EnergyFluxSplit::VanLeer => {
            let acoustic_energy =
                ((gamma - 1.0) * state.un + 2.0 * a).powi(2) / (2.0 * (gamma * gamma - 1.0));
            mass_plus * (acoustic_energy + tangential_ke)
        }
        EnergyFluxSplit::Hanel => mass_plus * specific_enthalpy(state, gamma),
    };
    FaceFrameFlux {
        mass: mass_plus,
        normal_momentum: mass_plus * normal_velocity_plus,
        tangential_momentum: [mass_plus * state.ut[0], mass_plus * state.ut[1]],
        energy,
    }
}

fn fvs_negative_flux(
    state: &FaceFrameState,
    gamma: Real,
    energy_split: EnergyFluxSplit,
) -> FaceFrameFlux {
    let full = physical_face_flux(state);
    let plus = fvs_positive_flux(state, gamma, energy_split);
    FaceFrameFlux {
        mass: full.mass - plus.mass,
        normal_momentum: full.normal_momentum - plus.normal_momentum,
        tangential_momentum: [
            full.tangential_momentum[0] - plus.tangential_momentum[0],
            full.tangential_momentum[1] - plus.tangential_momentum[1],
        ],
        energy: full.energy - plus.energy,
    }
}

fn add_face_fluxes(left: FaceFrameFlux, right: FaceFrameFlux) -> FaceFrameFlux {
    FaceFrameFlux {
        mass: left.mass + right.mass,
        normal_momentum: left.normal_momentum + right.normal_momentum,
        tangential_momentum: [
            left.tangential_momentum[0] + right.tangential_momentum[0],
            left.tangential_momentum[1] + right.tangential_momentum[1],
        ],
        energy: left.energy + right.energy,
    }
}

fn to_global_flux(face: FaceFrameFlux, normal: Vector3, t1: Vector3, t2: Vector3) -> InviscidFlux {
    InviscidFlux {
        mass: face.mass,
        momentum: [
            face.normal_momentum * normal.x
                + face.tangential_momentum[0] * t1.x
                + face.tangential_momentum[1] * t2.x,
            face.normal_momentum * normal.y
                + face.tangential_momentum[0] * t1.y
                + face.tangential_momentum[1] * t2.y,
            face.normal_momentum * normal.z
                + face.tangential_momentum[0] * t1.z
                + face.tangential_momentum[1] * t2.z,
        ],
        energy: face.energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::inviscid::physical_inviscid_flux;
    use crate::physics::PrimitiveState;

    #[test]
    fn split_recombines_to_physical_flux() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let prim = eos
            .freestream_primitive(0.8, 1.0, 1.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let n = Vector3::new(0.6, 0.8, 0.0);
        let nn = normalize_face_normal(n).expect("n");
        let (t1, t2) = face_tangent_basis(nn);
        let frame = face_frame_from_conserved(&cons, eos.gamma, nn, t1, t2).expect("frame");
        let plus = fvs_positive_flux(&frame, eos.gamma, EnergyFluxSplit::VanLeer);
        let minus = fvs_negative_flux(&frame, eos.gamma, EnergyFluxSplit::VanLeer);
        let split = add_face_fluxes(plus, minus);
        let phys = physical_face_flux(&frame);
        assert!(approx_eq(split.mass, phys.mass, 1.0e-12));
        assert!(approx_eq(
            split.normal_momentum,
            phys.normal_momentum,
            1.0e-10
        ));
        assert!(approx_eq(
            split.tangential_momentum[0],
            phys.tangential_momentum[0],
            1.0e-10
        ));
        assert!(approx_eq(split.energy, phys.energy, 1.0e-10));
    }

    #[test]
    fn van_leer_subsonic_positive_flux_matches_reference_formula() {
        let gamma = 1.4;
        let state = FaceFrameState {
            rho: 1.2,
            un: 0.4,
            ut: [0.3, -0.2],
            p: 1.1,
            rho_e: 3.0,
        };
        let a = sound_speed(state.rho, state.p, gamma);
        let mach = state.un / a;
        let mass = 0.25 * state.rho * a * (mach + 1.0).powi(2);
        let normal_velocity = ((gamma - 1.0) * state.un + 2.0 * a) / gamma;
        let tangential_ke = 0.5 * (state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1]);
        let acoustic_energy =
            ((gamma - 1.0) * state.un + 2.0 * a).powi(2) / (2.0 * (gamma * gamma - 1.0));
        let flux = fvs_positive_flux(&state, gamma, EnergyFluxSplit::VanLeer);

        assert!(approx_eq(flux.mass, mass, 1.0e-12));
        assert!(approx_eq(
            flux.normal_momentum,
            mass * normal_velocity,
            1.0e-12
        ));
        assert!(approx_eq(
            flux.tangential_momentum[0],
            mass * state.ut[0],
            1.0e-12
        ));
        assert!(approx_eq(
            flux.energy,
            mass * (acoustic_energy + tangential_ke),
            1.0e-12
        ));
    }

    #[test]
    fn identical_states_match_physical_flux() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let prim = eos
            .freestream_primitive(0.3, 1.0, 1.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let n = Vector3::new(1.0, 0.0, 0.0);
        let fvs = van_leer_flux(&cons, &cons, n, &eos).expect("van leer");
        let phys = physical_inviscid_flux(&cons, &prim, n);
        assert!(approx_eq(fvs.mass, phys.mass, 1.0e-10));
        assert!(approx_eq(fvs.momentum[0], phys.momentum[0], 1.0e-10));
        assert!(approx_eq(fvs.energy, phys.energy, 1.0e-10));
    }

    #[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
    #[test]
    fn cylinder_van_leer_rk4_step2_when_present() {
        use std::path::PathBuf;

        use crate::discretization::BoundaryGhostBuffer;
        use crate::io::{CaseMesh, load_case};
        use crate::solver::{
            CompressibleAdvanceContext3d, CompressibleEulerConfig, CompressibleEulerSolver,
            Rk4Storage, RungeKutta4Integrator, SolverState,
        };

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        let case = load_case(&case_path).expect("case");
        let CaseMesh::Structured3d(mesh) = &case.mesh else {
            panic!("3d");
        };
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("fs");
        let mut fields = case.build_conserved_fields().expect("fields");
        let mut ghosts = BoundaryGhostBuffer::new();
        let inviscid = case.euler.as_ref().expect("euler").inviscid();
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: crate::solver::RungeKutta4Config {
                dt: 0.0,
                max_steps: 10,
            },
            inviscid,
            cfl_schedule: case.cfl_schedule().expect("cfl"),
            local_time_step: true,
            time_mode: crate::solver::CompressibleTimeMode::Steady,
            ..CompressibleEulerConfig::default()
        });
        let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
        let mut state = SolverState::default();
        let mut integrator = RungeKutta4Integrator::new(solver.config.time);
        let mut ctx = CompressibleAdvanceContext3d {
            mesh,
            structured: mesh,
            patches: &case.boundary,
            ghosts: &mut ghosts,
            eos: &eos,
            freestream: &fs,
            reference: case.reference.as_ref(),
            primitive_scratch: crate::field::PrimitiveFields::zeros(mesh.num_cells())
                .expect("primitives"),
            gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
                .expect("gradients"),
            viscous: None,
            residual_correction: None,
        };
        for _ in 0..10 {
            solver
                .advance_step_3d(
                    &mut ctx,
                    &mut fields,
                    &mut storage,
                    &mut state,
                    &mut integrator,
                )
                .expect("advance");
        }
    }

    #[test]
    fn hanel_split_recombines_to_physical_flux() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let prim = eos
            .freestream_primitive(0.8, 1.0, 1.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let n = Vector3::new(0.6, 0.8, 0.0);
        let nn = normalize_face_normal(n).expect("n");
        let (t1, t2) = face_tangent_basis(nn);
        let frame = face_frame_from_conserved(&cons, eos.gamma, nn, t1, t2).expect("frame");
        let plus = fvs_positive_flux(&frame, eos.gamma, EnergyFluxSplit::Hanel);
        let minus = fvs_negative_flux(&frame, eos.gamma, EnergyFluxSplit::Hanel);
        let split = add_face_fluxes(plus, minus);
        let phys = physical_face_flux(&frame);
        assert!(approx_eq(split.mass, phys.mass, 1.0e-12));
        assert!(approx_eq(
            split.normal_momentum,
            phys.normal_momentum,
            1.0e-10
        ));
        assert!(approx_eq(split.energy, phys.energy, 1.0e-10));
    }

    #[test]
    fn hanel_subsonic_energy_differs_from_van_leer() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 1.0,
                velocity: [0.3, 0.0, 0.0],
                pressure: 1.0,
                temperature: 1.0,
            },
        )
        .expect("left");
        let right = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 0.8,
                velocity: [0.2, 0.0, 0.0],
                pressure: 0.9,
                temperature: 1.0,
            },
        )
        .expect("right");
        let n = Vector3::new(1.0, 0.0, 0.0);
        let vl = van_leer_flux(&left, &right, n, &eos).expect("vl");
        let hanel = hanel_van_leer_flux(&left, &right, n, &eos).expect("hanel");
        assert!(approx_eq(vl.mass, hanel.mass, 1.0e-12));
        assert!((vl.energy - hanel.energy).abs() > 1.0e-10);
    }

    #[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
    #[test]
    fn cylinder_hanel_van_leer_rk4_step10_when_present() {
        use std::path::PathBuf;

        use crate::discretization::BoundaryGhostBuffer;
        use crate::io::{CaseMesh, load_case};
        use crate::solver::{
            CompressibleAdvanceContext3d, CompressibleEulerConfig, CompressibleEulerSolver,
            Rk4Storage, RungeKutta4Integrator, SolverState,
        };

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        let mut case = load_case(&case_path).expect("case");
        case.euler.as_mut().expect("euler").flux = Some("hanel_van_leer".to_string());
        let CaseMesh::Structured3d(mesh) = &case.mesh else {
            panic!("3d");
        };
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("fs");
        let inviscid = case.euler.as_ref().expect("euler").inviscid();
        let mut fields = case.build_conserved_fields().expect("fields");
        let mut ghosts = BoundaryGhostBuffer::new();
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: crate::solver::RungeKutta4Config {
                dt: 0.0,
                max_steps: 10,
            },
            inviscid,
            cfl_schedule: case.cfl_schedule().expect("cfl"),
            local_time_step: true,
            time_mode: crate::solver::CompressibleTimeMode::Steady,
            ..CompressibleEulerConfig::default()
        });
        let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
        let mut state = SolverState::default();
        let mut integrator = RungeKutta4Integrator::new(solver.config.time);
        let mut ctx = CompressibleAdvanceContext3d {
            mesh,
            structured: mesh,
            patches: &case.boundary,
            ghosts: &mut ghosts,
            eos: &eos,
            freestream: &fs,
            reference: case.reference.as_ref(),
            primitive_scratch: crate::field::PrimitiveFields::zeros(mesh.num_cells())
                .expect("primitives"),
            gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
                .expect("gradients"),
            viscous: None,
            residual_correction: None,
        };
        for _ in 0..10 {
            solver
                .advance_step_3d(
                    &mut ctx,
                    &mut fields,
                    &mut storage,
                    &mut state,
                    &mut integrator,
                )
                .expect("advance");
        }
    }

    #[test]
    fn supersonic_uniform_flow_has_nonzero_mass_flux() {
        let eos = IdealGasEoS::new(1.4, 287.0).expect("eos");
        let prim_in = eos
            .freestream_primitive(8.0, 1000.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons_in = ConservedState::from_primitive(&eos, &prim_in).expect("cons");
        // 法向入流：un < 0
        let n = Vector3::new(-1.0, 0.0, 0.0);
        let flux_in = van_leer_flux(&cons_in, &cons_in, n, &eos).expect("flux");
        assert!(flux_in.mass.abs() > 0.0);
    }

    #[test]
    fn supersonic_wall_slip_has_zero_mass_flux() {
        let eos = IdealGasEoS::new(1.4, 287.0).expect("eos");
        let prim_in = eos
            .freestream_primitive(8.0, 1000.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons_in = ConservedState::from_primitive(&eos, &prim_in).expect("cons");
        let n = Vector3::new(-1.0, 0.0, 0.0);
        let prim = crate::field::primitive_from_conserved(&eos, &cons_in).expect("prim");
        let un = prim.velocity[0] * n.x;
        let mut ghost_v = prim.velocity;
        ghost_v[0] -= 2.0 * un * n.x;
        let ghost = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                velocity: ghost_v,
                ..prim
            },
        )
        .expect("ghost");
        let flux_wall = hanel_van_leer_flux(&cons_in, &ghost, n, &eos).expect("wall");
        assert!(flux_wall.mass.abs() < 1.0e-6);
    }

    #[test]
    fn sod_interface_flux_is_finite() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 1.0,
                velocity: [0.0, 0.0, 0.0],
                pressure: 1.0,
                temperature: 1.0,
            },
        )
        .expect("left");
        let right = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 0.125,
                velocity: [0.0, 0.0, 0.0],
                pressure: 0.1,
                temperature: 1.0,
            },
        )
        .expect("right");
        let flux = van_leer_flux(&left, &right, Vector3::new(1.0, 0.0, 0.0), &eos).expect("flux");
        assert!(flux.mass.is_finite());
        assert!(flux.energy.is_finite());
    }
}
