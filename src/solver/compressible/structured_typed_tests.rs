//! 结构化 typed 时间推进回归测试。

use super::*;
use crate::boundary::BoundarySet;
use crate::core::approx_eq;
use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
use crate::field::PrimitiveFields;
use crate::mesh::StructuredMesh3d;
use crate::physics::FreestreamParams;
use crate::solver::compressible::{CompressibleAdvanceContext3d, CompressibleEulerConfig};
use crate::solver::time::Rk4Storage;

fn freestream_box_context<T: crate::core::ComputeFloat>(
    side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
) -> (
    StructuredMesh3d,
    BoundarySet,
    crate::field::ConservedFieldsT<T>,
    crate::discretization::BoundaryGhostBuffer,
    crate::physics::IdealGasEoS,
    FreestreamParams,
) {
    let (mesh, boundary, fields, ghosts) = uniform_farfield_box(3, 3, 3, 1.0, 1.0, 1.0, side);
    let fields_t =
        crate::field::ConservedFieldsT::<T>::from_real_fields(&fields).expect("typed fields");
    (mesh, boundary, fields_t, ghosts, *side.eos, *side.fs)
}

#[test]
fn f32_explicit_step_matches_f64_on_uniform_box() {
    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: crate::solver::time::RungeKutta4Config {
            dt: 1.0e-4,
            max_steps: 1,
        },
        ..CompressibleEulerConfig::default()
    });
    let (mesh, patches, fields_f32, mut ghosts_f32, eos, freestream) =
        freestream_box_context::<f32>(&side);
    let (_, _, fields_f64, ghosts_f64, _, _) = freestream_box_context::<f64>(&side);
    let mut ghosts_f64 = ghosts_f64;
    let mut ctx_f32 = CompressibleAdvanceContext3dTyped {
        mesh: &mesh,
        structured: &mesh,
        patches: &patches,
        ghosts: &mut ghosts_f32,
        eos: &eos,
        freestream: &freestream,
        reference: None,
        primitive_scratch: crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells())
            .expect("prim f32"),
        spectral_primitives: PrimitiveFields::zeros(mesh.num_cells()).expect("prim f64"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("grad"),
        viscous: None,
        interface_residual: None,
    };
    let mut ctx_f64 = CompressibleAdvanceContext3d {
        mesh: &mesh,
        structured: &mesh,
        patches: &patches,
        ghosts: &mut ghosts_f64,
        eos: &eos,
        freestream: &freestream,
        reference: None,
        primitive_scratch: PrimitiveFields::zeros(mesh.num_cells()).expect("prim"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("grad"),
        viscous: None,
        residual_correction: None,
    };
    let mut fields_f32 = fields_f32;
    let mut fields_f64 = fields_f64;
    let mut storage_f32 = Rk4StorageT::<f32>::new(mesh.num_cells()).expect("storage f32");
    let mut storage_f64 = Rk4Storage::new(mesh.num_cells()).expect("storage f64");
    let mut state_f32 = SolverState::default();
    let mut state_f64 = SolverState::default();
    let mut integrator_f32 = RungeKutta4Integrator::new(solver.config.time);
    let mut integrator_f64 = RungeKutta4Integrator::new(solver.config.time);
    let info_f32 = solver
        .advance_step_3d_typed(
            &mut ctx_f32,
            &mut fields_f32,
            &mut storage_f32,
            &mut state_f32,
            &mut integrator_f32,
        )
        .expect("f32 step");
    let info_f64 = solver
        .advance_step_3d(
            &mut ctx_f64,
            &mut fields_f64,
            &mut storage_f64,
            &mut state_f64,
            &mut integrator_f64,
        )
        .expect("f64 step");
    assert!(approx_eq(
        info_f32.residual_rms,
        info_f64.residual_rms,
        1.0e-5
    ));
    for i in 0..mesh.num_cells() {
        let rho_f32 = fields_f32.density.values()[i].to_real();
        let rho_f64 = fields_f64.density.values()[i];
        let rel = (rho_f32 - rho_f64).abs() / rho_f64.max(1.0e-12);
        assert!(rel < 1.0e-3, "cell {i} rel={rel}");
    }
}

#[test]
fn f32_lusgs_step_on_uniform_box() {
    use crate::solver::time::{CflSchedule, TimeIntegrationScheme};
    use crate::solver::{CompressibleTimeMode, SolverState};

    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        cfl_schedule: CflSchedule::constant(0.1),
        time_mode: CompressibleTimeMode::Steady,
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::LuSgs,
        time: crate::solver::time::RungeKutta4Config {
            dt: 0.0,
            max_steps: 1,
        },
        ..CompressibleEulerConfig::default()
    });
    let (mesh, patches, mut fields, mut ghosts, eos, freestream) =
        freestream_box_context::<f32>(&side);
    let mut ctx = CompressibleAdvanceContext3dTyped {
        mesh: &mesh,
        structured: &mesh,
        patches: &patches,
        ghosts: &mut ghosts,
        eos: &eos,
        freestream: &freestream,
        reference: None,
        primitive_scratch: crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells())
            .expect("prim"),
        spectral_primitives: PrimitiveFields::zeros(mesh.num_cells()).expect("spec"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("grad"),
        viscous: None,
        interface_residual: None,
    };
    let mut storage = Rk4StorageT::<f32>::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let mut integrator = RungeKutta4Integrator::new(solver.config.time);
    let info = solver
        .advance_step_3d_typed(
            &mut ctx,
            &mut fields,
            &mut storage,
            &mut state,
            &mut integrator,
        )
        .expect("lusgs step");
    assert!(info.residual_rms.is_finite());
}

#[test]
fn f32_gmres_step_on_uniform_box() {
    use crate::field::is_physical_conserved;
    use crate::solver::time::{CflSchedule, TimeIntegrationScheme};
    use crate::solver::{CompressibleTimeMode, SolverState};

    let pair = FreestreamPairFixture::air_sutherland(0.2);
    let side = pair.inviscid_side();
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        cfl_schedule: CflSchedule::constant(0.1),
        time_mode: CompressibleTimeMode::Steady,
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::Gmres,
        time: crate::solver::time::RungeKutta4Config {
            dt: 0.0,
            max_steps: 1,
        },
        ..CompressibleEulerConfig::default()
    });
    let (mesh, patches, mut fields, mut ghosts, eos, freestream) =
        freestream_box_context::<f32>(&side);
    let mut ctx = CompressibleAdvanceContext3dTyped {
        mesh: &mesh,
        structured: &mesh,
        patches: &patches,
        ghosts: &mut ghosts,
        eos: &eos,
        freestream: &freestream,
        reference: None,
        primitive_scratch: crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells())
            .expect("prim"),
        spectral_primitives: PrimitiveFields::zeros(mesh.num_cells()).expect("spec"),
        gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
            .expect("grad"),
        viscous: None,
        interface_residual: None,
    };
    let mut storage = Rk4StorageT::<f32>::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let mut integrator = RungeKutta4Integrator::new(solver.config.time);
    let info = solver
        .advance_step_3d_typed(
            &mut ctx,
            &mut fields,
            &mut storage,
            &mut state,
            &mut integrator,
        )
        .expect("gmres step");
    assert!(info.residual_rms.is_finite());
    for cell in 0..mesh.num_cells() {
        assert!(is_physical_conserved(
            &fields.cell_state(cell).expect("cell"),
            eos.gamma,
            side.min_pressure
        ));
    }
}
