use super::*;
use crate::boundary::BoundarySet;
use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
use crate::discretization::{BoundaryGhostBuffer, GradientFields};
use crate::field::ConservedFields;
use crate::physics::ConservedState;
use crate::solver::time::{
    CflSchedule, Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme,
};
use crate::solver::{CompressibleEulerConfig, CompressibleTimeMode, SolverState};

#[test]
fn residual_vector_uses_cell_major_component_order() {
    let mut r = ConservedResidual::zeros(2).expect("r");
    r.density.values_mut()[1] = 1.0;
    r.momentum_x.values_mut()[1] = 2.0;
    r.momentum_y.values_mut()[1] = 3.0;
    r.momentum_z.values_mut()[1] = 4.0;
    r.total_energy.values_mut()[1] = 5.0;
    let v = residual_to_vector(&r);
    assert_eq!(&v[5..10], &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn perturbation_assigns_all_conserved_components() {
    let state = ConservedState {
        density: 1.0,
        momentum: [2.0, 3.0, 4.0],
        total_energy: 10.0,
    };
    let base = ConservedFields::uniform(1, state).expect("base");
    let mut out = base.clone();
    assign_perturbed_fields(&mut out, &base, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.1).expect("perturb");
    assert!((out.density.values()[0] - 1.1).abs() < 1.0e-12);
    assert!((out.momentum_x.values()[0] - 2.2).abs() < 1.0e-12);
    assert!((out.momentum_y.values()[0] - 3.3).abs() < 1.0e-12);
    assert!((out.momentum_z.values()[0] - 4.4).abs() < 1.0e-12);
    assert!((out.total_energy.values()[0] - 10.5).abs() < 1.0e-12);
}

#[test]
fn scaled_delta_assigns_all_conserved_components() {
    let state = ConservedState {
        density: 1.0,
        momentum: [2.0, 3.0, 4.0],
        total_energy: 10.0,
    };
    let base = ConservedFields::uniform(1, state).expect("base");
    let mut out = base.clone();
    assign_delta_scaled(&mut out, &base, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.25).expect("delta");
    assert!((out.density.values()[0] - 1.25).abs() < 1.0e-12);
    assert!((out.momentum_x.values()[0] - 2.5).abs() < 1.0e-12);
    assert!((out.momentum_y.values()[0] - 3.75).abs() < 1.0e-12);
    assert!((out.momentum_z.values()[0] - 5.0).abs() < 1.0e-12);
    assert!((out.total_energy.values()[0] - 11.25).abs() < 1.0e-12);
}

#[test]
fn limited_delta_clips_nonphysical_density_update() {
    let state = ConservedState {
        density: 1.0,
        momentum: [0.0, 0.0, 0.0],
        total_energy: 4.0,
    };
    let base = ConservedFields::uniform(1, state).expect("base");
    let mut out = base.clone();
    assign_delta_limited_scaled(&mut out, &base, &[-2.0, 0.0, 0.0, 0.0, 0.0], 1.0, 1.4, 0.0)
        .expect("limited");
    let limited = out.cell_state(0).expect("cell");
    assert!(limited.density > 0.0);
    assert!(limited.density < state.density);
    assert!(is_physical_conserved(&limited, 1.4, 0.0));
}

#[test]
fn physical_perturbation_reduces_epsilon_when_needed() {
    let state = ConservedState {
        density: 1.0,
        momentum: [0.0, 0.0, 0.0],
        total_energy: 4.0,
    };
    let base = ConservedFields::uniform(1, state).expect("base");
    let mut out = base.clone();
    let eps = assign_physical_perturbed_fields(
        &mut out,
        &base,
        &[-20.0, 0.0, 0.0, 0.0, 0.0],
        0.1,
        1.4,
        0.0,
    )
    .expect("perturb");
    assert!(eps < 0.1);
    let perturbed = out.cell_state(0).expect("cell");
    assert!(is_physical_conserved(&perturbed, 1.4, 0.0));
}

#[test]
fn validates_gmres_timestep_inputs() {
    assert!(validate_gmres_inputs(2, &[0.1, 0.2], &[1.0, 2.0], 1.0e-7).is_ok());
    assert!(validate_gmres_inputs(2, &[0.1], &[1.0, 2.0], 1.0e-7).is_err());
    assert!(validate_gmres_inputs(1, &[0.0], &[1.0], 1.0e-7).is_err());
    assert!(validate_gmres_inputs(1, &[0.1], &[-1.0], 1.0e-7).is_err());
    assert!(validate_gmres_inputs(1, &[0.1], &[1.0], 0.0).is_err());
}

#[test]
fn gmres_uniform_farfield_smoke_step_remains_physical() {
    let pair = FreestreamPairFixture::air_sutherland(0.3);
    let side = pair.inviscid_side();
    let (mesh, boundary, mut fields, mut ghosts) =
        uniform_farfield_box(2, 2, 2, 1.0, 1.0, 1.0, &side);
    let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
    let mut state = SolverState::default();
    let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
        dt: 0.0,
        max_steps: 1,
    });
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        cfl_schedule: CflSchedule::constant(0.1),
        time_mode: CompressibleTimeMode::Steady,
        local_time_step: true,
        time_scheme: TimeIntegrationScheme::Gmres,
        ..CompressibleEulerConfig::default()
    });
    let mut ctx = test_context(&mesh, &boundary, &mut ghosts, side.eos, side.fs);
    let info = solver
        .advance_step_3d(
            &mut ctx,
            &mut fields,
            &mut storage,
            &mut state,
            &mut integrator,
        )
        .expect("gmres step");
    assert!(info.residual_rms.is_finite());
    assert!(fields_are_physical(&fields, side.eos.gamma, side.min_pressure).expect("physical"));
}

#[test]
fn gmres_cell_block_preconditioner_solves_uniform_farfield_delta() {
    let pair = FreestreamPairFixture::air_sutherland(0.3);
    let side = pair.inviscid_side();
    let (mesh, boundary, fields, mut ghosts) = uniform_farfield_box(2, 2, 2, 1.0, 1.0, 1.0, &side);
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig::default());
    let mut ctx = test_context(&mesh, &boundary, &mut ghosts, side.eos, side.fs);
    let n = fields.num_cells();
    let delta = solver
        .solve_gmres_implicit_delta_3d(
            &mut ctx,
            &fields,
            &vec![0.01; n],
            &vec![1.0; n],
            side.min_pressure,
            GmresImplicitConfig {
                preconditioner: GmresPreconditionerKind::CellBlockDiagonal,
                ..GmresImplicitConfig::default()
            },
        )
        .expect("gmres delta");
    assert_eq!(
        delta.diagnostics.preconditioner,
        GmresPreconditionerKind::CellBlockDiagonal
    );
    assert!(delta.report.residual_norm.is_finite());
}

fn test_context<'a>(
    mesh: &'a crate::mesh::StructuredMesh3d,
    boundary: &'a BoundarySet,
    ghosts: &'a mut BoundaryGhostBuffer,
    eos: &'a crate::physics::IdealGasEoS,
    freestream: &'a crate::physics::FreestreamParams,
) -> CompressibleAdvanceContext3d<'a> {
    CompressibleAdvanceContext3d {
        mesh,
        structured: mesh,
        patches: boundary,
        ghosts,
        eos,
        freestream,
        reference: None,
        primitive_scratch: crate::field::PrimitiveFields::zeros(mesh.num_cells())
            .expect("primitives"),
        gradient_scratch: GradientFields::zeros(mesh.num_cells()).expect("gradients"),
        viscous: None,
        residual_correction: None,
    }
}
