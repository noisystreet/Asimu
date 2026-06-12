//! 算例编排集成测试（与 CLI 共用 `case::run_case_path`）。

use std::path::Path;

use asimu::case::{CaseRunKind, run_case_path};

#[test]
fn diffusion_benchmark_via_case_runner() {
    let result = run_case_path(Path::new(
        "tests/benchmarks/1d_diffusion_analytical/case.toml",
    ))
    .expect("run");
    assert_eq!(result.kind, CaseRunKind::Diffusion1dSteady);
    assert!(result.summary.contains("扩散"));
}

#[test]
fn sod_benchmark_via_case_runner() {
    let result = run_case_path(Path::new("tests/benchmarks/sod_1d/case.toml")).expect("run");
    assert_eq!(result.kind, CaseRunKind::Sod1dTransient);
    let metrics = result.sod.expect("sod metrics");
    assert_eq!(metrics.scheme, "muscl_roe");
    assert_eq!(metrics.limiter, "van_albada");
    assert!(metrics.l1_density < 0.02);
    assert!(result.summary.contains("van_albada/muscl_roe"));
}

#[test]
fn sod_muscl_hllc_via_case_runner() {
    let result =
        run_case_path(Path::new("tests/benchmarks/sod_1d/case_muscl_hllc.toml")).expect("run");
    assert_eq!(result.kind, CaseRunKind::Sod1dTransient);
    let metrics = result.sod.expect("sod metrics");
    assert_eq!(metrics.scheme, "muscl_hllc");
    assert_eq!(metrics.limiter, "van_albada");
    assert!(metrics.l1_density < 0.02);
}

#[test]
fn channel_poiseuille_incompressible_benchmark_runs() {
    let result =
        run_case_path(Path::new("tests/benchmarks/channel_poiseuille/case.toml")).expect("run");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    assert_eq!(result.benchmark_id.as_deref(), Some("channel_poiseuille"));
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert!(metrics.simplec_iterations <= 2000);
    assert!(metrics.simplec_converged);
    assert!(metrics.simplec_final_residual.is_finite());
    assert!(metrics.simplec_final_residual < 1.0e-8);
    assert!(
        metrics
            .max_abs_underrelaxed_corrected_divergence
            .is_finite()
    );
    assert!(metrics.max_abs_underrelaxed_corrected_divergence < 1.0e-8);
    assert!(
        metrics
            .max_abs_corrected_field_divergence_after_boundary
            .is_finite()
    );
    assert!(metrics.pressure_correction_rhs_active_sum.is_finite());
    assert!(metrics.simplec_final_momentum_residual.is_finite());
    assert!(metrics.pressure_solve_converged);
    assert!(metrics.pressure_solve_iterations <= 500);
    let profiles = metrics.centerline_profiles.expect("poiseuille profile");
    assert_eq!(profiles.vertical_u.len(), 8);
    assert!(profiles.horizontal_v.is_empty());
    assert!(
        profiles
            .vertical_u
            .iter()
            .all(|sample| sample.coordinate.is_finite() && sample.velocity_x.is_finite())
    );
    let error = metrics
        .poiseuille_profile_error
        .expect("poiseuille profile error");
    assert!(error.max_abs.is_finite());
    assert!(error.l2.is_finite());
    assert!(error.max_abs < 0.25, "max_abs={}", error.max_abs);
    assert!(error.l2 < 0.25, "l2={}", error.l2);
}

#[test]
fn lid_driven_cavity_re100_incompressible_benchmark_runs() {
    let expected =
        std::fs::read_to_string("tests/benchmarks/lid_driven_cavity_re100/expected.json")
            .expect("expected");
    assert!(expected.contains("coarse_grid_quantitative"));
    assert!(expected.contains("Ghia et al. 1982"));
    let result = run_case_path(Path::new(
        "tests/benchmarks/lid_driven_cavity_re100/case.toml",
    ))
    .expect("run");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    assert_eq!(
        result.benchmark_id.as_deref(),
        Some("lid_driven_cavity_re100")
    );
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert_eq!(metrics.algorithm, "piso");
    assert_eq!(metrics.pressure_correctors, 2);
    assert_eq!(metrics.simplec_iterations, 100);
    assert!(metrics.simplec_converged);
    assert_eq!(
        metrics.pressure_corrector_residual_history.len(),
        metrics.simplec_iterations * metrics.pressure_correctors
    );
    assert_eq!(
        metrics.pressure_corrector_max_correction_history.len(),
        metrics.pressure_corrector_residual_history.len()
    );
    assert!(metrics.simplec_final_residual.is_finite());
    assert!(metrics.max_abs_corrected_divergence < 1.0e-8);
    assert!(
        metrics
            .max_abs_underrelaxed_corrected_divergence
            .is_finite()
    );
    assert!(metrics.max_abs_underrelaxed_corrected_divergence < 3.0e-5);
    assert!(metrics.max_abs_corrected_velocity_delta_interior < 3.0e-5);
    assert!(
        metrics.max_abs_corrected_velocity_delta_boundary
            >= metrics.max_abs_corrected_velocity_delta_interior
    );
    assert!(
        metrics
            .max_abs_corrected_field_divergence_before_boundary
            .is_finite()
    );
    assert!(
        metrics
            .max_abs_corrected_field_divergence_after_boundary
            .is_finite()
    );
    assert!(metrics.pressure_correction_rhs_active_sum.is_finite());
    assert!(metrics.simplec_final_momentum_residual.is_finite());
    assert!(metrics.pressure_solve_converged);
    assert!(metrics.momentum_solve_converged);
    let profiles = metrics.centerline_profiles.expect("centerline profiles");
    assert_eq!(profiles.vertical_u.len(), 8);
    assert_eq!(profiles.horizontal_v.len(), 8);
    assert!(
        profiles
            .vertical_u
            .iter()
            .all(|sample| sample.coordinate.is_finite() && sample.velocity_x.is_finite())
    );
    assert!(
        profiles
            .horizontal_v
            .iter()
            .all(|sample| sample.coordinate.is_finite() && sample.velocity_y.is_finite())
    );
    let error = metrics
        .lid_cavity_profile_error
        .expect("lid cavity profile error");
    assert!(error.vertical_u.max_abs.is_finite());
    assert!(error.vertical_u.l2.is_finite());
    assert!(error.horizontal_v.max_abs.is_finite());
    assert!(error.horizontal_v.l2.is_finite());
    assert!(
        error.vertical_u.max_abs < 1.0,
        "vertical_u max_abs={}",
        error.vertical_u.max_abs
    );
    assert!(
        error.horizontal_v.max_abs < 1.0,
        "horizontal_v max_abs={}",
        error.horizontal_v.max_abs
    );
}

#[test]
fn lid_driven_cavity_re100_refined_grid_runs() {
    let result = run_case_path(Path::new(
        "tests/benchmarks/lid_driven_cavity_re100_refined/case.toml",
    ))
    .expect("run");
    assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
    assert_eq!(
        result.benchmark_id.as_deref(),
        Some("lid_driven_cavity_re100_refined")
    );
    let metrics = result.incompressible_3d.expect("incompressible metrics");
    assert_eq!(metrics.algorithm, "piso");
    assert_eq!(metrics.pressure_correctors, 2);
    assert_eq!(metrics.simplec_iterations, 100);
    assert!(metrics.simplec_final_residual.is_finite());
    assert!(metrics.pressure_solve_converged);
    assert!(metrics.momentum_solve_converged);
    let profiles = metrics.centerline_profiles.expect("centerline profiles");
    assert_eq!(profiles.vertical_u.len(), 12);
    assert_eq!(profiles.horizontal_v.len(), 12);
    let error = metrics
        .lid_cavity_profile_error
        .expect("lid cavity profile error");
    assert!(error.vertical_u.max_abs.is_finite());
    assert!(error.horizontal_v.max_abs.is_finite());
}
