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
    assert_eq!(metrics.simplec_iterations, 2);
    assert!(metrics.simplec_final_residual.is_finite());
    assert!(metrics.simplec_final_momentum_residual.is_finite());
    assert!(metrics.pressure_solve_residual.is_finite());
    assert!(metrics.pressure_solve_iterations <= 50);
}

#[test]
fn lid_driven_cavity_re100_incompressible_benchmark_runs() {
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
    assert_eq!(metrics.simplec_iterations, 2);
    assert!(metrics.simplec_final_residual.is_finite());
    assert!(metrics.simplec_final_momentum_residual.is_finite());
    assert!(metrics.momentum_solve_converged);
}
