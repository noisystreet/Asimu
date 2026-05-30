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
    assert!(metrics.l1_density < 0.04);
}
