//! CLI 集成测试（冒烟 + case 编排）。

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn cli_runs_diffusion_case() {
    Command::cargo_bin("asimu")
        .expect("failed to locate binary")
        .args([
            "--case",
            "tests/benchmarks/1d_diffusion_analytical/case.toml",
            "--log-level",
            "info",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("算例完成"));
}

#[test]
fn cli_requires_case_path() {
    Command::cargo_bin("asimu")
        .expect("failed to locate binary")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--case"));
}
