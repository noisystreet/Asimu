//! CLI 集成测试（冒烟）。

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn cli_runs_successfully() {
    Command::cargo_bin("asimu")
        .expect("failed to locate binary")
        .assert()
        .success()
        .stdout(predicate::str::contains("asimu").not())
        .stderr(predicate::str::contains("asimu 启动"));
}
