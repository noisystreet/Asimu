//! 双时间步内层成败矩阵：CI 内置探针 + 可选外部大算例。
//!
//! ## CI（默认 `cargo test --test dual_time_inner_regression`）
//!
//! 运行单四面体 uniform freestream 探针矩阵，打印 Markdown 汇总表。
//!
//! ## 单算例 ad-hoc
//!
//! ```bash
//! ASIMU_DUAL_TIME_CASE=/path/to/case.toml \
//! ASIMU_DUAL_TIME_MIN_DROP=0.05 \
//! cargo test --test dual_time_inner_regression dual_time_inner_single_case -- --nocapture
//! ```
//!
//! ## 外部探针 JSON
//!
//! ```bash
//! ASIMU_DUAL_TIME_MATRIX=tests/benchmarks/dual_time_inner_matrix/probes_external.json \
//! cargo test --test dual_time_inner_regression dual_time_inner_matrix_from_file -- --nocapture
//! ```

#[path = "common/dual_time_inner.rs"]
mod dual_time_inner;

use std::fs;
use std::path::{Path, PathBuf};

use dual_time_inner::{
    ProbeExpect, ProbeRunReport, build_freestream_dual_time_case, parse_inner_log10_residuals,
    print_matrix_header, print_matrix_row, repo_path, run_case_capture_logs, run_case_subprocess,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ExternalMatrixFile {
    schema_version: u32,
    probes: Vec<ExternalProbe>,
}

#[derive(Debug, Deserialize)]
struct ExternalProbe {
    id: String,
    case_path: String,
    #[serde(default = "default_min_drop")]
    min_drop: f64,
    #[serde(default)]
    expect: String,
    #[serde(default = "default_log_level")]
    log_level: String,
    #[serde(default)]
    skip_if_missing: bool,
}

fn default_min_drop() -> f64 {
    0.05
}

fn default_log_level() -> String {
    "info".to_string()
}

fn parse_expect(expect: &str, min_drop: f64) -> ProbeExpect {
    match expect {
        "observe" | "observe_only" => ProbeExpect::ObserveOnly,
        "converged" => ProbeExpect::Converged { max_log10: -2.0 },
        _ => ProbeExpect::Decrease { min_drop },
    }
}

fn run_builtin_probe(id: &str, case: asimu::io::CaseSpec, expect: ProbeExpect) -> ProbeRunReport {
    match run_case_capture_logs(&case) {
        Ok((_result, text)) => {
            let series = parse_inner_log10_residuals(&text);
            ProbeRunReport {
                id: id.to_string(),
                expect,
                series,
                run_ok: true,
                error: None,
            }
        }
        Err(err) => ProbeRunReport {
            id: id.to_string(),
            expect,
            series: parse_inner_log10_residuals(""),
            run_ok: false,
            error: Some(err),
        },
    }
}

fn run_external_probe(probe: &ExternalProbe) -> ProbeRunReport {
    let expect = parse_expect(&probe.expect, probe.min_drop);
    let path = resolve_probe_case_path(&probe.case_path);
    if probe.skip_if_missing && !path.is_file() {
        return ProbeRunReport {
            id: probe.id.clone(),
            expect,
            series: parse_inner_log10_residuals(""),
            run_ok: true,
            error: Some(format!("skip: 文件不存在 {}", path.display())),
        };
    }
    match run_case_subprocess(&path, &probe.log_level) {
        Ok(text) => ProbeRunReport {
            id: probe.id.clone(),
            expect,
            series: parse_inner_log10_residuals(&text),
            run_ok: true,
            error: None,
        },
        Err(err) => ProbeRunReport {
            id: probe.id.clone(),
            expect,
            series: parse_inner_log10_residuals(""),
            run_ok: false,
            error: Some(err),
        },
    }
}

fn resolve_probe_case_path(case_path: &str) -> PathBuf {
    let path = Path::new(case_path);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    repo_path(case_path)
}

/// CI 内置探针矩阵（单四面体 uniform freestream）。
#[test]
fn dual_time_inner_matrix_builtin_probes() {
    let probes: [(&str, asimu::io::CaseSpec, ProbeExpect); 5] = [
        (
            "freestream_f64_baseline",
            build_freestream_dual_time_case(|_| {}),
            ProbeExpect::DecreaseOrConverged {
                min_drop: 0.5,
                max_log10: -2.0,
            },
        ),
        (
            "freestream_f64_high_cfl",
            build_freestream_dual_time_case(|case| case.time.cfl = Some(100.0)),
            ProbeExpect::DecreaseOrConverged {
                min_drop: 0.3,
                max_log10: -2.0,
            },
        ),
        (
            "freestream_f64_sweep",
            build_freestream_dual_time_case(|case| {
                case.time.lusgs_sweep = Some(true);
                case.time.lusgs_sweep_backward_damping = Some(0.5);
            }),
            ProbeExpect::DecreaseOrConverged {
                min_drop: 0.3,
                max_log10: -2.0,
            },
        ),
        (
            "freestream_f64_low_omega",
            build_freestream_dual_time_case(|case| case.time.lusgs_omega = Some(0.1)),
            ProbeExpect::DecreaseOrConverged {
                min_drop: 0.2,
                max_log10: -2.0,
            },
        ),
        (
            "freestream_f32_cpu",
            build_freestream_dual_time_case(|case| {
                case.numerics.compute_precision = asimu::core::ComputePrecision::F32;
            }),
            ProbeExpect::DecreaseOrConverged {
                min_drop: 0.3,
                max_log10: -1.5,
            },
        ),
    ];

    print_matrix_header();
    for (id, case, expect) in probes {
        let report = run_builtin_probe(id, case, expect);
        print_matrix_row(&report);
        report.assert_expectation();
    }
}

/// 从 JSON 加载外部探针（默认跳过；见 `probes_external.json`）。
#[test]
fn dual_time_inner_matrix_from_file() {
    let Some(matrix_path) = std::env::var("ASIMU_DUAL_TIME_MATRIX").ok() else {
        eprintln!("skip dual_time_inner_matrix_from_file: ASIMU_DUAL_TIME_MATRIX 未设置");
        return;
    };
    let text = fs::read_to_string(&matrix_path)
        .unwrap_or_else(|e| panic!("读取矩阵文件 {} 失败: {e}", matrix_path));
    let matrix: ExternalMatrixFile = serde_json::from_str(&text).expect("解析 dual_time 探针 JSON");
    assert_eq!(matrix.schema_version, 1, "schema_version 须为 1");

    print_matrix_header();
    for probe in &matrix.probes {
        let report = run_external_probe(probe);
        if let Some(msg) = &report.error {
            if msg.starts_with("skip:") {
                eprintln!("| {} | - | - | - | SKIP |", report.id);
                continue;
            }
        }
        print_matrix_row(&report);
        if probe.expect != "observe" && probe.expect != "observe_only" {
            report.assert_expectation();
        }
    }
}

/// 单算例 ad-hoc 回归（`ASIMU_DUAL_TIME_CASE` 未设置则跳过）。
#[test]
fn dual_time_inner_single_case() {
    let Some(case_path) = std::env::var("ASIMU_DUAL_TIME_CASE").ok() else {
        eprintln!("skip dual_time_inner_single_case: ASIMU_DUAL_TIME_CASE 未设置");
        return;
    };
    let min_drop = std::env::var("ASIMU_DUAL_TIME_MIN_DROP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.05);
    let expect = ProbeExpect::Decrease { min_drop };
    let path = PathBuf::from(&case_path);
    let text = run_case_subprocess(&path, "info").expect("asimu 运行");
    let series = parse_inner_log10_residuals(&text);
    let report = ProbeRunReport {
        id: case_path,
        expect,
        series,
        run_ok: true,
        error: None,
    };
    print_matrix_header();
    print_matrix_row(&report);
    report.assert_expectation();
}
