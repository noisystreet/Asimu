//! 双时间步内层残差日志解析与探针判定（集成测试共享）。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use asimu::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use asimu::case::{CaseRunResult, run_case};
use asimu::core::FaceId;
use asimu::io::{CaseMesh, CaseSpec, parse_case_str};
use asimu::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

/// 内层残差采样（来自 `dual_time 内迭代残差` 日志）。
#[derive(Debug, Clone, PartialEq)]
pub struct InnerResidualSample {
    pub inner: u32,
    pub log10_residual: f64,
}

/// 单探针内层序列摘要。
#[derive(Debug, Clone, PartialEq)]
pub struct InnerResidualSeries {
    pub samples: Vec<InnerResidualSample>,
}

impl InnerResidualSeries {
    #[must_use]
    pub fn first_log10(&self) -> Option<f64> {
        self.samples.first().map(|s| s.log10_residual)
    }

    #[must_use]
    pub fn last_log10(&self) -> Option<f64> {
        self.samples.last().map(|s| s.log10_residual)
    }

    #[must_use]
    pub fn drop(&self) -> Option<f64> {
        Some(self.first_log10()? - self.last_log10()?)
    }

    #[must_use]
    pub fn monotonic_nonincreasing(&self) -> bool {
        self.samples
            .windows(2)
            .all(|w| w[1].log10_residual <= w[0].log10_residual + 1.0e-9)
    }
}

/// 探针期望类型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbeExpect {
    /// 末轮 log10(R_eff) 低于首轮，且总降幅 ≥ `min_drop`。
    Decrease { min_drop: f64 },
    /// 末轮 log10(R_eff) ≤ `max_log10`（均匀来流等已近乎零残差场景）。
    Converged { max_log10: f64 },
    /// 降幅达标，或末轮已充分收敛（二者满足其一即 PASS）。
    DecreaseOrConverged { min_drop: f64, max_log10: f64 },
    /// 记录观测结果，不做硬断言（用于已知失败探针）。
    ObserveOnly,
}

/// 探针运行结果（用于矩阵汇总）。
#[derive(Debug, Clone)]
pub struct ProbeRunReport {
    pub id: String,
    pub expect: ProbeExpect,
    pub series: InnerResidualSeries,
    pub run_ok: bool,
    pub error: Option<String>,
}

impl ProbeRunReport {
    #[must_use]
    pub fn verdict_label(&self) -> &'static str {
        if !self.run_ok {
            return "RUN_FAIL";
        }
        match self.expect {
            ProbeExpect::ObserveOnly => "OBSERVE",
            ProbeExpect::Decrease { min_drop } => {
                if self.passes_decrease(min_drop) {
                    "PASS"
                } else {
                    "FAIL"
                }
            }
            ProbeExpect::Converged { max_log10 } => {
                if self.passes_converged(max_log10) {
                    "PASS"
                } else {
                    "FAIL"
                }
            }
            ProbeExpect::DecreaseOrConverged {
                min_drop,
                max_log10,
            } => {
                if self.passes_decrease(min_drop) || self.passes_converged(max_log10) {
                    "PASS"
                } else {
                    "FAIL"
                }
            }
        }
    }

    fn passes_converged(&self, max_log10: f64) -> bool {
        self.series
            .last_log10()
            .is_some_and(|v| v <= max_log10 + 1.0e-12)
    }

    fn passes_decrease(&self, min_drop: f64) -> bool {
        self.series.samples.len() >= 2
            && self
                .series
                .drop()
                .is_some_and(|drop| drop >= min_drop - 1.0e-12)
    }

    pub fn assert_expectation(&self) {
        assert!(self.run_ok, "探针 {} 运行失败: {:?}", self.id, self.error);
        match self.expect {
            ProbeExpect::ObserveOnly => {}
            ProbeExpect::Decrease { min_drop } => {
                let first = self.series.first_log10().expect("inner samples");
                let last = self.series.last_log10().expect("inner samples");
                let drop = self.series.drop().expect("drop");
                assert!(
                    self.passes_decrease(min_drop),
                    "探针 {} 内层残差降幅不足: inner1={first:.4}, innerN={last:.4}, drop={drop:.4}, min_drop={min_drop:.4}",
                    self.id
                );
            }
            ProbeExpect::Converged { max_log10 } => {
                let last = self.series.last_log10().expect("inner samples");
                assert!(
                    self.passes_converged(max_log10),
                    "探针 {} 内层未收敛: innerN={last:.4}, max_log10={max_log10:.4}",
                    self.id
                );
            }
            ProbeExpect::DecreaseOrConverged {
                min_drop,
                max_log10,
            } => {
                let first = self.series.first_log10().unwrap_or(f64::NAN);
                let last = self.series.last_log10().unwrap_or(f64::NAN);
                let drop = self.series.drop().unwrap_or(0.0);
                assert!(
                    self.passes_decrease(min_drop) || self.passes_converged(max_log10),
                    "探针 {} 未达标: inner1={first:.4}, innerN={last:.4}, drop={drop:.4}, min_drop={min_drop:.4}, max_log10={max_log10:.4}",
                    self.id
                );
            }
        }
    }
}

/// 从 asimu 日志文本提取 `dual_time 内迭代残差` 的 log10 序列。
pub fn parse_inner_log10_residuals(text: &str) -> InnerResidualSeries {
    let mut samples = Vec::new();
    for line in text.lines() {
        if !line.contains("dual_time 内迭代残差") {
            continue;
        }
        let Some(inner) = parse_tracing_field_u32(line, "inner=") else {
            continue;
        };
        let Some(log10_residual) = parse_tracing_field_f64(line, "log10_residual=") else {
            continue;
        };
        samples.push(InnerResidualSample {
            inner,
            log10_residual,
        });
    }
    InnerResidualSeries { samples }
}

fn parse_tracing_field_u32(line: &str, key: &str) -> Option<u32> {
    let rest = line.split(key).nth(1)?;
    let token = rest.split_whitespace().next()?;
    token.parse().ok()
}

fn parse_tracing_field_f64(line: &str, key: &str) -> Option<f64> {
    let rest = line.split(key).nth(1)?;
    let token = rest.split_whitespace().next()?.trim_end_matches(',');
    token.parse().ok()
}

/// 单四面体远场网格（与 `compressible_unstructured_3d_tests` 一致）。
pub fn attach_single_tet_farfield(case: &mut CaseSpec) {
    let mesh = UnstructuredMesh3d::new(
        "tet",
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
    )
    .expect("mesh");
    let faces = (0..mesh.num_faces())
        .map(|face| FaceId(face as u32))
        .collect::<Vec<_>>();
    let fs = case.freestream.expect("freestream");
    case.mesh = CaseMesh::Unstructured3d(mesh);
    case.boundary = BoundarySet::new(vec![BoundaryPatch::new(
        "farfield",
        faces,
        BoundaryKind::Farfield {
            mach: fs.mach,
            pressure: fs.pressure,
            temperature: fs.temperature,
            alpha: fs.alpha,
            beta: fs.beta,
        },
    )]);
}

/// 在独立 tracing 订阅器下运行算例并捕获 info 级日志。
pub fn run_case_capture_logs(case: &CaseSpec) -> Result<(CaseRunResult, String), String> {
    let logs = Arc::new(Mutex::new(Vec::<u8>::new()));
    let writer_logs = Arc::clone(&logs);
    let make_writer = move || CapturingWriter(Arc::clone(&writer_logs));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(make_writer)
        .with_ansi(false)
        .without_time()
        .with_max_level(tracing::Level::INFO)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let result = run_case(case).map_err(|e| e.to_string())?;
    let text = String::from_utf8(logs.lock().expect("log lock").clone())
        .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).into_owned());
    Ok((result, text))
}

struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// 通过 CLI 子进程运行 case.toml 并收集 stdout/stderr。
pub fn run_case_subprocess(case_path: &Path, log_level: &str) -> Result<String, String> {
    let output = Command::new(env!("CARGO_BIN_EXE_asimu"))
        .args([
            "--case",
            case_path.to_str().expect("utf8 path"),
            "--log-level",
            log_level,
        ])
        .output()
        .map_err(|e| format!("启动 asimu 失败: {e}"))?;
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        Ok(text)
    } else {
        Err(format!(
            "asimu 退出码 {:?}\n{}",
            output.status.code(),
            summarize_tail(&text, 40)
        ))
    }
}

pub fn summarize_tail(text: &str, lines: usize) -> String {
    let mut buf = Vec::new();
    for line in text.lines().rev().take(lines) {
        buf.push(line);
    }
    buf.reverse();
    buf.join("\n")
}

/// 打印 Markdown 表格行（`--nocapture` 下可见）。
pub fn print_matrix_row(report: &ProbeRunReport) {
    let first = report
        .series
        .first_log10()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "-".to_string());
    let last = report
        .series
        .last_log10()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "-".to_string());
    let drop = report
        .series
        .drop()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "-".to_string());
    eprintln!(
        "| {} | {} | {} | {} | {} |",
        report.id,
        first,
        last,
        drop,
        report.verdict_label()
    );
}

pub fn print_matrix_header() {
    eprintln!("| probe | inner1 log10 | innerN log10 | drop | verdict |");
    eprintln!("| --- | ---: | ---: | ---: | --- |");
}

/// 解析仓库根目录下相对路径。
#[must_use]
pub fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// 构建 uniform freestream dual-time 探针算例（单四面体）。
pub fn build_freestream_dual_time_case(mutator: impl FnOnce(&mut CaseSpec)) -> CaseSpec {
    let base = r#"
name = "dual_time_inner_probe"
[mesh]
kind = "structured_3d"
nx = 1
ny = 1
nz = 1
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[euler]
flux = "hllc"
reconstruction = "first_order"
[time]
mode = "transient"
scheme = "dual_time"
dt = 1.0e-4
local_time_step = true
cfl = 0.4
max_inner_steps = 10
inner_tolerance = -2.0
max_steps = 1
"#;
    let mut case = parse_case_str(base).expect("parse probe case");
    attach_single_tet_farfield(&mut case);
    mutator(&mut case);
    case
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn parses_info_level_inner_residual_lines() {
        let text = r"
INFO dual_time 内迭代残差 inner=1 max_inner=10 log10_residual=1.5000 residual_rms=3.1623e0 inner_converged=false
INFO dual_time 内迭代残差 inner=2 max_inner=10 log10_residual=1.2000 residual_rms=1.5849e0 inner_converged=false
INFO dual_time 内迭代残差 inner=10 max_inner=10 log10_residual=0.8000 residual_rms=6.3096e-1 inner_converged=true
";
        let series = parse_inner_log10_residuals(text);
        assert_eq!(series.samples.len(), 3);
        assert!((series.drop().expect("drop") - 0.7).abs() < 1.0e-12);
        assert!(series.monotonic_nonincreasing());
    }

    #[test]
    fn ignores_unrelated_lines() {
        let text = "INFO dual_time 伪时间步诊断 inner=1 sigma_min=1.0e2\n";
        assert!(parse_inner_log10_residuals(text).samples.is_empty());
    }
}
