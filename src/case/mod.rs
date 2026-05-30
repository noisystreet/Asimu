//! 算例编排：case.toml → 求解器 dispatch → 结果摘要。
//!
//! 应用层（`app`）与集成测试共用本模块，避免在 CLI 中重复装配逻辑。

mod diffusion;
mod sod;

use std::path::Path;

use tracing::info;

use crate::error::{AsimuError, Result};
use crate::io::{CaseSpec, load_case};

pub use diffusion::DiffusionRunMetrics;
pub use sod::SodRunMetrics;

/// 算例运行模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseRunKind {
    Diffusion1dSteady,
    Sod1dTransient,
}

/// 算例运行结果摘要（写入日志 / 后续 manifest）。
#[derive(Debug, Clone, PartialEq)]
pub struct CaseRunResult {
    pub name: String,
    pub benchmark_id: Option<String>,
    pub kind: CaseRunKind,
    pub summary: String,
    pub diffusion: Option<DiffusionRunMetrics>,
    pub sod: Option<SodRunMetrics>,
}

/// 从 `case.toml` 路径加载并运行。
pub fn run_case_path(path: &Path) -> Result<CaseRunResult> {
    let case = load_case(path)?;
    run_case(&case)
}

/// 运行已解析算例。
pub fn run_case(case: &CaseSpec) -> Result<CaseRunResult> {
    let kind = detect_run_kind(case)?;
    info!(
        name = %case.name,
        benchmark_id = ?case.benchmark_id,
        ?kind,
        "开始算例编排"
    );
    match kind {
        CaseRunKind::Diffusion1dSteady => diffusion::run(case),
        CaseRunKind::Sod1dTransient => sod::run(case),
    }
}

fn detect_run_kind(case: &CaseSpec) -> Result<CaseRunKind> {
    if case.sod.is_some() {
        case.mesh.as_1d()?;
        return Ok(CaseRunKind::Sod1dTransient);
    }
    if case.is_compressible() {
        return Err(AsimuError::Config(
            "可压缩算例须包含 [sod] 段，或等待通用瞬态编排实现".to_string(),
        ));
    }
    case.mesh.as_1d()?;
    if case.time.mode == crate::io::CaseTimeMode::Transient {
        return Err(AsimuError::Config(
            "标量 1D 瞬态尚未实现；请使用 mode = \"steady\"".to_string(),
        ));
    }
    Ok(CaseRunKind::Diffusion1dSteady)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn runs_diffusion_benchmark_case() {
        let path = Path::new("tests/benchmarks/1d_diffusion_analytical/case.toml");
        let result = run_case_path(path).expect("run");
        assert_eq!(result.kind, CaseRunKind::Diffusion1dSteady);
        let metrics = result.diffusion.expect("metrics");
        assert!(metrics.max_abs_error > 0.0);
        assert!(metrics.max_abs_error < 0.05);
    }

    #[test]
    fn runs_sod_benchmark_case() {
        let path = Path::new("tests/benchmarks/sod_1d/case.toml");
        let result = run_case_path(path).expect("run");
        assert_eq!(result.kind, CaseRunKind::Sod1dTransient);
        let metrics = result.sod.expect("metrics");
        assert!(metrics.l1_density < 0.04);
    }
}
