//! Sod 激波管算例编排（case.toml `[sod]` → benchmark）。

use tracing::info;

use crate::case::{CaseRunKind, CaseRunResult};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::io::CaseSpec;
use crate::physics::SodProblem;
use crate::solver::{SodBenchmarkConfig, run_sod_benchmark};

/// Sod 运行指标。
#[derive(Debug, Clone, PartialEq)]
pub struct SodRunMetrics {
    pub l1_density: Real,
    pub l2_density: Real,
    pub final_time: Real,
    pub steps: u64,
    pub scheme: String,
    pub limiter: String,
}

pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    let mesh = case.mesh.as_1d()?;
    let sod_section = case
        .sod
        .as_ref()
        .ok_or_else(|| AsimuError::Config("Sod 算例缺少 [sod] 段".to_string()))?;
    let _eos = case.physics.eos()?;
    let inviscid = sod_section.inviscid();
    let config = SodBenchmarkConfig {
        ncells: mesh.num_cells(),
        length: mesh.length,
        diaphragm: sod_section.diaphragm,
        final_time: sod_section.final_time,
        cfl: sod_section.cfl,
        sod: SodProblem::CLASSIC,
        inviscid,
    };
    let scheme = inviscid.short_label().to_string();
    let limiter = inviscid.limiter_label().to_string();
    let result = run_sod_benchmark(&config)?;
    let metrics = SodRunMetrics {
        l1_density: result.l1_density,
        l2_density: result.l2_density,
        final_time: result.final_time,
        steps: result.steps,
        scheme: scheme.clone(),
        limiter: limiter.clone(),
    };
    info!(
        l1 = metrics.l1_density,
        l2 = metrics.l2_density,
        steps = metrics.steps,
        t = metrics.final_time,
        scheme = %scheme,
        limiter = %limiter,
        "Sod 激波管瞬态求解完成"
    );
    Ok(CaseRunResult {
        name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        kind: CaseRunKind::Sod1dTransient,
        summary: format!(
            "Sod {}/{} t={:.4}：L1(ρ)={:.6} L2(ρ)={:.6} steps={}",
            limiter,
            scheme,
            metrics.final_time,
            metrics.l1_density,
            metrics.l2_density,
            metrics.steps
        ),
        diffusion: None,
        sod: Some(metrics),
        compressible_3d: None,
        incompressible_3d: None,
    })
}
