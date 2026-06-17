//! 从 `CaseRunResult` 构建 Run Manifest（case 编排层）。

use std::time::{Instant, SystemTime};

use crate::case::{CaseRunKind, CaseRunResult};
use crate::core::Real;
use crate::error::Result;
use crate::io::CaseSpec;
use crate::io::{
    MANIFEST_SCHEMA_VERSION, ManifestObservability, ManifestSolveSummary, RunManifest, config_hash,
    maybe_write_run_manifest, time_mode_label, unix_timestamp_secs,
};

use super::benchmark_expected::try_load_benchmark_expected;
use super::time_advance::{incompressible_time_advance_kind, incompressible_time_advance_label};

/// 运行算例并写出 manifest（若配置了 `[output]`）。
pub fn run_case_with_manifest(case: &CaseSpec) -> Result<CaseRunResult> {
    let started_at = SystemTime::now();
    let started_unix = unix_timestamp_secs(started_at);
    let clock = Instant::now();
    let result = super::dispatch_case(case)?;
    let wall_sec = clock.elapsed().as_secs_f64();
    let finished_unix = unix_timestamp_secs(SystemTime::now());
    let manifest = build_run_manifest(
        case,
        &result,
        started_unix,
        finished_unix,
        wall_sec,
        collect_output_paths(&result),
    );
    let _ = maybe_write_run_manifest(case, &manifest)?;
    Ok(result)
}

fn build_run_manifest(
    case: &CaseSpec,
    result: &CaseRunResult,
    started_at_unix: Real,
    finished_at_unix: Real,
    wall_time_sec: Real,
    output_paths: Vec<std::path::PathBuf>,
) -> RunManifest {
    let benchmark_expected = try_load_benchmark_expected(case).ok().flatten();
    let solve = solve_fields(result);
    RunManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        run_id: format!("{}-{started_at_unix:.3}", case.name),
        asimu_version: env!("CARGO_PKG_VERSION").to_string(),
        config_hash: config_hash(case),
        case_name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        benchmark_status: benchmark_expected.and_then(|expected| expected.status),
        time_mode: time_mode_label(case.time.mode),
        incompressible_time_advance: incompressible_manifest_time_advance(case, result.kind),
        started_at_unix,
        finished_at_unix,
        solve: ManifestSolveSummary {
            kind: kind_label(result.kind).to_string(),
            summary: result.summary.clone(),
            steps: solve.steps,
            converged: solve.converged,
            residual_log10: solve.residual_log10,
            final_time: solve.final_time,
            inner_iterations: solve.inner_iterations,
        },
        observability: ManifestObservability { wall_time_sec },
        output_paths,
    }
}

fn incompressible_manifest_time_advance(
    case: &CaseSpec,
    kind: CaseRunKind,
) -> Option<&'static str> {
    if !matches!(
        kind,
        CaseRunKind::Incompressible3dSteady | CaseRunKind::Incompressible3dTransient
    ) {
        return None;
    }
    case.incompressible.as_ref()?;
    Some(incompressible_time_advance_label(
        incompressible_time_advance_kind(case),
    ))
}

fn kind_label(kind: CaseRunKind) -> &'static str {
    match kind {
        CaseRunKind::Diffusion1dSteady => "diffusion_1d_steady",
        CaseRunKind::Sod1dTransient => "sod_1d_transient",
        CaseRunKind::Compressible3dTransient => "compressible_3d_transient",
        CaseRunKind::Incompressible3dSteady => "incompressible_3d_steady",
        CaseRunKind::Incompressible3dTransient => "incompressible_3d_transient",
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ManifestSolveFields {
    steps: Option<u64>,
    converged: Option<bool>,
    residual_log10: Option<Real>,
    final_time: Option<Real>,
    inner_iterations: Option<u32>,
}

fn solve_fields(result: &CaseRunResult) -> ManifestSolveFields {
    if let Some(m) = &result.compressible_3d {
        return ManifestSolveFields {
            steps: Some(m.steps),
            converged: Some(m.converged),
            residual_log10: Some(m.residual_log10),
            final_time: Some(m.final_time),
            inner_iterations: Some(m.inner_iterations),
        };
    }
    if let Some(m) = &result.incompressible_3d {
        return ManifestSolveFields {
            steps: Some(m.steps),
            converged: Some(m.simplec_converged),
            residual_log10: Some(m.simplec_final_residual),
            final_time: Some(m.physical_time),
            ..ManifestSolveFields::default()
        };
    }
    if let Some(m) = &result.sod {
        return ManifestSolveFields {
            steps: Some(m.steps),
            final_time: Some(m.final_time),
            ..ManifestSolveFields::default()
        };
    }
    ManifestSolveFields::default()
}

fn collect_output_paths(result: &CaseRunResult) -> Vec<std::path::PathBuf> {
    result
        .incompressible_3d
        .as_ref()
        .map(|m| m.written.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use super::super::time_advance::{
        IncompressibleTimeAdvanceKind, incompressible_time_advance_from_config,
    };

    #[test]
    fn diffusion_run_writes_manifest_when_output_configured() {
        let path = Path::new("tests/benchmarks/1d_diffusion_analytical/case.toml");
        let case = crate::io::load_case(path).expect("load");
        let dir = std::env::temp_dir().join("asimu_manifest_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut case = case;
        case.output = Some(crate::io::CaseOutputConfig {
            dir: dir.clone(),
            residual_csv: None,
            residual_plot: None,
            solution_cgns: None,
            solution_every: None,
            solution_vtk: false,
            restart: None,
        });
        run_case_with_manifest(&case).expect("run");
        let manifest_path = dir.join("run-manifest.json");
        assert!(manifest_path.is_file(), "manifest should exist");
        let text = std::fs::read_to_string(manifest_path).expect("read");
        assert!(text.contains("\"schema_version\": 2"));
        assert!(text.contains("\"kind\": \"diffusion_1d_steady\""));
    }

    #[test]
    fn taylor_green_manifest_includes_benchmark_status_and_time_advance() {
        let path = Path::new("tests/benchmarks/taylor_green_3d/case.toml");
        let case = crate::io::load_case(path).expect("load");
        let dir = std::env::temp_dir().join("asimu_manifest_tg_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut case = case;
        case.output = Some(crate::io::CaseOutputConfig {
            dir: dir.clone(),
            residual_csv: Some("residual.csv".to_string()),
            residual_plot: None,
            solution_cgns: None,
            solution_every: None,
            solution_vtk: false,
            restart: None,
        });
        run_case_with_manifest(&case).expect("run");
        let text = std::fs::read_to_string(dir.join("run-manifest.json")).expect("read");
        assert!(text.contains("\"benchmark_status\": \"i3_piso_bdf1_kinetic_decay_vv\""));
        assert!(text.contains("\"incompressible_advance\": \"physical_transient\""));
        assert_eq!(
            incompressible_time_advance_from_config(&case.time),
            IncompressibleTimeAdvanceKind::PhysicalTransient
        );
    }
}
