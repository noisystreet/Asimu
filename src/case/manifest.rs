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
    let (steps, converged, residual_log10, final_time) = solve_fields(result);
    RunManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        run_id: format!("{}-{started_at_unix:.3}", case.name),
        asimu_version: env!("CARGO_PKG_VERSION").to_string(),
        config_hash: config_hash(case),
        case_name: case.name.clone(),
        benchmark_id: case.benchmark_id.clone(),
        time_mode: time_mode_label(case.time.mode),
        started_at_unix,
        finished_at_unix,
        solve: ManifestSolveSummary {
            kind: kind_label(result.kind).to_string(),
            summary: result.summary.clone(),
            steps,
            converged,
            residual_log10,
            final_time,
        },
        observability: ManifestObservability { wall_time_sec },
        output_paths,
    }
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

fn solve_fields(result: &CaseRunResult) -> (Option<u64>, Option<bool>, Option<Real>, Option<Real>) {
    if let Some(m) = &result.compressible_3d {
        return (
            Some(m.steps),
            Some(m.converged),
            Some(m.residual_log10),
            Some(m.final_time),
        );
    }
    if let Some(m) = &result.incompressible_3d {
        return (
            Some(m.steps),
            Some(m.simplec_converged),
            Some(m.simplec_final_residual),
            Some(m.physical_time),
        );
    }
    if let Some(m) = &result.sod {
        return (Some(m.steps), None, None, Some(m.final_time));
    }
    if let Some(_m) = &result.diffusion {
        return (None, None, None, None);
    }
    (None, None, None, None)
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
        });
        run_case_with_manifest(&case).expect("run");
        let manifest_path = dir.join("run-manifest.json");
        assert!(manifest_path.is_file(), "manifest should exist");
        let text = std::fs::read_to_string(manifest_path).expect("read");
        assert!(text.contains("\"kind\": \"diffusion_1d_steady\""));
    }
}
