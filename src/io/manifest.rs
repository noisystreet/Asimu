//! Run Manifest（`output/run-manifest.json`）序列化与写出。

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::info;

use crate::core::Real;
use crate::error::Result;
use crate::io::{CaseSpec, CaseTimeMode, resolve_case_output_path};

/// Manifest schema 版本（见 `docs/DATA_MODEL.md` §10、`docs/BENCHMARKS.md` §5）。
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// 运行清单（v0.3 最小字段集）。
#[derive(Debug, Clone, PartialEq)]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub asimu_version: String,
    pub config_hash: String,
    pub case_name: String,
    pub benchmark_id: Option<String>,
    pub benchmark_status: Option<String>,
    pub time_mode: &'static str,
    pub incompressible_time_advance: Option<&'static str>,
    pub started_at_unix: f64,
    pub finished_at_unix: f64,
    pub solve: ManifestSolveSummary,
    pub observability: ManifestObservability,
    pub output_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ManifestSolveSummary {
    pub kind: String,
    pub summary: String,
    pub steps: Option<u64>,
    pub converged: Option<bool>,
    pub residual_log10: Option<Real>,
    pub final_time: Option<Real>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ManifestObservability {
    pub wall_time_sec: Real,
}

/// 若算例配置了 `[output]`，写出 `{output.dir}/run-manifest.json`。
pub fn maybe_write_run_manifest(
    case: &CaseSpec,
    manifest: &RunManifest,
) -> Result<Option<PathBuf>> {
    let Some(output) = &case.output else {
        return Ok(None);
    };
    let path =
        resolve_case_output_path(case.case_dir.as_deref(), &output.dir, "run-manifest.json")?;
    write_run_manifest(&path, manifest)?;
    info!(path = %path.display(), run_id = %manifest.run_id, "已写出 run manifest");
    Ok(Some(path))
}

pub fn write_run_manifest(path: &Path, manifest: &RunManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = manifest.to_json()?;
    let mut file = fs::File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

impl RunManifest {
    fn to_json(&self) -> Result<String> {
        let benchmark = match &self.benchmark_id {
            Some(id) => format!("\"{}\"", escape_json(id)),
            None => "null".to_string(),
        };
        let benchmark_status = match &self.benchmark_status {
            Some(status) => format!("\"{}\"", escape_json(status)),
            None => "null".to_string(),
        };
        let incompressible_advance = match self.incompressible_time_advance {
            Some(label) => format!("\"{}\"", escape_json(label)),
            None => "null".to_string(),
        };
        let steps = optional_u64(self.solve.steps);
        let converged = optional_bool(self.solve.converged);
        let residual_log10 = optional_f64(self.solve.residual_log10);
        let final_time = optional_f64(self.solve.final_time);
        let output_paths: Vec<String> = self
            .output_paths
            .iter()
            .map(|p| format!("\"{}\"", escape_json(&p.display().to_string())))
            .collect();
        Ok(format!(
            r#"{{
  "schema_version": {},
  "run_id": "{}",
  "asimu_version": "{}",
  "config_hash": "{}",
  "case_name": "{}",
  "benchmark_id": {},
  "benchmark_status": {},
  "time": {{ "mode": "{}", "incompressible_advance": {} }},
  "started_at_unix": {:.6},
  "finished_at_unix": {:.6},
  "solve": {{
    "kind": "{}",
    "summary": "{}",
    "steps": {},
    "converged": {},
    "residual_log10": {},
    "final_time": {}
  }},
  "observability": {{ "wall_time_sec": {:.6} }},
  "output_paths": [{}]
}}"#,
            self.schema_version,
            escape_json(&self.run_id),
            escape_json(&self.asimu_version),
            escape_json(&self.config_hash),
            escape_json(&self.case_name),
            benchmark,
            benchmark_status,
            self.time_mode,
            incompressible_advance,
            self.started_at_unix,
            self.finished_at_unix,
            escape_json(&self.solve.kind),
            escape_json(&self.solve.summary),
            steps,
            converged,
            residual_log10,
            final_time,
            self.observability.wall_time_sec,
            output_paths.join(", "),
        ))
    }
}

#[must_use]
pub fn unix_timestamp_secs(now: SystemTime) -> f64 {
    now.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn config_hash(case: &CaseSpec) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    case.name.hash(&mut hasher);
    case.benchmark_id.hash(&mut hasher);
    case.mesh.num_cells().hash(&mut hasher);
    case.time.max_steps.hash(&mut hasher);
    case.numerics.compute_precision.hash(&mut hasher);
    case.numerics.exec_device.hash(&mut hasher);
    if let Some(tol) = case.time.tolerance {
        tol.to_bits().hash(&mut hasher);
    }
    case.time.mode.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub fn time_mode_label(mode: CaseTimeMode) -> &'static str {
    match mode {
        CaseTimeMode::Steady => "steady",
        CaseTimeMode::Transient => "transient",
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn optional_u64(v: Option<u64>) -> String {
    v.map(|n| n.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn optional_bool(v: Option<bool>) -> String {
    v.map(|b| b.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn optional_f64(v: Option<Real>) -> String {
    v.map(|x| format!("{x:.12}"))
        .unwrap_or_else(|| "null".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_json_contains_core_fields() {
        let manifest = RunManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            run_id: "test-run".to_string(),
            asimu_version: "0.1.0".to_string(),
            config_hash: "abc".to_string(),
            case_name: "demo".to_string(),
            benchmark_id: Some("bench".to_string()),
            benchmark_status: Some("smoke".to_string()),
            time_mode: "steady",
            incompressible_time_advance: None,
            started_at_unix: 1.0,
            finished_at_unix: 2.0,
            solve: ManifestSolveSummary {
                kind: "diffusion_1d".to_string(),
                summary: "ok".to_string(),
                steps: Some(10),
                converged: Some(true),
                residual_log10: Some(-8.0),
                final_time: None,
            },
            observability: ManifestObservability { wall_time_sec: 0.5 },
            output_paths: vec![PathBuf::from("output/a.csv")],
        };
        let json = manifest.to_json().expect("json");
        assert!(json.contains("\"schema_version\": 2"));
        assert!(json.contains("\"benchmark_id\": \"bench\""));
        assert!(json.contains("\"benchmark_status\": \"smoke\""));
        assert!(json.contains("\"incompressible_advance\": null"));
        assert!(json.contains("\"wall_time_sec\": 0.500000"));
    }

    #[test]
    fn config_hash_is_stable_for_same_inputs() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        "case-a".hash(&mut hasher);
        let a = format!("{:016x}", hasher.finish());
        let mut hasher = DefaultHasher::new();
        "case-a".hash(&mut hasher);
        let b = format!("{:016x}", hasher.finish());
        assert_eq!(a, b);
    }
}
