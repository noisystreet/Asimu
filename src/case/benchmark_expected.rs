//! `tests/benchmarks/*/expected.json` 解析与校验（V&V 机器可读契约）。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{AsimuError, Result};
use crate::io::CaseSpec;

/// `expected.json` schema 版本（见 `docs/BENCHMARKS.md` §5）。
pub const BENCHMARK_EXPECTED_SCHEMA_VERSION: u32 = 1;

/// 解析后的 benchmark 期望文件（CI / manifest 共用）。
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkExpected {
    pub schema_version: u32,
    pub benchmark_id: String,
    pub asimu_min_version: String,
    pub status: Option<String>,
    pub quantity_names: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkExpectedFile {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    benchmark_id: String,
    asimu_min_version: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    quantities: Vec<BenchmarkExpectedQuantity>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkExpectedQuantity {
    name: String,
}

const fn default_schema_version() -> u32 {
    BENCHMARK_EXPECTED_SCHEMA_VERSION
}

/// 算例目录下的 `expected.json` 路径（若存在）。
#[must_use]
pub fn benchmark_expected_path(case: &CaseSpec) -> Option<PathBuf> {
    let case_dir = case.case_dir.as_deref()?;
    case.benchmark_id.as_ref()?;
    Some(case_dir.join("expected.json"))
}

/// 读取并校验 `expected.json` 与算例 `benchmark_id` 一致。
pub fn load_benchmark_expected(path: &Path, case: &CaseSpec) -> Result<BenchmarkExpected> {
    let text = std::fs::read_to_string(path)?;
    parse_benchmark_expected(&text, case)
}

pub fn parse_benchmark_expected(text: &str, case: &CaseSpec) -> Result<BenchmarkExpected> {
    let raw: BenchmarkExpectedFile = serde_json::from_str(text)
        .map_err(|err| AsimuError::Config(format!("expected.json 解析失败: {err}")))?;
    if raw.schema_version > BENCHMARK_EXPECTED_SCHEMA_VERSION {
        return Err(AsimuError::Config(format!(
            "expected.json schema_version {} 高于支持的 {}",
            raw.schema_version, BENCHMARK_EXPECTED_SCHEMA_VERSION
        )));
    }
    if let Some(case_id) = case.benchmark_id.as_deref() {
        if raw.benchmark_id != case_id {
            return Err(AsimuError::Config(format!(
                "expected.json benchmark_id \"{}\" 与 case \"{}\" 不一致",
                raw.benchmark_id, case_id
            )));
        }
    }
    Ok(BenchmarkExpected {
        schema_version: raw.schema_version,
        benchmark_id: raw.benchmark_id,
        asimu_min_version: raw.asimu_min_version,
        status: raw.status,
        quantity_names: raw.quantities.into_iter().map(|q| q.name).collect(),
    })
}

/// 若算例目录存在 `expected.json` 则加载（缺失时不报错）。
pub fn try_load_benchmark_expected(case: &CaseSpec) -> Result<Option<BenchmarkExpected>> {
    let Some(path) = benchmark_expected_path(case) else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    load_benchmark_expected(&path, case).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn parses_taylor_green_expected_json() {
        let path = Path::new("tests/benchmarks/taylor_green_3d/expected.json");
        let case = crate::io::load_case(&path.with_file_name("case.toml")).expect("load case");
        let expected = load_benchmark_expected(path, &case).expect("parse expected");
        assert_eq!(expected.benchmark_id, "taylor_green_3d");
        assert_eq!(expected.schema_version, 1);
        assert_eq!(
            expected.status.as_deref(),
            Some("i3_piso_bdf1_kinetic_decay_vv")
        );
        assert!(!expected.quantity_names.is_empty());
    }
}
