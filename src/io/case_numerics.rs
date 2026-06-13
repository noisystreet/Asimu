//! 算例 `[numerics]` 段解析（ADR 0016）。

use serde::Deserialize;

use crate::core::{ComputePrecision, parse_compute_precision};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CaseNumericsConfig {
    pub compute_precision: ComputePrecision,
}

#[derive(Debug, Deserialize)]
pub(super) struct NumericsToml {
    compute_precision: Option<String>,
}

pub fn parse_numerics(raw: Option<&NumericsToml>) -> crate::error::Result<CaseNumericsConfig> {
    let Some(raw) = raw else {
        return Ok(CaseNumericsConfig::default());
    };
    let Some(ref value) = raw.compute_precision else {
        return Ok(CaseNumericsConfig::default());
    };
    Ok(CaseNumericsConfig {
        compute_precision: parse_compute_precision(value)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_f64_when_section_missing() {
        let config = parse_numerics(None).expect("default");
        assert_eq!(config.compute_precision, ComputePrecision::F64);
    }

    #[test]
    fn parses_explicit_f32() {
        let raw = NumericsToml {
            compute_precision: Some("f32".to_string()),
        };
        let config = parse_numerics(Some(&raw)).expect("f32");
        assert_eq!(config.compute_precision, ComputePrecision::F32);
    }
}
