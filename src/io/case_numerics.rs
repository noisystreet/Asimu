//! 算例 `[numerics]` 段解析（ADR 0016 / ADR 0017）。

use serde::Deserialize;

use crate::core::{ComputePrecision, ExecDevice, parse_compute_precision, parse_exec_backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CaseNumericsConfig {
    pub compute_precision: ComputePrecision,
    pub exec_device: ExecDevice,
}

#[derive(Debug, Deserialize)]
pub(super) struct NumericsToml {
    compute_precision: Option<String>,
    backend: Option<String>,
}

pub fn parse_numerics(raw: Option<&NumericsToml>) -> crate::error::Result<CaseNumericsConfig> {
    let Some(raw) = raw else {
        return Ok(CaseNumericsConfig::default());
    };
    let compute_precision = if let Some(ref value) = raw.compute_precision {
        parse_compute_precision(value)?
    } else {
        ComputePrecision::F64
    };
    let exec_device = if let Some(ref value) = raw.backend {
        parse_exec_backend(value)?
    } else {
        ExecDevice::Cpu
    };
    Ok(CaseNumericsConfig {
        compute_precision,
        exec_device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_f64_cpu_when_section_missing() {
        let config = parse_numerics(None).expect("default");
        assert_eq!(config.compute_precision, ComputePrecision::F64);
        assert_eq!(config.exec_device, ExecDevice::Cpu);
    }

    #[test]
    fn parses_explicit_f32_and_cuda() {
        let raw = NumericsToml {
            compute_precision: Some("f32".to_string()),
            backend: Some("cuda".to_string()),
        };
        let config = parse_numerics(Some(&raw)).expect("numerics");
        assert_eq!(config.compute_precision, ComputePrecision::F32);
        assert_eq!(config.exec_device, ExecDevice::GpuCuda);
    }

    #[test]
    fn rejects_unknown_backend() {
        let raw = NumericsToml {
            compute_precision: None,
            backend: Some("gpu-wgpu".to_string()),
        };
        let err = parse_numerics(Some(&raw)).expect_err("unknown backend");
        assert!(err.to_string().contains("backend"));
    }
}
