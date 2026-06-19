//! 低马赫预处理配置（可压缩非结构 P1）。

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 低马赫预处理（P1）：声速项按局部 \(\beta=\max(M, M_{\text{cut}})\) 缩放。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowMachPreconditioningConfig {
    pub mach_cutoff: Real,
}

impl LowMachPreconditioningConfig {
    /// 解析 `[time]` 低马赫配置。
    ///
    /// - `enabled = false` 时，无论是否给出 `mach_cutoff` 都返回 `None`
    /// - `enabled = true` 时，`mach_cutoff` 默认 `0.1`
    pub fn parse(enabled: bool, mach_cutoff: Option<Real>) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }
        let mach_cutoff = mach_cutoff.unwrap_or(0.1);
        if !(mach_cutoff.is_finite() && mach_cutoff > 0.0 && mach_cutoff <= 1.0) {
            return Err(AsimuError::Config(
                "[time].low_mach_mach_cutoff 须在 (0, 1] 内".to_string(),
            ));
        }
        Ok(Some(Self { mach_cutoff }))
    }
}

#[cfg(test)]
mod tests {
    use super::LowMachPreconditioningConfig;

    #[test]
    fn disabled_returns_none() {
        assert!(
            LowMachPreconditioningConfig::parse(false, Some(0.2))
                .expect("parse")
                .is_none()
        );
    }

    #[test]
    fn enabled_uses_default_cutoff() {
        let cfg = LowMachPreconditioningConfig::parse(true, None)
            .expect("parse")
            .expect("cfg");
        assert!((cfg.mach_cutoff - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_invalid_cutoff() {
        let err = LowMachPreconditioningConfig::parse(true, Some(0.0)).expect_err("invalid");
        assert!(err.to_string().contains("low_mach_mach_cutoff"));
    }
}
