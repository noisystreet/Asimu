//! 低马赫预处理配置（可压缩非结构 P1–P4）。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::physics::PrimitiveState;

/// 低马赫预处理向常规可压缩形式的退化方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowMachBlend {
    /// \(M_\text{cut}<M<M_\text{max}\) 线性混合预处理与常规声速项。
    Smooth,
    /// \(M\ge M_\text{max}\) 时直接退化为常规可压缩谱半径。
    HardCut,
}

impl LowMachBlend {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw {
            "smooth" => Ok(Self::Smooth),
            "hard_cut" => Ok(Self::HardCut),
            other => Err(AsimuError::Config(format!(
                "不支持的 [time].low_mach_blend \"{other}\"（允许 smooth | hard_cut）"
            ))),
        }
    }
}

/// 低马赫预处理：谱半径/扫掠与可选预处理 Jacobian。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowMachPreconditioningConfig {
    pub mach_cutoff: Real,
    pub max_mach: Real,
    pub blend: LowMachBlend,
    /// 为 true 时 block_lusgs / `lu_sgs` 块双扫使用预处理 Roe 面 Jacobian。
    pub jacobian: bool,
}

impl LowMachPreconditioningConfig {
    /// 解析 `[time]` 低马赫配置。
    pub fn parse(
        enabled: bool,
        mach_cutoff: Option<Real>,
        max_mach: Option<Real>,
        blend: Option<&str>,
        jacobian: Option<bool>,
    ) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }
        let mach_cutoff = mach_cutoff.unwrap_or(0.1);
        if !(mach_cutoff.is_finite() && mach_cutoff > 0.0 && mach_cutoff <= 1.0) {
            return Err(AsimuError::Config(
                "[time].low_mach_mach_cutoff 须在 (0, 1] 内".to_string(),
            ));
        }
        let max_mach = max_mach.unwrap_or(0.3);
        if !(max_mach.is_finite() && max_mach > mach_cutoff && max_mach <= 1.0) {
            return Err(AsimuError::Config(
                "[time].low_mach_max_mach 须满足 M_cut < M_max ≤ 1".to_string(),
            ));
        }
        let blend = match blend {
            Some(raw) => LowMachBlend::parse(raw)?,
            None => LowMachBlend::Smooth,
        };
        Ok(Some(Self {
            mach_cutoff,
            max_mach,
            blend,
            jacobian: jacobian.unwrap_or(false),
        }))
    }

    /// 预处理声速乘子 \(\beta_\text{eff}\)（\(M_\text{loc}=|u|/a\)）。
    #[must_use]
    pub fn sound_speed_multiplier(&self, mach: Real) -> Real {
        let mach = mach.max(0.0);
        let beta_lm = mach.max(self.mach_cutoff).min(1.0);

        match self.blend {
            LowMachBlend::HardCut => {
                if mach >= self.max_mach {
                    1.0
                } else {
                    beta_lm
                }
            }
            LowMachBlend::Smooth => {
                if mach >= self.max_mach {
                    1.0
                } else if mach <= self.mach_cutoff {
                    beta_lm
                } else {
                    let span = self.max_mach - self.mach_cutoff;
                    debug_assert!(span > 0.0);
                    let weight = (self.max_mach - mach) / span;
                    weight * beta_lm + (1.0 - weight)
                }
            }
        }
    }

    #[must_use]
    pub fn sound_speed_multiplier_f32(&self, mach: f32) -> f32 {
        self.sound_speed_multiplier(mach as Real) as f32
    }

    #[must_use]
    pub fn sound_speed_multiplier_from_primitive(
        &self,
        prim: &PrimitiveState,
        gamma: Real,
    ) -> Real {
        let rho = prim.density.max(1.0e-30);
        let u = prim.velocity;
        let speed = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let a = (gamma * prim.pressure.max(1.0e-30) / rho).sqrt();
        let mach = if a > 0.0 { speed / a } else { 0.0 };
        self.sound_speed_multiplier(mach)
    }

    /// 内界面两侧 \(\beta\) 算术平均（预处理 Jacobian / 特征速度共用）。
    #[must_use]
    pub fn face_average_sound_speed_multiplier(
        &self,
        prim_l: &PrimitiveState,
        prim_r: &PrimitiveState,
        gamma: Real,
    ) -> Real {
        0.5 * (self.sound_speed_multiplier_from_primitive(prim_l, gamma)
            + self.sound_speed_multiplier_from_primitive(prim_r, gamma))
    }
}

#[cfg(test)]
mod tests {
    use super::{LowMachBlend, LowMachPreconditioningConfig};

    fn cfg(blend: LowMachBlend) -> LowMachPreconditioningConfig {
        LowMachPreconditioningConfig {
            mach_cutoff: 0.1,
            max_mach: 0.3,
            blend,
            jacobian: false,
        }
    }

    #[test]
    fn disabled_returns_none() {
        assert!(
            LowMachPreconditioningConfig::parse(false, Some(0.2), Some(0.4), None, None)
                .expect("parse")
                .is_none()
        );
    }

    #[test]
    fn enabled_uses_defaults() {
        let cfg = LowMachPreconditioningConfig::parse(true, None, None, None, None)
            .expect("parse")
            .expect("cfg");
        assert!((cfg.mach_cutoff - 0.1).abs() < 1.0e-12);
        assert!((cfg.max_mach - 0.3).abs() < 1.0e-12);
        assert_eq!(cfg.blend, LowMachBlend::Smooth);
        assert!(!cfg.jacobian);
    }

    #[test]
    fn parses_jacobian_flag() {
        let cfg = LowMachPreconditioningConfig::parse(true, None, None, None, Some(true))
            .expect("parse")
            .expect("cfg");
        assert!(cfg.jacobian);
    }

    #[test]
    fn rejects_invalid_cutoff() {
        let err = LowMachPreconditioningConfig::parse(true, Some(0.0), None, None, None)
            .expect_err("invalid");
        assert!(err.to_string().contains("low_mach_mach_cutoff"));
    }

    #[test]
    fn rejects_max_mach_not_above_cutoff() {
        let err = LowMachPreconditioningConfig::parse(true, Some(0.2), Some(0.15), None, None)
            .expect_err("invalid");
        assert!(err.to_string().contains("low_mach_max_mach"));
    }

    #[test]
    fn smooth_multiplier_at_cutoff_and_above_max() {
        let cfg = cfg(LowMachBlend::Smooth);
        assert!((cfg.sound_speed_multiplier(0.05) - 0.1).abs() < 1.0e-12);
        assert!((cfg.sound_speed_multiplier(0.5) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn smooth_multiplier_linear_blend_mid_range() {
        let cfg = cfg(LowMachBlend::Smooth);
        assert!((cfg.sound_speed_multiplier(0.2) - 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn hard_cut_multiplier_switches_at_max_mach() {
        let cfg = cfg(LowMachBlend::HardCut);
        assert!((cfg.sound_speed_multiplier(0.15) - 0.15).abs() < 1.0e-12);
        assert!((cfg.sound_speed_multiplier(0.35) - 1.0).abs() < 1.0e-12);
    }
}
