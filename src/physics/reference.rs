//! 可压缩流参考量与 Reynolds 数（无量纲化用）。
//!
//! 约定：\(L_{\mathrm{ref}}=1\,\mathrm{m}\)，\(U_{\mathrm{ref}}=a_\infty\)，
//! \(T_{\mathrm{ref}}=T_\infty\)，\(\mu_{\mathrm{ref}}=\mu(T_\infty)\)。
//!
//! 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §1。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::physics::{FreestreamParams, IdealGasEoS, ViscousPhysicsConfig};

/// 特征长度恒为 1 m。
pub const REFERENCE_LENGTH: Real = 1.0;

/// 无量纲化参考量（由来流与物性自动计算）。
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceScales {
    pub length: Real,
    pub velocity: Real,
    pub temperature: Real,
    pub viscosity: Real,
    pub density: Real,
    /// \(p_{\mathrm{ref}} = \rho_{\mathrm{ref}} U_{\mathrm{ref}}^2 = \gamma p_\infty\)。
    pub pressure: Real,
    pub reynolds: Real,
    /// 有量纲气体常数 \(R\)（输出还原用）。
    pub dimensional_gas_constant: Real,
}

impl ReferenceScales {
    /// 由来流静参数、EOS 与粘性配置构造参考量。
    pub fn from_freestream(
        eos: &IdealGasEoS,
        freestream: &FreestreamParams,
        viscous: Option<&ViscousPhysicsConfig>,
    ) -> Result<Self> {
        if freestream.temperature <= 0.0 {
            return Err(AsimuError::Config(
                "来流温度必须大于 0 才能构造参考量".to_string(),
            ));
        }
        if freestream.pressure <= 0.0 {
            return Err(AsimuError::Config(
                "来流压力必须大于 0 才能构造参考量".to_string(),
            ));
        }
        let velocity = eos.sound_speed(freestream.temperature)?;
        let density = eos.density(freestream.pressure, freestream.temperature)?;
        let pressure = density * velocity * velocity;
        let viscosity = match viscous {
            Some(v) => v
                .model
                .dynamic_viscosity(freestream.temperature)
                .map_err(|e| AsimuError::Config(format!("来流粘度参考量计算失败: {e}")))?,
            None => 0.0,
        };
        let reynolds = if viscosity > Real::EPSILON {
            density * velocity * REFERENCE_LENGTH / viscosity
        } else {
            Real::INFINITY
        };
        Ok(Self {
            length: REFERENCE_LENGTH,
            velocity,
            temperature: freestream.temperature,
            viscosity,
            density,
            pressure,
            reynolds,
            dimensional_gas_constant: eos.gas_constant,
        })
    }

    #[must_use]
    pub fn inv_reynolds(&self) -> Real {
        if self.reynolds.is_finite() && self.reynolds > Real::EPSILON {
            1.0 / self.reynolds
        } else {
            0.0
        }
    }

    /// \(t_{\mathrm{ref}} = L_{\mathrm{ref}}/U_{\mathrm{ref}}\)。
    #[must_use]
    pub fn time_scale(&self) -> Real {
        self.length / self.velocity
    }

    /// 无量纲 EOS：\(R^* = U_{\mathrm{ref}}^2/T_{\mathrm{ref}}\)。
    #[must_use]
    pub fn nondimensional_gas_constant(&self) -> Real {
        self.velocity * self.velocity / self.temperature
    }

    /// 输出还原用的有量纲 EOS。
    pub fn dimensional_eos(&self, gamma: Real) -> Result<IdealGasEoS> {
        IdealGasEoS::new(gamma, self.dimensional_gas_constant)
    }

    /// 热流密度尺度 \(\rho_{\mathrm{ref}} U_{\mathrm{ref}}^3\)。
    #[must_use]
    pub fn heat_flux_scale(&self) -> Real {
        self.density * self.velocity * self.velocity * self.velocity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ViscosityModel;

    #[test]
    fn freestream_reference_scales_match_mach8_air() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 8.0,
            pressure: 1000.0,
            temperature: 300.0,
            ..FreestreamParams::default()
        };
        let viscous = ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("v");
        let ref_scales = ReferenceScales::from_freestream(&eos, &fs, Some(&viscous)).expect("ref");
        assert!((ref_scales.length - 1.0).abs() < 1.0e-12);
        let a = eos.sound_speed(300.0).expect("a");
        assert!((ref_scales.velocity - a).abs() / a < 1.0e-10);
        let rho = eos.density(1000.0, 300.0).expect("rho");
        assert!((ref_scales.density - rho).abs() / rho < 1.0e-10);
        assert!((ref_scales.pressure - rho * a * a).abs() / (rho * a * a) < 1.0e-10);
        assert!(ref_scales.reynolds > 0.0 && ref_scales.reynolds.is_finite());
        assert!(
            (ref_scales.nondimensional_gas_constant() - eos.gamma * eos.gas_constant).abs()
                / (eos.gamma * eos.gas_constant)
                < 1.0e-10
        );
    }

    #[test]
    fn euler_reference_has_infinite_reynolds() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let ref_scales = ReferenceScales::from_freestream(&eos, &fs, None).expect("ref");
        assert!(!ref_scales.reynolds.is_finite());
        assert_eq!(ref_scales.inv_reynolds(), 0.0);
    }
}
