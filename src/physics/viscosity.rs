//! 层流动力粘度与热传导系数（Sutherland / 常数）。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::physics::IdealGasEoS;

/// 动力粘度模型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViscosityModel {
    /// \(\mu = \mathrm{const}\)。
    Constant { mu: Real },
    /// Sutherland 空气：\(\mu = \mu_{\mathrm{ref}} (T/T_{\mathrm{ref}})^{3/2} (T_{\mathrm{ref}}+S)/(T+S)\)。
    Sutherland {
        mu_ref: Real,
        t_ref: Real,
        sutherland_constant: Real,
    },
}

impl ViscosityModel {
    /// 海平面标准空气（\(\mu_{\mathrm{ref}}\) @ 273.15 K，\(S=110.4\) K）。
    pub const AIR_SUTHERLAND: Self = Self::Sutherland {
        mu_ref: 1.716e-5,
        t_ref: 273.15,
        sutherland_constant: 110.4,
    };

    pub fn constant(mu: Real) -> Result<Self> {
        if mu <= 0.0 {
            return Err(AsimuError::Config("动力粘度 mu 必须大于 0".to_string()));
        }
        Ok(Self::Constant { mu })
    }

    pub fn dynamic_viscosity(&self, temperature: Real) -> Result<Real> {
        if temperature <= 0.0 {
            return Err(AsimuError::Config(
                "温度必须大于 0 才能计算粘度".to_string(),
            ));
        }
        match self {
            Self::Constant { mu } => Ok(*mu),
            Self::Sutherland {
                mu_ref,
                t_ref,
                sutherland_constant,
            } => {
                if *t_ref <= 0.0 {
                    return Err(AsimuError::Config(
                        "Sutherland T_ref 必须大于 0".to_string(),
                    ));
                }
                let tr = temperature / t_ref;
                Ok(mu_ref * tr.powf(1.5) * (t_ref + sutherland_constant)
                    / (temperature + sutherland_constant))
            }
        }
    }

    /// f32 动力粘度（Sutherland / 常数；热路径用）。
    pub fn dynamic_viscosity_f32(&self, temperature: f32) -> Result<f32> {
        if temperature <= 0.0 {
            return Err(AsimuError::Config(
                "温度必须大于 0 才能计算粘度".to_string(),
            ));
        }
        match self {
            Self::Constant { mu } => Ok(*mu as f32),
            Self::Sutherland {
                mu_ref,
                t_ref,
                sutherland_constant,
            } => {
                if *t_ref <= 0.0 {
                    return Err(AsimuError::Config(
                        "Sutherland T_ref 必须大于 0".to_string(),
                    ));
                }
                let t_ref = *t_ref as f32;
                let tr = temperature / t_ref;
                Ok(
                    (*mu_ref as f32) * tr.powf(1.5) * (t_ref + *sutherland_constant as f32)
                        / (temperature + *sutherland_constant as f32),
                )
            }
        }
    }

    /// 热传导系数 \(\lambda = \mu c_p / \mathrm{Pr}\)。
    pub fn thermal_conductivity(
        &self,
        temperature: Real,
        eos: &IdealGasEoS,
        prandtl: Real,
    ) -> Result<Real> {
        if prandtl <= 0.0 {
            return Err(AsimuError::Config("Prandtl 数必须大于 0".to_string()));
        }
        let mu = self.dynamic_viscosity(temperature)?;
        let cp = eos.gamma * eos.gas_constant / (eos.gamma - 1.0);
        Ok(mu * cp / prandtl)
    }
}

/// 粘性通量物性配置（case `[physics]` / `[navier_stokes]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct ViscousPhysicsConfig {
    pub model: ViscosityModel,
    pub prandtl: Real,
    /// 无量纲 NS：粘性项乘子 \(1/\mathrm{Re}\)；有量纲时为 1.0。
    pub inv_reynolds: Real,
    /// 无量纲 Sutherland：\(T^*\) 还原为有量纲温度时的 \(T_{\mathrm{ref}}\)。
    pub viscosity_ref: Option<Real>,
    pub temperature_ref: Option<Real>,
}

impl Default for ViscousPhysicsConfig {
    fn default() -> Self {
        Self {
            model: ViscosityModel::AIR_SUTHERLAND,
            prandtl: 0.72,
            inv_reynolds: 1.0,
            viscosity_ref: None,
            temperature_ref: None,
        }
    }
}

impl ViscousPhysicsConfig {
    pub fn new(model: ViscosityModel, prandtl: Real) -> Result<Self> {
        if prandtl <= 0.0 {
            return Err(AsimuError::Config("Prandtl 数必须大于 0".to_string()));
        }
        Ok(Self {
            model,
            prandtl,
            inv_reynolds: 1.0,
            viscosity_ref: None,
            temperature_ref: None,
        })
    }

    fn dimensional_temperature(&self, temperature: Real) -> Real {
        self.dimensional_temperature_from_static(temperature)
    }

    /// 是否为无量纲 NS（`temperature_ref` 已设）。
    #[must_use]
    pub fn is_nondimensional(&self) -> bool {
        self.temperature_ref.is_some()
    }

    /// 静温：有量纲式 (1) \(T=p/(\rho R)\)；无量纲式 (2) \(T^*=p^*\gamma/\rho^*\)。
    ///
    /// 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §3.1。
    #[must_use]
    pub fn static_temperature(&self, pressure: Real, density: Real, eos: &IdealGasEoS) -> Real {
        let rho = density.max(1.0e-30);
        if self.is_nondimensional() {
            pressure / rho * eos.gamma
        } else {
            pressure / (rho * eos.gas_constant)
        }
    }

    /// 静温 f32（与 [`static_temperature`] 语义一致）。
    #[must_use]
    pub fn static_temperature_f32(&self, pressure: f32, density: f32, eos: &IdealGasEoS) -> f32 {
        let rho = density.max(1.0e-30_f32);
        let gamma = eos.gamma as f32;
        if self.is_nondimensional() {
            pressure / rho * gamma
        } else {
            pressure / (rho * eos.gas_constant as f32)
        }
    }

    /// 将 `static_temperature` 返回值转为 Sutherland 等模型所需的有量纲 \(T\) (K)。
    #[must_use]
    pub fn dimensional_temperature_from_static(&self, temperature: Real) -> Real {
        self.temperature_ref
            .map(|t_ref| temperature * t_ref)
            .unwrap_or(temperature)
    }

    /// 将 f32 静温转为 Sutherland 等有量纲温度 (K)。
    #[must_use]
    pub fn dimensional_temperature_from_static_f32(&self, temperature: f32) -> f32 {
        self.temperature_ref
            .map(|t_ref| temperature * t_ref as f32)
            .unwrap_or(temperature)
    }

    /// 比热：有量纲 \(c_p=\gamma R/(\gamma-1)\)；无量纲 \(c_p^*=1/(\gamma-1)\)（`T^*=p^*\gamma/\rho^*` 约定）。
    #[must_use]
    pub fn specific_heat_capacity(&self, eos: &IdealGasEoS) -> Real {
        if self.is_nondimensional() {
            1.0 / (eos.gamma - 1.0)
        } else {
            eos.gamma * eos.gas_constant / (eos.gamma - 1.0)
        }
    }

    #[must_use]
    pub fn specific_heat_capacity_f32(&self, eos: &IdealGasEoS) -> f32 {
        let gamma = eos.gamma as f32;
        if self.is_nondimensional() {
            1.0 / (gamma - 1.0)
        } else {
            gamma * eos.gas_constant as f32 / (gamma - 1.0)
        }
    }

    /// 热传导系数 \(\lambda\)（或无量纲 \(\lambda^*\)），含 \(1/\mathrm{Re}\) 缩放。
    pub fn thermal_conductivity_coefficient(
        &self,
        temperature_static: Real,
        eos: &IdealGasEoS,
    ) -> Result<Real> {
        let t_dim = self.dimensional_temperature(temperature_static);
        let mu = self.model.dynamic_viscosity(t_dim)?;
        let mut lambda = mu * self.specific_heat_capacity(eos) / self.prandtl;
        if let Some(mu_ref) = self.viscosity_ref {
            lambda *= self.inv_reynolds / mu_ref;
        }
        Ok(lambda)
    }

    /// 面平均 \(\mu,\lambda\)；无量纲模式下含 \(1/\mathrm{Re}\) 与 \(\mu/\mu_{\mathrm{ref}}\)。
    pub fn face_transport_coefficients(
        &self,
        t_left: Real,
        t_right: Real,
        eos: &IdealGasEoS,
    ) -> Result<(Real, Real)> {
        let t_l = self.dimensional_temperature(t_left);
        let t_r = self.dimensional_temperature(t_right);
        let mu_l = self.model.dynamic_viscosity(t_l)?;
        let mu_r = self.model.dynamic_viscosity(t_r)?;
        let mut mu = 0.5 * (mu_l + mu_r);
        let cp = self.specific_heat_capacity(eos);
        let mut lambda = mu * cp / self.prandtl;
        if let Some(mu_ref) = self.viscosity_ref {
            let scale = self.inv_reynolds / mu_ref;
            mu *= scale;
            lambda *= scale;
        }
        Ok((mu, lambda))
    }

    /// 面平均 \(\mu,\lambda\)（f32 热路径；无量纲缩放与 f64 一致）。
    pub fn face_transport_coefficients_f32(
        &self,
        t_left: f32,
        t_right: f32,
        eos: &IdealGasEoS,
    ) -> Result<(f32, f32)> {
        let t_l = self.dimensional_temperature_from_static_f32(t_left);
        let t_r = self.dimensional_temperature_from_static_f32(t_right);
        let mu_l = self.model.dynamic_viscosity_f32(t_l)?;
        let mu_r = self.model.dynamic_viscosity_f32(t_r)?;
        let mut mu = 0.5 * (mu_l + mu_r);
        let cp = self.specific_heat_capacity_f32(eos);
        let prandtl = self.prandtl as f32;
        let mut lambda = mu * cp / prandtl;
        if let Some(mu_ref) = self.viscosity_ref {
            let scale = (self.inv_reynolds / mu_ref) as f32;
            mu *= scale;
            lambda *= scale;
        }
        Ok((mu, lambda))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::physics::IdealGasEoS;
    use crate::physics::{FreestreamParams, ReferenceScales};

    #[test]
    fn static_temperature_matches_freestream_in_nondimensional_mode() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = ViscousPhysicsConfig {
            temperature_ref: Some(300.0),
            viscosity_ref: Some(1.716e-5),
            ..Default::default()
        };
        let p_star = 1.0 / eos.gamma;
        let t_star = viscous.static_temperature(p_star, 1.0, &eos);
        assert!((t_star - 1.0).abs() < 1.0e-12);
        let mu_ref = viscous.model.dynamic_viscosity(300.0).expect("mu ref");
        let mu_at_fs = viscous
            .model
            .dynamic_viscosity(viscous.dimensional_temperature_from_static(t_star))
            .expect("mu");
        assert!((mu_at_fs - mu_ref).abs() / mu_ref < 1.0e-10);
    }

    #[test]
    fn nondimensional_lambda_uses_cp_star_not_gamma_times_dimensional_cp() {
        let dim_eos = IdealGasEoS::AIR_STANDARD;
        let mut nd_eos = dim_eos;
        nd_eos.gas_constant = dim_eos.gamma * dim_eos.gas_constant;
        let viscous = ViscousPhysicsConfig {
            inv_reynolds: 1.0 / 1.0e6,
            viscosity_ref: Some(1.716e-5),
            temperature_ref: Some(300.0),
            ..Default::default()
        };
        let (mu, lambda) = viscous
            .face_transport_coefficients(1.0, 1.0, &nd_eos)
            .expect("tc");
        let cp_star = 1.0 / (dim_eos.gamma - 1.0);
        assert!((lambda / mu - cp_star / viscous.prandtl).abs() < 1.0e-12);
        let cp_wrong = dim_eos.gamma * nd_eos.gas_constant / (dim_eos.gamma - 1.0);
        assert!((lambda / mu - cp_wrong / viscous.prandtl).abs() > 1.0e-6);
    }

    #[test]
    fn nondimensional_face_mu_at_freestream_equals_inv_re() {
        let dim_eos = IdealGasEoS::AIR_STANDARD;
        let mut nd_eos = dim_eos;
        nd_eos.gas_constant = dim_eos.gamma * dim_eos.gas_constant;
        let fs = FreestreamParams {
            mach: 8.0,
            pressure: 1000.0,
            temperature: 300.0,
            ..FreestreamParams::default()
        };
        let viscous = ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("v");
        let reference =
            ReferenceScales::from_freestream(&dim_eos, &fs, Some(&viscous)).expect("ref");
        let mut nd_viscous = viscous.clone();
        nd_viscous.inv_reynolds = reference.inv_reynolds();
        nd_viscous.viscosity_ref = Some(reference.viscosity);
        nd_viscous.temperature_ref = Some(reference.temperature);
        let (mu, _) = nd_viscous
            .face_transport_coefficients(1.0, 1.0, &nd_eos)
            .expect("tc");
        let inv_re = reference.inv_reynolds();
        assert!(
            (mu - inv_re).abs() / inv_re < 1.0e-10,
            "freestream mu* should be 1/Re, got {mu} vs {inv_re}"
        );
    }

    #[test]
    fn face_transport_coefficients_f32_matches_f64_at_300k() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = ViscousPhysicsConfig::default();
        let t = 300.0_f32;
        let (mu_f64, lambda_f64) = viscous
            .face_transport_coefficients(t as Real, t as Real, &eos)
            .expect("f64");
        let (mu_f32, lambda_f32) = viscous
            .face_transport_coefficients_f32(t, t, &eos)
            .expect("f32");
        assert!(approx_eq(mu_f32 as Real, mu_f64, 1.0e-4));
        assert!(approx_eq(lambda_f32 as Real, lambda_f64, 1.0e-4));
    }

    #[test]
    fn static_temperature_dimensional_ideal_gas() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = ViscousPhysicsConfig::default();
        let p = 101_325.0;
        let t = 300.0;
        let rho = eos.density(p, t).expect("rho");
        let back = viscous.static_temperature(p, rho, &eos);
        assert!((back - t).abs() / t < 1.0e-10);
    }
}
