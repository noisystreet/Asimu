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
}

impl Default for ViscousPhysicsConfig {
    fn default() -> Self {
        Self {
            model: ViscosityModel::AIR_SUTHERLAND,
            prandtl: 0.72,
        }
    }
}

impl ViscousPhysicsConfig {
    pub fn new(model: ViscosityModel, prandtl: Real) -> Result<Self> {
        if prandtl <= 0.0 {
            return Err(AsimuError::Config("Prandtl 数必须大于 0".to_string()));
        }
        Ok(Self { model, prandtl })
    }
}
