//! 物性参数与状态方程。

mod eos;
mod riemann_exact;
mod viscosity;

pub use eos::{ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState};
pub use riemann_exact::{
    RiemannPrimitive1d, RiemannProblem1d, SodProblem, sample_exact, sod_sample,
    solve_star_pressure_velocity,
};
pub use viscosity::{ViscosityModel, ViscousPhysicsConfig};

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 算例物性配置（扩散 + 可压缩流）。
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsConfig {
    pub diffusivity: Option<Real>,
    pub eos: Option<IdealGasEoS>,
    /// 层流粘性（`[navier_stokes]` 算例启用）。
    pub viscous: Option<ViscousPhysicsConfig>,
}

impl PhysicsConfig {
    #[must_use]
    pub fn diffusion_only(diffusivity: Real) -> Self {
        Self {
            diffusivity: Some(diffusivity),
            eos: None,
            viscous: None,
        }
    }

    #[must_use]
    pub fn compressible(eos: IdealGasEoS) -> Self {
        Self {
            diffusivity: None,
            eos: Some(eos),
            viscous: None,
        }
    }

    #[must_use]
    pub fn is_navier_stokes(&self) -> bool {
        self.viscous.is_some()
    }

    pub fn eos(&self) -> Result<IdealGasEoS> {
        self.eos
            .ok_or_else(|| AsimuError::Config("算例未配置 EOS".to_string()))
    }
}
