//! 物性参数与状态方程。

mod eos;

pub use eos::{
    ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState,
};

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 算例物性配置（扩散 + 可压缩流）。
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsConfig {
    pub diffusivity: Option<Real>,
    pub eos: Option<IdealGasEoS>,
}

impl PhysicsConfig {
    #[must_use]
    pub fn diffusion_only(diffusivity: Real) -> Self {
        Self {
            diffusivity: Some(diffusivity),
            eos: None,
        }
    }

    #[must_use]
    pub fn compressible(eos: IdealGasEoS) -> Self {
        Self {
            diffusivity: None,
            eos: Some(eos),
        }
    }

    pub fn eos(&self) -> Result<IdealGasEoS> {
        self.eos
            .ok_or_else(|| AsimuError::Config("算例未配置 EOS".to_string()))
    }
}
