//! 来流与参考量状态的单一构造入口（无量纲 \(*\) 求解）。
//!
//! 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §2、§6。

use crate::core::Real;
use crate::error::Result;

use super::{
    ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState, ReferenceScales,
    ViscousPhysicsConfig,
};

/// 来流构造上下文：EOS。可压缩算例在 \(*\) 变量下求解，来流 primitive 经 `FreestreamContext` 构造。
#[derive(Debug, Clone, Copy)]
pub struct FreestreamContext<'a> {
    pub eos: &'a IdealGasEoS,
}

impl<'a> FreestreamContext<'a> {
    /// `reference` / `viscous` 保留参数位，供算例编排与 BC 路径统一签名。
    #[must_use]
    pub fn new(
        eos: &'a IdealGasEoS,
        _reference: Option<&ReferenceScales>,
        _viscous: Option<&ViscousPhysicsConfig>,
    ) -> Self {
        Self { eos }
    }

    /// 来流原始变量（BC ghost、诊断）。
    pub fn primitive(&self, params: &FreestreamParams) -> Result<PrimitiveState> {
        Ok(nondimensional_freestream_primitive(params))
    }

    /// 单 cell 来流守恒状态。
    pub fn conserved(&self, params: &FreestreamParams) -> Result<ConservedState> {
        ConservedState::from_primitive(self.eos, &self.primitive(params)?)
    }

    /// 由 \((p^*, T^*)\) 求 \(\rho^*\)（壁面 ghost 等）。
    #[must_use]
    pub fn density_from_pressure_temperature(&self, pressure: Real, temperature: Real) -> Real {
        let t = temperature.max(1.0e-30);
        pressure * self.eos.gamma / t
    }
}

/// 无量纲来流：\(\rho^*=1\)，\(p^*\)/\(T^*\) 来自缩放后的 `[freestream]`，\(u^*=M\)（\(a^*=1\)）。
fn nondimensional_freestream_primitive(params: &FreestreamParams) -> PrimitiveState {
    let dir = params.effective_direction();
    let speed = params.mach;
    PrimitiveState {
        density: 1.0,
        velocity: [dir[0] * speed, dir[1] * speed, dir[2] * speed],
        pressure: params.pressure,
        temperature: params.temperature,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{FreestreamContext, FreestreamParams, IdealGasEoS, ReferenceScales};

    #[test]
    fn context_nondimensional_freestream_has_unit_density_and_temperature() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let params = FreestreamParams {
            mach: 0.5,
            pressure: 1.0 / eos.gamma,
            temperature: 1.0,
            ..FreestreamParams::default()
        };
        let viscous = ViscousPhysicsConfig {
            temperature_ref: Some(300.0),
            ..ViscousPhysicsConfig::default()
        };
        let ctx = FreestreamContext::new(&eos, None, Some(&viscous));
        let prim = ctx.primitive(&params).expect("prim");
        assert!((prim.density - 1.0).abs() < 1.0e-12);
        assert!((prim.temperature - 1.0).abs() < 1.0e-12);
        assert!((prim.velocity[0] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn reference_scales_do_not_change_freestream_primitive() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let reference = ReferenceScales::from_freestream(&eos, &fs, None).expect("ref");
        let ctx = FreestreamContext::new(&eos, Some(&reference), None);
        let prim = ctx
            .primitive(&FreestreamParams {
                pressure: 1.0 / eos.gamma,
                temperature: 1.0,
                ..fs
            })
            .expect("prim");
        assert!((prim.density - 1.0).abs() < 1.0e-12);
    }
}
