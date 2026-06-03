//! 来流与参考量状态的单一构造入口（有量纲 / 无量纲）。
//!
//! 理论：[`docs/theory/nondimensional.md`](../../docs/theory/nondimensional.md) §2、§6。

use crate::core::Real;
use crate::error::Result;

use super::{
    ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState, ReferenceScales,
    ViscousPhysicsConfig,
};

/// 来流构造模式（与 `CaseSpec.reference` / `ViscousPhysicsConfig.temperature_ref` 一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreestreamMode {
    Dimensional,
    Nondimensional,
}

impl FreestreamMode {
    /// 算例编排层：`reference` 存在即无量纲。
    #[must_use]
    pub fn from_reference(reference: Option<&ReferenceScales>) -> Self {
        if reference.is_some() {
            Self::Nondimensional
        } else {
            Self::Dimensional
        }
    }

    /// BC / 单测：无 `reference` 时由粘性配置推断。
    #[must_use]
    pub fn from_viscous(viscous: Option<&ViscousPhysicsConfig>) -> Self {
        if viscous.is_some_and(|v| v.is_nondimensional()) {
            Self::Nondimensional
        } else {
            Self::Dimensional
        }
    }

    #[must_use]
    pub fn is_nondimensional(self) -> bool {
        matches!(self, Self::Nondimensional)
    }
}

/// 来流构造上下文：EOS + 模式。初始场、远场/入口 ghost、测试 fixture 均经此构造。
#[derive(Debug, Clone, Copy)]
pub struct FreestreamContext<'a> {
    pub eos: &'a IdealGasEoS,
    pub mode: FreestreamMode,
}

impl<'a> FreestreamContext<'a> {
    /// 优先 `reference`（算例路径），否则 `viscous`（单测 / BC 路径）。
    #[must_use]
    pub fn new(
        eos: &'a IdealGasEoS,
        reference: Option<&ReferenceScales>,
        viscous: Option<&ViscousPhysicsConfig>,
    ) -> Self {
        let mode = if reference.is_some() {
            FreestreamMode::Nondimensional
        } else {
            FreestreamMode::from_viscous(viscous)
        };
        Self { eos, mode }
    }

    #[must_use]
    pub fn dimensional(eos: &'a IdealGasEoS) -> Self {
        Self {
            eos,
            mode: FreestreamMode::Dimensional,
        }
    }

    #[must_use]
    pub fn nondimensional(eos: &'a IdealGasEoS) -> Self {
        Self {
            eos,
            mode: FreestreamMode::Nondimensional,
        }
    }

    /// 来流原始变量（BC ghost、诊断）。
    pub fn primitive(&self, params: &FreestreamParams) -> Result<PrimitiveState> {
        if self.mode.is_nondimensional() {
            Ok(nondimensional_freestream_primitive(params))
        } else {
            self.eos.freestream_primitive(
                params.mach,
                params.pressure,
                params.temperature,
                params.effective_direction(),
            )
        }
    }

    /// 单 cell 来流守恒状态。
    pub fn conserved(&self, params: &FreestreamParams) -> Result<ConservedState> {
        ConservedState::from_primitive(self.eos, &self.primitive(params)?)
    }

    /// 由 \((p, T)\) 求 \(\rho\)（壁面 ghost 等）；无量纲用 \(T^*=p^*\gamma/\rho^*\) 的逆关系。
    #[must_use]
    pub fn density_from_pressure_temperature(&self, pressure: Real, temperature: Real) -> Real {
        let t = temperature.max(1.0e-30);
        if self.mode.is_nondimensional() {
            pressure * self.eos.gamma / t
        } else {
            pressure / (self.eos.gas_constant * t)
        }
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
    fn context_primitive_matches_dimensional_eos() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let params = FreestreamParams {
            mach: 0.5,
            pressure: 101_325.0,
            temperature: 288.15,
            ..FreestreamParams::default()
        };
        let ctx = FreestreamContext::dimensional(&eos);
        let from_ctx = ctx.primitive(&params).expect("ctx");
        let from_eos = eos
            .freestream_primitive(
                params.mach,
                params.pressure,
                params.temperature,
                params.effective_direction(),
            )
            .expect("eos");
        assert_eq!(from_ctx, from_eos);
    }

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
    fn reference_overrides_viscous_for_mode() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let reference = ReferenceScales::from_freestream(&eos, &fs, None).expect("ref");
        let ctx = FreestreamContext::new(&eos, Some(&reference), None);
        assert!(ctx.mode.is_nondimensional());
    }
}
