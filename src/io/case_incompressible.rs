//! 不可压缩 case 段解析。

use serde::Deserialize;

use crate::core::Real;
use crate::error::{AsimuError, Result};

/// 不可压缩 Navier-Stokes I0/I1 配置（SI 输入，解析后切换为星号量）。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleCaseConfig {
    pub pressure: Real,
    pub velocity: [Real; 3],
    pub density: Real,
    pub kinematic_viscosity: Real,
    pub reference: IncompressibleReferenceConfig,
}

/// 不可压缩无量纲化参考量配置（SI 输入）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleReferenceConfig {
    pub length: Real,
    pub velocity: Real,
}

#[derive(Debug, Deserialize)]
pub(super) struct IncompressibleToml {
    pressure: Option<Real>,
    velocity: Option<[Real; 3]>,
    density: Option<Real>,
    kinematic_viscosity: Option<Real>,
    reference: Option<IncompressibleReferenceToml>,
}

#[derive(Debug, Deserialize)]
struct IncompressibleReferenceToml {
    length: Option<Real>,
    velocity: Option<Real>,
}

pub(super) fn parse_incompressible_config(
    raw: &IncompressibleToml,
) -> Result<IncompressibleCaseConfig> {
    let density = raw.density.unwrap_or(1.0);
    let kinematic_viscosity = raw.kinematic_viscosity.unwrap_or(1.0e-3);
    if density <= 0.0 {
        return Err(AsimuError::Config(
            "[incompressible].density 必须大于 0".to_string(),
        ));
    }
    if kinematic_viscosity < 0.0 {
        return Err(AsimuError::Config(
            "[incompressible].kinematic_viscosity 不能为负".to_string(),
        ));
    }
    Ok(IncompressibleCaseConfig {
        pressure: raw.pressure.unwrap_or(0.0),
        velocity: raw.velocity.unwrap_or([0.0, 0.0, 0.0]),
        density,
        kinematic_viscosity,
        reference: parse_incompressible_reference(raw.reference.as_ref())?,
    })
}

fn parse_incompressible_reference(
    raw: Option<&IncompressibleReferenceToml>,
) -> Result<IncompressibleReferenceConfig> {
    let raw = raw.ok_or_else(|| {
        AsimuError::Config(
            "不可压缩算例须指定 [incompressible.reference] length 与 velocity".to_string(),
        )
    })?;
    let length = raw
        .length
        .ok_or_else(|| AsimuError::Config("[incompressible.reference] 缺少 length".to_string()))?;
    let velocity = raw.velocity.ok_or_else(|| {
        AsimuError::Config("[incompressible.reference] 缺少 velocity".to_string())
    })?;
    if length <= 0.0 {
        return Err(AsimuError::Config(
            "[incompressible.reference].length 必须大于 0".to_string(),
        ));
    }
    if velocity <= 0.0 {
        return Err(AsimuError::Config(
            "[incompressible.reference].velocity 必须大于 0".to_string(),
        ));
    }
    Ok(IncompressibleReferenceConfig { length, velocity })
}
