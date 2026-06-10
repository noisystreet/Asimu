//! 不可压缩 case 段解析。

use serde::Deserialize;

use crate::core::Real;
use crate::discretization::IncompressibleConvectionScheme;
use crate::error::{AsimuError, Result};
use crate::linalg::GmresConfig;
use crate::solver::IncompressibleLinearSolverConfig;

/// 不可压缩 Navier-Stokes I0/I1 配置（SI 输入，解析后切换为星号量）。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleCaseConfig {
    pub pressure: Real,
    pub velocity: [Real; 3],
    pub body_force: [Real; 3],
    pub density: Real,
    pub kinematic_viscosity: Real,
    pub velocity_under_relaxation: Real,
    pub pressure_under_relaxation: Real,
    pub convection_scheme: IncompressibleConvectionScheme,
    pub piso_correctors: usize,
    pub linear_solvers: IncompressibleLinearSolverConfig,
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
    body_force: Option<[Real; 3]>,
    density: Option<Real>,
    kinematic_viscosity: Option<Real>,
    velocity_under_relaxation: Option<Real>,
    pressure_under_relaxation: Option<Real>,
    convection_scheme: Option<String>,
    piso_correctors: Option<usize>,
    linear: Option<IncompressibleLinearToml>,
    reference: Option<IncompressibleReferenceToml>,
}

#[derive(Debug, Deserialize)]
struct IncompressibleLinearToml {
    momentum: Option<IncompressibleGmresToml>,
    pressure: Option<IncompressibleGmresToml>,
}

#[derive(Debug, Deserialize)]
struct IncompressibleGmresToml {
    solver: Option<String>,
    restart: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<Real>,
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
    let velocity_under_relaxation = raw.velocity_under_relaxation.unwrap_or(1.0);
    let pressure_under_relaxation = raw.pressure_under_relaxation.unwrap_or(1.0);
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
    if !(0.0..=1.0).contains(&velocity_under_relaxation) || velocity_under_relaxation == 0.0 {
        return Err(AsimuError::Config(
            "[incompressible].velocity_under_relaxation 必须位于 (0, 1]".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&pressure_under_relaxation) || pressure_under_relaxation == 0.0 {
        return Err(AsimuError::Config(
            "[incompressible].pressure_under_relaxation 必须位于 (0, 1]".to_string(),
        ));
    }
    Ok(IncompressibleCaseConfig {
        pressure: raw.pressure.unwrap_or(0.0),
        velocity: raw.velocity.unwrap_or([0.0, 0.0, 0.0]),
        body_force: validate_body_force(raw.body_force.unwrap_or([0.0, 0.0, 0.0]))?,
        density,
        kinematic_viscosity,
        velocity_under_relaxation,
        pressure_under_relaxation,
        convection_scheme: parse_convection_scheme(raw.convection_scheme.as_deref())?,
        piso_correctors: parse_piso_correctors(raw.piso_correctors)?,
        linear_solvers: parse_linear_solvers(raw.linear.as_ref())?,
        reference: parse_incompressible_reference(raw.reference.as_ref())?,
    })
}

fn parse_convection_scheme(raw: Option<&str>) -> Result<IncompressibleConvectionScheme> {
    match raw.unwrap_or("upwind").trim().to_ascii_lowercase().as_str() {
        "upwind" | "first_order" | "first-order" => Ok(IncompressibleConvectionScheme::Upwind),
        "central" | "central2" | "second_order" | "second-order" => {
            Ok(IncompressibleConvectionScheme::Central)
        }
        other => Err(AsimuError::Config(format!(
            "[incompressible].convection_scheme 不支持 \"{other}\""
        ))),
    }
}

fn parse_piso_correctors(raw: Option<usize>) -> Result<usize> {
    let value = raw.unwrap_or(1);
    if value == 0 {
        return Err(AsimuError::Config(
            "[incompressible].piso_correctors 必须大于 0".to_string(),
        ));
    }
    Ok(value)
}

fn validate_body_force(value: [Real; 3]) -> Result<[Real; 3]> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(AsimuError::Config(
            "[incompressible].body_force 分量必须为有限值".to_string(),
        ));
    }
    Ok(value)
}

fn parse_linear_solvers(
    raw: Option<&IncompressibleLinearToml>,
) -> Result<IncompressibleLinearSolverConfig> {
    let defaults = IncompressibleLinearSolverConfig::default();
    let Some(raw) = raw else {
        return Ok(defaults);
    };
    Ok(IncompressibleLinearSolverConfig {
        momentum: parse_gmres_config(raw.momentum.as_ref(), defaults.momentum, "momentum")?,
        pressure: parse_gmres_config(raw.pressure.as_ref(), defaults.pressure, "pressure")?,
    })
}

fn parse_gmres_config(
    raw: Option<&IncompressibleGmresToml>,
    defaults: GmresConfig,
    name: &str,
) -> Result<GmresConfig> {
    let Some(raw) = raw else {
        return Ok(defaults);
    };
    if let Some(solver) = raw.solver.as_deref() {
        if solver != "gmres" {
            return Err(AsimuError::Config(format!(
                "[incompressible.linear.{name}].solver 当前仅支持 \"gmres\""
            )));
        }
    }
    let config = GmresConfig {
        restart: raw.restart.unwrap_or(defaults.restart),
        max_iters: raw.max_iters.unwrap_or(defaults.max_iters),
        tolerance: raw.tolerance.unwrap_or(defaults.tolerance),
    };
    GmresSolverConfigValidator::validate(config, name)
}

struct GmresSolverConfigValidator;

impl GmresSolverConfigValidator {
    fn validate(config: GmresConfig, name: &str) -> Result<GmresConfig> {
        if config.restart == 0
            || config.max_iters == 0
            || !config.tolerance.is_finite()
            || config.tolerance <= 0.0
        {
            return Err(AsimuError::Config(format!(
                "[incompressible.linear.{name}] GMRES restart/max_iters/tolerance 参数无效"
            )));
        }
        Ok(config)
    }
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
