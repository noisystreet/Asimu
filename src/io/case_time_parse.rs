use serde::Deserialize;

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::CaseTimeConfig;
use super::CaseTimeMode;

#[derive(Debug, Deserialize)]
pub(super) struct TimeToml {
    pub mode: Option<String>,
    pub dt: Option<Real>,
    pub cfl: Option<Real>,
    pub cfl_max: Option<Real>,
    pub final_time: Option<Real>,
    pub max_steps: Option<u64>,
    pub min_steps: Option<u64>,
    pub tolerance: Option<Real>,
    pub local_time_step: Option<bool>,
    pub cfl_ramp_steps: Option<u64>,
    pub scheme: Option<String>,
    pub lusgs_omega: Option<Real>,
    pub lusgs_sweep: Option<bool>,
    pub lusgs_sweep_backward_damping: Option<Real>,
    pub gmres_preconditioner: Option<String>,
    pub gmres_tolerance: Option<Real>,
    pub gmres_max_iters: Option<usize>,
    pub gmres_restart: Option<usize>,
    pub residual_smoothing: Option<bool>,
    pub residual_smoothing_epsilon: Option<Real>,
    pub residual_smoothing_sweeps: Option<usize>,
    pub max_inner_steps: Option<u32>,
    pub inner_tolerance: Option<Real>,
    pub low_mach_preconditioning: Option<bool>,
    pub low_mach_mach_cutoff: Option<Real>,
    pub low_mach_max_mach: Option<Real>,
    pub low_mach_blend: Option<String>,
    pub low_mach_jacobian: Option<bool>,
}

pub(super) fn parse_time_config(raw: Option<&TimeToml>, has_sod: bool) -> Result<CaseTimeConfig> {
    let Some(raw) = raw else {
        return Ok(if has_sod {
            CaseTimeConfig {
                mode: CaseTimeMode::Transient,
                ..CaseTimeConfig::default()
            }
        } else {
            CaseTimeConfig::default()
        });
    };
    let mode = match raw.mode.as_deref().unwrap_or("steady") {
        "steady" => CaseTimeMode::Steady,
        "transient" => CaseTimeMode::Transient,
        other => {
            return Err(AsimuError::Config(format!(
                "不支持的 time.mode \"{other}\""
            )));
        }
    };
    let scheme = raw
        .scheme
        .as_deref()
        .map(crate::solver::time::TimeIntegrationScheme::parse)
        .transpose()?;
    let lusgs_omega = raw.lusgs_omega;
    let lusgs_sweep = raw.lusgs_sweep;
    let lusgs_sweep_backward_damping = raw.lusgs_sweep_backward_damping;
    let _ = crate::solver::time::LuSgsConfig::parse(
        lusgs_omega,
        lusgs_sweep,
        lusgs_sweep_backward_damping,
    )?;
    let residual_smoothing = crate::solver::time::ResidualSmoothingConfig::parse(
        raw.residual_smoothing.unwrap_or(false),
        raw.residual_smoothing_epsilon,
        raw.residual_smoothing_sweeps,
    )?;
    let gmres_preconditioner = raw
        .gmres_preconditioner
        .as_deref()
        .map(crate::solver::GmresPreconditionerKind::parse)
        .transpose()?;
    let low_mach_preconditioning = crate::solver::time::LowMachPreconditioningConfig::parse(
        raw.low_mach_preconditioning.unwrap_or(false),
        raw.low_mach_mach_cutoff,
        raw.low_mach_max_mach,
        raw.low_mach_blend.as_deref(),
        raw.low_mach_jacobian,
    )?;
    Ok(CaseTimeConfig {
        mode,
        dt: raw.dt,
        cfl: raw.cfl,
        cfl_max: raw.cfl_max,
        final_time: raw.final_time,
        max_steps: raw.max_steps,
        min_steps: raw.min_steps,
        tolerance: raw.tolerance,
        local_time_step: raw.local_time_step.unwrap_or(false),
        cfl_ramp_steps: raw.cfl_ramp_steps,
        scheme,
        lusgs_omega,
        lusgs_sweep,
        lusgs_sweep_backward_damping,
        gmres_preconditioner,
        gmres_tolerance: raw.gmres_tolerance,
        gmres_max_iters: raw.gmres_max_iters,
        gmres_restart: raw.gmres_restart,
        residual_smoothing,
        max_inner_steps: raw.max_inner_steps,
        inner_tolerance: raw.inner_tolerance,
        low_mach_preconditioning,
    })
}
