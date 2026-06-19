use crate::error::Result;
use crate::linalg::GmresConfig;
use crate::solver::{GmresImplicitConfig, GmresPreconditionerKind, time::TimeIntegrationScheme};

use super::CaseTimeConfig;

pub(super) fn resolve_gmres_config(time: &CaseTimeConfig) -> Result<GmresImplicitConfig> {
    if (time.gmres_tolerance.is_some()
        || time.gmres_max_iters.is_some()
        || time.gmres_restart.is_some())
        && time.resolved_time_scheme() != TimeIntegrationScheme::Gmres
    {
        return Err(crate::error::AsimuError::Config(
            "gmres_tolerance / gmres_max_iters / gmres_restart 仅用于 time.scheme = \"gmres\""
                .to_string(),
        ));
    }
    let defaults = GmresImplicitConfig::default();
    let gmres = GmresConfig {
        restart: time.gmres_restart.unwrap_or(defaults.gmres.restart),
        max_iters: time.gmres_max_iters.unwrap_or(defaults.gmres.max_iters),
        tolerance: time.gmres_tolerance.unwrap_or(defaults.gmres.tolerance),
    };
    gmres.validate()?;
    Ok(GmresImplicitConfig {
        gmres,
        epsilon: defaults.epsilon,
        preconditioner: time
            .gmres_preconditioner
            .unwrap_or(GmresPreconditionerKind::ScalarDiagonal),
    })
}
