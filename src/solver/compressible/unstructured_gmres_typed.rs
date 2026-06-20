//! 非结构 typed GMRES 稳态伪时间步。

use std::time::Instant;

use tracing::info_span;

use super::gmres_implicit_unstructured_typed::solve_gmres_implicit_delta_unstructured_typed;
use super::unstructured_prepare_timestep_typed::{
    UnstructuredCudaPrepareSync, UnstructuredTimestepFromSigma,
};
use super::{UnstructuredComputeBackend, UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};
use crate::core::{ComputeFloat, Real, elapsed_ms};
use crate::error::{AsimuError, Result};
use crate::field::ConservedFieldsT;
use crate::solver::compressible::gmres_implicit_3d::gmres_implicit_typed_common::apply_delta_with_line_search_typed;
use crate::solver::compressible::gmres_implicit_3d::{
    GmresStepLog, GmresStepTiming, log_gmres_step_diagnostics,
};

pub(crate) struct UnstructuredGmresStepOutcome {
    pub gmres_iterations: u32,
}

pub(crate) fn advance_unstructured_gmres_typed<
    T: ComputeFloat
        + UnstructuredComputeBackend
        + UnstructuredCudaPrepareSync
        + UnstructuredTimestepFromSigma,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    min_dt: Real,
    p_floor: Real,
) -> Result<UnstructuredGmresStepOutcome> {
    if !env.config.local_time_step {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"gmres\" 须配合 local_time_step = true".to_string(),
        ));
    }
    let step = work.state.time_step.saturating_add(1);
    let cells = fields.num_cells();
    let gmres_cfg = env.config.solver.config.gmres;
    let _span = info_span!(
        "unstructured_gmres_step_typed",
        step,
        cells,
        precision = T::PRECISION.label(),
        gmres_preconditioner = gmres_cfg.preconditioner.as_str(),
        gmres_restart = gmres_cfg.gmres.restart,
        gmres_max_iters = gmres_cfg.gmres.max_iters,
    )
    .entered();
    let step_start = Instant::now();
    let compute_dt_ms = 0.0;
    let (dt, sigma) = unstructured_timestep_real_slices(work);
    work.storage.ensure_capacity(cells)?;
    let implicit_solve_start = Instant::now();
    let solve = {
        let _span = info_span!(
            "gmres_implicit_solve",
            cells,
            gmres_preconditioner = gmres_cfg.preconditioner.as_str(),
        )
        .entered();
        solve_gmres_implicit_delta_unstructured_typed(
            env, work, fields, &dt, &sigma, p_floor, gmres_cfg,
        )?
    };
    let implicit_solve_ms = elapsed_ms(implicit_solve_start);
    work.storage.u0.copy_from(fields)?;
    let line_search_start = Instant::now();
    let update = {
        let _span = info_span!("gmres_line_search", cells).entered();
        apply_delta_with_line_search_typed(
            fields,
            &mut work.storage.stage,
            &work.storage.u0,
            &solve.delta,
            env.config.eos,
            p_floor,
        )?
    };
    let line_search_ms = elapsed_ms(line_search_start);
    let step_total_ms = elapsed_ms(step_start);
    log_gmres_step_diagnostics(GmresStepLog {
        step,
        dt: min_dt,
        cfl: env.config.cfl_schedule.at_step(step, env.config.max_steps),
        delta: &solve.delta,
        update,
        residual_rms: solve.delta.base_residual_rms,
        timing: GmresStepTiming {
            compute_dt_ms,
            implicit_solve_ms,
            line_search_ms,
            post_residual_ms: 0.0,
            step_total_ms,
        },
    });
    work.density_rms_after_rhs = Some(solve.delta.base_residual_rms);
    Ok(UnstructuredGmresStepOutcome {
        gmres_iterations: u32::try_from(solve.delta.report.iterations).unwrap_or(u32::MAX),
    })
}

fn unstructured_timestep_real_slices<T: ComputeFloat>(
    work: &UnstructuredStepWorkTyped<T>,
) -> (Vec<Real>, Vec<Real>) {
    if T::PRECISION == crate::core::ComputePrecision::F64 {
        (work.timestep.cell_dts.clone(), work.timestep.sigma.clone())
    } else {
        (
            work.timestep
                .cell_dts_f32
                .iter()
                .map(|v| *v as Real)
                .collect(),
            work.timestep.sigma_f32.iter().map(|v| *v as Real).collect(),
        )
    }
}
