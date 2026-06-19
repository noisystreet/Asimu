//! 非结构 typed GMRES 稳态伪时间步。

use std::time::Instant;

use tracing::info_span;

use crate::core::{ComputeFloat, Real, elapsed_ms};
use crate::error::{AsimuError, Result};
use crate::field::ConservedFieldsT;
use crate::solver::compressible::gmres_implicit_3d::gmres_implicit_typed_common::apply_delta_with_line_search_typed;
use crate::solver::compressible::gmres_implicit_3d::{
    GmresStepLog, GmresStepTiming, log_gmres_step_diagnostics,
};

use super::gmres_implicit_unstructured_typed::solve_gmres_implicit_delta_unstructured_typed;
use super::unstructured_prepare_timestep_typed::{
    UnstructuredCudaPrepareSync, UnstructuredTimestepFromSigma, prepare_unstructured_timestep_typed,
};
use super::{UnstructuredComputeBackend, UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};

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
    cfl: Real,
    p_floor: Real,
) -> Result<UnstructuredGmresStepOutcome> {
    if !env.config.local_time_step {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"gmres\" 须配合 local_time_step = true".to_string(),
        ));
    }
    let _span = info_span!(
        "unstructured_gmres_step_typed",
        precision = T::PRECISION.label(),
    )
    .entered();
    let step_start = Instant::now();
    prepare_unstructured_timestep_typed(env, fields, work, cfl, p_floor)?;
    let (dt, sigma) = unstructured_timestep_real_slices(work);
    work.storage.ensure_capacity(fields.num_cells())?;
    let solve = solve_gmres_implicit_delta_unstructured_typed(
        env,
        work,
        fields,
        &dt,
        &sigma,
        p_floor,
        env.config.solver.config.gmres,
    )?;
    work.storage.u0.copy_from(fields)?;
    let update = apply_delta_with_line_search_typed(
        fields,
        &mut work.storage.stage,
        &work.storage.u0,
        &solve.delta,
        env.config.eos,
        p_floor,
    )?;
    let step = work.state.time_step.saturating_add(1);
    log_gmres_step_diagnostics(GmresStepLog {
        step,
        dt: dt.iter().copied().fold(Real::INFINITY, |a, b| a.min(b)),
        cfl,
        delta: &solve.delta,
        update,
        residual_rms: solve.delta.base_residual_rms,
        timing: GmresStepTiming {
            compute_dt_ms: 0.0,
            implicit_solve_ms: solve.delta.diagnostics.timing.total_ms,
            line_search_ms: 0.0,
            post_residual_ms: 0.0,
            step_total_ms: elapsed_ms(step_start),
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
