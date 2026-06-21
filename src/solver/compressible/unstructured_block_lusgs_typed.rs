//! 非结构 typed 块 LU-SGS 稳态伪时间步（一阶无粘 5×5 面块对称双扫）。

use std::time::Instant;

use tracing::info_span;

use super::gmres_implicit_unstructured_typed::take_and_refresh_block_lusgs_preconditioner;
use super::unstructured_prepare_timestep_typed::{
    UnstructuredCudaPrepareSync, UnstructuredTimestepFromSigma,
};
use super::{
    UnstructuredComputeBackend, UnstructuredRunEnvTyped, UnstructuredStepWorkTyped,
    UnstructuredTypedRhsWork, assemble_unstructured_typed_rhs,
};
use crate::core::{ComputeFloat, ComputePrecision, Real, elapsed_ms};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedFieldsT};
use crate::linalg::{GmresReport, Preconditioner};
use crate::solver::compressible::gmres_implicit_3d::gmres_implicit_typed_common::{
    apply_delta_with_line_search_typed, residual_to_vector_typed,
};
use crate::solver::compressible::gmres_implicit_3d::{
    GmresImplicitDelta, GmresImplicitDiagnostics, GmresPreconditionerKind,
};
use crate::solver::compressible::lu_sgs_common::{
    LuSgsSweepScalars, apply_diagonal_fallback_typed,
};
use tracing::warn;

pub(crate) fn advance_unstructured_block_lusgs_typed<
    T: ComputeFloat
        + UnstructuredComputeBackend
        + UnstructuredCudaPrepareSync
        + UnstructuredTimestepFromSigma,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    p_floor: Real,
) -> Result<()> {
    if !env.config.local_time_step {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"block_lusgs\" 须配合 local_time_step = true".to_string(),
        ));
    }
    if T::PRECISION != ComputePrecision::F64 {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"block_lusgs\" 暂仅支持 compute_precision = \"f64\"".to_string(),
        ));
    }
    if env.config.inviscid.reconstruction != crate::discretization::ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"block_lusgs\" 暂要求 reconstruction = first_order".to_string(),
        ));
    }
    let cells = fields.num_cells();
    let step = work.state.time_step.saturating_add(1);
    let _span = info_span!(
        "unstructured_block_lusgs_step_typed",
        step,
        cells,
        precision = T::PRECISION.label(),
    )
    .entered();
    let step_start = Instant::now();
    let (dt, _sigma) = unstructured_timestep_real_slices(work);
    work.storage.ensure_capacity(cells)?;
    {
        let _span = info_span!("unstructured_block_lusgs_copy_base_typed").entered();
        work.storage.u0.copy_from(fields)?;
    }
    let rhs_start = Instant::now();
    {
        let _span = info_span!("unstructured_block_lusgs_rhs_typed", cells).entered();
        let mut rhs_work = UnstructuredTypedRhsWork {
            ghosts: &mut work.ghosts,
            primitives: &mut work.primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
            viscous_grad_scratch_f32: &mut work.viscous_grad_scratch_f32,
            mesh_cache: &work.mesh_cache,
            exec: &mut work.exec,
        };
        assemble_unstructured_typed_rhs(
            env,
            &mut rhs_work,
            &work.storage.u0,
            &mut work.storage.k1,
            true,
            p_floor,
        )?;
    }
    let rhs_ms = elapsed_ms(rhs_start);
    let base_residual_rms = work.storage.k1.density_rms_norm();
    let rhs = residual_to_vector_typed(&work.storage.k1);
    let epsilon_rel = crate::solver::GmresImplicitConfig::default().epsilon;
    let build_start = Instant::now();
    let mut precond = {
        let _span = info_span!("unstructured_block_lusgs_build", cells).entered();
        take_and_refresh_block_lusgs_preconditioner(env, work, fields, &dt, p_floor, epsilon_rel)?
    };
    let build_ms = elapsed_ms(build_start);
    let omega = env.config.lu_sgs.omega;
    let mut delta = vec![0.0; rhs.len()];
    let sweep_start = Instant::now();
    {
        let _span = info_span!("unstructured_block_lusgs_sweep", cells).entered();
        precond.apply(&rhs, &mut delta)?;
    }
    let sweep_ms = elapsed_ms(sweep_start);
    work.block_lusgs_preconditioner = Some(precond);
    if omega < 1.0 - 1.0e-12 {
        for entry in &mut delta {
            *entry *= omega;
        }
    }
    work.block_lusgs_inner_iterations = 1;
    let line_search_start = Instant::now();
    let implicit_delta = GmresImplicitDelta {
        delta,
        report: GmresReport {
            converged: true,
            iterations: 1,
            residual_norm: 0.0,
        },
        base_residual_rms,
        diagnostics: GmresImplicitDiagnostics::new(GmresPreconditionerKind::BlockLusgs),
    };
    {
        let _span = info_span!("unstructured_block_lusgs_line_search", cells).entered();
        apply_block_lusgs_update_typed(BlockLusgsUpdateParams {
            fields,
            stage: &mut work.storage.stage,
            base: &work.storage.u0,
            residual: &work.storage.k1,
            delta: &implicit_delta,
            eos: env.config.eos,
            p_floor,
            dt: &work.timestep.cell_dts,
            sigma: &work.timestep.sigma,
            volumes: &work.volumes,
            omega,
        })?;
    }
    let line_search_ms = elapsed_ms(line_search_start);
    work.density_rms_after_rhs = Some(base_residual_rms);
    tracing::debug!(
        step,
        profile_rhs_ms = %crate::core::format_log_fixed4(rhs_ms),
        profile_build_ms = %crate::core::format_log_fixed4(build_ms),
        profile_sweep_ms = %crate::core::format_log_fixed4(sweep_ms),
        profile_line_search_ms = %crate::core::format_log_fixed4(line_search_ms),
        profile_step_total_ms = %crate::core::format_log_fixed4(elapsed_ms(step_start)),
        log10_residual = %crate::core::format_log_fixed4(crate::core::log10_positive(base_residual_rms)),
        "非结构 block_lusgs 步 profiling",
    );
    Ok(())
}

struct BlockLusgsUpdateParams<'a, T: ComputeFloat> {
    fields: &'a mut ConservedFieldsT<T>,
    stage: &'a mut ConservedFieldsT<T>,
    base: &'a ConservedFieldsT<T>,
    residual: &'a crate::field::ConservedResidualT<T>,
    delta: &'a GmresImplicitDelta,
    eos: &'a crate::physics::IdealGasEoS,
    p_floor: Real,
    dt: &'a [Real],
    sigma: &'a [Real],
    volumes: &'a [Real],
    omega: Real,
}

fn apply_block_lusgs_update_typed<T: ComputeFloat>(
    params: BlockLusgsUpdateParams<'_, T>,
) -> Result<()> {
    let BlockLusgsUpdateParams {
        fields,
        stage,
        base,
        residual,
        delta,
        eos,
        p_floor,
        dt,
        sigma,
        volumes,
        omega,
    } = params;
    match apply_delta_with_line_search_typed(fields, stage, base, delta, eos, p_floor) {
        Ok(_) => Ok(()),
        Err(err) => {
            warn!(omega, "block_lusgs 线搜索失败，回退对角 LU-SGS 更新: {err}");
            let scalars = LuSgsSweepScalars {
                dt,
                sigma,
                volumes,
                omega,
                gamma: eos.gamma,
                inv_dt_phys: 0.0,
            };
            apply_diagonal_fallback_typed(fields, base, residual, eos.gamma, p_floor, &scalars)?;
            Ok(())
        }
    }
}

fn unstructured_timestep_real_slices<T: ComputeFloat>(
    work: &UnstructuredStepWorkTyped<T>,
) -> (Vec<Real>, Vec<Real>) {
    if T::PRECISION == ComputePrecision::F64 {
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

/// `lu_sgs` + `low_mach_jacobian`：block LU-SGS 双扫替代标量扫掠（f64）。
pub(crate) fn apply_lusgs_block_jacobian_sweep_f64(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFields,
    work: &mut UnstructuredStepWorkTyped<f64>,
    p_floor: Real,
    omega: Real,
) -> Result<()> {
    use super::gmres_implicit_unstructured_typed::take_and_refresh_block_lusgs_preconditioner;

    let (dt, _sigma) = unstructured_timestep_real_slices(work);
    let epsilon_rel = crate::solver::GmresImplicitConfig::default().epsilon;
    let mut precond =
        take_and_refresh_block_lusgs_preconditioner(env, work, fields, &dt, p_floor, epsilon_rel)?;
    let rhs = residual_to_vector_typed(&work.storage.k1);
    let mut delta = vec![0.0; rhs.len()];
    precond.apply(&rhs, &mut delta)?;
    work.block_lusgs_preconditioner = Some(precond);
    if omega < 1.0 - 1.0e-12 {
        for entry in &mut delta {
            *entry *= omega;
        }
    }
    let base_residual_rms = work.storage.k1.density_rms_norm();
    let implicit_delta = GmresImplicitDelta {
        delta,
        report: GmresReport {
            converged: true,
            iterations: 1,
            residual_norm: 0.0,
        },
        base_residual_rms,
        diagnostics: GmresImplicitDiagnostics::new(GmresPreconditionerKind::BlockLusgs),
    };
    apply_block_lusgs_update_typed(BlockLusgsUpdateParams {
        fields,
        stage: &mut work.storage.stage,
        base: &work.storage.u0,
        residual: &work.storage.k1,
        delta: &implicit_delta,
        eos: env.config.eos,
        p_floor,
        dt: &work.timestep.cell_dts,
        sigma: &work.timestep.sigma,
        volumes: &work.volumes,
        omega,
    })
}
