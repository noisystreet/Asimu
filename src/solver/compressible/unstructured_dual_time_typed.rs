//! 非结构可压缩双时间步内外循环（理论见 `docs/theory/dual_time_stepping.md` §3.2）。

use tracing::info_span;

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::solver::UnstructuredComputeBackend;
use crate::solver::time::{DualTimeConfig, add_physical_storage_residual};

use super::unstructured_lusgs_typed::UnstructuredLusgsSweepContext;
use super::unstructured_prepare_timestep_typed::UnstructuredCudaPrepareSync;
use super::{
    UnstructuredRunEnvTyped, UnstructuredStepWorkTyped, UnstructuredTypedRhsWork,
    assemble_unstructured_typed_rhs, prepare_unstructured_timestep_typed,
};

struct DualTimeInnerCtx<'a> {
    env: &'a UnstructuredRunEnvTyped<'a>,
    dual: DualTimeConfig,
    cfl: Real,
    p_floor: Real,
    lu_sgs: crate::solver::LuSgsConfig,
    inv_dt_phys: Real,
    inner: u32,
}

/// 单物理步双时间推进：内层伪时间 LU-SGS + 存储项，返回末次内层 \(\|R_{\mathrm{eff}}\|_{\mathrm{rms}}\)。
pub(crate) fn advance_unstructured_dual_time_typed<
    T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut crate::field::ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    dual: DualTimeConfig,
    cfl: Real,
    p_floor: Real,
) -> Result<Real> {
    if !env.config.local_time_step {
        return Err(AsimuError::Config(
            "非结构 time.scheme = \"dual_time\" 须配合 local_time_step = true".to_string(),
        ));
    }
    let lu_sgs = env.config.lu_sgs;
    let inv_dt_phys = dual.inv_dt_phys();
    work.dual_time_state.snapshot_u_n(fields)?;
    let mut effective_residual_rms = 0.0;
    let base_ctx = DualTimeInnerCtx {
        env,
        dual,
        cfl,
        p_floor,
        lu_sgs,
        inv_dt_phys,
        inner: 0,
    };
    for inner in 0..dual.max_inner_steps {
        work.dual_time_state.inner_iterations = inner + 1;
        let ctx = DualTimeInnerCtx { inner, ..base_ctx };
        effective_residual_rms = dual_time_inner_iteration(&ctx, fields, work)?;
        if dual.inner_converged(effective_residual_rms) {
            break;
        }
    }
    Ok(effective_residual_rms)
}

fn dual_time_inner_iteration<T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync>(
    ctx: &DualTimeInnerCtx<'_>,
    fields: &mut crate::field::ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
) -> Result<Real> {
    let _span = info_span!(
        "unstructured_dual_time_inner",
        inner = ctx.inner + 1,
        max_inner = ctx.dual.max_inner_steps,
        precision = T::PRECISION.label(),
    )
    .entered();
    prepare_unstructured_timestep_typed(ctx.env, fields, work, ctx.cfl, ctx.p_floor)?;
    work.storage.u0.copy_from(fields)?;
    T::maybe_upload_lusgs_integration_base(work)?;
    dual_time_assemble_effective_rhs(ctx.env, work, fields, ctx.dual.dt_phys, ctx.p_floor)?;
    let effective_residual_rms = T::step_density_residual_rms(work)?;
    dual_time_apply_lusgs_update(
        ctx.env,
        fields,
        work,
        ctx.p_floor,
        ctx.lu_sgs,
        ctx.inv_dt_phys,
    )?;
    Ok(effective_residual_rms)
}

fn dual_time_assemble_effective_rhs<T: UnstructuredComputeBackend>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    fields: &crate::field::ConservedFieldsT<T>,
    dt_phys: Real,
    p_floor: Real,
) -> Result<()> {
    let _rhs_span = info_span!("unstructured_dual_time_rhs_typed").entered();
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
    add_physical_storage_residual(
        &mut work.storage.k1,
        fields,
        &work.dual_time_state.u_at_physical_level,
        &work.volumes,
        dt_phys,
    )
}

fn dual_time_apply_lusgs_update<T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut crate::field::ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    p_floor: Real,
    lu_sgs: crate::solver::LuSgsConfig,
    inv_dt_phys: Real,
) -> Result<()> {
    if lu_sgs.sweep {
        T::run_lusgs_sweep(
            fields,
            work,
            &UnstructuredLusgsSweepContext {
                env,
                p_floor,
                sweep: true,
                omega: lu_sgs.omega,
                backward_damping: lu_sgs.sweep_backward_damping,
                inv_dt_phys,
            },
        )?;
    } else {
        T::assign_lusgs_diagonal_update(
            work,
            lu_sgs.omega,
            env.config.eos.gamma,
            p_floor,
            inv_dt_phys,
        )?;
        if !T::lusgs_skip_copy_stage_after_diagonal(work) {
            fields.copy_from(&work.storage.stage)?;
        }
    }
    #[cfg(feature = "cuda")]
    if work.exec.device() == ExecDevice::GpuCuda {
        work.exec.mark_cuda_primitives_stale_after_integration();
    }
    Ok(())
}
