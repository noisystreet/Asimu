//! 非结构可压缩双时间步内外循环（理论见 `docs/theory/dual_time_stepping.md` §3.2）。

use tracing::{debug, info_span};

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::{ComputeFloat, Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::error::{AsimuError, Result};
use crate::field::ConservedFieldsT;
use crate::solver::UnstructuredComputeBackend;
use crate::solver::time::DualTimeConfig;

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
    let _physical_span = info_span!(
        "unstructured_dual_time_physical_step",
        dt_phys = dual.dt_phys,
        max_inner_steps = dual.max_inner_steps,
        precision = T::PRECISION.label(),
    )
    .entered();
    {
        let _span = info_span!("unstructured_dual_time_snapshot_u_n_typed").entered();
        T::snapshot_dual_time_u_n(work, fields)?;
    }
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
        let log10_residual = log10_positive(effective_residual_rms);
        if dual.inner_converged(effective_residual_rms) {
            debug!(
                inner = inner + 1,
                max_inner = dual.max_inner_steps,
                log10_residual = %format_log_fixed4(log10_residual),
                inner_converged = true,
                "dual_time 内层早停",
            );
            break;
        }
    }
    debug!(
        inner_iterations = work.dual_time_state.inner_iterations,
        log10_residual = %format_log_fixed4(log10_positive(effective_residual_rms)),
        "dual_time 物理步内层结束",
    );
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
    {
        let _span = info_span!("unstructured_dual_time_copy_base_typed").entered();
        T::prepare_dual_time_inner_base(work, fields)?;
    }
    T::maybe_upload_lusgs_integration_base(work)?;
    dual_time_assemble_effective_rhs(ctx.env, work, fields, ctx.dual.dt_phys, ctx.p_floor)?;
    let effective_residual_rms = {
        let _span = info_span!("unstructured_dual_time_rhs_monitor").entered();
        let rms = T::step_density_residual_rms(work)?;
        debug!(
            inner = ctx.inner + 1,
            log10_residual = %format_log_fixed4(log10_positive(rms)),
            "dual_time 内层 R_eff 监控",
        );
        rms
    };
    dual_time_apply_lusgs_update(
        ctx.env,
        fields,
        work,
        ctx.p_floor,
        ctx.lu_sgs,
        ctx.inv_dt_phys,
    )?;
    debug_log_conserved_health_after_update(fields, ctx.inner + 1);
    Ok(effective_residual_rms)
}

/// 内层 LU-SGS 更新后 host 守恒场健康度（debug 级；定位非物 KE/内能）。
struct ConservedHealthSummary {
    worst_cell: usize,
    min_internal: Real,
    rho: Real,
    ke: Real,
    total_energy: Real,
    speed: Real,
}

fn summarize_conserved_health<T: ComputeFloat>(
    fields: &ConservedFieldsT<T>,
) -> ConservedHealthSummary {
    let n = fields.num_cells();
    let mut worst_cell = 0usize;
    let mut min_internal = Real::INFINITY;
    let mut rho_at_worst = 0.0;
    let mut ke_at_worst = 0.0;
    let mut energy_at_worst = 0.0;
    let mut speed_at_worst = 0.0;
    for i in 0..n {
        let rho = fields.density.values()[i].to_real();
        let mx = fields.momentum_x.values()[i].to_real();
        let my = fields.momentum_y.values()[i].to_real();
        let mz = fields.momentum_z.values()[i].to_real();
        let energy = fields.total_energy.values()[i].to_real();
        if !(rho.is_finite() && energy.is_finite()) {
            worst_cell = i;
            min_internal = Real::NEG_INFINITY;
            rho_at_worst = rho;
            ke_at_worst = Real::NAN;
            energy_at_worst = energy;
            speed_at_worst = Real::NAN;
            break;
        }
        let inv_rho = if rho > 0.0 { 1.0 / rho } else { Real::INFINITY };
        let ux = mx * inv_rho;
        let uy = my * inv_rho;
        let uz = mz * inv_rho;
        let ke = 0.5 * rho * (ux * ux + uy * uy + uz * uz);
        let internal = energy - ke;
        if internal < min_internal {
            min_internal = internal;
            worst_cell = i;
            rho_at_worst = rho;
            ke_at_worst = ke;
            energy_at_worst = energy;
            speed_at_worst = (ux * ux + uy * uy + uz * uz).sqrt();
        }
    }
    ConservedHealthSummary {
        worst_cell,
        min_internal,
        rho: rho_at_worst,
        ke: ke_at_worst,
        total_energy: energy_at_worst,
        speed: speed_at_worst,
    }
}

fn debug_log_conserved_health_after_update<T: ComputeFloat>(
    fields: &ConservedFieldsT<T>,
    inner: u32,
) {
    let h = summarize_conserved_health(fields);
    debug!(
        inner,
        worst_cell = h.worst_cell,
        min_internal = %format_log_sci4(h.min_internal),
        rho = %format_log_sci4(h.rho),
        ke = %format_log_sci4(h.ke),
        total_energy = %format_log_sci4(h.total_energy),
        speed = %format_log_sci4(h.speed),
        "dual_time 内层更新后守恒场健康度",
    );
}

fn dual_time_assemble_effective_rhs<T: UnstructuredComputeBackend + UnstructuredCudaPrepareSync>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    fields: &ConservedFieldsT<T>,
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
    let spatial_rho_rms = work.storage.k1.density_rms_norm();
    let _storage_span = info_span!("unstructured_dual_time_storage_typed").entered();
    T::add_dual_time_storage_residual(work, fields, dt_phys)?;
    debug!(
        spatial_log10_rho_rms = %format_log_fixed4(log10_positive(spatial_rho_rms)),
        effective_log10_rho_rms = %format_log_fixed4(log10_positive(work.storage.k1.density_rms_norm())),
        "dual_time 内层 RHS 分项（密度 RMS）",
    );
    Ok(())
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
        let _span = info_span!(
            "unstructured_dual_time_lusgs_sweep_typed",
            precision = T::PRECISION.label(),
        )
        .entered();
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
        {
            let _span = info_span!("unstructured_dual_time_lusgs_diagonal_typed").entered();
            T::assign_lusgs_diagonal_update(
                work,
                lu_sgs.omega,
                env.config.eos.gamma,
                p_floor,
                inv_dt_phys,
            )?;
        }
        if !T::lusgs_skip_copy_stage_after_diagonal(work) {
            let _span = info_span!("unstructured_dual_time_lusgs_copy_stage_typed").entered();
            fields.copy_from(&work.storage.stage)?;
        }
    }
    #[cfg(feature = "cuda")]
    if work.exec.device() == ExecDevice::GpuCuda {
        work.exec.mark_cuda_primitives_stale_after_integration();
    }
    Ok(())
}

#[cfg(test)]
mod health_tests {
    use super::*;
    use crate::field::ConservedFieldsT;
    use crate::physics::ConservedState;

    #[test]
    fn summarize_conserved_health_finds_low_internal_energy_cell() {
        let state_ok = ConservedState {
            density: 1.0,
            momentum: [0.1, 0.0, 0.0],
            total_energy: 2.5,
        };
        let mut bad = state_ok;
        bad.momentum = [68.0, 0.0, 0.0];
        bad.total_energy = 8.6;
        let mut fields = ConservedFieldsT::<f64>::uniform(2, state_ok).expect("fields");
        fields.density.values_mut()[1] = bad.density;
        fields.momentum_x.values_mut()[1] = bad.momentum[0];
        fields.total_energy.values_mut()[1] = bad.total_energy;
        let h = summarize_conserved_health(&fields);
        assert_eq!(h.worst_cell, 1);
        assert!(h.min_internal < 0.0);
        assert!(h.ke > 2000.0);
    }
}
