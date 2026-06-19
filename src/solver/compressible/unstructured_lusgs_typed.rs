//! 非结构 typed 驱动 LU-SGS 扫掠精度分发（f32 预打包耦合）。

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFieldsT, LusgsDiagonalCoeffs, LusgsDiagonalCoeffsF32, assign_lusgs_diagonal_update_f32,
};
#[cfg(feature = "cuda")]
use tracing::warn;

use super::{UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};
use crate::solver::{
    LuSgsSweepUnstructuredF32Input, LuSgsSweepUnstructuredInput, LuSgsSweepUnstructuredTypedParams,
    LuSgsUnstructuredCouplingsRef, lu_sgs_sweep_unstructured_f32, lu_sgs_sweep_unstructured_typed,
};

/// LU-SGS 扫掠上下文（驱动层传入）。
pub(crate) struct UnstructuredLusgsSweepContext<'a> {
    pub env: &'a UnstructuredRunEnvTyped<'a>,
    pub p_floor: Real,
    pub sweep: bool,
    pub omega: Real,
    pub backward_damping: Real,
    /// \(1/\Delta t_{\mathrm{phys}}\)；稳态伪时间为 0。
    pub inv_dt_phys: Real,
}

/// LU-SGS 扫掠精度分发（f32 用 `mesh_cache.lusgs_couplings_f32`）。
pub(crate) trait UnstructuredLusgsSweep: ComputeFloat {
    fn run_lusgs_sweep(
        fields: &mut ConservedFieldsT<Self>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        ctx: &UnstructuredLusgsSweepContext<'_>,
    ) -> Result<()>;
}

impl UnstructuredLusgsSweep for f32 {
    fn run_lusgs_sweep(
        fields: &mut ConservedFieldsT<f32>,
        work: &mut UnstructuredStepWorkTyped<f32>,
        ctx: &UnstructuredLusgsSweepContext<'_>,
    ) -> Result<()> {
        if !ctx.sweep {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if try_cuda_lusgs_sweep_f32(fields, work, ctx)? {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        warn_cuda_lusgs_sweep_cpu_fallback(work);
        #[cfg(feature = "cuda")]
        if work.exec.cuda_rhs_pipeline_active() && work.exec.cuda_residual_on_device() {
            work.exec.cuda_flush_rhs_residual(&mut work.storage.k1)?;
        }
        ensure_f32_host_timestep_for_sweep(
            work,
            fields.num_cells(),
            ctx.env.config.local_time_step,
        )?;
        let couplings = LuSgsUnstructuredCouplingsRef::F32(&work.mesh_cache.lusgs_couplings_f32);
        let residual = &work.storage.k1;
        let mut sweep_params = LuSgsSweepUnstructuredTypedParams {
            mesh: ctx.env.config.mesh,
            eos: ctx.env.config.eos,
            primitives: &mut work.primitives,
            min_pressure: ctx.p_floor,
            backward_damping: ctx.backward_damping,
            low_mach_preconditioning: ctx.env.config.low_mach_preconditioning,
        };
        lu_sgs_sweep_unstructured_f32(
            fields,
            residual,
            &mut sweep_params,
            LuSgsSweepUnstructuredF32Input {
                dt: &work.timestep.cell_dts_f32,
                sigma: &work.timestep.sigma_f32,
                volumes: &work.volumes_f32,
                couplings,
                solver_order: &work.mesh_cache.solver_order,
                solver_rank: &work.mesh_cache.solver_rank,
                omega: ctx.omega as f32,
                gamma: ctx.env.config.eos.gamma as f32,
                inv_dt_phys: ctx.inv_dt_phys as f32,
            },
        )
    }
}

impl UnstructuredLusgsSweep for f64 {
    fn run_lusgs_sweep(
        fields: &mut ConservedFieldsT<f64>,
        work: &mut UnstructuredStepWorkTyped<f64>,
        ctx: &UnstructuredLusgsSweepContext<'_>,
    ) -> Result<()> {
        if !ctx.sweep {
            return Ok(());
        }
        let couplings = LuSgsUnstructuredCouplingsRef::F64(&work.lusgs_couplings);
        let residual = &work.storage.k1;
        let mut sweep_params = LuSgsSweepUnstructuredTypedParams {
            mesh: ctx.env.config.mesh,
            eos: ctx.env.config.eos,
            primitives: &mut work.primitives,
            min_pressure: ctx.p_floor,
            backward_damping: ctx.backward_damping,
            low_mach_preconditioning: ctx.env.config.low_mach_preconditioning,
        };
        lu_sgs_sweep_unstructured_typed(
            fields,
            residual,
            &mut sweep_params,
            LuSgsSweepUnstructuredInput {
                dt: &work.timestep.cell_dts,
                sigma: &work.timestep.sigma,
                volumes: &work.volumes,
                couplings,
                solver_order: &work.mesh_cache.solver_order,
                solver_rank: &work.mesh_cache.solver_rank,
                omega: ctx.omega,
                gamma: ctx.env.config.eos.gamma,
                inv_dt_phys: ctx.inv_dt_phys,
            },
        )
    }
}

/// LU-SGS 非扫掠对角更新（f32 用原生 \(\sigma,\Delta t_i\) 缓冲）。
pub(crate) trait UnstructuredLusgsDiagonalUpdate: ComputeFloat {
    fn assign_lusgs_diagonal_update(
        work: &mut UnstructuredStepWorkTyped<Self>,
        omega: Real,
        gamma: Real,
        p_floor: Real,
        inv_dt_phys: Real,
    ) -> Result<()>;
}

impl UnstructuredLusgsDiagonalUpdate for f32 {
    fn assign_lusgs_diagonal_update(
        work: &mut UnstructuredStepWorkTyped<f32>,
        omega: Real,
        _gamma: Real,
        _p_floor: Real,
        inv_dt_phys: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if try_cuda_lusgs_diagonal_update_f32(work, omega, inv_dt_phys)? {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if work.exec.cuda_rhs_pipeline_active() && work.exec.cuda_residual_on_device() {
            work.exec.cuda_flush_rhs_residual(&mut work.storage.k1)?;
        }
        assign_lusgs_diagonal_update_f32(
            &mut work.storage.stage,
            &work.storage.u0,
            &work.storage.k1,
            &work.timestep.sigma_f32,
            &work.timestep.cell_dts_f32,
            LusgsDiagonalCoeffsF32 {
                omega: omega as f32,
                inv_dt_phys: inv_dt_phys as f32,
            },
        )
    }
}

#[cfg(feature = "cuda")]
fn warn_cuda_lusgs_sweep_cpu_fallback(work: &UnstructuredStepWorkTyped<f32>) {
    if work.exec.device() != ExecDevice::GpuCuda {
        return;
    }
    if work.state.time_step > 0 {
        return;
    }
    let reason = if !work.exec.cuda_timestep_on_device() {
        "σ/Δtᵢ 未驻留 device（非 CUDA 或 prepare 未上传谱半径）"
    } else if !work.exec.cuda_residual_on_device() {
        "RHS 残差尚未在 device 上（如无粘路径未走 device 装配）"
    } else {
        "CUDA 双扫前置条件未满足"
    };
    warn!(
        reason,
        rhs_pipeline = work.exec.cuda_rhs_pipeline_active(),
        "CUDA f32 非结构 LU-SGS 双扫回落 CPU host 扫掠（后续步不再重复告警）"
    );
}

#[cfg(feature = "cuda")]
fn try_cuda_lusgs_sweep_f32(
    fields: &mut ConservedFieldsT<f32>,
    work: &mut UnstructuredStepWorkTyped<f32>,
    ctx: &UnstructuredLusgsSweepContext<'_>,
) -> Result<bool> {
    if work.exec.device() != ExecDevice::GpuCuda {
        return Ok(false);
    }
    if !work.exec.cuda_timestep_on_device() || !work.exec.cuda_residual_on_device() {
        return Ok(false);
    }
    let topo_key = std::ptr::from_ref(&work.mesh_cache).addr();
    work.exec.cuda_lusgs_sweep_update_f32(
        crate::exec::gpu::cuda::lusgs_sweep::LusgsSweepCudaHostInput {
            fields,
            u0: &work.storage.u0,
            residual: &mut work.storage.k1,
            sweep_topo: &work.mesh_cache.lusgs_sweep_topo,
            topo_key,
            primitives: &work.primitives,
            host_sigma: &work.timestep.sigma_f32,
            host_cell_dts: &work.timestep.cell_dts_f32,
            host_volumes: &work.volumes_f32,
            local_time_step: ctx.env.config.local_time_step,
            scalars: crate::exec::gpu::cuda::lusgs_sweep::LusgsSweepCudaScalars {
                omega: ctx.omega as f32,
                gamma: ctx.env.config.eos.gamma as f32,
                min_pressure: ctx.p_floor as f32,
                inv_dt_phys: ctx.inv_dt_phys as f32,
                backward_damping: ctx.backward_damping as f32,
            },
        },
    )?;
    Ok(true)
}

/// device 驻留 σ/Δtᵢ 时 host 缓冲为空；双扫 stabilize 与校验须先镜像到 host。
fn ensure_f32_host_timestep_for_sweep(
    work: &mut UnstructuredStepWorkTyped<f32>,
    num_cells: usize,
    local_time_step: bool,
) -> Result<()> {
    if work.timestep.sigma_f32.len() == num_cells && work.timestep.cell_dts_f32.len() == num_cells {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        if work.exec.device() == ExecDevice::GpuCuda && work.exec.cuda_timestep_on_device() {
            work.timestep.sigma_f32.resize(num_cells, 0.0);
            work.timestep.cell_dts_f32.resize(num_cells, 0.0);
            return work.exec.cuda_mirror_timestep_f32_to_host(
                &mut work.timestep.sigma_f32,
                &mut work.timestep.cell_dts_f32,
                local_time_step,
            );
        }
    }
    Err(AsimuError::Exec(format!(
        "LU-SGS 双扫 host σ/Δt 长度 {}/{} 与单元数 {num_cells} 不一致（local_time_step={local_time_step}）",
        work.timestep.sigma_f32.len(),
        work.timestep.cell_dts_f32.len(),
    )))
}

#[cfg(feature = "cuda")]
fn try_cuda_lusgs_diagonal_update_f32(
    work: &mut UnstructuredStepWorkTyped<f32>,
    omega: Real,
    inv_dt_phys: Real,
) -> Result<bool> {
    if work.exec.device() != ExecDevice::GpuCuda || !work.exec.cuda_timestep_on_device() {
        return Ok(false);
    }
    work.exec.cuda_lusgs_diagonal_update_f32(
        &work.storage.u0,
        &work.storage.k1,
        omega as f32,
        inv_dt_phys as f32,
    )?;
    Ok(true)
}

impl UnstructuredLusgsDiagonalUpdate for f64 {
    fn assign_lusgs_diagonal_update(
        work: &mut UnstructuredStepWorkTyped<f64>,
        omega: Real,
        gamma: Real,
        p_floor: Real,
        inv_dt_phys: Real,
    ) -> Result<()> {
        work.storage.stage.assign_lusgs_diagonal_update(
            &work.storage.u0,
            &work.storage.k1,
            &work.timestep.sigma,
            &work.timestep.cell_dts,
            LusgsDiagonalCoeffs::steady_pseudo_time(omega, gamma, p_floor)
                .with_inv_dt_phys(inv_dt_phys),
        )
    }
}
