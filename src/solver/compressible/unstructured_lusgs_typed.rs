//! 非结构 typed 驱动 LU-SGS 扫掠精度分发（f32 预打包耦合）。

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::{ConservedFieldsT, assign_lusgs_diagonal_update_f32};

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
        let couplings = LuSgsUnstructuredCouplingsRef::F32(&work.mesh_cache.lusgs_couplings_f32);
        let residual = &work.storage.k1;
        let mut sweep_params = LuSgsSweepUnstructuredTypedParams {
            mesh: ctx.env.config.mesh,
            eos: ctx.env.config.eos,
            primitives: &mut work.primitives,
            min_pressure: ctx.p_floor,
            backward_damping: ctx.backward_damping,
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
                omega: ctx.omega as f32,
                gamma: ctx.env.config.eos.gamma as f32,
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
                omega: ctx.omega,
                gamma: ctx.env.config.eos.gamma,
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
    ) -> Result<()>;
}

impl UnstructuredLusgsDiagonalUpdate for f32 {
    fn assign_lusgs_diagonal_update(
        work: &mut UnstructuredStepWorkTyped<f32>,
        omega: Real,
        gamma: Real,
        p_floor: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if try_cuda_lusgs_diagonal_update_f32(work, omega)? {
            if work.exec.cuda_residual_on_device() {
                work.exec.cuda_flush_rhs_residual(&mut work.storage.k1)?;
            }
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
            omega as f32,
            gamma,
            p_floor,
        )
    }
}

#[cfg(feature = "cuda")]
fn try_cuda_lusgs_diagonal_update_f32(
    work: &mut UnstructuredStepWorkTyped<f32>,
    omega: Real,
) -> Result<bool> {
    if work.exec.device() != ExecDevice::GpuCuda || !work.exec.cuda_timestep_on_device() {
        return Ok(false);
    }
    work.exec.cuda_lusgs_diagonal_update_f32(
        &work.storage.u0,
        &work.storage.k1,
        &mut work.storage.stage,
        omega as f32,
    )?;
    Ok(true)
}

impl UnstructuredLusgsDiagonalUpdate for f64 {
    fn assign_lusgs_diagonal_update(
        work: &mut UnstructuredStepWorkTyped<f64>,
        omega: Real,
        gamma: Real,
        p_floor: Real,
    ) -> Result<()> {
        work.storage.stage.assign_lusgs_diagonal_update(
            &work.storage.u0,
            &work.storage.k1,
            &work.timestep.sigma,
            &work.timestep.cell_dts,
            omega,
            gamma,
            p_floor,
        )
    }
}
