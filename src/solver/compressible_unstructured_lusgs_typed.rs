//! 非结构 typed 驱动 LU-SGS 扫掠精度分发（f32 预打包耦合）。

use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::ConservedFieldsT;

use super::{UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};
use crate::solver::{
    LuSgsSweepUnstructuredInput, LuSgsSweepUnstructuredTypedParams, LuSgsUnstructuredCouplingsRef,
    lu_sgs_sweep_unstructured_f32, lu_sgs_sweep_unstructured_typed,
};

/// LU-SGS 扫掠上下文（驱动层传入）。
pub(crate) struct UnstructuredLusgsSweepContext<'a> {
    pub env: &'a UnstructuredRunEnvTyped<'a>,
    pub cell_dts: &'a [Real],
    pub sigma: &'a [Real],
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
        let volumes = &work.volumes;
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
            LuSgsSweepUnstructuredInput {
                dt: ctx.cell_dts,
                sigma: ctx.sigma,
                volumes,
                couplings,
                omega: ctx.omega,
                gamma: ctx.env.config.eos.gamma,
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
        let volumes = &work.volumes;
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
                dt: ctx.cell_dts,
                sigma: ctx.sigma,
                volumes,
                couplings,
                omega: ctx.omega,
                gamma: ctx.env.config.eos.gamma,
            },
        )
    }
}
