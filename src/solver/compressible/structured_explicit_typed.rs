//! 结构化 3D typed 显式 RK4/Euler 推进（ADR 0019 S0-b / S1-c）。

use tracing::info_span;

use crate::core::{ComputePrecision, Real, log10_positive};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::physics::IdealGasEoS;
use crate::solver::compressible::structured_compute_backend::StructuredComputeBackend;
use crate::solver::compressible::structured_timestep_buffers::StructuredExplicitTimeAdvance;
use crate::solver::compressible::{
    CompressibleAdvanceContext3dTyped, CompressibleEulerSolver, CompressibleStepInfo,
};
use crate::solver::state::SolverState;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator, euler_step,
    euler_step_local, euler_step_local_f32, min_positive_dt, min_positive_dt_f32, rk4_step,
    rk4_step_local, rk4_step_local_f32,
};

impl StructuredExplicitTimeAdvance for f32 {
    fn advance_structured_explicit(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, f32>,
        fields: &mut ConservedFieldsT<f32>,
        storage: &mut Rk4StorageT<f32>,
        dt_global: Real,
        local_time_step: bool,
        p_floor: Real,
        eos: &IdealGasEoS,
    ) -> Result<()> {
        let inviscid = solver.config.inviscid;
        match (solver.config.time_scheme, local_time_step) {
            (TimeIntegrationScheme::Rk4, true) => {
                let cell_dts = ctx.timestep.cell_dts_f32.clone();
                let evaluate = |u: &ConservedFieldsT<f32>, r: &mut ConservedResidualT<f32>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                rk4_step_local_f32(fields, storage, &cell_dts, evaluate, Some(eos), p_floor)
            }
            (TimeIntegrationScheme::Rk4, false) => {
                let evaluate = |u: &ConservedFieldsT<f32>, r: &mut ConservedResidualT<f32>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                rk4_step(fields, storage, dt_global, evaluate)
            }
            (TimeIntegrationScheme::Euler, true) => {
                let cell_dts = ctx.timestep.cell_dts_f32.clone();
                let evaluate = |u: &ConservedFieldsT<f32>, r: &mut ConservedResidualT<f32>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                euler_step_local_f32(fields, storage, &cell_dts, evaluate, Some(eos), p_floor)
            }
            (TimeIntegrationScheme::Euler, false) => {
                let evaluate = |u: &ConservedFieldsT<f32>, r: &mut ConservedResidualT<f32>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                euler_step(fields, storage, dt_global, evaluate, Some(eos), p_floor)
            }
            _ => Err(AsimuError::Solver(
                "结构化 typed 显式推进收到不支持的时间格式".to_string(),
            )),
        }
    }
}

impl StructuredExplicitTimeAdvance for f64 {
    fn advance_structured_explicit(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, f64>,
        fields: &mut ConservedFieldsT<f64>,
        storage: &mut Rk4StorageT<f64>,
        dt_global: Real,
        local_time_step: bool,
        p_floor: Real,
        eos: &IdealGasEoS,
    ) -> Result<()> {
        let inviscid = solver.config.inviscid;
        match (solver.config.time_scheme, local_time_step) {
            (TimeIntegrationScheme::Rk4, true) => {
                let cell_dts = ctx.timestep.cell_dts.clone();
                let evaluate = |u: &ConservedFieldsT<f64>, r: &mut ConservedResidualT<f64>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                rk4_step_local(fields, storage, &cell_dts, evaluate, Some(eos), p_floor)
            }
            (TimeIntegrationScheme::Rk4, false) => {
                let evaluate = |u: &ConservedFieldsT<f64>, r: &mut ConservedResidualT<f64>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                rk4_step(fields, storage, dt_global, evaluate)
            }
            (TimeIntegrationScheme::Euler, true) => {
                let cell_dts = ctx.timestep.cell_dts.clone();
                let evaluate = |u: &ConservedFieldsT<f64>, r: &mut ConservedResidualT<f64>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                euler_step_local(fields, storage, &cell_dts, evaluate, Some(eos), p_floor)
            }
            (TimeIntegrationScheme::Euler, false) => {
                let evaluate = |u: &ConservedFieldsT<f64>, r: &mut ConservedResidualT<f64>| {
                    solver
                        .rhs_context_3d_typed(ctx, &inviscid, p_floor)
                        .run(u, r)
                };
                euler_step(fields, storage, dt_global, evaluate, Some(eos), p_floor)
            }
            (scheme, _) => Err(AsimuError::Solver(format!(
                "typed 显式推进不支持 {}",
                scheme.label()
            ))),
        }
    }
}

impl CompressibleEulerSolver {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance_explicit_step_3d_typed<T: StructuredComputeBackend>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        storage: &mut Rk4StorageT<T>,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
        cfl: Real,
        p_floor: Real,
    ) -> Result<CompressibleStepInfo> {
        let inviscid = self.config.inviscid;
        let dt = {
            let _span = info_span!(
                "compute_dt",
                cells = ctx.structured.num_cells(),
                local_time_step = self.config.local_time_step,
                precision = T::PRECISION.label(),
            )
            .entered();
            {
                let _span = info_span!("enforce_positivity_pre").entered();
                fields.enforce_positivity(ctx.eos, p_floor);
            }
            self.prepare_spectral_timestep_3d_typed(ctx, fields, cfl, p_floor)?;
            match (T::PRECISION, self.config.local_time_step) {
                (ComputePrecision::F32, true) => {
                    min_positive_dt_f32(&ctx.timestep.cell_dts_f32) as Real
                }
                (ComputePrecision::F32, false) => ctx.timestep.cell_dts[0],
                (_, true) => min_positive_dt(&ctx.timestep.cell_dts),
                (_, false) => min_positive_dt(&ctx.timestep.cell_dts),
            }
        };
        integrator.config.dt = dt;
        let eos = *ctx.eos;
        let step_residual = {
            let _span = info_span!("rhs_monitor").entered();
            self.rhs_context_3d_typed(ctx, &inviscid, p_floor)
                .run(fields, &mut storage.k1)?;
            storage.k1.density_rms_norm()
        };
        {
            let _span = info_span!(
                "time_integration",
                scheme = self.config.time_scheme.label(),
                local_time_step = self.config.local_time_step,
                precision = T::PRECISION.label(),
            )
            .entered();
            T::advance_structured_explicit(
                self,
                ctx,
                fields,
                storage,
                dt,
                self.config.local_time_step,
                p_floor,
                &eos,
            )?;
        }
        {
            let _span = info_span!("enforce_positivity_post").entered();
            fields.enforce_positivity(&eos, p_floor);
        }
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: step_residual,
            residual_log10: log10_positive(step_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
        })
    }
}
