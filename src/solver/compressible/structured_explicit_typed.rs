//! 结构化 3D typed 显式 RK4/Euler 推进（ADR 0019 S0-b）。

use tracing::info_span;

use crate::core::{ComputeFloat, Real, log10_positive};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::physics::IdealGasEoS;
use crate::solver::compressible::structured_compute_backend::StructuredComputeBackend;
use crate::solver::compressible::{
    CompressibleAdvanceContext3dTyped, CompressibleEulerSolver, CompressibleStepInfo,
};
use crate::solver::state::SolverState;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator, euler_step,
    euler_step_local, min_positive_dt, rk4_step, rk4_step_local,
};

impl CompressibleEulerSolver {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance_explicit_step_3d_typed<T: ComputeFloat + StructuredComputeBackend>(
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
        let (dt, cell_dts) = {
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
            let cell_dts = self.compute_cell_dts_3d_typed(ctx, fields, cfl, p_floor)?;
            (min_positive_dt(&cell_dts), cell_dts)
        };
        integrator.config.dt = dt;
        let eos = *ctx.eos;
        let step_residual = {
            let _span = info_span!("rhs_monitor").entered();
            self.rhs_context_3d_typed(ctx, &inviscid, p_floor)
                .run(fields, &mut storage.k1)?;
            storage.k1.density_rms_norm()
        };
        let cell_dts_arg = if self.config.local_time_step {
            Some(cell_dts.as_slice())
        } else {
            None
        };
        {
            let _span = info_span!(
                "time_integration",
                scheme = self.config.time_scheme.label(),
                local_time_step = self.config.local_time_step,
                precision = T::PRECISION.label(),
            )
            .entered();
            let evaluate = |u: &ConservedFieldsT<T>, r: &mut ConservedResidualT<T>| {
                self.rhs_context_3d_typed(ctx, &inviscid, p_floor).run(u, r)
            };
            self.advance_explicit_step_typed(
                fields,
                storage,
                dt,
                cell_dts_arg,
                evaluate,
                Some((&eos, p_floor)),
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

    fn advance_explicit_step_typed<T, F>(
        &self,
        fields: &mut ConservedFieldsT<T>,
        storage: &mut Rk4StorageT<T>,
        dt_global: Real,
        cell_dts: Option<&[Real]>,
        evaluate_rhs: F,
        positivity: Option<(&IdealGasEoS, Real)>,
    ) -> Result<()>
    where
        T: ComputeFloat,
        F: FnMut(&ConservedFieldsT<T>, &mut ConservedResidualT<T>) -> Result<()>,
    {
        let (eos, min_pressure) = match positivity {
            Some((eos, p)) => (Some(eos), p),
            None => (None, 1.0e-6),
        };
        match (self.config.time_scheme, cell_dts) {
            (TimeIntegrationScheme::Rk4, Some(dt)) => {
                rk4_step_local(fields, storage, dt, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::Rk4, None) => {
                rk4_step(fields, storage, dt_global, evaluate_rhs)
            }
            (TimeIntegrationScheme::Euler, Some(dt)) => {
                euler_step_local(fields, storage, dt, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::Euler, None) => {
                euler_step(fields, storage, dt_global, evaluate_rhs, eos, min_pressure)
            }
            (scheme, _) => Err(AsimuError::Solver(format!(
                "typed 显式推进不支持 {}",
                scheme.label()
            ))),
        }
    }
}
