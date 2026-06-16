//! 结构化 3D 可压缩 typed 时间推进入口（ADR 0016 P2/P4；子模块见 ADR 0019 S0-b）。

use std::time::Instant;

use tracing::info_span;

use super::gmres_implicit_3d::{GmresStepLog, GmresStepTiming, log_gmres_step_diagnostics};
use super::rhs_typed::EvaluateRhs3dTyped;
use super::structured_compute_backend::StructuredComputeBackend;
use crate::core::{ComputeFloat, Real, elapsed_ms, log10_positive};
use crate::discretization::InviscidFluxConfig;
use crate::error::{AsimuError, Result};
use crate::field::ConservedFieldsT;
use crate::solver::compressible::{
    CompressibleAdvanceContext3dTyped, CompressibleEulerSolver, CompressibleStepInfo,
};
use crate::solver::state::SolverState;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator, min_positive_dt,
};

#[path = "gmres_implicit_3d_typed.rs"]
mod gmres_implicit_3d_typed;
#[path = "structured_explicit_typed.rs"]
mod structured_explicit_typed;
#[path = "structured_lusgs_typed.rs"]
mod structured_lusgs_typed;
#[path = "structured_prepare_timestep_typed.rs"]
mod structured_prepare_timestep_typed;

use gmres_implicit_3d_typed::apply_delta_with_line_search_typed;

impl CompressibleEulerSolver {
    pub(crate) fn rhs_context_3d_typed<'a, T: ComputeFloat + StructuredComputeBackend>(
        &'a self,
        ctx: &'a mut CompressibleAdvanceContext3dTyped<'_, T>,
        inviscid: &'a InviscidFluxConfig,
        min_pressure: Real,
    ) -> EvaluateRhs3dTyped<'a, T> {
        EvaluateRhs3dTyped {
            mesh: ctx.mesh,
            structured: ctx.structured,
            patches: ctx.patches,
            ghosts: ctx.ghosts,
            eos: ctx.eos,
            freestream: ctx.freestream,
            reference: ctx.reference,
            inviscid,
            viscous: self.config.viscous.as_ref(),
            min_pressure,
            primitive_scratch: &mut ctx.primitive_scratch,
            gradient_scratch: &mut ctx.gradient_scratch,
            interface_residual: ctx.interface_residual,
            face_cache_f32: ctx.face_cache_f32,
        }
    }

    /// typed 3D 时间推进（显式 rk4/euler；隐式 lu_sgs/gmres）。
    #[allow(private_bounds)]
    pub fn advance_step_3d_typed<T: StructuredComputeBackend>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        storage: &mut Rk4StorageT<T>,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let cfl = self.cfl_for_step(state);
        let p_floor = Self::positivity_pressure_floor(ctx.freestream);
        if self.config.time_scheme == TimeIntegrationScheme::Gmres {
            return self.advance_gmres_step_3d_typed(
                ctx, fields, storage, state, integrator, cfl, p_floor,
            );
        }
        if self.config.time_scheme == TimeIntegrationScheme::LuSgs {
            return self.advance_lusgs_step_3d_typed(
                ctx, fields, storage, state, integrator, cfl, p_floor,
            );
        }
        self.advance_explicit_step_3d_typed(ctx, fields, storage, state, integrator, cfl, p_floor)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_gmres_step_3d_typed<T: ComputeFloat + StructuredComputeBackend>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        storage: &mut Rk4StorageT<T>,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
        cfl: Real,
        p_floor: Real,
    ) -> Result<CompressibleStepInfo> {
        let step_start = Instant::now();
        if !self.config.local_time_step {
            return Err(AsimuError::Config(
                "time.scheme = gmres 须配合 [time].local_time_step = true（稳态伪时间）"
                    .to_string(),
            ));
        }
        let compute_dt_start = Instant::now();
        let (dt, cell_dts, sigma) = {
            let _span = info_span!("compute_dt").entered();
            let (cell_dts, sigma) =
                self.prepare_lusgs_timestep_3d_typed(ctx, fields, cfl, p_floor)?;
            (min_positive_dt(&cell_dts), cell_dts, sigma)
        };
        let compute_dt_ms = elapsed_ms(compute_dt_start);
        integrator.config.dt = dt;
        storage.ensure_capacity(fields.num_cells())?;
        storage.u0.copy_from(fields)?;
        let implicit_solve_start = Instant::now();
        let delta = {
            let _span = info_span!("gmres_implicit_solve").entered();
            self.solve_gmres_implicit_delta_3d_typed(
                ctx,
                &storage.u0,
                &cell_dts,
                &sigma,
                p_floor,
                self.config.gmres,
            )?
        };
        let implicit_solve_ms = elapsed_ms(implicit_solve_start);
        let line_search_start = Instant::now();
        let update = {
            let _span = info_span!("gmres_line_search").entered();
            apply_delta_with_line_search_typed(
                fields,
                &mut storage.stage,
                &storage.u0,
                &delta,
                ctx.eos,
                p_floor,
            )?
        };
        let line_search_ms = elapsed_ms(line_search_start);
        let step_residual = delta.base_residual_rms;
        let step_total_ms = elapsed_ms(step_start);
        log_gmres_step_diagnostics(GmresStepLog {
            step: state.time_step.saturating_add(1),
            dt,
            cfl,
            delta: &delta,
            update,
            residual_rms: step_residual,
            timing: GmresStepTiming {
                compute_dt_ms,
                implicit_solve_ms,
                line_search_ms,
                post_residual_ms: 0.0,
                step_total_ms,
            },
        });
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
            inner_iterations: 0,
        })
    }
}

#[cfg(test)]
#[path = "structured_typed_tests.rs"]
mod structured_typed_tests;
