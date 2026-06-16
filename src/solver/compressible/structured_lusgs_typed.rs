//! 结构化 3D typed LU-SGS 对角更新（扫掠见 S4；ADR 0019 S0-b / S1-c）。

use tracing::info_span;

use crate::core::{ComputePrecision, Real, log10_positive};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFieldsT, LusgsDiagonalCoeffs, LusgsDiagonalCoeffsF32, assign_lusgs_diagonal_update_f32,
};
use crate::solver::compressible::structured_compute_backend::StructuredComputeBackend;
use crate::solver::compressible::structured_timestep_buffers::StructuredLusgsDiagonalUpdate;
use crate::solver::compressible::{
    CompressibleAdvanceContext3dTyped, CompressibleEulerSolver, CompressibleStepInfo,
};
use crate::solver::state::SolverState;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Integrator, TimeIntegrator, min_positive_dt, min_positive_dt_f32,
};

impl CompressibleEulerSolver {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance_lusgs_step_3d_typed<T: StructuredComputeBackend>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        storage: &mut Rk4StorageT<T>,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
        cfl: Real,
        p_floor: Real,
    ) -> Result<CompressibleStepInfo> {
        if !self.config.local_time_step {
            return Err(AsimuError::Config(
                "time.scheme = lu_sgs 须配合 [time].local_time_step = true（稳态伪时间）"
                    .to_string(),
            ));
        }
        if self.config.lu_sgs.sweep {
            return Err(AsimuError::Config(
                "compute_precision f32 typed 路径暂不支持 lusgs_sweep = true".to_string(),
            ));
        }
        let inviscid = self.config.inviscid;
        let dt = {
            let _span = info_span!(
                "compute_dt",
                cells = ctx.structured.num_cells(),
                scheme = "lu_sgs",
                precision = T::PRECISION.label(),
            )
            .entered();
            self.prepare_lusgs_timestep_3d_typed(ctx, fields, cfl, p_floor)?;
            match T::PRECISION {
                ComputePrecision::F32 => min_positive_dt_f32(&ctx.timestep.cell_dts_f32) as Real,
                ComputePrecision::F64 => min_positive_dt(&ctx.timestep.cell_dts),
            }
        };
        integrator.config.dt = dt;
        let eos = *ctx.eos;
        let lu_sgs = self.config.lu_sgs;
        {
            let _span = info_span!(
                "time_integration",
                scheme = "lu_sgs",
                local_time_step = true,
                precision = T::PRECISION.label(),
            )
            .entered();
            fields.enforce_positivity(&eos, p_floor);
            storage.u0.copy_from(fields)?;
            {
                let _span = info_span!("lu_sgs_rhs").entered();
                self.rhs_context_3d_typed(ctx, &inviscid, p_floor)
                    .run(&storage.u0, &mut storage.k1)?;
            }
            T::apply_structured_lusgs_diagonal_update(
                &mut storage.stage,
                &storage.u0,
                &storage.k1,
                ctx,
                lu_sgs.omega,
                eos.gamma,
                p_floor,
            )?;
            fields.copy_from(&storage.stage)?;
            fields.enforce_positivity(&eos, p_floor);
        }
        let step_residual = storage.k1.density_rms_norm();
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

impl StructuredLusgsDiagonalUpdate for f32 {
    fn apply_structured_lusgs_diagonal_update(
        out: &mut ConservedFieldsT<f32>,
        base: &ConservedFieldsT<f32>,
        residual: &crate::field::ConservedResidualT<f32>,
        ctx: &CompressibleAdvanceContext3dTyped<'_, f32>,
        omega: Real,
        _gamma: Real,
        _min_pressure: Real,
    ) -> Result<()> {
        assign_lusgs_diagonal_update_f32(
            out,
            base,
            residual,
            &ctx.timestep.sigma_f32,
            &ctx.timestep.cell_dts_f32,
            LusgsDiagonalCoeffsF32 {
                omega: omega as f32,
                inv_dt_phys: 0.0,
            },
        )
    }
}

impl StructuredLusgsDiagonalUpdate for f64 {
    fn apply_structured_lusgs_diagonal_update(
        out: &mut ConservedFieldsT<f64>,
        base: &ConservedFieldsT<f64>,
        residual: &crate::field::ConservedResidualT<f64>,
        ctx: &CompressibleAdvanceContext3dTyped<'_, f64>,
        omega: Real,
        gamma: Real,
        min_pressure: Real,
    ) -> Result<()> {
        out.assign_lusgs_diagonal_update(
            base,
            residual,
            &ctx.timestep.sigma,
            &ctx.timestep.cell_dts,
            LusgsDiagonalCoeffs::steady_pseudo_time(omega, gamma, min_pressure),
        )
    }
}
