//! 结构化 3D 可压缩 typed 时间推进（ADR 0016 P2）。

use tracing::info_span;

use super::compressible_rhs_typed::EvaluateRhs3dTyped;
use crate::core::{ComputeFloat, Real, log10_positive};
use crate::discretization::InviscidFluxConfig;
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::physics::IdealGasEoS;
use crate::solver::compressible::{
    CompressibleAdvanceContext3dTyped, CompressibleEulerSolver, CompressibleStepInfo,
    positive_fixed_dt,
};
use crate::solver::compressible_helpers::{
    RefreshCompressibleStateTypedInput, refresh_compressible_ghosts_and_primitives_typed,
};
use crate::solver::spectral_radius::{
    SpectralRadius3dParams, cell_local_dt_spectral, cell_spectral_radius_3d,
};
use crate::solver::state::SolverState;
use crate::solver::time::{
    Rk4StorageT, RungeKutta4Integrator, TimeIntegrationScheme, TimeIntegrator, euler_step,
    euler_step_local, min_positive_dt, rk4_step, rk4_step_local,
};

impl CompressibleEulerSolver {
    fn rhs_context_3d_typed<'a, T: ComputeFloat>(
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
            spectral_primitives: &mut ctx.spectral_primitives,
            gradient_scratch: &mut ctx.gradient_scratch,
        }
    }

    /// typed 3D 显式推进（rk4/euler；一阶无粘）。
    pub fn advance_step_3d_typed<T: ComputeFloat>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        storage: &mut Rk4StorageT<T>,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let cfl = self.cfl_for_step(state);
        let p_floor = Self::positivity_pressure_floor(ctx.freestream);
        self.advance_explicit_step_3d_typed(ctx, fields, storage, state, integrator, cfl, p_floor)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_explicit_step_3d_typed<T: ComputeFloat>(
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
            (scheme, _) => Err(crate::error::AsimuError::Solver(format!(
                "typed 显式推进不支持 {}",
                scheme.label()
            ))),
        }
    }

    fn compute_cell_dts_3d_typed<T: ComputeFloat>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<Vec<Real>> {
        let n = fields.num_cells();
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(vec![dt; n]);
        }
        let (cell_dts, _) = self.prepare_spectral_timestep_3d_typed(ctx, fields, cfl, p_floor)?;
        if self.config.local_time_step {
            Ok(cell_dts)
        } else {
            let dt = min_positive_dt(&cell_dts);
            Ok(vec![dt; n])
        }
    }

    fn prepare_spectral_timestep_3d_typed<T: ComputeFloat>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        fields.enforce_positivity(ctx.eos, p_floor);
        refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
            boundary_mesh: ctx.mesh,
            patches: ctx.patches,
            fields,
            ghosts: ctx.ghosts,
            eos: ctx.eos,
            freestream: ctx.freestream,
            reference: ctx.reference,
            viscous: ctx.viscous,
            min_pressure: p_floor,
            primitives: &mut ctx.primitive_scratch,
            spectral_primitives: &mut ctx.spectral_primitives,
        })?;
        let params = SpectralRadius3dParams {
            mesh: ctx.structured,
            boundary_mesh: ctx.mesh,
            boundaries: ctx.patches,
            ghosts: ctx.ghosts,
            primitives: &ctx.spectral_primitives,
            eos: ctx.eos,
            min_pressure: p_floor,
            viscous: ctx.viscous,
        };
        let sigma = cell_spectral_radius_3d(&params)?;
        let volumes = params.mesh.cell_volumes();
        let cell_dts = cell_local_dt_spectral(&volumes, &sigma, cfl)?;
        Ok((cell_dts, sigma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundarySet;
    use crate::core::approx_eq;
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
    use crate::field::PrimitiveFields;
    use crate::mesh::StructuredMesh3d;
    use crate::physics::FreestreamParams;
    use crate::solver::compressible::{CompressibleAdvanceContext3d, CompressibleEulerConfig};
    use crate::solver::time::Rk4Storage;

    fn freestream_box_context<T: ComputeFloat>(
        side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
    ) -> (
        StructuredMesh3d,
        BoundarySet,
        ConservedFieldsT<T>,
        crate::discretization::BoundaryGhostBuffer,
        IdealGasEoS,
        FreestreamParams,
    ) {
        let (mesh, boundary, fields, ghosts) = uniform_farfield_box(3, 3, 3, 1.0, 1.0, 1.0, side);
        let fields_t = ConservedFieldsT::<T>::from_real_fields(&fields).expect("typed fields");
        (mesh, boundary, fields_t, ghosts, *side.eos, *side.fs)
    }

    #[test]
    fn f32_explicit_step_matches_f64_on_uniform_box() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: crate::solver::time::RungeKutta4Config {
                dt: 1.0e-4,
                max_steps: 1,
            },
            ..CompressibleEulerConfig::default()
        });
        let (mesh, patches, fields_f32, mut ghosts_f32, eos, freestream) =
            freestream_box_context::<f32>(&side);
        let (_, _, fields_f64, ghosts_f64, _, _) = freestream_box_context::<f64>(&side);
        let mut ghosts_f64 = ghosts_f64;
        let mut ctx_f32 = CompressibleAdvanceContext3dTyped {
            mesh: &mesh,
            structured: &mesh,
            patches: &patches,
            ghosts: &mut ghosts_f32,
            eos: &eos,
            freestream: &freestream,
            reference: None,
            primitive_scratch: crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells())
                .expect("prim f32"),
            spectral_primitives: PrimitiveFields::zeros(mesh.num_cells()).expect("prim f64"),
            gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
                .expect("grad"),
            viscous: None,
        };
        let mut ctx_f64 = CompressibleAdvanceContext3d {
            mesh: &mesh,
            structured: &mesh,
            patches: &patches,
            ghosts: &mut ghosts_f64,
            eos: &eos,
            freestream: &freestream,
            reference: None,
            primitive_scratch: PrimitiveFields::zeros(mesh.num_cells()).expect("prim"),
            gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
                .expect("grad"),
            viscous: None,
            residual_correction: None,
        };
        let mut fields_f32 = fields_f32;
        let mut fields_f64 = fields_f64;
        let mut storage_f32 = Rk4StorageT::<f32>::new(mesh.num_cells()).expect("storage f32");
        let mut storage_f64 = Rk4Storage::new(mesh.num_cells()).expect("storage f64");
        let mut state_f32 = SolverState::default();
        let mut state_f64 = SolverState::default();
        let mut integrator_f32 = RungeKutta4Integrator::new(solver.config.time);
        let mut integrator_f64 = RungeKutta4Integrator::new(solver.config.time);
        let info_f32 = solver
            .advance_step_3d_typed(
                &mut ctx_f32,
                &mut fields_f32,
                &mut storage_f32,
                &mut state_f32,
                &mut integrator_f32,
            )
            .expect("f32 step");
        let info_f64 = solver
            .advance_step_3d(
                &mut ctx_f64,
                &mut fields_f64,
                &mut storage_f64,
                &mut state_f64,
                &mut integrator_f64,
            )
            .expect("f64 step");
        assert!(approx_eq(
            info_f32.residual_rms,
            info_f64.residual_rms,
            1.0e-5
        ));
        for i in 0..mesh.num_cells() {
            let rho_f32 = fields_f32.density.values()[i].to_real();
            let rho_f64 = fields_f64.density.values()[i];
            let rel = (rho_f32 - rho_f64).abs() / rho_f64.max(1.0e-12);
            assert!(rel < 1.0e-3, "cell {i} rel={rel}");
        }
    }
}
