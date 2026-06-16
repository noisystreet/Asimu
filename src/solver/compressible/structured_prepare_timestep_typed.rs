//! 结构化 3D typed 谱半径与当地时间步准备（ADR 0019 S0-b / S1-c）。

use tracing::info_span;

use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::ConservedFieldsT;
use crate::solver::compressible::helpers::{
    RefreshCompressibleStateTypedInput, finalize_cell_dts_from_sigma,
    finalize_cell_dts_from_sigma_f32, refresh_compressible_ghosts_and_primitives_typed,
};
use crate::solver::compressible::spectral_radius_3d_f32::{
    SpectralRadius3dTypedParams, StructuredSpectralRadiusTyped,
};
use crate::solver::compressible::structured_timestep_buffers::StructuredSpectralTimestepPrepare;
use crate::solver::compressible::{CompressibleAdvanceContext3dTyped, CompressibleEulerSolver};
use crate::solver::time::positive_fixed_dt;

impl StructuredSpectralTimestepPrepare for f64 {
    fn prepare_spectral_timestep_3d(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, f64>,
        fields: &mut ConservedFieldsT<f64>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        refresh_for_spectral(ctx, fields, p_floor)?;
        let params = spectral_params(ctx, p_floor);
        ctx.timestep.sigma = f64::cell_spectral_radius_3d_typed(&params)?;
        ctx.timestep.cell_dts = finalize_cell_dts_from_sigma(
            &ctx.structured.cell_volumes(),
            &ctx.timestep.sigma,
            cfl,
            positive_fixed_dt(solver.config.time.dt),
            solver.config.local_time_step,
        )?;
        Ok((ctx.timestep.cell_dts.clone(), ctx.timestep.sigma.clone()))
    }

    fn prepare_lusgs_timestep_3d(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, f64>,
        fields: &mut ConservedFieldsT<f64>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        let _span = info_span!("prepare_lusgs_timestep_typed", precision = "f64").entered();
        Self::prepare_spectral_timestep_3d(solver, ctx, fields, cfl, p_floor)?;
        ctx.timestep.cell_dts = finalize_cell_dts_from_sigma(
            &ctx.structured.cell_volumes(),
            &ctx.timestep.sigma,
            cfl,
            positive_fixed_dt(solver.config.time.dt),
            true,
        )?;
        Ok((ctx.timestep.cell_dts.clone(), ctx.timestep.sigma.clone()))
    }
}

impl StructuredSpectralTimestepPrepare for f32 {
    fn prepare_spectral_timestep_3d(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, f32>,
        fields: &mut ConservedFieldsT<f32>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        refresh_for_spectral(ctx, fields, p_floor)?;
        let params = spectral_params(ctx, p_floor);
        ctx.timestep.sigma_f32 = f32::cell_spectral_radius_3d_typed(&params)?;
        ctx.timestep.cell_dts_f32 = finalize_cell_dts_from_sigma_f32(
            ctx.volumes_f32,
            &ctx.timestep.sigma_f32,
            cfl as f32,
            positive_fixed_dt(solver.config.time.dt).map(|dt| dt as f32),
            solver.config.local_time_step,
        )?;
        ctx.timestep.sigma = f32::sigma_to_real(ctx.timestep.sigma_f32.clone());
        ctx.timestep.cell_dts = ctx
            .timestep
            .cell_dts_f32
            .iter()
            .map(|dt| f64::from(*dt))
            .collect();
        Ok((ctx.timestep.cell_dts.clone(), ctx.timestep.sigma.clone()))
    }

    fn prepare_lusgs_timestep_3d(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, f32>,
        fields: &mut ConservedFieldsT<f32>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        let _span = info_span!("prepare_lusgs_timestep_typed", precision = "f32").entered();
        Self::prepare_spectral_timestep_3d(solver, ctx, fields, cfl, p_floor)?;
        ctx.timestep.cell_dts_f32 = finalize_cell_dts_from_sigma_f32(
            ctx.volumes_f32,
            &ctx.timestep.sigma_f32,
            cfl as f32,
            positive_fixed_dt(solver.config.time.dt).map(|dt| dt as f32),
            true,
        )?;
        ctx.timestep.cell_dts = ctx
            .timestep
            .cell_dts_f32
            .iter()
            .map(|dt| f64::from(*dt))
            .collect();
        Ok((ctx.timestep.cell_dts.clone(), ctx.timestep.sigma.clone()))
    }
}

impl CompressibleEulerSolver {
    pub(crate) fn prepare_spectral_timestep_3d_typed<T: StructuredSpectralTimestepPrepare>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        T::prepare_spectral_timestep_3d(self, ctx, fields, cfl, p_floor)
    }

    pub(crate) fn prepare_lusgs_timestep_3d_typed<T: StructuredSpectralTimestepPrepare>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        T::prepare_lusgs_timestep_3d(self, ctx, fields, cfl, p_floor)
    }
}

fn refresh_for_spectral<T: ComputeFloat + crate::field::PrimitiveFillFromConserved>(
    ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
    fields: &mut ConservedFieldsT<T>,
    p_floor: Real,
) -> Result<()> {
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
    })?;
    ctx.spectral_primitives = ctx.primitive_scratch.cast_real()?;
    Ok(())
}

fn spectral_params<'a, T: ComputeFloat>(
    ctx: &'a CompressibleAdvanceContext3dTyped<'_, T>,
    p_floor: Real,
) -> SpectralRadius3dTypedParams<'a, T> {
    SpectralRadius3dTypedParams {
        mesh: ctx.structured,
        boundary_mesh: ctx.mesh,
        boundaries: ctx.patches,
        ghosts: ctx.ghosts,
        primitives: &ctx.primitive_scratch,
        face_cache_f32: ctx.face_cache_f32,
        eos: ctx.eos,
        min_pressure: p_floor,
        viscous: ctx.viscous,
    }
}
