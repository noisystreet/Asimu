//! 结构化 3D typed 谱半径与当地时间步准备（ADR 0019 S0-b）。

use tracing::info_span;

use crate::core::Real;
use crate::error::Result;
use crate::field::ConservedFieldsT;
use crate::solver::compressible::helpers::{
    RefreshCompressibleStateTypedInput, finalize_cell_dts_from_sigma,
    refresh_compressible_ghosts_and_primitives_typed,
};
use crate::solver::compressible::spectral_radius::cell_local_dt_spectral;
use crate::solver::compressible::spectral_radius_3d_f32::SpectralRadius3dTypedParams;
use crate::solver::compressible::structured_compute_backend::StructuredComputeBackend;
use crate::solver::compressible::{CompressibleAdvanceContext3dTyped, CompressibleEulerSolver};
use crate::solver::time::{min_positive_dt, positive_fixed_dt};

impl CompressibleEulerSolver {
    pub(crate) fn compute_cell_dts_3d_typed<T: StructuredComputeBackend>(
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

    pub(crate) fn prepare_spectral_timestep_3d_typed<T: StructuredComputeBackend>(
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
        })?;
        ctx.spectral_primitives = ctx.primitive_scratch.cast_real()?;
        let params = SpectralRadius3dTypedParams {
            mesh: ctx.structured,
            boundary_mesh: ctx.mesh,
            boundaries: ctx.patches,
            ghosts: ctx.ghosts,
            primitives: &ctx.primitive_scratch,
            face_cache_f32: ctx.face_cache_f32,
            eos: ctx.eos,
            min_pressure: p_floor,
            viscous: ctx.viscous,
        };
        let sigma_typed = T::cell_spectral_radius_3d_typed(&params)?;
        let sigma = T::sigma_to_real(sigma_typed);
        let volumes = ctx.structured.cell_volumes();
        let cell_dts = cell_local_dt_spectral(&volumes, &sigma, cfl)?;
        Ok((cell_dts, sigma))
    }

    pub(crate) fn prepare_lusgs_timestep_3d_typed<T: StructuredComputeBackend>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &mut ConservedFieldsT<T>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)> {
        let _span = info_span!(
            "prepare_lusgs_timestep_typed",
            precision = T::PRECISION.label(),
        )
        .entered();
        let (_, sigma) = self.prepare_spectral_timestep_3d_typed(ctx, fields, cfl, p_floor)?;
        let cell_dts = finalize_cell_dts_from_sigma(
            &ctx.structured.cell_volumes(),
            &sigma,
            cfl,
            positive_fixed_dt(self.config.time.dt),
            true,
        )?;
        Ok((cell_dts, sigma))
    }
}
