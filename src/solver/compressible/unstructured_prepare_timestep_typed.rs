//! 非结构 typed 时间步准备（BC/原变量刷新、谱半径、局部 \(\Delta t\)）。

use tracing::info_span;

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::{ComputeFloat, Real};
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::exec::ExecutionContext;
use crate::field::{ConservedFieldsT, PrimitiveFillFromConserved};
use crate::solver::compressible::spectral_radius_unstructured::{
    SpectralRadiusUnstructuredTypedParams, UnstructuredSpectralRadiusTyped,
};
#[cfg(feature = "cuda")]
use crate::solver::time::TimeIntegrationScheme;
use crate::solver::{
    RefreshCompressibleStateTypedInput, finalize_cell_dts_from_sigma,
    finalize_cell_dts_from_sigma_f32, min_positive_dt, min_positive_dt_f32,
    refresh_compressible_ghosts_and_primitives_typed,
};

use super::{UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};

/// 时间步准备阶段谱半径输出；`cell_dts` 为 `Some` 时表示已在 GPU/CPU 同路径完成 finalize。
pub(crate) struct TimestepPrepareSpectral<T> {
    pub sigma: Vec<T>,
    pub cell_dts: Option<Vec<T>>,
}

pub(crate) fn prepare_unstructured_timestep_typed<
    T: ComputeFloat
        + UnstructuredSpectralRadiusAtPrepare
        + UnstructuredTimestepFromSigma
        + UnstructuredCudaPrepareSync
        + PrimitiveFillFromConserved,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    cfl: Real,
    p_floor: Real,
) -> Result<Real> {
    let n = env.config.mesh.num_cells();
    let _prepare_span = info_span!(
        "unstructured_prepare_timestep_typed",
        cells = n,
        precision = T::PRECISION.label(),
        local_time_step = env.config.local_time_step,
    )
    .entered();
    fields.enforce_positivity(env.config.eos, p_floor);
    work.ghosts
        .ensure_face_capacity(env.config.mesh.num_faces());
    {
        let _span = info_span!("unstructured_refresh_state_typed", cells = n).entered();
        T::refresh_state_for_prepare(env, fields, work, p_floor)?;
        T::maybe_prepare_cuda_rhs_device_state(env, work, p_floor)?;
    }
    let fixed_dt = if env.config.dual_time.is_some() {
        None
    } else {
        env.config.fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite())
    };
    let prepared = compute_spectral_radius_at_prepare(env, work, p_floor, cfl, fixed_dt)?;
    {
        let _span = info_span!(
            "unstructured_finalize_cell_dts_typed",
            cells = n,
            fixed_dt = fixed_dt.is_some(),
        )
        .entered();
        T::store_sigma_and_cell_dts(work, prepared, cfl, fixed_dt, env.config.local_time_step)
    }
}

fn compute_spectral_radius_at_prepare<
    T: ComputeFloat + UnstructuredSpectralRadiusAtPrepare + UnstructuredTimestepFromSigma,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    p_floor: Real,
    cfl: Real,
    fixed_dt: Option<Real>,
) -> Result<TimestepPrepareSpectral<T>> {
    let n = env.config.mesh.num_cells();
    let _span = info_span!("unstructured_spectral_radius_typed", cells = n).entered();
    T::compute_spectral_radius_at_prepare(
        env,
        work,
        p_floor,
        cfl,
        fixed_dt,
        env.config.local_time_step,
    )
}

/// BC/原变量刷新后同步 device primitive（f32 CUDA 单步一次 H2D）。
pub(crate) trait UnstructuredCudaPrepareSync:
    UnstructuredSpectralRadiusTyped + PrimitiveFillFromConserved
{
    fn sync_primitives_after_refresh(work: &mut UnstructuredStepWorkTyped<Self>) -> Result<()>;

    fn refresh_state_for_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        fields: &mut ConservedFieldsT<Self>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        p_floor: Real,
    ) -> Result<()> {
        refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
            boundary_mesh: env.config.mesh,
            patches: env.config.patches,
            fields,
            ghosts: &mut work.ghosts,
            eos: env.config.eos,
            freestream: env.config.freestream,
            reference: env.config.reference,
            viscous: env.config.viscous,
            min_pressure: p_floor,
            primitives: &mut work.primitives,
        })?;
        Self::sync_primitives_after_refresh(work)
    }

    fn maybe_prepare_cuda_rhs_device_state(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        p_floor: Real,
    ) -> Result<()> {
        let _ = (env, work, p_floor);
        Ok(())
    }

    fn step_density_residual_rms(work: &mut UnstructuredStepWorkTyped<Self>) -> Result<Real>;

    fn maybe_upload_lusgs_integration_base(
        work: &mut UnstructuredStepWorkTyped<Self>,
    ) -> Result<()>;

    fn lusgs_skip_copy_stage_after_diagonal(work: &UnstructuredStepWorkTyped<Self>) -> bool;

    fn maybe_enforce_conserved_after_integration(
        work: &mut UnstructuredStepWorkTyped<Self>,
        eos: &crate::physics::IdealGasEoS,
        min_pressure: Real,
    ) -> Result<()>;

    fn maybe_download_conserved_for_output(
        work: &mut UnstructuredStepWorkTyped<Self>,
        fields: &mut ConservedFieldsT<Self>,
    ) -> Result<()>;
}

impl UnstructuredCudaPrepareSync for f64 {
    fn sync_primitives_after_refresh(_work: &mut UnstructuredStepWorkTyped<f64>) -> Result<()> {
        Ok(())
    }

    fn step_density_residual_rms(work: &mut UnstructuredStepWorkTyped<f64>) -> Result<Real> {
        Ok(work.storage.k1.density_rms_norm())
    }

    fn maybe_upload_lusgs_integration_base(
        _work: &mut UnstructuredStepWorkTyped<f64>,
    ) -> Result<()> {
        Ok(())
    }

    fn lusgs_skip_copy_stage_after_diagonal(_work: &UnstructuredStepWorkTyped<f64>) -> bool {
        false
    }

    fn maybe_enforce_conserved_after_integration(
        _work: &mut UnstructuredStepWorkTyped<f64>,
        _eos: &crate::physics::IdealGasEoS,
        _min_pressure: Real,
    ) -> Result<()> {
        Ok(())
    }

    fn maybe_download_conserved_for_output(
        _work: &mut UnstructuredStepWorkTyped<f64>,
        _fields: &mut ConservedFieldsT<f64>,
    ) -> Result<()> {
        Ok(())
    }
}

/// 时间步准备阶段的谱半径（f32 可走 CUDA）。
pub(crate) trait UnstructuredSpectralRadiusAtPrepare:
    UnstructuredSpectralRadiusTyped
{
    fn compute_spectral_radius_at_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        p_floor: Real,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<TimestepPrepareSpectral<Self>>;
}

impl UnstructuredSpectralRadiusAtPrepare for f32 {
    fn compute_spectral_radius_at_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<f32>,
        p_floor: Real,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<TimestepPrepareSpectral<f32>> {
        let params = SpectralRadiusUnstructuredTypedParams {
            mesh: env.config.mesh,
            mesh_cache: &work.mesh_cache,
            boundaries: env.config.patches,
            ghosts: &work.ghosts,
            primitives: &work.primitives,
            eos: env.config.eos,
            min_pressure: p_floor,
            viscous: env.config.viscous,
        };
        #[cfg(feature = "cuda")]
        {
            let keep_timestep_on_device = env.config.time_scheme == TimeIntegrationScheme::LuSgs
                && !env.config.lu_sgs.sweep
                && work.exec.device() == ExecDevice::GpuCuda;
            let (sigma, cell_dts) =
                crate::solver::compressible::spectral_radius_unstructured_f32_cuda::compute_spectral_radius_f32_with_exec(
                    &params,
                    &mut work.exec,
                    cfl,
                    fixed_dt,
                    local_time_step,
                    keep_timestep_on_device,
                )?;
            Ok(TimestepPrepareSpectral { sigma, cell_dts })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (cfl, fixed_dt, local_time_step);
            let sigma = Self::cell_spectral_radius_unstructured_typed(&params)?;
            Ok(TimestepPrepareSpectral {
                sigma,
                cell_dts: None,
            })
        }
    }
}

impl UnstructuredSpectralRadiusAtPrepare for f64 {
    fn compute_spectral_radius_at_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        work: &mut UnstructuredStepWorkTyped<f64>,
        p_floor: Real,
        _cfl: Real,
        _fixed_dt: Option<Real>,
        _local_time_step: bool,
    ) -> Result<TimestepPrepareSpectral<Real>> {
        let sigma = Self::cell_spectral_radius_unstructured_typed(
            &SpectralRadiusUnstructuredTypedParams {
                mesh: env.config.mesh,
                mesh_cache: &work.mesh_cache,
                boundaries: env.config.patches,
                ghosts: &work.ghosts,
                primitives: &work.primitives,
                eos: env.config.eos,
                min_pressure: p_floor,
                viscous: env.config.viscous,
            },
        )?;
        Ok(TimestepPrepareSpectral {
            sigma,
            cell_dts: None,
        })
    }
}

/// 谱半径结果写入时间步缓冲（f32 原生 \(\sigma_i\)，无 prepare 边界转换）。
pub(crate) trait UnstructuredTimestepFromSigma: UnstructuredSpectralRadiusTyped {
    fn store_sigma_and_cell_dts(
        work: &mut UnstructuredStepWorkTyped<Self>,
        prepared: TimestepPrepareSpectral<Self>,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<Real>;
}

impl UnstructuredTimestepFromSigma for f32 {
    fn store_sigma_and_cell_dts(
        work: &mut UnstructuredStepWorkTyped<f32>,
        prepared: TimestepPrepareSpectral<f32>,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<Real> {
        work.timestep.sigma_f32 = prepared.sigma;
        if let Some(cell_dts) = prepared.cell_dts {
            work.timestep.cell_dts_f32 = cell_dts;
            Ok(min_positive_dt_f32(&work.timestep.cell_dts_f32) as Real)
        } else {
            #[cfg(feature = "cuda")]
            if work.exec.cuda_timestep_on_device() {
                work.timestep.cell_dts_f32.clear();
                if let Some(dt) = fixed_dt {
                    return Ok(dt);
                }
                return Ok(work.exec.cuda_download_min_cell_dt_f32()? as Real);
            }
            work.timestep.cell_dts_f32 = finalize_cell_dts_from_sigma_f32(
                &work.volumes_f32,
                &work.timestep.sigma_f32,
                cfl as f32,
                fixed_dt.map(|d| d as f32),
                local_time_step,
            )?;
            Ok(min_positive_dt_f32(&work.timestep.cell_dts_f32) as Real)
        }
    }
}

impl UnstructuredTimestepFromSigma for f64 {
    fn store_sigma_and_cell_dts(
        work: &mut UnstructuredStepWorkTyped<f64>,
        prepared: TimestepPrepareSpectral<Real>,
        cfl: Real,
        fixed_dt: Option<Real>,
        local_time_step: bool,
    ) -> Result<Real> {
        work.timestep.sigma = prepared.sigma;
        work.timestep.cell_dts = if let Some(cell_dts) = prepared.cell_dts {
            cell_dts
        } else {
            finalize_cell_dts_from_sigma(
                &work.volumes,
                &work.timestep.sigma,
                cfl,
                fixed_dt,
                local_time_step,
            )?
        };
        Ok(min_positive_dt(&work.timestep.cell_dts))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn f32_cuda_prepare_device_refresh(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &UnstructuredStepWorkTyped<f32>,
) -> bool {
    env.config.viscous.is_some() && work.exec.device() == ExecDevice::GpuCuda
}

#[cfg(feature = "cuda")]
pub(crate) fn f32_cuda_viscous_rhs_pipeline(
    env: &UnstructuredRunEnvTyped<'_>,
    exec: &ExecutionContext,
) -> bool {
    env.config.viscous.is_some() && exec.device() == ExecDevice::GpuCuda
}
