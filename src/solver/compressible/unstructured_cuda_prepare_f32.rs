//! f32 CUDA prepare 同步（`UnstructuredCudaPrepareSync` 实现）。

#[cfg(feature = "cuda")]
use crate::core::ExecDevice;
use crate::core::Real;
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::exec::gpu::cuda::cuda_compressible_bc_supported;
use crate::field::ConservedFieldsT;
#[cfg(feature = "cuda")]
use crate::solver::compressible::helpers::refresh_compressible_ghosts_only_typed;
use crate::solver::{
    RefreshCompressibleStateTypedInput, refresh_compressible_ghosts_and_primitives_typed,
};

use super::unstructured_prepare_timestep_typed::UnstructuredCudaPrepareSync;
#[cfg(feature = "cuda")]
use super::unstructured_prepare_timestep_typed::f32_cuda_prepare_device_refresh;
use super::{UnstructuredRunEnvTyped, UnstructuredStepWorkTyped};

impl UnstructuredCudaPrepareSync for f32 {
    fn sync_primitives_after_refresh(work: &mut UnstructuredStepWorkTyped<f32>) -> Result<()> {
        work.exec.sync_cuda_primitives_to_device(&work.primitives)
    }

    fn refresh_state_for_prepare(
        env: &UnstructuredRunEnvTyped<'_>,
        fields: &mut ConservedFieldsT<f32>,
        work: &mut UnstructuredStepWorkTyped<f32>,
        p_floor: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if f32_cuda_prepare_device_refresh(env, work) {
            let device_bc = cuda_compressible_bc_supported(env.config.patches);
            if !device_bc {
                refresh_compressible_ghosts_only_typed(RefreshCompressibleStateTypedInput {
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
            }
            let viscous = env.config.viscous.expect("cuda prepare 需粘性配置");
            work.exec.cuda_fill_primitives_and_diffusivity_on_device(
                fields,
                &work.mesh_cache,
                env.config.eos,
                viscous,
                p_floor,
            )?;
            return Ok(());
        }
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
        work: &mut UnstructuredStepWorkTyped<f32>,
        p_floor: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if work.exec.device() != ExecDevice::GpuCuda {
                return Ok(());
            }
            let Some(viscous) = env.config.viscous else {
                return Ok(());
            };
            work.exec.cuda_prepare_rhs_device_state(
                crate::exec::gpu::cuda::CudaPrepareRhsDeviceInput {
                    mesh_cache: &work.mesh_cache,
                    patches: env.config.patches,
                    ghosts: &work.ghosts,
                    primitives: &work.primitives,
                    eos: env.config.eos,
                    viscous,
                    freestream: env.config.freestream,
                    min_pressure: p_floor,
                },
            )?;
        }
        let _ = (env, work, p_floor);
        Ok(())
    }

    fn step_density_residual_rms(work: &mut UnstructuredStepWorkTyped<f32>) -> Result<Real> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda && work.exec.cuda_residual_on_device() {
            return Ok(work.exec.cuda_density_residual_rms_f32()? as Real);
        }
        Ok(work.storage.k1.density_rms_norm())
    }

    fn maybe_upload_lusgs_integration_base(
        work: &mut UnstructuredStepWorkTyped<f32>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda {
            work.exec
                .cuda_upload_conserved_for_integration(&work.storage.u0)?;
        }
        let _ = work;
        Ok(())
    }

    fn lusgs_skip_copy_stage_after_diagonal(work: &UnstructuredStepWorkTyped<f32>) -> bool {
        #[cfg(feature = "cuda")]
        {
            work.exec.device() == ExecDevice::GpuCuda && work.exec.cuda_lusgs_diagonal_on_device()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = work;
            false
        }
    }

    fn maybe_enforce_conserved_after_integration(
        work: &mut UnstructuredStepWorkTyped<f32>,
        eos: &crate::physics::IdealGasEoS,
        min_pressure: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda && work.exec.cuda_conserved_on_device() {
            work.exec
                .cuda_enforce_conserved_positivity_on_device(eos, min_pressure)?;
            return Ok(());
        }
        let _ = (work, eos, min_pressure);
        Ok(())
    }

    fn maybe_download_conserved_for_output(
        work: &mut UnstructuredStepWorkTyped<f32>,
        fields: &mut ConservedFieldsT<f32>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda {
            work.exec.cuda_download_conserved_if_on_device(fields)?;
        }
        let _ = (work, fields);
        Ok(())
    }

    fn snapshot_dual_time_u_n(
        work: &mut UnstructuredStepWorkTyped<f32>,
        fields: &ConservedFieldsT<f32>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda {
            work.exec.cuda_snapshot_u_n_on_device(fields)?;
            work.dual_time_state.inner_iterations = 0;
            if !work.exec.cuda_rhs_pipeline_active() {
                if work.exec.cuda_u_n_on_device() {
                    work.exec.cuda_download_u_n_on_device(
                        &mut work.dual_time_state.u_at_physical_level,
                    )?;
                } else {
                    work.dual_time_state.snapshot_u_n(fields)?;
                }
            }
            return Ok(());
        }
        work.dual_time_state.snapshot_u_n(fields)
    }

    fn prepare_dual_time_inner_base(
        work: &mut UnstructuredStepWorkTyped<f32>,
        fields: &mut ConservedFieldsT<f32>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda {
            work.exec.cuda_reset_pipeline_step()?;
            if work.exec.cuda_conserved_on_device() {
                work.exec.cuda_download_conserved_if_on_device(fields)?;
            }
        }
        work.storage.u0.copy_from(fields)
    }

    fn add_dual_time_storage_residual(
        work: &mut UnstructuredStepWorkTyped<f32>,
        fields: &ConservedFieldsT<f32>,
        dt_phys: Real,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if work.exec.device() == ExecDevice::GpuCuda
            && work.exec.cuda_residual_on_device()
            && work.exec.cuda_u_n_on_device()
        {
            if !work.exec.cuda_conserved_on_device() {
                work.exec.cuda_upload_conserved_for_integration(fields)?;
            }
            let mesh_key = std::ptr::from_ref(&work.mesh_cache).addr();
            return work.exec.cuda_add_physical_storage_residual_f32(
                dt_phys as f32,
                mesh_key,
                &work.volumes_f32,
            );
        }
        crate::solver::time::add_physical_storage_residual(
            &mut work.storage.k1,
            fields,
            &work.dual_time_state.u_at_physical_level,
            &work.volumes,
            dt_phys,
        )
    }
}
