//! [`ExecutionContext`](super::context::ExecutionContext) 设备状态（ADR 0017 G0）。

use crate::error::{AsimuError, Result};

use super::context::ExecConfig;
use super::device::ExecDevice;

/// 算例级后端状态；构造后不在步间切换设备族。
pub(crate) enum BackendState {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(Box<super::gpu::cuda::CudaBackendState>),
}

impl BackendState {
    pub(crate) fn try_new(config: &ExecConfig) -> Result<Self> {
        match config.device {
            ExecDevice::Cpu => Ok(Self::Cpu),
            ExecDevice::GpuCuda => Self::try_new_cuda(),
        }
    }

    fn try_new_cuda() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self::Cuda(Box::new(
                super::gpu::cuda::CudaBackendState::try_new()?,
            )))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(AsimuError::Exec(
                "backend = \"cuda\" 需要启用 Cargo feature cuda".to_string(),
            ))
        }
    }

    pub(crate) fn sync_to_host(&mut self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(state) => state.sync_to_host(),
        }
    }

    pub(crate) fn sync_to_device(&mut self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(state) => state.sync_to_device(None),
        }
    }

    pub(crate) fn mark_cuda_primitives_stale(&mut self) {
        #[cfg(feature = "cuda")]
        if let Self::Cuda(state) = self {
            state.mark_host_primitives_updated();
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn mark_cuda_primitives_stale_after_integration(&mut self) {
        if let Self::Cuda(state) = self {
            state.mark_primitives_stale_after_integration();
        }
    }

    pub(crate) fn sync_cuda_primitives_to_device(
        &mut self,
        primitives: &crate::field::PrimitiveFieldsT<f32>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Self::Cuda(state) = self {
            return state.sync_primitives_to_device(primitives);
        }
        let _ = primitives;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rhs_pipeline_active(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.rhs_pipeline_active()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_timestep_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.timestep_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_residual_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.residual_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_host_bc_primitives_synced(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.host_bc_primitives_synced()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_conserved_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.conserved_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_boundary_ghosts_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.boundary_ghosts_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_spectral_diffusivity_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.spectral_diffusivity_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_lusgs_diagonal_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.lusgs_diagonal_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_u_n_on_device(&self) -> Option<bool> {
        match self {
            Self::Cuda(state) => Some(state.u_n_on_device()),
            Self::Cpu => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_mut(&mut self) -> Result<&mut super::gpu::cuda::CudaBackendState> {
        match self {
            Self::Cuda(state) => Ok(state.as_mut()),
            Self::Cpu => Err(AsimuError::Exec("CUDA 装配需要 backend = cuda".to_string())),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn try_csr_spmv_cuda(
        &mut self,
        matrix: &crate::exec::CsrSpmvView<'_>,
        x: &[crate::core::Real],
        y: &mut [crate::core::Real],
    ) -> Result<()> {
        if let Self::Cuda(state) = self {
            return state.csr_spmv(matrix, x, y);
        }
        Err(AsimuError::Exec(
            "CUDA SpMV 需要 backend = cuda".to_string(),
        ))
    }
}
