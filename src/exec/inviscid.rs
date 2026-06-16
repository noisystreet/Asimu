//! 非结构一阶无粘内面装配入口（CPU / CUDA 分发；ADR 0017 G1）。

use crate::core::ExecDevice;
use crate::error::Result;
use crate::exec::context::ExecutionContext;
use crate::exec::gpu::cuda::CudaFirstOrderInviscidParams;
use crate::exec::gpu::cuda::ExecInteriorFaceTopology;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

/// 若当前 `ExecutionContext` 为 CUDA，在 device 上装配内面残差并返回 `Ok(true)`。
///
/// `params.flux_scheme`：`CUDA_FLUX_SCHEME_ROE`、`CUDA_FLUX_SCHEME_HVL` 或 `CUDA_FLUX_SCHEME_SLAU2`。
pub fn try_assemble_first_order_interior_f32(
    exec: &mut ExecutionContext,
    residual: &mut ConservedResidualT<f32>,
    primitives: &PrimitiveFieldsT<f32>,
    topo: &ExecInteriorFaceTopology,
    topo_key: usize,
    params: CudaFirstOrderInviscidParams,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_assemble_first_order_inviscid_interior(
                residual, primitives, topo, topo_key, params,
            )?;
            return Ok(true);
        }
    }
    let _ = (exec, residual, primitives, topo, topo_key, params);
    Ok(false)
}

/// 若当前 `ExecutionContext` 为 CUDA，在 device 上装配无粘边界面残差并返回 `Ok(true)`。
pub fn try_assemble_first_order_boundary_f32(
    exec: &mut ExecutionContext,
    residual: &mut ConservedResidualT<f32>,
    primitives: &PrimitiveFieldsT<f32>,
    topo: &crate::exec::gpu::cuda::ExecInviscidBoundaryTopology,
    topo_key: usize,
    boundary_ghosts: &[crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost],
    params: CudaFirstOrderInviscidParams,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_assemble_first_order_inviscid_boundary(
                residual,
                primitives,
                topo,
                topo_key,
                boundary_ghosts,
                params,
            )?;
            return Ok(true);
        }
    }
    let _ = (
        exec,
        residual,
        primitives,
        topo,
        topo_key,
        boundary_ghosts,
        params,
    );
    Ok(false)
}
