//! IDWLS 粘性 RHS CUDA 分发（ADR 0017 P4）。

use crate::core::ExecDevice;
use crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost;
use crate::error::Result;
use crate::exec::context::ExecutionContext;
use crate::exec::gpu::cuda::ExecIdwlsViscousTopology;
use crate::field::PrimitiveFieldsT;

/// CUDA 路径累加粘性 IDWLS RHS；成功返回 `Ok(true)`。
pub fn try_accumulate_viscous_rhs_f32_cuda(
    exec: &mut ExecutionContext,
    primitives: &PrimitiveFieldsT<f32>,
    topo: &ExecIdwlsViscousTopology,
    topo_key: usize,
    temperatures: &[f32],
    boundary_ghosts: &[IdwlsGhostSampleHost],
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_accumulate_idwls_viscous_rhs(
                primitives,
                topo,
                topo_key,
                temperatures,
                boundary_ghosts,
            )?;
            return Ok(true);
        }
    }
    let _ = (
        exec,
        primitives,
        topo,
        topo_key,
        temperatures,
        boundary_ghosts,
    );
    Ok(false)
}
