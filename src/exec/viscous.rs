//! 非结构粘性内面装配入口（CPU / CUDA 分发；ADR 0017 G2）。

use crate::core::ExecDevice;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::error::Result;
use crate::exec::context::ExecutionContext;
use crate::exec::gpu::cuda::ExecViscousInteriorTopology;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

/// 若当前 `ExecutionContext` 为 CUDA，在 device 上累加粘性内面动量/能量残差并返回 `Ok(true)`。
pub fn try_assemble_viscous_interior_f32(
    exec: &mut ExecutionContext,
    residual: &mut ConservedResidualT<f32>,
    primitives: &PrimitiveFieldsT<f32>,
    gradients: &GradientFieldsT<f32>,
    topo: &ExecViscousInteriorTopology,
    topo_key: usize,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_assemble_viscous_interior(residual, primitives, gradients, topo, topo_key)?;
            return Ok(true);
        }
    }
    let _ = (exec, residual, primitives, gradients, topo, topo_key);
    Ok(false)
}
