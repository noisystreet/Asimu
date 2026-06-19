//! 非结构粘性内面装配入口（CPU / CUDA 分发；ADR 0017 G2）。

use crate::core::ExecDevice;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::error::Result;
use crate::exec::context::ExecutionContext;
use crate::exec::gpu::cuda::ExecViscousInteriorTopology;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

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

/// CUDA：在 device 上计算内面 \(\mu,\lambda\)（对齐 CPU `prepare_unstructured_viscous_transport_f32`）。
pub fn try_prepare_unstructured_viscous_transport_f32_cuda(
    exec: &mut ExecutionContext,
    topo: &ExecViscousInteriorTopology,
    topo_key: usize,
    num_cells: usize,
    temperatures: &[f32],
    viscous: &ViscousPhysicsConfig,
    eos: &IdealGasEoS,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            if temperatures.is_empty() && exec.cuda_rhs_pipeline_active() {
                exec.cuda_ensure_cell_temperatures_from_device_primitives(num_cells, eos, viscous)?;
            }
            exec.cuda_prepare_viscous_face_transport_f32(
                topo,
                topo_key,
                temperatures,
                viscous,
                eos,
            )?;
            return Ok(true);
        }
    }
    let _ = (exec, topo, topo_key, num_cells, temperatures, viscous, eos);
    Ok(false)
}

/// 若当前 `ExecutionContext` 为 CUDA，在 device 上装配粘性边界面残差并返回 `Ok(true)`。
pub fn try_assemble_viscous_boundary_f32(
    exec: &mut ExecutionContext,
    residual: &mut ConservedResidualT<f32>,
    primitives: &PrimitiveFieldsT<f32>,
    gradients: &GradientFieldsT<f32>,
    input: crate::exec::gpu::cuda::CudaViscousBoundaryInput<'_>,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_assemble_viscous_boundary_f32(residual, primitives, gradients, input)?;
            return Ok(true);
        }
    }
    let _ = (exec, residual, primitives, gradients, input);
    Ok(false)
}
