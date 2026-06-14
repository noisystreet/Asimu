//! 非结构谱半径 CUDA 分发。

use crate::core::ExecDevice;
use crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost;
use crate::error::Result;
use crate::exec::context::ExecutionContext;
use crate::exec::gpu::cuda::ExecSpectralRadiusTopology;
use crate::field::PrimitiveFieldsT;

/// 谱半径 CUDA 步内输入（原变量、静态拓扑、边界面 ghost、可选粘性扩散系数）。
pub struct SpectralRadiusCudaInput<'a> {
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub topo: &'a ExecSpectralRadiusTopology,
    pub topo_key: usize,
    pub gamma: f32,
    pub boundary_ghosts: &'a [SpectralGhostPrimHost],
    pub diffusivity: Option<&'a [f32]>,
    pub cfl: f32,
    pub fixed_dt: Option<f32>,
    /// `true` 时 \(\sigma\)/`cell_dts` 留在 device，由 `download_timestep_f32` 批量 D2H。
    pub defer_timestep_d2h: bool,
}

/// 批量 D2H 谱半径时间步缓冲（\(\sigma_i\) + `cell_dts`）。
pub fn download_timestep_f32(
    exec: &mut ExecutionContext,
    sigma_out: &mut [f32],
    cell_dts_out: &mut [f32],
    local_time_step: bool,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_download_timestep_f32(sigma_out, cell_dts_out, local_time_step)?;
            return Ok(true);
        }
    }
    let _ = (exec, sigma_out, cell_dts_out, local_time_step);
    Ok(false)
}

/// CUDA 路径计算非结构单元谱半径；成功返回 `Ok(true)`。
pub fn try_compute_spectral_radius_unstructured_f32(
    exec: &mut ExecutionContext,
    input: &SpectralRadiusCudaInput<'_>,
    sigma_out: &mut [f32],
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == ExecDevice::GpuCuda {
            exec.cuda_compute_spectral_radius_unstructured_f32(input, sigma_out)?;
            return Ok(true);
        }
    }
    let _ = (exec, input, sigma_out);
    Ok(false)
}
