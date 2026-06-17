//! LU-SGS 扫掠 CUDA 参数与 launch。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::buffers::CudaFieldBuffers;
use super::lusgs_sweep_mesh_cache::CudaLusgsSweepMeshDeviceCache;
use crate::discretization::unstructured_lusgs_sweep_exec_topo::LuSgsSweepHostTopology;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};

/// device 扫掠 launch 标量参数。
pub struct LusgsSweepCudaScalars {
    pub omega: f32,
    pub gamma: f32,
    pub min_pressure: f32,
    pub inv_dt_phys: f32,
    pub backward_damping: f32,
}

/// device 扫掠 launch 缓冲引用。
pub struct LusgsSweepCudaLaunchBuffers<'a> {
    pub fields: &'a CudaFieldBuffers,
    pub sweep_mesh: &'a CudaLusgsSweepMeshDeviceCache,
    pub sigma: &'a CudaSlice<f32>,
    pub cell_dts: &'a CudaSlice<f32>,
    pub u0_rho: &'a CudaSlice<f32>,
    pub u0_mx: &'a CudaSlice<f32>,
    pub u0_my: &'a CudaSlice<f32>,
    pub u0_mz: &'a CudaSlice<f32>,
    pub u0_e: &'a CudaSlice<f32>,
}

/// host 侧扫掠 + stabilize 输入。
pub struct LusgsSweepCudaHostInput<'a> {
    pub fields: &'a mut ConservedFieldsT<f32>,
    pub u0: &'a ConservedFieldsT<f32>,
    pub residual: &'a mut ConservedResidualT<f32>,
    pub sweep_topo: &'a LuSgsSweepHostTopology,
    pub topo_key: usize,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub host_sigma: &'a [f32],
    pub host_cell_dts: &'a [f32],
    pub host_volumes: &'a [f32],
    pub local_time_step: bool,
    pub scalars: LusgsSweepCudaScalars,
}

pub fn launch_lusgs_sweep_unstructured_serial(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    buffers: &LusgsSweepCudaLaunchBuffers<'_>,
    scalars: &LusgsSweepCudaScalars,
) -> Result<()> {
    let num_cells = buffers.fields.num_cells() as u32;
    let _span = info_span!(
        "cuda_lusgs_sweep_unstructured",
        cells = num_cells,
        inv_dt_phys = scalars.inv_dt_phys,
    )
    .entered();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&scalars.omega);
    builder.arg(&scalars.gamma);
    builder.arg(&scalars.min_pressure);
    builder.arg(&scalars.inv_dt_phys);
    builder.arg(&scalars.backward_damping);
    builder.arg(buffers.sweep_mesh.cell_offsets());
    builder.arg(buffers.sweep_mesh.neighbors());
    builder.arg(buffers.sweep_mesh.areas());
    builder.arg(buffers.sweep_mesh.normals());
    builder.arg(buffers.sweep_mesh.volumes());
    builder.arg(buffers.sigma);
    builder.arg(buffers.cell_dts);
    builder.arg(&buffers.fields.res_rho);
    builder.arg(&buffers.fields.res_mx);
    builder.arg(&buffers.fields.res_my);
    builder.arg(&buffers.fields.res_mz);
    builder.arg(&buffers.fields.res_e);
    builder.arg(buffers.u0_rho);
    builder.arg(buffers.u0_mx);
    builder.arg(buffers.u0_my);
    builder.arg(buffers.u0_mz);
    builder.arg(buffers.u0_e);
    builder.arg(&buffers.fields.cons_rho);
    builder.arg(&buffers.fields.cons_mx);
    builder.arg(&buffers.fields.cons_my);
    builder.arg(&buffers.fields.cons_mz);
    builder.arg(&buffers.fields.cons_e);
    builder.arg(&buffers.fields.prim_rho);
    builder.arg(&buffers.fields.prim_p);
    builder.arg(&buffers.fields.prim_ux);
    builder.arg(&buffers.fields.prim_uy);
    builder.arg(&buffers.fields.prim_uz);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| AsimuError::Exec(format!("CUDA LU-SGS 扫掠 kernel launch 失败: {e:?}")))?;
    }
    Ok(())
}

const SWEEP_BLOCK_THREADS: u32 = 256;

/// 并行检查 device 守恒场是否全部正性；`any_bad[0]==0` 表示可跳过 host stabilize。
pub fn launch_lusgs_any_nonphysical_conserved(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    fields: &CudaFieldBuffers,
    gamma: f32,
    min_pressure: f32,
    any_bad: &mut CudaSlice<i32>,
) -> Result<()> {
    let num_cells = fields.num_cells() as u32;
    let _span = info_span!("cuda_lusgs_any_nonphysical", cells = num_cells,).entered();
    stream
        .memset_zeros(any_bad)
        .map_err(|e| AsimuError::Exec(format!("CUDA any_nonphysical memset 失败: {e:?}")))?;
    let num_blocks = num_cells.div_ceil(SWEEP_BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (SWEEP_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&gamma);
    builder.arg(&min_pressure);
    builder.arg(&fields.cons_rho);
    builder.arg(&fields.cons_mx);
    builder.arg(&fields.cons_my);
    builder.arg(&fields.cons_mz);
    builder.arg(&fields.cons_e);
    builder.arg(any_bad);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA LU-SGS 正性检查 kernel launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}
