//! 守恒场 → 原变量 / 粘性扩散系数 CUDA launch。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use tracing::info_span;

use super::boundary_face_geom::{BoundaryConservedGhostHost, ViscousBoundaryGhostHost};
use super::spectral_radius_topology::DeviceSpectralGhostPrim;
use super::viscous_transport_params::DeviceViscousTransportParams;
use crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost;
use crate::error::{AsimuError, Result};

const BLOCK_THREADS: u32 = 256;

/// device 守恒场切片（`CudaFieldBuffers` 内五分量）。
pub struct FieldConservedSlices<'a> {
    pub rho: &'a CudaSlice<f32>,
    pub mx: &'a CudaSlice<f32>,
    pub my: &'a CudaSlice<f32>,
    pub mz: &'a CudaSlice<f32>,
    pub e: &'a CudaSlice<f32>,
}

/// device 原变量切片。
pub struct FieldPrimitiveSlices<'a> {
    pub rho: &'a CudaSlice<f32>,
    pub p: &'a CudaSlice<f32>,
    pub ux: &'a CudaSlice<f32>,
    pub uy: &'a CudaSlice<f32>,
    pub uz: &'a CudaSlice<f32>,
}

/// 单元粘性扩散系数 kernel 参数。
pub struct ViscousDiffusivityLaunchArgs<'a> {
    pub num_cells: u32,
    pub gamma: f32,
    pub gas_r: f32,
    pub nondim_flag: f32,
    pub transport: DeviceViscousTransportParams,
    pub prim_rho: &'a CudaSlice<f32>,
    pub prim_p: &'a CudaSlice<f32>,
    pub diffusivity_out: &'a mut CudaSlice<f32>,
}

/// 单元静温 kernel 参数。
pub struct CellStaticTemperatureLaunchArgs<'a> {
    pub num_cells: u32,
    pub gamma: f32,
    pub gas_r: f32,
    pub nondim_flag: f32,
    pub prim_rho: &'a CudaSlice<f32>,
    pub prim_p: &'a CudaSlice<f32>,
    pub temp_out: &'a mut CudaSlice<f32>,
}

/// 守恒 ghost → 各套边界面缓冲 kernel 参数。
pub struct BoundaryGhostBuffersFromConservedLaunch<'a> {
    pub num_faces: u32,
    pub gamma: f32,
    pub min_pressure: f32,
    pub gas_r: f32,
    pub nondim_flag: f32,
    pub cons_in: &'a CudaSlice<BoundaryConservedGhostHost>,
    pub idwls_out: &'a mut CudaSlice<IdwlsGhostSampleHost>,
    pub inviscid_out: &'a mut CudaSlice<DeviceSpectralGhostPrim>,
    pub spectral_out: &'a mut CudaSlice<DeviceSpectralGhostPrim>,
    pub viscous_out: &'a mut CudaSlice<ViscousBoundaryGhostHost>,
}

pub fn launch_fill_primitives_from_conserved(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    num_cells: u32,
    gamma: f32,
    min_pressure: f32,
    conserved: &FieldConservedSlices<'_>,
    primitives: &FieldPrimitiveSlices<'_>,
) -> Result<()> {
    let _span = info_span!("cuda_fill_primitives_from_conserved", cells = num_cells).entered();
    let num_blocks = num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&num_cells);
    builder.arg(&gamma);
    builder.arg(&min_pressure);
    builder.arg(conserved.rho);
    builder.arg(conserved.mx);
    builder.arg(conserved.my);
    builder.arg(conserved.mz);
    builder.arg(conserved.e);
    builder.arg(primitives.rho);
    builder.arg(primitives.p);
    builder.arg(primitives.ux);
    builder.arg(primitives.uy);
    builder.arg(primitives.uz);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA fill_primitives kernel launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}

pub fn launch_fill_boundary_ghost_buffers_from_conserved(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    args: BoundaryGhostBuffersFromConservedLaunch<'_>,
) -> Result<()> {
    let _span = info_span!(
        "cuda_fill_boundary_ghost_buffers_from_conserved",
        faces = args.num_faces
    )
    .entered();
    let num_blocks = args.num_faces.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&args.num_faces);
    builder.arg(&args.gamma);
    builder.arg(&args.min_pressure);
    builder.arg(&args.gas_r);
    builder.arg(&args.nondim_flag);
    builder.arg(args.cons_in);
    builder.arg(args.idwls_out);
    builder.arg(args.inviscid_out);
    builder.arg(args.spectral_out);
    builder.arg(args.viscous_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!(
                "CUDA fill_boundary_ghost_buffers kernel launch 失败: {e:?}"
            ))
        })?;
    }
    Ok(())
}

pub fn launch_cell_static_temperature_f32(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    args: CellStaticTemperatureLaunchArgs<'_>,
) -> Result<()> {
    let _span = info_span!("cuda_cell_static_temperature", cells = args.num_cells).entered();
    let num_blocks = args.num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&args.num_cells);
    builder.arg(&args.gamma);
    builder.arg(&args.gas_r);
    builder.arg(&args.nondim_flag);
    builder.arg(args.prim_rho);
    builder.arg(args.prim_p);
    builder.arg(args.temp_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!(
                "CUDA cell_static_temperature kernel launch 失败: {e:?}"
            ))
        })?;
    }
    Ok(())
}

pub fn launch_cell_viscous_diffusivity_max(
    stream: &Arc<CudaStream>,
    function: &cudarc::driver::CudaFunction,
    args: ViscousDiffusivityLaunchArgs<'_>,
) -> Result<()> {
    let _span = info_span!("cuda_cell_viscous_diffusivity_max", cells = args.num_cells).entered();
    let num_blocks = args.num_cells.div_ceil(BLOCK_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(&args.num_cells);
    builder.arg(&args.gamma);
    builder.arg(&args.gas_r);
    builder.arg(&args.nondim_flag);
    builder.arg(&args.transport);
    builder.arg(args.prim_rho);
    builder.arg(args.prim_p);
    builder.arg(args.diffusivity_out);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            AsimuError::Exec(format!("CUDA diffusivity_max kernel launch 失败: {e:?}"))
        })?;
    }
    Ok(())
}
