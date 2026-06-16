//! LU-SGS 扫掠 CSR 拓扑 device 缓存。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use crate::discretization::unstructured_lusgs_sweep_exec_topo::LuSgsSweepHostTopology;
use crate::error::{AsimuError, Result};
use crate::exec::gpu::cuda::transfer::{clone_htod, memcpy_htod};

pub struct CudaLusgsSweepMeshDeviceCache {
    cell_offsets: CudaSlice<u32>,
    neighbors: CudaSlice<u32>,
    areas: CudaSlice<f32>,
    normals: CudaSlice<f32>,
    volumes: CudaSlice<f32>,
}

impl CudaLusgsSweepMeshDeviceCache {
    pub fn try_upload(stream: &Arc<CudaStream>, topo: &LuSgsSweepHostTopology) -> Result<Self> {
        Ok(Self {
            cell_offsets: clone_htod(stream, "lusgs_sweep_cell_offsets", &topo.cell_offsets)?,
            neighbors: clone_htod(stream, "lusgs_sweep_neighbors", &topo.neighbors)?,
            areas: clone_htod(stream, "lusgs_sweep_areas", &topo.areas)?,
            normals: clone_htod(stream, "lusgs_sweep_normals", &topo.normals)?,
            volumes: clone_htod(stream, "lusgs_sweep_volumes", &topo.volumes)?,
        })
    }

    pub fn cell_offsets(&self) -> &CudaSlice<u32> {
        &self.cell_offsets
    }

    pub fn neighbors(&self) -> &CudaSlice<u32> {
        &self.neighbors
    }

    pub fn areas(&self) -> &CudaSlice<f32> {
        &self.areas
    }

    pub fn normals(&self) -> &CudaSlice<f32> {
        &self.normals
    }

    pub fn volumes(&self) -> &CudaSlice<f32> {
        &self.volumes
    }
}

/// 上传 u0 至 device 专用缓冲（与当前 cons 分离）。
pub fn upload_u0_snapshot(
    stream: &Arc<CudaStream>,
    u0: &crate::field::ConservedFieldsT<f32>,
    u0_rho: &mut CudaSlice<f32>,
    u0_mx: &mut CudaSlice<f32>,
    u0_my: &mut CudaSlice<f32>,
    u0_mz: &mut CudaSlice<f32>,
    u0_e: &mut CudaSlice<f32>,
) -> Result<()> {
    let n = u0.num_cells();
    ensure_u0_buffers(stream, n, u0_rho, u0_mx, u0_my, u0_mz, u0_e)?;
    memcpy_htod(stream, "lusgs_sweep_u0_rho", u0.density.values(), u0_rho)?;
    memcpy_htod(stream, "lusgs_sweep_u0_mx", u0.momentum_x.values(), u0_mx)?;
    memcpy_htod(stream, "lusgs_sweep_u0_my", u0.momentum_y.values(), u0_my)?;
    memcpy_htod(stream, "lusgs_sweep_u0_mz", u0.momentum_z.values(), u0_mz)?;
    memcpy_htod(stream, "lusgs_sweep_u0_e", u0.total_energy.values(), u0_e)?;
    Ok(())
}

fn ensure_u0_buffers(
    stream: &Arc<CudaStream>,
    n: usize,
    u0_rho: &mut CudaSlice<f32>,
    u0_mx: &mut CudaSlice<f32>,
    u0_my: &mut CudaSlice<f32>,
    u0_mz: &mut CudaSlice<f32>,
    u0_e: &mut CudaSlice<f32>,
) -> Result<()> {
    if u0_rho.len() != n {
        *u0_rho = stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| AsimuError::Exec(format!("lusgs_sweep u0_rho 分配失败: {e:?}")))?;
        *u0_mx = stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| AsimuError::Exec(format!("lusgs_sweep u0_mx 分配失败: {e:?}")))?;
        *u0_my = stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| AsimuError::Exec(format!("lusgs_sweep u0_my 分配失败: {e:?}")))?;
        *u0_mz = stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| AsimuError::Exec(format!("lusgs_sweep u0_mz 分配失败: {e:?}")))?;
        *u0_e = stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| AsimuError::Exec(format!("lusgs_sweep u0_e 分配失败: {e:?}")))?;
    }
    Ok(())
}
