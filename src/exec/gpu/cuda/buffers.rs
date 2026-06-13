//! 设备侧场缓冲（原始变量 SoA + 残差 SoA）。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};

use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

/// 与 CUDA kernel `FaceGeom` 一致的设备布局。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceFaceGeom {
    pub owner: u32,
    pub neighbor: u32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub owner_scale: f32,
    pub neighbor_scale: f32,
}

unsafe impl DeviceRepr for DeviceFaceGeom {}

/// 步间常驻：原始变量 + 残差 device 缓冲。
pub struct CudaFieldBuffers {
    pub(crate) prim_rho: CudaSlice<f32>,
    pub(crate) prim_p: CudaSlice<f32>,
    pub(crate) prim_ux: CudaSlice<f32>,
    pub(crate) prim_uy: CudaSlice<f32>,
    pub(crate) prim_uz: CudaSlice<f32>,
    pub(crate) res_rho: CudaSlice<f32>,
    pub(crate) res_mx: CudaSlice<f32>,
    pub(crate) res_my: CudaSlice<f32>,
    pub(crate) res_mz: CudaSlice<f32>,
    pub(crate) res_e: CudaSlice<f32>,
    pub(crate) num_cells: usize,
}

impl CudaFieldBuffers {
    #[must_use]
    pub(crate) fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub fn try_new(stream: &Arc<CudaStream>, num_cells: usize) -> Result<Self> {
        if num_cells == 0 {
            return Err(AsimuError::Field(
                "CUDA 场缓冲需要 num_cells > 0".to_string(),
            ));
        }
        let alloc = |n: usize| -> Result<CudaSlice<f32>> {
            stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| AsimuError::Exec(format!("CUDA 分配失败: {e:?}")))
        };
        Ok(Self {
            prim_rho: alloc(num_cells)?,
            prim_p: alloc(num_cells)?,
            prim_ux: alloc(num_cells)?,
            prim_uy: alloc(num_cells)?,
            prim_uz: alloc(num_cells)?,
            res_rho: alloc(num_cells)?,
            res_mx: alloc(num_cells)?,
            res_my: alloc(num_cells)?,
            res_mz: alloc(num_cells)?,
            res_e: alloc(num_cells)?,
            num_cells,
        })
    }

    pub fn upload_primitives(
        &mut self,
        stream: &Arc<CudaStream>,
        primitives: &PrimitiveFieldsT<f32>,
    ) -> Result<()> {
        let n = primitives.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "primitive 长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        htod(stream, &mut self.prim_rho, primitives.density.values())?;
        htod(stream, &mut self.prim_p, primitives.pressure.values())?;
        htod(stream, &mut self.prim_ux, primitives.velocity_x.values())?;
        htod(stream, &mut self.prim_uy, primitives.velocity_y.values())?;
        htod(stream, &mut self.prim_uz, primitives.velocity_z.values())?;
        Ok(())
    }

    pub fn zero_residual(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        zero_slice(stream, &mut self.res_rho)?;
        zero_slice(stream, &mut self.res_mx)?;
        zero_slice(stream, &mut self.res_my)?;
        zero_slice(stream, &mut self.res_mz)?;
        zero_slice(stream, &mut self.res_e)?;
        Ok(())
    }

    pub fn download_residual(
        &self,
        stream: &Arc<CudaStream>,
        residual: &mut ConservedResidualT<f32>,
    ) -> Result<()> {
        let n = residual.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "残差长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        dtoh_into(stream, &self.res_rho, residual.density.values_mut())?;
        dtoh_into(stream, &self.res_mx, residual.momentum_x.values_mut())?;
        dtoh_into(stream, &self.res_my, residual.momentum_y.values_mut())?;
        dtoh_into(stream, &self.res_mz, residual.momentum_z.values_mut())?;
        dtoh_into(stream, &self.res_e, residual.total_energy.values_mut())?;
        Ok(())
    }
}

fn htod(stream: &Arc<CudaStream>, dst: &mut CudaSlice<f32>, src: &[f32]) -> Result<()> {
    stream
        .memcpy_htod(src, dst)
        .map_err(|e| AsimuError::Exec(format!("CUDA H2D 失败: {e:?}")))
}

fn dtoh_into(stream: &Arc<CudaStream>, src: &CudaSlice<f32>, dst: &mut [f32]) -> Result<()> {
    let host = stream
        .clone_dtoh(src)
        .map_err(|e| AsimuError::Exec(format!("CUDA D2H 失败: {e:?}")))?;
    dst.copy_from_slice(host.as_slice());
    Ok(())
}

fn zero_slice(stream: &Arc<CudaStream>, buf: &mut CudaSlice<f32>) -> Result<()> {
    stream
        .memset_zeros(buf)
        .map_err(|e| AsimuError::Exec(format!("CUDA memset 失败: {e:?}")))
}
