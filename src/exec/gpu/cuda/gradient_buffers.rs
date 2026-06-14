//! 速度/温度梯度 device 缓冲（粘性 G2）。

use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::transfer::{clone_dtoh_unchecked, d2h_batch, h2d_batch, memcpy_htod_unchecked};
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::error::{AsimuError, Result};

/// 粘性装配所需的 12 个梯度 SoA 分量。
pub struct CudaGradientBuffers {
    pub(crate) du_dx: CudaSlice<f32>,
    pub(crate) du_dy: CudaSlice<f32>,
    pub(crate) du_dz: CudaSlice<f32>,
    pub(crate) dv_dx: CudaSlice<f32>,
    pub(crate) dv_dy: CudaSlice<f32>,
    pub(crate) dv_dz: CudaSlice<f32>,
    pub(crate) dw_dx: CudaSlice<f32>,
    pub(crate) dw_dy: CudaSlice<f32>,
    pub(crate) dw_dz: CudaSlice<f32>,
    pub(crate) dt_dx: CudaSlice<f32>,
    pub(crate) dt_dy: CudaSlice<f32>,
    pub(crate) dt_dz: CudaSlice<f32>,
    num_cells: usize,
}

impl CudaGradientBuffers {
    pub fn try_new(stream: &Arc<CudaStream>, num_cells: usize) -> Result<Self> {
        if num_cells == 0 {
            return Err(AsimuError::Field(
                "CUDA 梯度缓冲需要 num_cells > 0".to_string(),
            ));
        }
        let alloc = |n: usize| -> Result<CudaSlice<f32>> {
            stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| AsimuError::Exec(format!("CUDA 梯度分配失败: {e:?}")))
        };
        Ok(Self {
            du_dx: alloc(num_cells)?,
            du_dy: alloc(num_cells)?,
            du_dz: alloc(num_cells)?,
            dv_dx: alloc(num_cells)?,
            dv_dy: alloc(num_cells)?,
            dv_dz: alloc(num_cells)?,
            dw_dx: alloc(num_cells)?,
            dw_dy: alloc(num_cells)?,
            dw_dz: alloc(num_cells)?,
            dt_dx: alloc(num_cells)?,
            dt_dy: alloc(num_cells)?,
            dt_dz: alloc(num_cells)?,
            num_cells,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub fn upload(
        &mut self,
        stream: &Arc<CudaStream>,
        gradients: &GradientFieldsT<f32>,
    ) -> Result<()> {
        let n = gradients.du_dx.len();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "梯度长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        h2d_batch("gradients", n * 12 * size_of::<f32>(), n, || {
            memcpy_htod_unchecked(stream, gradients.du_dx.values(), &mut self.du_dx)?;
            memcpy_htod_unchecked(stream, gradients.du_dy.values(), &mut self.du_dy)?;
            memcpy_htod_unchecked(stream, gradients.du_dz.values(), &mut self.du_dz)?;
            memcpy_htod_unchecked(stream, gradients.dv_dx.values(), &mut self.dv_dx)?;
            memcpy_htod_unchecked(stream, gradients.dv_dy.values(), &mut self.dv_dy)?;
            memcpy_htod_unchecked(stream, gradients.dv_dz.values(), &mut self.dv_dz)?;
            memcpy_htod_unchecked(stream, gradients.dw_dx.values(), &mut self.dw_dx)?;
            memcpy_htod_unchecked(stream, gradients.dw_dy.values(), &mut self.dw_dy)?;
            memcpy_htod_unchecked(stream, gradients.dw_dz.values(), &mut self.dw_dz)?;
            memcpy_htod_unchecked(stream, gradients.dt_dx.values(), &mut self.dt_dx)?;
            memcpy_htod_unchecked(stream, gradients.dt_dy.values(), &mut self.dt_dy)?;
            memcpy_htod_unchecked(stream, gradients.dt_dz.values(), &mut self.dt_dz)?;
            Ok(())
        })
    }

    pub fn download_to_host(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut GradientFieldsT<f32>,
    ) -> Result<()> {
        let n = out.num_cells();
        if n != self.num_cells {
            return Err(AsimuError::Field(format!(
                "梯度场长度 {n} 与 device 缓冲 {} 不一致",
                self.num_cells
            )));
        }
        let bytes = n
            .checked_mul(12)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| AsimuError::Field("梯度 D2H 字节数溢出".to_string()))?;
        d2h_batch("gradients_d2h", bytes, n, || {
            copy_gradient_components_d2h(stream, self, out)
        })
    }
}

fn copy_gradient_components_d2h(
    stream: &Arc<CudaStream>,
    src: &CudaGradientBuffers,
    out: &mut GradientFieldsT<f32>,
) -> Result<()> {
    dtoh_into(stream, &src.du_dx, out.du_dx.values_mut())?;
    dtoh_into(stream, &src.du_dy, out.du_dy.values_mut())?;
    dtoh_into(stream, &src.du_dz, out.du_dz.values_mut())?;
    dtoh_into(stream, &src.dv_dx, out.dv_dx.values_mut())?;
    dtoh_into(stream, &src.dv_dy, out.dv_dy.values_mut())?;
    dtoh_into(stream, &src.dv_dz, out.dv_dz.values_mut())?;
    dtoh_into(stream, &src.dw_dx, out.dw_dx.values_mut())?;
    dtoh_into(stream, &src.dw_dy, out.dw_dy.values_mut())?;
    dtoh_into(stream, &src.dw_dz, out.dw_dz.values_mut())?;
    dtoh_into(stream, &src.dt_dx, out.dt_dx.values_mut())?;
    dtoh_into(stream, &src.dt_dy, out.dt_dy.values_mut())?;
    dtoh_into(stream, &src.dt_dz, out.dt_dz.values_mut())?;
    Ok(())
}

fn dtoh_into(stream: &Arc<CudaStream>, src: &CudaSlice<f32>, dst: &mut [f32]) -> Result<()> {
    let flat = clone_dtoh_unchecked(stream, src)?;
    dst.copy_from_slice(flat.as_slice());
    Ok(())
}
