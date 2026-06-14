//! 速度/温度梯度 device 缓冲（粘性 G2）。

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

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
        htod(stream, &mut self.du_dx, gradients.du_dx.values())?;
        htod(stream, &mut self.du_dy, gradients.du_dy.values())?;
        htod(stream, &mut self.du_dz, gradients.du_dz.values())?;
        htod(stream, &mut self.dv_dx, gradients.dv_dx.values())?;
        htod(stream, &mut self.dv_dy, gradients.dv_dy.values())?;
        htod(stream, &mut self.dv_dz, gradients.dv_dz.values())?;
        htod(stream, &mut self.dw_dx, gradients.dw_dx.values())?;
        htod(stream, &mut self.dw_dy, gradients.dw_dy.values())?;
        htod(stream, &mut self.dw_dz, gradients.dw_dz.values())?;
        htod(stream, &mut self.dt_dx, gradients.dt_dx.values())?;
        htod(stream, &mut self.dt_dy, gradients.dt_dy.values())?;
        htod(stream, &mut self.dt_dz, gradients.dt_dz.values())?;
        Ok(())
    }
}

fn htod(stream: &Arc<CudaStream>, dst: &mut CudaSlice<f32>, src: &[f32]) -> Result<()> {
    stream
        .memcpy_htod(src, dst)
        .map_err(|e| AsimuError::Exec(format!("CUDA 梯度 H2D 失败: {e:?}")))
}
