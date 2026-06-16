//! 双时间步 device 存储项（`inviscid` 子模块）。

use super::super::dual_time_storage::launch_dual_time_storage;
use super::CudaBackendState;
use crate::error::{AsimuError, Result};

impl CudaBackendState {
    /// device 叠加 BDF1 物理存储项（须 `u_n` / 残差 / 守恒场均在 device）。
    pub fn add_physical_storage_residual_f32(&mut self, inv_dt_phys: f32) -> Result<()> {
        if !self.pipeline.u_n_on_device {
            return Err(AsimuError::Exec(
                "CUDA 存储项需要 device U^n；请先 snapshot_u_n_on_device".to_string(),
            ));
        }
        if !self.pipeline.residual_on_device {
            return Err(AsimuError::Exec(
                "CUDA 存储项需要 spatial 残差在 device 上".to_string(),
            ));
        }
        if !self.pipeline.conserved_on_device {
            return Err(AsimuError::Exec(
                "CUDA 存储项需要当前守恒场在 device 上".to_string(),
            ));
        }
        if !(inv_dt_phys.is_finite() && inv_dt_phys > 0.0) {
            return Err(AsimuError::Field(
                "dual_time: inv_dt_phys 须为正有限".to_string(),
            ));
        }
        let fields = self.fields.as_ref().expect("field buffers");
        launch_dual_time_storage(
            &self.stream,
            &self.dual_time_module.storage,
            fields,
            inv_dt_phys,
        )?;
        Ok(())
    }
}
