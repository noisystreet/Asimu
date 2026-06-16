//! LU-SGS 对角更新与 device timestep 下载（`inviscid` 子模块，可访问私有字段）。

use super::super::lusgs_diagonal::{launch_lusgs_diagonal_update, launch_residual_density_sum_sq};
use super::CudaBackendState;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};

impl CudaBackendState {
    pub fn download_timestep_f32(
        &mut self,
        sigma_out: &mut [f32],
        cell_dts_out: &mut [f32],
        local_time_step: bool,
    ) -> Result<()> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA timestep 未在 device 上；请先调用谱半径 CUDA 路径".to_string(),
            ));
        }
        let mesh = self.spectral_mesh.as_ref().expect("spectral mesh");
        mesh.download_timestep(&self.stream, sigma_out, cell_dts_out)?;
        if !local_time_step {
            let min_dt = cell_dts_out
                .iter()
                .copied()
                .filter(|d| d.is_finite() && *d > 0.0)
                .fold(f32::INFINITY, f32::min);
            if min_dt.is_finite() {
                cell_dts_out.fill(min_dt);
            }
        }
        self.pipeline.timestep_on_device = false;
        Ok(())
    }

    pub fn download_min_cell_dt_f32(&mut self) -> Result<f32> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA timestep 未在 device 上；请先调用谱半径 CUDA 路径".to_string(),
            ));
        }
        let mesh = self.spectral_mesh.as_mut().expect("spectral mesh");
        mesh.download_min_cell_dt(&self.stream, &self.spectral_module)
    }

    /// LU-SGS 对角更新：守恒场已在 device 时跳过 H2D/D2H（P4）。
    pub fn lusgs_diagonal_update_f32(
        &mut self,
        base: &ConservedFieldsT<f32>,
        residual: &ConservedResidualT<f32>,
        omega: f32,
        inv_dt_phys: f32,
    ) -> Result<()> {
        if !self.pipeline.timestep_on_device {
            return Err(AsimuError::Exec(
                "CUDA LU-SGS 对角更新需要 device 上 σ/Δt_i".to_string(),
            ));
        }
        self.ensure_fields(base.num_cells())?;
        let fields = self.fields.as_mut().expect("field buffers after ensure");
        if !self.pipeline.conserved_on_device {
            fields.upload_conserved(&self.stream, base)?;
            self.pipeline.conserved_on_device = true;
        }
        if !self.pipeline.residual_on_device {
            fields.upload_full_residual(&self.stream, residual)?;
        }
        let mesh = self.spectral_mesh.as_ref().expect("spectral mesh");
        launch_lusgs_diagonal_update(
            &self.stream,
            &self.lusgs_module.diagonal_update,
            fields,
            mesh.sigma(),
            mesh.cell_dts(),
            omega,
            inv_dt_phys,
        )?;
        self.pipeline.residual_on_device = true;
        self.pipeline.timestep_on_device = false;
        self.pipeline.conserved_on_device = true;
        self.pipeline.lusgs_diagonal_on_device = true;
        Ok(())
    }

    /// device 密度残差 RMS（单 float D2H；替代全量残差 D2H）。
    pub fn density_residual_rms_f32(&mut self) -> Result<f32> {
        if !self.pipeline.residual_on_device {
            return Err(AsimuError::Exec(
                "CUDA 密度残差 RMS 需要 residual 在 device 上".to_string(),
            ));
        }
        let fields = self.fields.as_ref().expect("field buffers");
        let n = fields.num_cells();
        if n == 0 {
            return Ok(0.0);
        }
        if self
            .residual_sum_sq_scratch
            .as_ref()
            .is_none_or(|s| s.len() != 1)
        {
            self.residual_sum_sq_scratch = Some(
                self.stream
                    .alloc_zeros::<f32>(1)
                    .map_err(|e| AsimuError::Exec(format!("CUDA sum_sq 分配失败: {e:?}")))?,
            );
        }
        let sum_buf = self
            .residual_sum_sq_scratch
            .as_mut()
            .expect("sum_sq scratch after ensure");
        launch_residual_density_sum_sq(
            &self.stream,
            &self.lusgs_module.residual_density_sum_sq,
            &fields.res_rho,
            n as u32,
            sum_buf,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| AsimuError::Exec(format!("CUDA 同步失败: {e:?}")))?;
        let sum_sq =
            super::super::transfer::clone_dtoh(&self.stream, "residual_density_sum_sq", sum_buf)?;
        let rms = (sum_sq[0] / n as f32).sqrt();
        Ok(rms)
    }
}
