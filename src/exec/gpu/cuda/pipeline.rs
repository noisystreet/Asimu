//! CUDA 步内管线状态（P1：减少 H2D/D2H 乒乓）。

/// device 侧数据就绪标志；步初由 `reset_step` 清零。
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CudaPipelineState {
    /// 无粘/粘性残差在 `CudaFieldBuffers` 上有效（尚未 D2H）。
    pub residual_on_device: bool,
    /// IDWLS 梯度在 `CudaGradientBuffers` 上有效。
    pub gradients_on_device: bool,
    /// 谱半径 \(\sigma_i\) 与 `cell_dts` 在 spectral device 缓冲上有效。
    pub timestep_on_device: bool,
    /// RHS 粘性链：边界面先 CPU scatter，内面/粘性在 device 累加。
    pub rhs_pipeline_active: bool,
    /// 面 \(\mu,\lambda\) 已在 device `CudaViscousFaceGeomBuffer` 上刷新（跳过 H2D refresh）。
    pub viscous_transport_on_device: bool,
    /// 本步 prepare 已刷新 BC/原变量并 H2D；RHS 可跳过重复 refresh。
    pub host_bc_primitives_synced: bool,
    /// 边界面 ghost 已上传至 IDWLS/无粘/粘性 boundary device 缓冲。
    pub boundary_ghosts_on_device: bool,
    /// 单元静温已在 device（与 IDWLS `temperature` 缓冲一致）。
    pub cell_temps_on_device: bool,
    /// 守恒场在 `CudaFieldBuffers` 上有效（与最近一次 upload / 对角更新一致）。
    pub conserved_on_device: bool,
    /// 谱半径粘性扩散系数已在 spectral device 缓冲上（跳过 H2D）。
    pub spectral_diffusivity_on_device: bool,
    /// LU-SGS 对角更新已在 device 上写回守恒场（P4）。
    pub lusgs_diagonal_on_device: bool,
    /// LU-SGS 双扫已在 device 上写回守恒场。
    pub lusgs_sweep_on_device: bool,
    /// 双时间步 \(U^n\) 快照在 `cons_u_n_*` 上有效（P3b）。
    pub u_n_on_device: bool,
}

impl CudaPipelineState {
    pub(crate) fn reset_step(&mut self) {
        *self = Self::default();
    }

    /// RHS 管线步初：保留谱半径 timestep / 守恒场 / 原变量 device 驻留（LU-SGS 步内）。
    pub(crate) fn reset_rhs_step(&mut self) {
        self.residual_on_device = false;
        self.gradients_on_device = false;
        self.viscous_transport_on_device = false;
    }

    /// 步间重置：保留守恒场 device 驻留；BC/积分后原变量与边界面数据失效。
    pub(crate) fn reset_between_timesteps(&mut self) {
        self.reset_rhs_step();
        self.timestep_on_device = false;
        self.lusgs_diagonal_on_device = false;
        self.lusgs_sweep_on_device = false;
        self.u_n_on_device = false;
        self.boundary_ghosts_on_device = false;
        self.cell_temps_on_device = false;
        self.spectral_diffusivity_on_device = false;
        // 保留 conserved_on_device；间隔输出前再 D2H。
    }
}

#[cfg(test)]
mod tests {
    use super::CudaPipelineState;

    #[test]
    fn reset_between_timesteps_preserves_conserved_on_device() {
        let mut p = CudaPipelineState {
            conserved_on_device: true,
            host_bc_primitives_synced: true,
            residual_on_device: true,
            gradients_on_device: true,
            timestep_on_device: true,
            ..CudaPipelineState::default()
        };
        p.reset_between_timesteps();
        assert!(p.conserved_on_device);
        assert!(p.host_bc_primitives_synced);
        assert!(!p.residual_on_device);
        assert!(!p.timestep_on_device);
        assert!(!p.boundary_ghosts_on_device);
    }

    #[test]
    fn reset_rhs_step_preserves_u_n_on_device() {
        let mut p = CudaPipelineState {
            u_n_on_device: true,
            conserved_on_device: true,
            residual_on_device: true,
            ..CudaPipelineState::default()
        };
        p.reset_rhs_step();
        assert!(p.u_n_on_device);
        assert!(p.conserved_on_device);
        assert!(!p.residual_on_device);
    }
}
