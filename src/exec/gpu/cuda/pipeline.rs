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
}

impl CudaPipelineState {
    pub(crate) fn reset_step(&mut self) {
        *self = Self::default();
    }

    /// RHS 管线步初：保留谱半径 timestep device 驻留状态。
    pub(crate) fn reset_rhs_step(&mut self) {
        self.residual_on_device = false;
        self.gradients_on_device = false;
        self.viscous_transport_on_device = false;
    }
}
