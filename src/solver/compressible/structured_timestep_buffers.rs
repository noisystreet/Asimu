//! 结构化 typed 时间步缓冲（f64 / f32 热路径分离；ADR 0019 S1-c）。

use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::ConservedFieldsT;
use crate::solver::compressible::spectral_radius_3d_f32::StructuredSpectralRadiusTyped;
use crate::solver::compressible::{CompressibleAdvanceContext3dTyped, CompressibleEulerSolver};

/// 谱半径与单元 \(\Delta t_i\) 缓冲（显式 LTS / LU-SGS 对角复用）。
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct StructuredTimestepBuffers {
    pub sigma: Vec<Real>,
    pub cell_dts: Vec<Real>,
    pub sigma_f32: Vec<f32>,
    pub cell_dts_f32: Vec<f32>,
}

/// 结构化 typed 谱半径与时间步准备（f32 原生缓冲；GMRES 仍读 Real 镜像）。
pub(crate) trait StructuredSpectralTimestepPrepare:
    ComputeFloat + StructuredSpectralRadiusTyped
{
    fn prepare_spectral_timestep_3d(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, Self>,
        fields: &mut ConservedFieldsT<Self>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)>;

    fn prepare_lusgs_timestep_3d(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, Self>,
        fields: &mut ConservedFieldsT<Self>,
        cfl: Real,
        p_floor: Real,
    ) -> Result<(Vec<Real>, Vec<Real>)>;
}

/// 结构化 typed 显式 RK4/Euler 推进（f32 LTS 原生 \(\Delta t_i\)）。
pub(crate) trait StructuredExplicitTimeAdvance: StructuredSpectralTimestepPrepare {
    #[allow(clippy::too_many_arguments)]
    fn advance_structured_explicit(
        solver: &CompressibleEulerSolver,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, Self>,
        fields: &mut ConservedFieldsT<Self>,
        storage: &mut crate::solver::time::Rk4StorageT<Self>,
        dt_global: Real,
        local_time_step: bool,
        p_floor: Real,
        eos: &crate::physics::IdealGasEoS,
    ) -> Result<()>;
}

/// 结构化 typed LU-SGS 非扫掠对角更新（f32 用原生 \(\sigma,\Delta t_i\) 缓冲）。
pub(crate) trait StructuredLusgsDiagonalUpdate: StructuredSpectralTimestepPrepare {
    fn apply_structured_lusgs_diagonal_update(
        out: &mut ConservedFieldsT<Self>,
        base: &ConservedFieldsT<Self>,
        residual: &crate::field::ConservedResidualT<Self>,
        ctx: &CompressibleAdvanceContext3dTyped<'_, Self>,
        omega: Real,
        gamma: Real,
        min_pressure: Real,
    ) -> Result<()>;
}
