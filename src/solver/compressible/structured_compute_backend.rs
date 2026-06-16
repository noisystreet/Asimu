//! 结构化可压缩求解热路径精度后端聚合（ADR 0019 S0）。
//!
//! 随 S1–S4 里程碑扩展子 trait（谱半径、粘性、LU-SGS 扫掠等），case / 驱动边界仅写
//! `T: StructuredComputeBackend`。

use crate::core::ComputeFloat;
use crate::discretization::InviscidFaceFluxTyped;
use crate::discretization::compressible::residual::StructuredInviscidAssembly3dTyped;
use crate::field::{LusgsDiagonalUpdateBackend, PrimitiveFillFromConserved};
use crate::solver::compressible::spectral_radius_3d_f32::StructuredSpectralRadiusTyped;
use crate::solver::compressible::structured_timestep_buffers::{
    StructuredExplicitTimeAdvance, StructuredLusgsDiagonalUpdate, StructuredSpectralTimestepPrepare,
};

/// 结构化 3D 可压缩 typed 热路径所需精度后端（密封于 `f32` / `f64`）。
pub(crate) trait StructuredComputeBackend:
    ComputeFloat
    + LusgsDiagonalUpdateBackend
    + InviscidFaceFluxTyped
    + PrimitiveFillFromConserved
    + StructuredInviscidAssembly3dTyped
    + StructuredSpectralRadiusTyped
    + StructuredSpectralTimestepPrepare
    + StructuredExplicitTimeAdvance
    + StructuredLusgsDiagonalUpdate
{
}

impl StructuredComputeBackend for f32 {}
impl StructuredComputeBackend for f64 {}
