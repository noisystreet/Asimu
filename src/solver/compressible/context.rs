//! 可压缩 1D/3D 单步推进上下文。

use std::{cell::RefCell, rc::Rc};

use crate::boundary::BoundarySet;
use crate::discretization::{BoundaryGhostBuffer, GradientFields, StructuredFaceCacheF32};
use crate::error::Result;
use crate::field::{ConservedResidual, PrimitiveFields, PrimitiveFieldsT};
use crate::mesh::{BoundaryMesh3d, StructuredMesh1d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig};
use crate::solver::compressible::structured_timestep_buffers::StructuredTimestepBuffers;

pub trait ResidualCorrection3d {
    fn apply(&mut self, residual: &mut ConservedResidual) -> Result<()>;
}

pub type ResidualCorrection3dHandle = Rc<RefCell<dyn ResidualCorrection3d>>;

/// 3D 单步推进上下文（减少参数个数）。
pub struct CompressibleAdvanceContext3d<'a> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    /// 每步 RHS 复用的原始变量缓冲（避免每 `evaluate_rhs` 重新分配）。
    pub primitive_scratch: PrimitiveFields,
    /// 粘性梯度缓冲（仅 NS 算例使用）。
    pub gradient_scratch: GradientFields,
    /// NS 物性（谱半径 / CFL 粘性扩散项；与 `CompressibleEulerConfig::viscous` 一致）。
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    /// RHS 后处理修正（多块共享接口通量等）；单块路径保持 `None`。
    pub residual_correction: Option<ResidualCorrection3dHandle>,
}

/// typed 3D 单步推进上下文（P2：结构化显式 rk4/euler + 一阶无粘）。
pub struct CompressibleAdvanceContext3dTyped<'a, T: crate::core::ComputeFloat> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    pub primitive_scratch: PrimitiveFieldsT<T>,
    /// 谱半径 / BC 刷新用的 f64 原始变量缓冲。
    pub spectral_primitives: PrimitiveFields,
    pub gradient_scratch: GradientFields,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    /// 多块 1-to-1 共享接口残差修正（单块路径保持 `None`）。
    pub(crate) interface_residual: Option<
        &'a [crate::solver::compressible::multiblock_interface::InterfaceResidualContribution],
    >,
    /// f32 结构化内面几何预打包（S1-a；`T=f32` 时由驱动构建）。
    pub face_cache_f32: Option<&'a StructuredFaceCacheF32>,
    /// 单元体积 f32（S1-c；`T=f32` 时预计算）。
    pub volumes_f32: &'a [f32],
    /// 谱半径 / 局部 \(\Delta t_i\) 缓冲（S1-c）。
    pub(crate) timestep: &'a mut StructuredTimestepBuffers,
}

impl<'a, T: crate::core::ComputeFloat> CompressibleAdvanceContext3dTyped<'a, T> {
    /// 构造 f64 预条件器装配上下文（复用 ghost / 谱半径原始变量缓冲）。
    pub fn f64_preconditioner_context(&mut self) -> CompressibleAdvanceContext3d<'_> {
        CompressibleAdvanceContext3d {
            mesh: self.mesh,
            structured: self.structured,
            patches: self.patches,
            ghosts: self.ghosts,
            eos: self.eos,
            freestream: self.freestream,
            reference: self.reference,
            primitive_scratch: self.spectral_primitives.clone(),
            gradient_scratch: self.gradient_scratch.clone(),
            viscous: self.viscous,
            residual_correction: None,
        }
    }
}

/// 1D 多步推进上下文。
pub struct CompressibleAdvanceContext1d<'a> {
    pub mesh: &'a StructuredMesh1d,
    pub boundary: crate::discretization::InviscidBoundary1d,
    pub eos: &'a IdealGasEoS,
}
