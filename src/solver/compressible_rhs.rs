//! 3D 可压缩 Euler 右端项（边界 + 无粘残差），供时间推进与 Chrome trace 复用。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::discretization::{
    BoundaryGhostBuffer, InviscidFluxConfig, apply_compressible_boundary_conditions,
    assemble_inviscid_residual_3d,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS};

/// 单步 RHS 求值上下文（避免过多函数参数）。
pub(crate) struct EvaluateRhs3d<'a> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub inviscid: &'a InviscidFluxConfig,
}

impl EvaluateRhs3d<'_> {
    pub fn run(
        &mut self,
        fields: &ConservedFields,
        residual: &mut ConservedResidual,
    ) -> Result<()> {
        let _span = info_span!("evaluate_rhs").entered();
        apply_compressible_boundary_conditions(
            self.mesh,
            self.patches,
            fields,
            self.ghosts,
            self.eos,
            self.freestream,
        )?;
        assemble_inviscid_residual_3d(
            self.structured,
            fields,
            residual,
            self.eos,
            self.inviscid,
            self.patches,
            self.ghosts,
        )
    }
}
