//! 3D 可压缩 Euler 右端项（边界 + 无粘残差），供时间推进与 Chrome trace 复用。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::residual::InviscidAssembly3dParams;
use crate::discretization::{
    BoundaryGhostBuffer, InviscidFluxConfig, apply_compressible_boundary_conditions,
    assemble_inviscid_residual_3d,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
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
    pub min_pressure: Real,
    pub primitive_scratch: &'a mut PrimitiveFields,
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
        self.primitive_scratch
            .fill_from_conserved(fields, self.eos, self.min_pressure)?;
        let assembly = InviscidAssembly3dParams {
            mesh: self.structured,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitive_scratch,
            min_pressure: self.min_pressure,
        };
        assemble_inviscid_residual_3d(fields, residual, &assembly)
    }
}
