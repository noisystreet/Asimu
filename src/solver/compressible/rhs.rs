//! 3D 可压缩 Euler 右端项（边界 + 无粘残差），供时间推进与 Chrome trace 复用。

use tracing::info_span;

use super::ResidualCorrection3dHandle;
use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::compressible::residual::InviscidAssembly3dParams;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, InviscidFluxConfig, assemble_inviscid_residual_3d,
    compute_gradients_and_assemble_viscous_3d,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig};
use crate::solver::compressible::helpers::{
    RefreshCompressibleStateInput, refresh_compressible_ghosts_and_primitives,
};

/// 单步 RHS 求值上下文（避免过多函数参数）。
pub(crate) struct EvaluateRhs3d<'a> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    pub inviscid: &'a InviscidFluxConfig,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub min_pressure: Real,
    pub primitive_scratch: &'a mut PrimitiveFields,
    pub gradient_scratch: &'a mut GradientFields,
    pub residual_correction: Option<ResidualCorrection3dHandle>,
}

impl EvaluateRhs3d<'_> {
    pub fn run(
        &mut self,
        fields: &ConservedFields,
        residual: &mut ConservedResidual,
    ) -> Result<()> {
        let _span = info_span!("evaluate_rhs").entered();
        refresh_compressible_ghosts_and_primitives(RefreshCompressibleStateInput {
            boundary_mesh: self.mesh,
            patches: self.patches,
            fields,
            ghosts: self.ghosts,
            eos: self.eos,
            freestream: self.freestream,
            reference: self.reference,
            viscous: self.viscous,
            min_pressure: self.min_pressure,
            primitives: self.primitive_scratch,
        })?;
        let assembly = InviscidAssembly3dParams {
            mesh: self.structured,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitive_scratch,
            min_pressure: self.min_pressure,
        };
        assemble_inviscid_residual_3d(fields, residual, &assembly)?;
        if let Some(viscous) = self.viscous {
            let mut input = crate::discretization::ViscousAssembly3dInput {
                mesh: self.structured,
                eos: self.eos,
                viscous,
                boundaries: self.patches,
                ghosts: self.ghosts,
                primitives: self.primitive_scratch,
                min_pressure: self.min_pressure,
                gradient_scratch: self.gradient_scratch,
            };
            compute_gradients_and_assemble_viscous_3d(residual, &mut input)?;
        }
        if let Some(correction) = &self.residual_correction {
            correction.borrow_mut().apply(residual)?;
        }
        Ok(())
    }
}
