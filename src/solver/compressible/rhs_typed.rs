//! 3D 可压缩 Euler typed 右端项（边界 + 一阶无粘残差）。

use tracing::info_span;

use super::structured_compute_backend::StructuredComputeBackend;
use crate::boundary::BoundarySet;
use crate::core::ComputeFloat;
use crate::core::Real;
use crate::discretization::compressible::residual::InviscidAssembly3dTypedParams;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, InviscidFluxConfig, StructuredFaceCacheF32,
    assemble_inviscid_residual_3d_typed,
};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig};
use crate::solver::compressible::helpers::{
    RefreshCompressibleStateTypedInput, refresh_compressible_ghosts_and_primitives_typed,
};

/// typed 单步 RHS 求值上下文。
pub(crate) struct EvaluateRhs3dTyped<'a, T: ComputeFloat> {
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
    pub primitive_scratch: &'a mut PrimitiveFieldsT<T>,
    pub gradient_scratch: &'a mut GradientFields,
    pub interface_residual: Option<
        &'a [crate::solver::compressible::multiblock_interface::InterfaceResidualContribution],
    >,
    pub face_cache_f32: Option<&'a StructuredFaceCacheF32>,
}

impl<T: StructuredComputeBackend> EvaluateRhs3dTyped<'_, T> {
    pub fn run(
        &mut self,
        fields: &ConservedFieldsT<T>,
        residual: &mut ConservedResidualT<T>,
    ) -> Result<()> {
        let _span = info_span!("evaluate_rhs_typed", precision = T::PRECISION.label()).entered();
        refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
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
        let assembly = InviscidAssembly3dTypedParams {
            mesh: self.structured,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitive_scratch,
            min_pressure: self.min_pressure,
            face_cache_f32: self.face_cache_f32,
        };
        assemble_inviscid_residual_3d_typed(fields, residual, &assembly)?;
        if let Some(contributions) = self.interface_residual {
            T::apply_interface_residuals(residual, contributions)?;
        }
        let _ = self.gradient_scratch;
        Ok(())
    }
}
