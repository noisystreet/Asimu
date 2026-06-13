//! 非结构 3D 可压缩 Euler typed 右端项。

use crate::boundary::BoundarySet;
use crate::core::ComputeFloat;
use crate::core::Real;
use crate::discretization::residual::InviscidAssemblyUnstructuredTypedParams;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, InviscidFluxConfig, UnstructuredSolverMeshCache,
    assemble_inviscid_residual_unstructured_typed,
};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFields, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig};
use crate::solver::compressible_helpers::{
    RefreshCompressibleStateTypedInput, refresh_compressible_ghosts_and_primitives_typed,
};
use tracing::info_span;

/// typed 非结构单步 RHS 求值上下文（供非闭包路径复用；驱动层暂 inline 以避免泛型闭包借用）。
#[allow(dead_code)]
pub(crate) struct EvaluateRhsUnstructuredTyped<'a, T: ComputeFloat> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    pub inviscid: &'a InviscidFluxConfig,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub min_pressure: Real,
    pub primitives: &'a mut PrimitiveFieldsT<T>,
    pub spectral_primitives: &'a mut PrimitiveFields,
    pub gradient_scratch: &'a mut GradientFields,
}

impl<T: ComputeFloat> EvaluateRhsUnstructuredTyped<'_, T> {
    #[allow(dead_code)]
    pub fn run(
        &mut self,
        fields: &ConservedFieldsT<T>,
        residual: &mut ConservedResidualT<T>,
    ) -> Result<()> {
        let _span = info_span!(
            "evaluate_rhs_unstructured_typed",
            precision = T::PRECISION.label()
        )
        .entered();
        if self.viscous.is_some() {
            return Err(crate::error::AsimuError::Config(format!(
                "compute_precision = \"{}\" 的非结构 typed 路径暂不支持粘性通量",
                T::PRECISION.label()
            )));
        }
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
            primitives: self.primitives,
            spectral_primitives: self.spectral_primitives,
        })?;
        let assembly = InviscidAssemblyUnstructuredTypedParams {
            mesh: self.mesh,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitives,
            mesh_cache: self.mesh_cache,
            min_pressure: self.min_pressure,
        };
        assemble_inviscid_residual_unstructured_typed(fields, residual, &assembly)?;
        let _ = self.gradient_scratch;
        Ok(())
    }

    /// ghost/primitive 已由调用方刷新时，仅装配无粘残差。
    #[allow(dead_code)]
    pub fn assemble_from_current_state(
        &mut self,
        fields: &ConservedFieldsT<T>,
        residual: &mut ConservedResidualT<T>,
    ) -> Result<()> {
        let assembly = InviscidAssemblyUnstructuredTypedParams {
            mesh: self.mesh,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitives,
            mesh_cache: self.mesh_cache,
            min_pressure: self.min_pressure,
        };
        {
            let _span = info_span!("assemble_unstructured_inviscid_residual_typed").entered();
            assemble_inviscid_residual_unstructured_typed(fields, residual, &assembly)?;
        }
        Ok(())
    }
}
