//! 非结构 3D 可压缩 Euler typed 右端项。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::gradient_unstructured_f32::UnstructuredGradientLsqInputF32;
use crate::discretization::residual::InviscidAssemblyUnstructuredTypedParams;
use crate::discretization::residual::InviscidTypedScatterBackend;
use crate::discretization::residual::ViscousTypedScatterBackend;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFieldsT, InviscidFluxConfig, ReconstructionKind,
    UnstructuredGradientLsqInput, UnstructuredGradientScratchF32, UnstructuredSolverMeshCache,
    ViscousAssemblyUnstructuredF32Input, ViscousAssemblyUnstructuredScratch,
    ViscousAssemblyUnstructuredTypedInput, assemble_inviscid_residual_unstructured_typed,
    compute_gradients_and_assemble_viscous_unstructured_f32,
    compute_gradients_and_assemble_viscous_unstructured_typed,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32,
};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig};
use crate::solver::compressible_helpers::{
    RefreshCompressibleStateTypedInput, refresh_compressible_ghosts_and_primitives_typed,
};
use tracing::info_span;

/// typed 非结构单步 RHS 求值上下文（驱动层当前 inline 装配；供 LU-SGS typed 复用）。
#[allow(dead_code)]
pub(crate) struct EvaluateRhsUnstructuredTyped<
    'a,
    T: InviscidTypedScatterBackend + ViscousTypedScatterBackend,
> {
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
    pub gradient_scratch: &'a mut GradientFieldsT<T>,
    pub viscous_scratch: &'a mut ViscousAssemblyUnstructuredScratch,
    pub viscous_grad_scratch_f32: &'a mut UnstructuredGradientScratchF32,
    pub exec: &'a mut crate::exec::ExecutionContext,
}

#[allow(dead_code)]
impl EvaluateRhsUnstructuredTyped<'_, f32> {
    pub fn run(
        &mut self,
        fields: &ConservedFieldsT<f32>,
        residual: &mut ConservedResidualT<f32>,
    ) -> Result<()> {
        let _span = info_span!("evaluate_rhs_unstructured_typed", precision = "f32").entered();
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
        })?;
        self.assemble_from_current_state(fields, residual)
    }

    pub fn assemble_from_current_state(
        &mut self,
        fields: &ConservedFieldsT<f32>,
        residual: &mut ConservedResidualT<f32>,
    ) -> Result<()> {
        if self.inviscid.reconstruction == ReconstructionKind::Muscl {
            let grad_input = UnstructuredGradientLsqInputF32 {
                mesh: self.mesh,
                mesh_cache: self.mesh_cache,
                primitives: self.primitives,
                eos: self.eos,
                ghosts: self.ghosts,
                min_pressure: self.min_pressure,
                viscous: self.viscous,
            };
            compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32(
                grad_input,
                self.gradient_scratch,
                self.exec,
            )?;
        }
        let gradients = match self.inviscid.reconstruction {
            ReconstructionKind::Muscl => Some(&*self.gradient_scratch),
            ReconstructionKind::FirstOrder => None,
        };
        let mut assembly = InviscidAssemblyUnstructuredTypedParams {
            mesh: self.mesh,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitives,
            mesh_cache: self.mesh_cache,
            gradients,
            min_pressure: self.min_pressure,
            exec: self.exec,
        };
        assemble_inviscid_residual_unstructured_typed(fields, residual, &mut assembly)?;
        if let Some(viscous) = self.viscous {
            let mut input = ViscousAssemblyUnstructuredF32Input {
                mesh: self.mesh,
                mesh_cache: self.mesh_cache,
                eos: self.eos,
                viscous,
                boundaries: self.patches,
                ghosts: self.ghosts,
                primitives: self.primitives,
                min_pressure: self.min_pressure,
                gradient_scratch: self.gradient_scratch,
                exec: self.exec,
            };
            compute_gradients_and_assemble_viscous_unstructured_f32(
                residual,
                &mut input,
                self.viscous_scratch,
                self.viscous_grad_scratch_f32,
            )?;
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl EvaluateRhsUnstructuredTyped<'_, f64> {
    pub fn run(
        &mut self,
        fields: &ConservedFieldsT<f64>,
        residual: &mut ConservedResidualT<f64>,
    ) -> Result<()> {
        let _span = info_span!("evaluate_rhs_unstructured_typed", precision = "f64").entered();
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
        })?;
        self.assemble_from_current_state(fields, residual)
    }

    pub fn assemble_from_current_state(
        &mut self,
        fields: &ConservedFieldsT<f64>,
        residual: &mut ConservedResidualT<f64>,
    ) -> Result<()> {
        if self.inviscid.reconstruction == ReconstructionKind::Muscl {
            let grad_input = UnstructuredGradientLsqInput {
                mesh: self.mesh,
                mesh_cache: self.mesh_cache,
                primitives: self.primitives,
                eos: self.eos,
                ghosts: self.ghosts,
                min_pressure: self.min_pressure,
                viscous: self.viscous,
            };
            compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
                grad_input,
                self.gradient_scratch,
                &mut self.viscous_scratch.gradient,
                self.exec,
            )?;
        }
        let gradients = match self.inviscid.reconstruction {
            ReconstructionKind::Muscl => Some(&*self.gradient_scratch),
            ReconstructionKind::FirstOrder => None,
        };
        let mut assembly = InviscidAssemblyUnstructuredTypedParams {
            mesh: self.mesh,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitives,
            mesh_cache: self.mesh_cache,
            gradients,
            min_pressure: self.min_pressure,
            exec: self.exec,
        };
        assemble_inviscid_residual_unstructured_typed(fields, residual, &mut assembly)?;
        if let Some(viscous) = self.viscous {
            let mut input = ViscousAssemblyUnstructuredTypedInput {
                mesh: self.mesh,
                mesh_cache: self.mesh_cache,
                eos: self.eos,
                viscous,
                boundaries: self.patches,
                ghosts: self.ghosts,
                primitives: self.primitives,
                min_pressure: self.min_pressure,
                gradient_scratch: self.gradient_scratch,
                exec: self.exec,
            };
            compute_gradients_and_assemble_viscous_unstructured_typed(
                residual,
                &mut input,
                self.viscous_scratch,
            )?;
        }
        Ok(())
    }
}
