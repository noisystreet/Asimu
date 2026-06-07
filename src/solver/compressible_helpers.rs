//! 可压缩求解共用：BC/原始变量刷新、谱半径时间步策略、非结构 RHS 求值。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, InviscidAssemblyUnstructuredParams,
    UnstructuredSolverMeshCache, ViscousAssemblyUnstructuredInput,
    ViscousAssemblyUnstructuredScratch, apply_compressible_boundary_conditions,
    assemble_inviscid_residual_unstructured,
    compute_gradients_and_assemble_viscous_unstructured_with_scratch,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh3d, UnstructuredMesh3d};
use crate::physics::{
    FreestreamContext, FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig,
};
use crate::solver::spectral_radius::cell_local_dt_spectral;
use crate::solver::time::min_positive_dt;

/// BC + 原始变量刷新输入（结构/非结构共用）。
pub struct RefreshCompressibleStateInput<'a> {
    pub boundary_mesh: &'a dyn BoundaryMesh3d,
    pub patches: &'a BoundarySet,
    pub fields: &'a ConservedFields,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub min_pressure: Real,
    pub primitives: &'a mut PrimitiveFields,
}

/// 刷新 BC ghost 与原始变量（结构/非结构共用）。
pub fn refresh_compressible_ghosts_and_primitives(
    input: RefreshCompressibleStateInput<'_>,
) -> Result<()> {
    let fs_ctx = FreestreamContext::new(input.eos, input.reference, input.viscous);
    apply_compressible_boundary_conditions(
        input.boundary_mesh,
        input.patches,
        input.fields,
        input.ghosts,
        &fs_ctx,
        input.freestream,
        input.viscous,
    )?;
    input
        .primitives
        .fill_from_conserved(input.fields, input.eos, input.min_pressure)
}

/// 由谱半径计算单元时间步并应用固定 dt / 全局 dt 策略。
pub fn finalize_cell_dts_from_sigma(
    volumes: &[Real],
    sigma: &[Real],
    cfl: Real,
    fixed_dt: Option<Real>,
    local_time_step: bool,
) -> Result<Vec<Real>> {
    let mut cell_dts = cell_local_dt_spectral(volumes, sigma, cfl)?;
    if let Some(dt) = fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite()) {
        cell_dts.fill(dt);
    } else if !local_time_step {
        let dt = min_positive_dt(&cell_dts);
        cell_dts.fill(dt);
    }
    Ok(cell_dts)
}

/// 非结构 3D RHS 求值上下文（镜像 `EvaluateRhs3d`）。
pub struct EvaluateRhsUnstructured<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    pub inviscid: &'a crate::discretization::InviscidFluxConfig,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub min_pressure: Real,
    pub primitives: &'a mut PrimitiveFields,
    pub gradients: &'a mut GradientFields,
    pub viscous_scratch: &'a mut ViscousAssemblyUnstructuredScratch,
}

impl EvaluateRhsUnstructured<'_> {
    pub fn run(
        &mut self,
        fields: &ConservedFields,
        residual: &mut ConservedResidual,
    ) -> Result<()> {
        let _span = info_span!("evaluate_rhs_unstructured").entered();
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
            primitives: self.primitives,
        })?;
        let params = InviscidAssemblyUnstructuredParams {
            mesh: self.mesh,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitives,
        };
        assemble_inviscid_residual_unstructured(fields, residual, &params)?;
        if let Some(viscous) = self.viscous {
            let mut input = ViscousAssemblyUnstructuredInput {
                mesh: self.mesh,
                mesh_cache: self.mesh_cache,
                eos: self.eos,
                viscous,
                boundaries: self.patches,
                ghosts: self.ghosts,
                primitives: self.primitives,
                min_pressure: self.min_pressure,
                gradient_scratch: self.gradients,
            };
            compute_gradients_and_assemble_viscous_unstructured_with_scratch(
                residual,
                &mut input,
                self.viscous_scratch,
            )?;
        }
        Ok(())
    }

    /// 在 ghost/primitive 已由调用方刷新时，仅装配残差（LU-SGS 内层复用）。
    pub fn assemble_from_current_state(
        &mut self,
        fields: &ConservedFields,
        residual: &mut ConservedResidual,
    ) -> Result<()> {
        let params = InviscidAssemblyUnstructuredParams {
            mesh: self.mesh,
            eos: self.eos,
            config: self.inviscid,
            boundaries: self.patches,
            ghosts: self.ghosts,
            primitives: self.primitives,
        };
        {
            let _span = info_span!("assemble_unstructured_inviscid_residual").entered();
            assemble_inviscid_residual_unstructured(fields, residual, &params)?;
        }
        if let Some(viscous) = self.viscous {
            let mut input = ViscousAssemblyUnstructuredInput {
                mesh: self.mesh,
                mesh_cache: self.mesh_cache,
                eos: self.eos,
                viscous,
                boundaries: self.patches,
                ghosts: self.ghosts,
                primitives: self.primitives,
                min_pressure: self.min_pressure,
                gradient_scratch: self.gradients,
            };
            let _span = info_span!("assemble_unstructured_viscous_residual").entered();
            compute_gradients_and_assemble_viscous_unstructured_with_scratch(
                residual,
                &mut input,
                self.viscous_scratch,
            )?;
        }
        Ok(())
    }
}
