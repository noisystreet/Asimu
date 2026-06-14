//! 可压缩求解共用：BC/原始变量刷新、谱半径时间步策略、非结构 RHS 求值。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    BoundaryGhostBuffer, GradientFields, InviscidAssemblyUnstructuredParams, ReconstructionKind,
    UnstructuredGradientLsqInput, UnstructuredSolverMeshCache, ViscousAssemblyUnstructuredInput,
    ViscousAssemblyUnstructuredScratch, apply_compressible_boundary_conditions,
    apply_compressible_boundary_conditions_typed, assemble_inviscid_residual_unstructured,
    compute_gradients_and_assemble_viscous_unstructured_with_scratch,
    compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh3d, UnstructuredMesh3d};
use crate::physics::{
    FreestreamContext, FreestreamParams, IdealGasEoS, ReferenceScales, ViscousPhysicsConfig,
};
use crate::solver::spectral_radius::{cell_local_dt_spectral, cell_local_dt_spectral_f32};
use crate::solver::time::{min_positive_dt, min_positive_dt_f32};

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

/// typed 场 BC + 原始变量刷新（ghost 逐面读 typed `cell_state`，不再整场合 `cast_real`）。
pub struct RefreshCompressibleStateTypedInput<'a, T: crate::core::ComputeFloat> {
    pub boundary_mesh: &'a dyn BoundaryMesh3d,
    pub patches: &'a BoundarySet,
    pub fields: &'a crate::field::ConservedFieldsT<T>,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
    pub reference: Option<&'a ReferenceScales>,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub min_pressure: Real,
    pub primitives: &'a mut crate::field::PrimitiveFieldsT<T>,
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

/// typed 守恒场：ghost 与 primitive 均基于 typed 场刷新。
pub fn refresh_compressible_ghosts_and_primitives_typed<
    T: crate::core::ComputeFloat + crate::field::PrimitiveFillFromConserved,
>(
    input: RefreshCompressibleStateTypedInput<'_, T>,
) -> Result<()> {
    let fs_ctx = FreestreamContext::new(input.eos, input.reference, input.viscous);
    apply_compressible_boundary_conditions_typed(
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

/// f32 谱半径 → 单元时间步（固定 dt / 全局 dt 策略与 f64 一致）。
pub fn finalize_cell_dts_from_sigma_f32(
    volumes: &[f32],
    sigma: &[f32],
    cfl: f32,
    fixed_dt: Option<f32>,
    local_time_step: bool,
) -> Result<Vec<f32>> {
    let mut cell_dts = cell_local_dt_spectral_f32(volumes, sigma, cfl)?;
    if let Some(dt) = fixed_dt.filter(|dt| *dt > 0.0 && dt.is_finite()) {
        cell_dts.fill(dt);
    } else if !local_time_step {
        let dt = min_positive_dt_f32(&cell_dts);
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
    pub exec: &'a mut crate::exec::ExecutionContext,
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
                self.gradients,
                &mut self.viscous_scratch.gradient,
                self.exec,
            )?;
        }
        let params = inviscid_assembly_params(self);
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
                exec: self.exec,
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
        let params = inviscid_assembly_params(self);
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
                exec: self.exec,
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

fn inviscid_assembly_params<'a>(
    ctx: &'a EvaluateRhsUnstructured<'a>,
) -> InviscidAssemblyUnstructuredParams<'a> {
    InviscidAssemblyUnstructuredParams {
        mesh: ctx.mesh,
        eos: ctx.eos,
        config: ctx.inviscid,
        boundaries: ctx.patches,
        ghosts: ctx.ghosts,
        primitives: ctx.primitives,
        face_topology: Some(&ctx.mesh_cache.face_topology),
        mesh_cache: Some(ctx.mesh_cache),
        gradients: Some(ctx.gradients),
        min_pressure: ctx.min_pressure,
        exec: ctx.exec,
    }
}

#[cfg(test)]
mod refresh_typed_tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::approx_eq;
    use crate::discretization::freestream_pair::FreestreamPairFixture;
    use crate::field::ConservedFieldsT;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    #[test]
    fn f32_typed_ghost_refresh_matches_f64_per_face() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces.clone(),
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let fields_f32 = ConservedFieldsT::<f32>::from_real_fields(
            &crate::field::ConservedFields::from_freestream_context(
                mesh.num_cells(),
                &side.ctx,
                side.fs,
            )
            .expect("fields"),
        )
        .expect("f32");
        let fields_f64 = fields_f32.cast_real().expect("f64");
        let mut ghosts_f32 = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let mut ghosts_f64 = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let mut prim_f32 =
            crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim f32");
        let mut prim_f64 = PrimitiveFields::zeros(mesh.num_cells()).expect("prim f64");
        refresh_compressible_ghosts_and_primitives_typed(RefreshCompressibleStateTypedInput {
            boundary_mesh: &mesh,
            patches: &boundary,
            fields: &fields_f32,
            ghosts: &mut ghosts_f32,
            eos: side.eos,
            freestream: side.fs,
            reference: None,
            viscous: None,
            min_pressure: side.min_pressure,
            primitives: &mut prim_f32,
        })
        .expect("f32 refresh");
        refresh_compressible_ghosts_and_primitives(RefreshCompressibleStateInput {
            boundary_mesh: &mesh,
            patches: &boundary,
            fields: &fields_f64,
            ghosts: &mut ghosts_f64,
            eos: side.eos,
            freestream: side.fs,
            reference: None,
            viscous: None,
            min_pressure: side.min_pressure,
            primitives: &mut prim_f64,
        })
        .expect("f64 refresh");
        for &face in &faces {
            let g32 = ghosts_f32.get_face(face).expect("f32 ghost");
            let g64 = ghosts_f64.get_face(face).expect("f64 ghost");
            assert!(approx_eq(
                g32.conserved.density,
                g64.conserved.density,
                1.0e-5
            ));
            assert!(approx_eq(
                g32.conserved.total_energy,
                g64.conserved.total_energy,
                1.0e-5
            ));
        }
    }
}
