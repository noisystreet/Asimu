//! 非结构 3D 网格无粘残差装配（typed 场；P3 首版仅一阶、串行 accumulate）。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::{ComputeFloat, Real};
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::{
    BoundaryGhostBuffer, FaceFluxInput, InviscidFluxConfig, ReconstructionKind,
    UnstructuredSolverMeshCache, face_inviscid_flux,
};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT, primitive_from_conserved_relaxed,
};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use super::{accumulate_boundary_face_typed, accumulate_interior_face_typed, is_degenerate_volume};

/// typed 非结构无粘残差装配上下文。
pub struct InviscidAssemblyUnstructuredTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub min_pressure: Real,
}

/// 装配非结构 3D 无粘 Euler 残差（一阶；`T=f32`/`f64`）。
pub fn assemble_inviscid_residual_unstructured_typed<T: ComputeFloat>(
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
) -> Result<()> {
    if params.config.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(format!(
            "compute_precision = \"{}\" 的非结构 typed 路径暂仅支持一阶重构",
            T::PRECISION.label()
        )));
    }
    let n = params.mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构 typed 场/残差/primitive 长度须等于网格单元数 {n}"
        )));
    }
    residual.clear();
    let topology = &params.mesh_cache.face_topology;
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_faces_typed",
            faces = topology.interior.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        assemble_interior_faces_typed(residual, params, topology)?;
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_boundary_faces_typed",
            faces = topology.boundary.len(),
            precision = T::PRECISION.label(),
        )
        .entered();
        assemble_boundary_faces_typed(residual, params, topology)?;
    }
    Ok(())
}

fn first_order_interior_flux<T: ComputeFloat>(
    primitives: &PrimitiveFieldsT<T>,
    owner: usize,
    neighbor: usize,
    normal: crate::core::Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<crate::discretization::InviscidFlux> {
    let owner_prim = primitives.cell_primitive(owner);
    let neighbor_prim = primitives.cell_primitive(neighbor);
    face_inviscid_flux(
        FaceFluxInput::first_order(&owner_prim, &neighbor_prim),
        normal,
        eos,
        config,
    )
}

fn assemble_interior_faces_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    for face in &topology.interior {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume(face.owner_volume) || is_degenerate_volume(face.neighbor_volume) {
            continue;
        }
        let flux = first_order_interior_flux(
            params.primitives,
            face.owner,
            face.neighbor,
            face.normal,
            params.eos,
            params.config,
        )?;
        accumulate_interior_face_typed(
            residual,
            face.owner,
            face.neighbor,
            &flux,
            face.area,
            face.owner_volume,
            face.neighbor_volume,
        )?;
    }
    Ok(())
}

fn assemble_boundary_faces_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssemblyUnstructuredTypedParams<'_, T>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    for bface in &topology.boundary {
        if bface.owner_rhs_scale == 0.0 {
            continue;
        }
        if is_degenerate_volume(bface.owner_volume) {
            continue;
        }
        let ghost = params.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let owner_prim = params.primitives.cell_primitive(bface.owner);
        let ghost_prim =
            primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
        let flux = face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &ghost_prim),
            bface.normal,
            params.eos,
            params.config,
        )?;
        accumulate_boundary_face_typed(
            residual,
            bface.owner,
            &flux,
            bface.area,
            bface.owner_volume,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::freestream_pair::FreestreamPairFixture;
    use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    #[test]
    fn f32_single_tet_uniform_freestream_has_near_zero_rhs() {
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
            faces,
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let fields = ConservedFieldsT::<f32>::from_real_fields(
            &crate::field::ConservedFields::from_freestream_context(
                mesh.num_cells(),
                &side.ctx,
                side.fs,
            )
            .expect("fields"),
        )
        .expect("typed");
        let mut ghosts = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary,
            &fields.cast_real().expect("real"),
            &mut ghosts,
            &side.ctx,
            side.fs,
            None,
        )
        .expect("bc");
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let mut primitives = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, side.eos, side.min_pressure)
            .expect("fill");
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let config = InviscidFluxConfig::default();
        let params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            min_pressure: side.min_pressure,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut rhs, &params)
            .expect("assemble");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-5),
            "f32 tet density rhs"
        );
    }
}
