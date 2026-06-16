//! 3D 结构化 MUSCL typed 面循环（复用 `muscl_stencil_3d_typed` 宽模板）。

use crate::core::ComputeFloat;
use crate::error::Result;
use crate::field::ConservedResidualT;
use crate::mesh::StructuredMesh3d;

use super::assembly_3d_typed::{BoundaryAssembly3dTyped, InviscidAssembly3dTypedParams};
use super::muscl_stencil_3d_typed::{
    BoundaryFaceFlux3dTyped, InteriorFaceFlux3dTyped, flux_at_boundary_face_typed,
    flux_at_i_face_typed, flux_at_j_face_typed, flux_at_k_face_typed,
};
use super::{accumulate_boundary_face_typed, accumulate_interior_face_typed, is_degenerate_volume};

pub(super) fn assemble_muscl_faces_3d_typed<T: ComputeFloat>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
) -> Result<()> {
    assemble_i_faces_muscl_typed(mesh, residual, params)?;
    assemble_j_faces_muscl_typed(mesh, residual, params)?;
    assemble_k_faces_muscl_typed(mesh, residual, params)?;
    assemble_boundary_faces_muscl_typed(
        residual,
        &BoundaryAssembly3dTyped {
            mesh,
            structured: mesh,
            params,
        },
    )
}

fn assemble_i_faces_muscl_typed<T: ComputeFloat>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                let ctx = InteriorFaceFlux3dTyped {
                    primitives: params.primitives,
                    mesh,
                    eos: params.eos,
                    config: params.config,
                    normal: face.normal,
                };
                let flux = flux_at_i_face_typed(&ctx, i, j, k)?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i + 1, j, k).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face_typed(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    owner_volume,
                    neighbor_volume,
                )?;
            }
        }
    }
    Ok(())
}

fn assemble_j_faces_muscl_typed<T: ComputeFloat>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j + 1, k);
                let face = mesh.j_face_metric(i, j, k);
                let ctx = InteriorFaceFlux3dTyped {
                    primitives: params.primitives,
                    mesh,
                    eos: params.eos,
                    config: params.config,
                    normal: face.normal,
                };
                let flux = flux_at_j_face_typed(&ctx, i, j, k)?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i, j + 1, k).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face_typed(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    owner_volume,
                    neighbor_volume,
                )?;
            }
        }
    }
    Ok(())
}

fn assemble_k_faces_muscl_typed<T: ComputeFloat>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j, k + 1);
                let face = mesh.k_face_metric(i, j, k);
                let ctx = InteriorFaceFlux3dTyped {
                    primitives: params.primitives,
                    mesh,
                    eos: params.eos,
                    config: params.config,
                    normal: face.normal,
                };
                let flux = flux_at_k_face_typed(&ctx, i, j, k)?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i, j, k + 1).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face_typed(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    owner_volume,
                    neighbor_volume,
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn assemble_boundary_faces_muscl_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    ctx: &BoundaryAssembly3dTyped<'_, T>,
) -> Result<()> {
    let mesh = ctx.structured;
    let params = ctx.params;
    for patch in params.boundaries.patches() {
        if matches!(patch.kind, crate::boundary::BoundaryKind::Periodic { .. }) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner_id = ctx.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let geom = ctx.mesh.face_geometry_3d(face)?;
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                crate::error::AsimuError::Boundary(format!(
                    "边界面 FaceId({}) 缺少 ghost 状态",
                    face.index()
                ))
            })?;
            let bctx = BoundaryFaceFlux3dTyped {
                primitives: params.primitives,
                mesh,
                eos: params.eos,
                config: params.config,
                min_pressure: params.min_pressure,
            };
            let flux = flux_at_boundary_face_typed(&bctx, face, ghost.conserved, geom.normal)?;
            let (logical, local) = crate::mesh::LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            let owner_volume = mesh.cell_metric(i, j, k).volume;
            if is_degenerate_volume(owner_volume) {
                continue;
            }
            accumulate_boundary_face_typed(residual, owner, &flux, geom.area, owner_volume)?;
        }
    }
    Ok(())
}
