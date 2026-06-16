//! 3D 结构化无粘残差 f32 内面装配（读取 `StructuredFaceCacheF32`；ADR 0019 S1-a）。

use tracing::info_span;

use crate::discretization::ReconstructionKind;
use crate::discretization::face_flux_typed::face_inviscid_flux_first_order_interior_soa_f32;
use crate::discretization::structured_face_cache_f32::{
    StructuredFaceCacheF32, i_face_cache_index, j_face_cache_index, k_face_cache_index,
    vec3_from_f32,
};
use crate::error::Result;
use crate::field::ConservedFieldsT;
use crate::field::ConservedResidualT;
use crate::mesh::StructuredMesh3d;

use super::assembly_3d_muscl_typed::assemble_boundary_faces_muscl_typed;
use super::assembly_3d_typed::{
    BoundaryAssembly3dTyped, InviscidAssembly3dTypedParams, assemble_boundary_faces_3d_typed,
};
use super::muscl_stencil_3d_typed::{
    InteriorFaceFlux3dTyped, flux_at_i_face_typed, flux_at_j_face_typed, flux_at_k_face_typed,
};
use super::{
    accumulate_interior_face_f32, accumulate_interior_face_typed, is_degenerate_volume_f32,
};

pub(super) fn assemble_inviscid_residual_3d_f32(
    _fields: &ConservedFieldsT<f32>,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
    cache: &StructuredFaceCacheF32,
) -> Result<()> {
    match params.config.reconstruction {
        ReconstructionKind::FirstOrder => {
            assemble_first_order_interior_f32(cache, residual, params)?;
            assemble_boundary_faces_3d_typed(
                residual,
                &BoundaryAssembly3dTyped {
                    mesh: params.mesh,
                    structured: params.mesh,
                    params,
                },
            )
        }
        ReconstructionKind::Muscl => {
            assemble_muscl_interior_f32(params.mesh, cache, residual, params)?;
            assemble_boundary_faces_muscl_typed(
                residual,
                &BoundaryAssembly3dTyped {
                    mesh: params.mesh,
                    structured: params.mesh,
                    params,
                },
            )
        }
    }
}

fn assemble_first_order_interior_f32(
    cache: &StructuredFaceCacheF32,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
) -> Result<()> {
    {
        let _span = info_span!("assemble_faces_f32", dim = "i").entered();
        for face in &cache.i_faces {
            accumulate_first_order_face_f32(residual, face, params)?;
        }
    }
    {
        let _span = info_span!("assemble_faces_f32", dim = "j").entered();
        for face in &cache.j_faces {
            accumulate_first_order_face_f32(residual, face, params)?;
        }
    }
    {
        let _span = info_span!("assemble_faces_f32", dim = "k").entered();
        for face in &cache.k_faces {
            accumulate_first_order_face_f32(residual, face, params)?;
        }
    }
    Ok(())
}

fn accumulate_first_order_face_f32(
    residual: &mut ConservedResidualT<f32>,
    face: &crate::discretization::structured_face_cache_f32::StructuredInteriorFaceF32,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
) -> Result<()> {
    if is_degenerate_volume_f32(face.owner_volume) || is_degenerate_volume_f32(face.neighbor_volume)
    {
        return Ok(());
    }
    let flux = face_inviscid_flux_first_order_interior_soa_f32(
        face.owner,
        face.neighbor,
        params.primitives,
        face.normal,
        params.eos,
        params.config,
    )?;
    accumulate_interior_face_f32(
        residual,
        face.owner,
        face.neighbor,
        &flux,
        face.area,
        face.owner_volume,
        face.neighbor_volume,
    )
}

fn assemble_muscl_interior_f32(
    mesh: &StructuredMesh3d,
    cache: &StructuredFaceCacheF32,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
) -> Result<()> {
    assemble_i_faces_muscl_f32(mesh, cache, residual, params)?;
    assemble_j_faces_muscl_f32(mesh, cache, residual, params)?;
    assemble_k_faces_muscl_f32(mesh, cache, residual, params)
}

fn assemble_i_faces_muscl_f32(
    mesh: &StructuredMesh3d,
    cache: &StructuredFaceCacheF32,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let geom = &cache.i_faces[i_face_cache_index(nx, ny, i, j, k)];
                if is_degenerate_volume_f32(geom.owner_volume)
                    || is_degenerate_volume_f32(geom.neighbor_volume)
                {
                    continue;
                }
                let ctx = InteriorFaceFlux3dTyped {
                    primitives: params.primitives,
                    mesh,
                    eos: params.eos,
                    config: params.config,
                    normal: vec3_from_f32(geom.normal),
                };
                let flux = flux_at_i_face_typed(&ctx, i, j, k)?;
                accumulate_interior_face_typed(
                    residual,
                    geom.owner,
                    geom.neighbor,
                    &flux,
                    f64::from(geom.area),
                    f64::from(geom.owner_volume),
                    f64::from(geom.neighbor_volume),
                )?;
            }
        }
    }
    Ok(())
}

fn assemble_j_faces_muscl_f32(
    mesh: &StructuredMesh3d,
    cache: &StructuredFaceCacheF32,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let geom = &cache.j_faces[j_face_cache_index(nx, ny, i, j, k)];
                if is_degenerate_volume_f32(geom.owner_volume)
                    || is_degenerate_volume_f32(geom.neighbor_volume)
                {
                    continue;
                }
                let ctx = InteriorFaceFlux3dTyped {
                    primitives: params.primitives,
                    mesh,
                    eos: params.eos,
                    config: params.config,
                    normal: vec3_from_f32(geom.normal),
                };
                let flux = flux_at_j_face_typed(&ctx, i, j, k)?;
                accumulate_interior_face_typed(
                    residual,
                    geom.owner,
                    geom.neighbor,
                    &flux,
                    f64::from(geom.area),
                    f64::from(geom.owner_volume),
                    f64::from(geom.neighbor_volume),
                )?;
            }
        }
    }
    Ok(())
}

fn assemble_k_faces_muscl_f32(
    mesh: &StructuredMesh3d,
    cache: &StructuredFaceCacheF32,
    residual: &mut ConservedResidualT<f32>,
    params: &InviscidAssembly3dTypedParams<'_, f32>,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let geom = &cache.k_faces[k_face_cache_index(nx, ny, i, j, k)];
                if is_degenerate_volume_f32(geom.owner_volume)
                    || is_degenerate_volume_f32(geom.neighbor_volume)
                {
                    continue;
                }
                let ctx = InteriorFaceFlux3dTyped {
                    primitives: params.primitives,
                    mesh,
                    eos: params.eos,
                    config: params.config,
                    normal: vec3_from_f32(geom.normal),
                };
                let flux = flux_at_k_face_typed(&ctx, i, j, k)?;
                accumulate_interior_face_typed(
                    residual,
                    geom.owner,
                    geom.neighbor,
                    &flux,
                    f64::from(geom.area),
                    f64::from(geom.owner_volume),
                    f64::from(geom.neighbor_volume),
                )?;
            }
        }
    }
    Ok(())
}
