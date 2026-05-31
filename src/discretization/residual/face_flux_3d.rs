//! 3D 内/边界面通量求值（供 LU-SGS 扫掠等复用）。

use crate::core::{Real, Vector3};
use crate::discretization::InviscidFlux;
use crate::error::Result;
use crate::field::PrimitiveFields;
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::{ConservedState, IdealGasEoS};

use super::InviscidAssembly3dParams;
use super::muscl_stencil_3d::{
    BoundaryFaceFlux3d, InteriorFaceFlux3d, flux_at_boundary_face, flux_at_i_face, flux_at_j_face,
    flux_at_k_face,
};

pub fn inviscid_i_face_flux(
    params: &InviscidAssembly3dParams<'_>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let normal = params.mesh.i_face_metric(i, j, k).normal;
    interior_flux(params, normal, |ctx| flux_at_i_face(ctx, i, j, k))
}

pub fn inviscid_j_face_flux(
    params: &InviscidAssembly3dParams<'_>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let normal = params.mesh.j_face_metric(i, j, k).normal;
    interior_flux(params, normal, |ctx| flux_at_j_face(ctx, i, j, k))
}

pub fn inviscid_k_face_flux(
    params: &InviscidAssembly3dParams<'_>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let normal = params.mesh.k_face_metric(i, j, k).normal;
    interior_flux(params, normal, |ctx| flux_at_k_face(ctx, i, j, k))
}

#[allow(clippy::too_many_arguments)]
pub fn inviscid_boundary_face_flux(
    mesh: &dyn BoundaryMesh3d,
    structured: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    config: &crate::discretization::InviscidFluxConfig,
    min_pressure: Real,
    face: crate::core::FaceId,
    ghost: ConservedState,
) -> Result<InviscidFlux> {
    let geom = mesh.face_geometry_3d(face)?;
    let ctx = BoundaryFaceFlux3d {
        primitives,
        mesh: structured,
        eos,
        config,
        min_pressure,
    };
    flux_at_boundary_face(&ctx, face, ghost, geom.normal)
}

fn interior_flux(
    params: &InviscidAssembly3dParams<'_>,
    normal: Vector3,
    flux_fn: impl FnOnce(&InteriorFaceFlux3d<'_>) -> Result<InviscidFlux>,
) -> Result<InviscidFlux> {
    let ctx = InteriorFaceFlux3d {
        primitives: params.primitives,
        mesh: params.mesh,
        eos: params.eos,
        config: params.config,
        normal,
    };
    flux_fn(&ctx)
}
