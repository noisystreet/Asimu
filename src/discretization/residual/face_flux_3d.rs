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

/// 3D 边界面无粘通量输入。
///
/// `exterior` 是边界模型给出的面外侧状态；它可以来自 ghost cell、特征边界条件
/// 或其他边界 Riemann 模型。
pub struct BoundaryInviscidFluxInput<'a> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub primitives: &'a PrimitiveFields,
    pub eos: &'a IdealGasEoS,
    pub config: &'a crate::discretization::InviscidFluxConfig,
    pub min_pressure: Real,
    pub face: crate::core::FaceId,
    pub exterior: ConservedState,
}

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

pub fn inviscid_boundary_face_flux(input: BoundaryInviscidFluxInput<'_>) -> Result<InviscidFlux> {
    let geom = input.mesh.face_geometry_3d(input.face)?;
    let ctx = BoundaryFaceFlux3d {
        primitives: input.primitives,
        mesh: input.structured,
        eos: input.eos,
        config: input.config,
        min_pressure: input.min_pressure,
    };
    flux_at_boundary_face(&ctx, input.face, input.exterior, geom.normal)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::InviscidFluxConfig;
    use crate::field::ConservedFields;
    use crate::mesh::{BoundaryMesh, StructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS};

    #[test]
    fn boundary_flux_input_computes_finite_flux() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("primitives");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let face = mesh.resolve_logical_boundary("i_max").expect("faces")[0];
        let owner = mesh.face_owner(face).expect("owner").index() as usize;
        let exterior = fields.cell_state(owner).expect("owner state");
        let config = InviscidFluxConfig::roe_first_order();
        let flux = inviscid_boundary_face_flux(BoundaryInviscidFluxInput {
            mesh: &mesh,
            structured: &mesh,
            primitives: &primitives,
            eos: &eos,
            config: &config,
            min_pressure: 1.0e-6,
            face,
            exterior,
        })
        .expect("flux");
        assert!(flux.mass.is_finite());
        assert!(flux.energy.is_finite());
    }
}
