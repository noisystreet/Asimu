//! 3D 结构化网格 MUSCL 宽模板（沿面法向 i/j/k 四点；边界面含 ghost 宽模板）。

use crate::core::{FaceId, Vector3};
use crate::discretization::{
    FaceFluxInput, InviscidFlux, InviscidFluxConfig, ReconstructionKind, face_inviscid_flux,
};
use crate::error::Result;
use crate::field::ConservedFields;
use crate::mesh::{LogicalFace3d, StructuredMesh3d};
use crate::physics::{ConservedState, IdealGasEoS};

struct LoadedStencil4 {
    owner: ConservedState,
    neighbor: ConservedState,
    left_of_owner: Option<ConservedState>,
    right_of_neighbor: Option<ConservedState>,
}

impl LoadedStencil4 {
    fn face_input(&self) -> FaceFluxInput<'_> {
        FaceFluxInput {
            owner: &self.owner,
            neighbor: &self.neighbor,
            left_of_owner: self.left_of_owner.as_ref(),
            right_of_neighbor: self.right_of_neighbor.as_ref(),
        }
    }
}

pub(crate) struct InteriorFaceFlux3d<'a> {
    pub fields: &'a ConservedFields,
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub normal: Vector3,
}

pub(crate) struct BoundaryFaceFlux3d<'a> {
    pub fields: &'a ConservedFields,
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
}

fn load_stencil(
    fields: &ConservedFields,
    owner_idx: usize,
    neighbor_idx: usize,
    left_idx: Option<usize>,
    right_idx: Option<usize>,
) -> Result<LoadedStencil4> {
    let owner = fields.cell_state(owner_idx)?;
    let neighbor = fields.cell_state(neighbor_idx)?;
    let left_of_owner = match left_idx {
        Some(idx) => Some(fields.cell_state(idx)?),
        None => None,
    };
    let right_of_neighbor = match right_idx {
        Some(idx) => Some(fields.cell_state(idx)?),
        None => None,
    };
    Ok(LoadedStencil4 {
        owner,
        neighbor,
        left_of_owner,
        right_of_neighbor,
    })
}

fn flux_from_stencil(
    stencil: LoadedStencil4,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    let input = match config.reconstruction {
        ReconstructionKind::FirstOrder => {
            FaceFluxInput::first_order(&stencil.owner, &stencil.neighbor)
        }
        ReconstructionKind::Muscl => stencil.face_input(),
    };
    face_inviscid_flux(input, normal, eos, config)
}

fn flux_from_indices(
    ctx: &InteriorFaceFlux3d<'_>,
    owner_idx: usize,
    neighbor_idx: usize,
    left_idx: Option<usize>,
    right_idx: Option<usize>,
) -> Result<InviscidFlux> {
    let stencil = load_stencil(ctx.fields, owner_idx, neighbor_idx, left_idx, right_idx)?;
    flux_from_stencil(stencil, ctx.normal, ctx.eos, ctx.config)
}

fn load_boundary_stencil(
    fields: &ConservedFields,
    owner_idx: usize,
    ghost: ConservedState,
    left_idx: Option<usize>,
    right_idx: Option<usize>,
) -> Result<LoadedStencil4> {
    let owner = fields.cell_state(owner_idx)?;
    let left_of_owner = match left_idx {
        Some(idx) => Some(fields.cell_state(idx)?),
        None => None,
    };
    let right_of_neighbor = match right_idx {
        Some(idx) => Some(fields.cell_state(idx)?),
        None => None,
    };
    Ok(LoadedStencil4 {
        owner,
        neighbor: ghost,
        left_of_owner,
        right_of_neighbor,
    })
}

fn boundary_wide_indices(
    logical: LogicalFace3d,
    i: usize,
    j: usize,
    k: usize,
    mesh: &StructuredMesh3d,
) -> (Option<usize>, Option<usize>) {
    match logical {
        LogicalFace3d::IMin => {
            let left = (mesh.nx > 1).then(|| mesh.cell_index(1, j, k));
            let right = (mesh.nx > 2).then(|| mesh.cell_index(2, j, k));
            (left, right)
        }
        LogicalFace3d::IMax => {
            let left = (mesh.nx > 1).then(|| mesh.cell_index(mesh.nx - 2, j, k));
            let right = (mesh.nx > 2).then(|| mesh.cell_index(mesh.nx - 3, j, k));
            (left, right)
        }
        LogicalFace3d::JMin => {
            let left = (mesh.ny > 1).then(|| mesh.cell_index(i, 1, k));
            let right = (mesh.ny > 2).then(|| mesh.cell_index(i, 2, k));
            (left, right)
        }
        LogicalFace3d::JMax => {
            let left = (mesh.ny > 1).then(|| mesh.cell_index(i, mesh.ny - 2, k));
            let right = (mesh.ny > 2).then(|| mesh.cell_index(i, mesh.ny - 3, k));
            (left, right)
        }
        LogicalFace3d::KMin => {
            let left = (mesh.nz > 1).then(|| mesh.cell_index(i, j, 1));
            let right = (mesh.nz > 2).then(|| mesh.cell_index(i, j, 2));
            (left, right)
        }
        LogicalFace3d::KMax => {
            let left = (mesh.nz > 1).then(|| mesh.cell_index(i, j, mesh.nz - 2));
            let right = (mesh.nz > 2).then(|| mesh.cell_index(i, j, mesh.nz - 3));
            (left, right)
        }
    }
}

pub(crate) fn flux_at_boundary_face(
    ctx: &BoundaryFaceFlux3d<'_>,
    face: FaceId,
    ghost: ConservedState,
    normal: Vector3,
) -> Result<InviscidFlux> {
    let (logical, local) = LogicalFace3d::decode(face)?;
    let (i, j, k) = ctx.mesh.face_ij(logical, local)?;
    let owner_idx = ctx.mesh.cell_index(i, j, k);
    let (left_idx, right_idx) = boundary_wide_indices(logical, i, j, k, ctx.mesh);
    let stencil = load_boundary_stencil(ctx.fields, owner_idx, ghost, left_idx, right_idx)?;
    flux_from_stencil(stencil, normal, ctx.eos, ctx.config)
}

pub(crate) fn flux_at_i_face(
    ctx: &InteriorFaceFlux3d<'_>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let neighbor = ctx.mesh.cell_index(i + 1, j, k);
    let left = (i > 0).then(|| ctx.mesh.cell_index(i - 1, j, k));
    let right = (i + 2 < ctx.mesh.nx).then(|| ctx.mesh.cell_index(i + 2, j, k));
    flux_from_indices(ctx, owner, neighbor, left, right)
}

pub(crate) fn flux_at_j_face(
    ctx: &InteriorFaceFlux3d<'_>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let neighbor = ctx.mesh.cell_index(i, j + 1, k);
    let left = (j > 0).then(|| ctx.mesh.cell_index(i, j - 1, k));
    let right = (j + 2 < ctx.mesh.ny).then(|| ctx.mesh.cell_index(i, j + 2, k));
    flux_from_indices(ctx, owner, neighbor, left, right)
}

pub(crate) fn flux_at_k_face(
    ctx: &InteriorFaceFlux3d<'_>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let neighbor = ctx.mesh.cell_index(i, j, k + 1);
    let left = (k > 0).then(|| ctx.mesh.cell_index(i, j, k - 1));
    let right = (k + 2 < ctx.mesh.nz).then(|| ctx.mesh.cell_index(i, j, k + 2));
    flux_from_indices(ctx, owner, neighbor, left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::InviscidFluxConfig;
    use crate::mesh::{BoundaryMesh, BoundaryMesh3d};
    use crate::physics::FreestreamParams;

    #[test]
    fn boundary_imin_uses_ghost_and_interior_wide_stencil() {
        let mesh = StructuredMesh3d::uniform_box("box", 4, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &FreestreamParams::default())
                .expect("fields");
        let face = mesh.resolve_logical_boundary("i_min").expect("faces")[0];
        let owner = mesh.face_owner(face).expect("owner");
        let ghost = fields.cell_state(owner.index() as usize).expect("ghost");
        let ctx = BoundaryFaceFlux3d {
            fields: &fields,
            mesh: &mesh,
            eos: &eos,
            config: &InviscidFluxConfig::muscl_hllc(),
        };
        let geom = mesh.face_geometry_3d(face).expect("geom");
        let flux = flux_at_boundary_face(&ctx, face, ghost, geom.normal).expect("flux");
        assert!(flux.mass.abs() < 1.0e-10);
    }

    #[test]
    fn i_face_stencil_uses_x_neighbors_only() {
        let mesh = StructuredMesh3d::uniform_box("box", 4, 3, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &FreestreamParams::default())
                .expect("fields");
        let config = InviscidFluxConfig::muscl_hllc();
        let ctx = InteriorFaceFlux3d {
            fields: &fields,
            mesh: &mesh,
            eos: &eos,
            config: &config,
            normal: Vector3::new(1.0, 0.0, 0.0),
        };
        let flux = flux_at_i_face(&ctx, 1, 1, 0).expect("flux");
        assert!(flux.mass.abs() < 1.0e-10);
    }
}
