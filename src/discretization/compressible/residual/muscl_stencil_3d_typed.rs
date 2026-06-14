//! 3D 结构化 MUSCL typed 原变量加载（逐 lane 转 f64 后复用 `muscl_stencil_3d` 重构）。

use crate::core::{ComputeFloat, Real, Vector3};
use crate::discretization::{InviscidFlux, InviscidFluxConfig};
use crate::error::Result;
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed};
use crate::mesh::{LogicalFace3d, StructuredMesh3d};
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use super::muscl_stencil_3d::{LoadedPrimitiveStencil4, flux_from_primitive_stencil};

pub(crate) struct InteriorFaceFlux3dTyped<'a, T: ComputeFloat> {
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub normal: Vector3,
}

pub(crate) struct BoundaryFaceFlux3dTyped<'a, T: ComputeFloat> {
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub min_pressure: Real,
}

fn primitive_lane_as_f64<T: ComputeFloat>(
    primitives: &PrimitiveFieldsT<T>,
    index: usize,
) -> PrimitiveState {
    PrimitiveState {
        density: primitives.density.values()[index].to_real(),
        velocity: [
            primitives.velocity_x.values()[index].to_real(),
            primitives.velocity_y.values()[index].to_real(),
            primitives.velocity_z.values()[index].to_real(),
        ],
        pressure: primitives.pressure.values()[index].to_real(),
        temperature: 0.0,
    }
}

fn load_primitive_stencil_typed<T: ComputeFloat>(
    cache: &PrimitiveFieldsT<T>,
    owner_idx: usize,
    neighbor_idx: usize,
    left_idx: Option<usize>,
    right_idx: Option<usize>,
) -> LoadedPrimitiveStencil4 {
    LoadedPrimitiveStencil4 {
        owner: primitive_lane_as_f64(cache, owner_idx),
        neighbor: primitive_lane_as_f64(cache, neighbor_idx),
        left_of_owner: left_idx.map(|idx| primitive_lane_as_f64(cache, idx)),
        right_of_neighbor: right_idx.map(|idx| primitive_lane_as_f64(cache, idx)),
    }
}

fn flux_from_indices_typed<T: ComputeFloat>(
    ctx: &InteriorFaceFlux3dTyped<'_, T>,
    owner_idx: usize,
    neighbor_idx: usize,
    left_idx: Option<usize>,
    right_idx: Option<usize>,
) -> Result<InviscidFlux> {
    let stencil =
        load_primitive_stencil_typed(ctx.primitives, owner_idx, neighbor_idx, left_idx, right_idx);
    flux_from_primitive_stencil(stencil, ctx.normal, ctx.eos, ctx.config)
}

pub(crate) fn flux_at_i_face_typed<T: ComputeFloat>(
    ctx: &InteriorFaceFlux3dTyped<'_, T>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let neighbor = ctx.mesh.cell_index(i + 1, j, k);
    let left = (i > 0).then(|| ctx.mesh.cell_index(i - 1, j, k));
    let right = (i + 2 < ctx.mesh.nx).then(|| ctx.mesh.cell_index(i + 2, j, k));
    flux_from_indices_typed(ctx, owner, neighbor, left, right)
}

pub(crate) fn flux_at_j_face_typed<T: ComputeFloat>(
    ctx: &InteriorFaceFlux3dTyped<'_, T>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let neighbor = ctx.mesh.cell_index(i, j + 1, k);
    let left = (j > 0).then(|| ctx.mesh.cell_index(i, j - 1, k));
    let right = (j + 2 < ctx.mesh.ny).then(|| ctx.mesh.cell_index(i, j + 2, k));
    flux_from_indices_typed(ctx, owner, neighbor, left, right)
}

pub(crate) fn flux_at_k_face_typed<T: ComputeFloat>(
    ctx: &InteriorFaceFlux3dTyped<'_, T>,
    i: usize,
    j: usize,
    k: usize,
) -> Result<InviscidFlux> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let neighbor = ctx.mesh.cell_index(i, j, k + 1);
    let left = (k > 0).then(|| ctx.mesh.cell_index(i, j, k - 1));
    let right = (k + 2 < ctx.mesh.nz).then(|| ctx.mesh.cell_index(i, j, k + 2));
    flux_from_indices_typed(ctx, owner, neighbor, left, right)
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

pub(crate) fn flux_at_boundary_face_typed<T: ComputeFloat>(
    ctx: &BoundaryFaceFlux3dTyped<'_, T>,
    face: crate::core::FaceId,
    ghost: ConservedState,
    normal: Vector3,
) -> Result<InviscidFlux> {
    let (logical, local) = LogicalFace3d::decode(face)?;
    let (i, j, k) = ctx.mesh.face_ij(logical, local)?;
    let owner_idx = ctx.mesh.cell_index(i, j, k);
    let (left_idx, right_idx) = boundary_wide_indices(logical, i, j, k, ctx.mesh);
    let owner = primitive_lane_as_f64(ctx.primitives, owner_idx);
    let neighbor = primitive_from_conserved_relaxed(ctx.eos, &ghost, ctx.min_pressure)?;
    let left_of_owner = left_idx.map(|idx| primitive_lane_as_f64(ctx.primitives, idx));
    let right_of_neighbor = right_idx.map(|idx| primitive_lane_as_f64(ctx.primitives, idx));
    let stencil = LoadedPrimitiveStencil4 {
        owner,
        neighbor,
        left_of_owner,
        right_of_neighbor,
    };
    flux_from_primitive_stencil(stencil, normal, ctx.eos, ctx.config)
}
