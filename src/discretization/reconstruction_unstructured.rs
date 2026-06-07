//! 非结构网格二阶线性重构（IDWLS 梯度外推 + 梯度限制器）。
//!
//! 理论：[`docs/theory/unstructured_fvm.md`](../../docs/theory/unstructured_fvm.md)、
//! [`docs/adr/0012`](../../docs/adr/0012-unstructured-gradient-limiters.md)

use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::InviscidPrimitiveGradients;
use crate::discretization::reconstruction::InterfacePrimitiveStates;
use crate::discretization::unstructured_face_cache::{
    GradientLimiterSampleKind, UnstructuredBoundaryFace, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache,
};
use crate::discretization::unstructured_limiter::{
    UnstructuredGradientLimiter, limit_cell_gradient_factor,
};
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::physics::{IdealGasEoS, PrimitiveState};

/// 非结构 MUSCL 面重构共享上下文（避免热路径参数过多）。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredMusclReconstructionCtx<'a> {
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub primitives: &'a PrimitiveFields,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    pub limiter: UnstructuredGradientLimiter,
}

/// 非结构内部面原始变量重构。
pub fn reconstruct_unstructured_interior_face(
    face: &UnstructuredInteriorFace,
    ctx: UnstructuredMusclReconstructionCtx<'_>,
    owner_grad: InviscidPrimitiveGradients,
    neighbor_grad: InviscidPrimitiveGradients,
) -> Result<InterfacePrimitiveStates> {
    let owner = face.owner;
    let neighbor = face.neighbor;
    let owner_prim = ctx.primitives.cell_primitive(owner);
    let neighbor_prim = ctx.primitives.cell_primitive(neighbor);
    let left =
        extrapolate_cell_primitive(owner, &owner_prim, owner_grad, face.dr_owner_to_face, ctx)?;
    let right = extrapolate_cell_primitive(
        neighbor,
        &neighbor_prim,
        neighbor_grad,
        face.dr_neighbor_to_face,
        ctx,
    )?;
    Ok(InterfacePrimitiveStates { left, right })
}

/// 非结构边界面 owner 侧外推；ghost 侧取 BC 原始变量。
pub fn reconstruct_unstructured_boundary_face(
    face: &UnstructuredBoundaryFace,
    ctx: UnstructuredMusclReconstructionCtx<'_>,
    owner_grad: InviscidPrimitiveGradients,
) -> Result<InterfacePrimitiveStates> {
    let owner = face.owner;
    let owner_prim = ctx.primitives.cell_primitive(owner);
    let ghost = ctx.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "非结构边界面 FaceId({}) 缺少 ghost",
            face.face.index()
        ))
    })?;
    let ghost_prim = primitive_from_conserved_relaxed(ctx.eos, &ghost.conserved, ctx.min_pressure)?;
    let left =
        extrapolate_cell_primitive(owner, &owner_prim, owner_grad, face.dr_owner_to_face, ctx)?;
    Ok(InterfacePrimitiveStates {
        left,
        right: ghost_prim,
    })
}

fn extrapolate_cell_primitive(
    cell: usize,
    prim: &PrimitiveState,
    grad: InviscidPrimitiveGradients,
    dr_to_face: Vector3,
    ctx: UnstructuredMusclReconstructionCtx<'_>,
) -> Result<PrimitiveState> {
    let samples = sample_phi_extrema_and_list(
        cell,
        ctx.mesh_cache,
        ctx.primitives,
        ctx.ghosts,
        ctx.eos,
        ctx.min_pressure,
    )?;
    let psi_rho = limit_cell_gradient_factor(
        ctx.limiter,
        prim.density,
        samples.phi_min.density,
        samples.phi_max.density,
        grad.drho,
        &samples.density,
    );
    let psi_u = limit_cell_gradient_factor(
        ctx.limiter,
        prim.velocity[0],
        samples.phi_min.velocity[0],
        samples.phi_max.velocity[0],
        grad.du,
        &samples.velocity_u,
    );
    let psi_v = limit_cell_gradient_factor(
        ctx.limiter,
        prim.velocity[1],
        samples.phi_min.velocity[1],
        samples.phi_max.velocity[1],
        grad.dv,
        &samples.velocity_v,
    );
    let psi_w = limit_cell_gradient_factor(
        ctx.limiter,
        prim.velocity[2],
        samples.phi_min.velocity[2],
        samples.phi_max.velocity[2],
        grad.dw,
        &samples.velocity_w,
    );
    let psi_p = limit_cell_gradient_factor(
        ctx.limiter,
        prim.pressure,
        samples.phi_min.pressure,
        samples.phi_max.pressure,
        grad.dp,
        &samples.pressure,
    );
    Ok(PrimitiveState {
        density: (prim.density + psi_rho * dot(grad.drho, dr_to_face)).max(1.0e-30),
        velocity: [
            prim.velocity[0] + psi_u * dot(grad.du, dr_to_face),
            prim.velocity[1] + psi_v * dot(grad.dv, dr_to_face),
            prim.velocity[2] + psi_w * dot(grad.dw, dr_to_face),
        ],
        pressure: (prim.pressure + psi_p * dot(grad.dp, dr_to_face)).max(ctx.min_pressure),
        temperature: prim.temperature,
    })
}

struct SamplePhiData {
    phi_min: PrimitiveState,
    phi_max: PrimitiveState,
    density: Vec<([Real; 3], Real)>,
    pressure: Vec<([Real; 3], Real)>,
    velocity_u: Vec<([Real; 3], Real)>,
    velocity_v: Vec<([Real; 3], Real)>,
    velocity_w: Vec<([Real; 3], Real)>,
}

fn sample_phi_extrema_and_list(
    cell: usize,
    mesh_cache: &UnstructuredSolverMeshCache,
    primitives: &PrimitiveFields,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<SamplePhiData> {
    let center = primitives.cell_primitive(cell);
    let mut phi_min = center;
    let mut phi_max = center;
    let mut density = Vec::new();
    let mut pressure = Vec::new();
    let mut velocity_u = Vec::new();
    let mut velocity_v = Vec::new();
    let mut velocity_w = Vec::new();
    for sample in &mesh_cache.cell_gradient_samples[cell] {
        let phi = sample_phi(sample, mesh_cache, primitives, ghosts, eos, min_pressure)?;
        phi_min = min_primitive(&phi_min, &phi);
        phi_max = max_primitive(&phi_max, &phi);
        let dr = [sample.dr.x, sample.dr.y, sample.dr.z];
        density.push((dr, phi.density));
        pressure.push((dr, phi.pressure));
        velocity_u.push((dr, phi.velocity[0]));
        velocity_v.push((dr, phi.velocity[1]));
        velocity_w.push((dr, phi.velocity[2]));
    }
    Ok(SamplePhiData {
        phi_min,
        phi_max,
        density,
        pressure,
        velocity_u,
        velocity_v,
        velocity_w,
    })
}

fn sample_phi(
    sample: &crate::discretization::unstructured_face_cache::GradientLimiterSample,
    mesh_cache: &UnstructuredSolverMeshCache,
    primitives: &PrimitiveFields,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<PrimitiveState> {
    match sample.kind {
        GradientLimiterSampleKind::NeighborCell(idx) => Ok(primitives.cell_primitive(idx)),
        GradientLimiterSampleKind::Boundary(bidx) => {
            let bface = &mesh_cache.face_topology.boundary[bidx];
            let ghost = ghosts.get_face(bface.face).ok_or_else(|| {
                AsimuError::Boundary(format!(
                    "非结构限制器样本 FaceId({}) 缺少 ghost",
                    bface.face.index()
                ))
            })?;
            primitive_from_conserved_relaxed(eos, &ghost.conserved, min_pressure)
        }
    }
}

fn min_primitive(a: &PrimitiveState, b: &PrimitiveState) -> PrimitiveState {
    PrimitiveState {
        density: a.density.min(b.density),
        velocity: [
            a.velocity[0].min(b.velocity[0]),
            a.velocity[1].min(b.velocity[1]),
            a.velocity[2].min(b.velocity[2]),
        ],
        pressure: a.pressure.min(b.pressure),
        temperature: a.temperature,
    }
}

fn max_primitive(a: &PrimitiveState, b: &PrimitiveState) -> PrimitiveState {
    PrimitiveState {
        density: a.density.max(b.density),
        velocity: [
            a.velocity[0].max(b.velocity[0]),
            a.velocity[1].max(b.velocity[1]),
            a.velocity[2].max(b.velocity[2]),
        ],
        pressure: a.pressure.max(b.pressure),
        temperature: a.temperature,
    }
}

fn dot(grad: [Real; 3], dr: Vector3) -> Real {
    grad[0] * dr.x + grad[1] * dr.y + grad[2] * dr.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::approx_eq;
    use crate::discretization::GradientFields;
    use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS};

    #[test]
    fn uniform_freestream_muscl_reconstruction_matches_cell_values() {
        let mesh = UnstructuredMesh3d::new(
            "two_tets",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![
                UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell"),
                UnstructuredCell::new(CellKind::Tet, vec![1, 2, 3, 4]).expect("cell"),
            ],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|f| crate::core::FaceId(f as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces.clone(),
            BoundaryKind::Farfield {
                mach: 0.0,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &FreestreamParams::default())
                .expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-8)
            .expect("fill");
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        for &face in &faces {
            ghosts.insert_face(
                face,
                crate::discretization::GhostCellState { conserved: state },
            );
        }
        let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let input = crate::discretization::UnstructuredGradientLsqInput {
            mesh: &mesh,
            mesh_cache: &cache,
            primitives: &primitives,
            eos: &eos,
            ghosts: &ghosts,
            min_pressure: 1.0e-8,
            viscous: None,
        };
        let mut scratch = crate::discretization::UnstructuredGradientScratch::new(mesh.num_cells());
        crate::discretization::compute_unstructured_inviscid_muscl_gradients_idw_lsq(
            input,
            &mut gradients,
            &mut scratch,
        )
        .expect("grad");
        let face = &cache.face_topology.interior[0];
        let ctx = UnstructuredMusclReconstructionCtx {
            mesh_cache: &cache,
            primitives: &primitives,
            ghosts: &ghosts,
            eos: &eos,
            min_pressure: 1.0e-8,
            limiter: UnstructuredGradientLimiter::BarthJespersen,
        };
        let iface = reconstruct_unstructured_interior_face(
            face,
            ctx,
            gradients.inviscid_primitive_grad_at(face.owner),
            gradients.inviscid_primitive_grad_at(face.neighbor),
        )
        .expect("iface");
        let owner = primitives.cell_primitive(face.owner);
        assert!(approx_eq(iface.left.density, owner.density, 1.0e-10));
        assert!(approx_eq(iface.left.pressure, owner.pressure, 1.0e-6));
    }
}
