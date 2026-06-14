//! 非结构网格二阶线性重构（f32 路径；几何/限制器样本距离仍 f64）。

use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient_typed::InviscidPrimitiveGradientsT;
use crate::discretization::reconstruction::InterfacePrimitiveStates;
use crate::discretization::unstructured_face_cache::{
    GradientLimiterSampleKind, UnstructuredBoundaryFace, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache,
};
use crate::discretization::unstructured_limiter::UnstructuredGradientLimiter;
use crate::discretization::viscous_boundary_f32::{
    PrimitiveStateF32, primitive_state_f32_from_real, primitive_state_f32_to_real,
};
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed};
use crate::physics::IdealGasEoS;

/// f32 面左右原始变量态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfacePrimitiveStatesF32 {
    pub left: PrimitiveStateF32,
    pub right: PrimitiveStateF32,
}

#[must_use]
pub fn interface_primitive_states_f32_to_f64(
    iface: InterfacePrimitiveStatesF32,
) -> InterfacePrimitiveStates {
    InterfacePrimitiveStates {
        left: primitive_state_f32_to_real(iface.left),
        right: primitive_state_f32_to_real(iface.right),
    }
}

/// 非结构二阶线性重构面重构共享上下文（f32）。
#[derive(Debug, Clone, Copy)]
pub struct UnstructuredLinearReconstructionCtxF32<'a> {
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    pub limiter: UnstructuredGradientLimiter,
}

/// 非结构内部面原始变量重构（f32）。
pub fn reconstruct_unstructured_interior_face_f32(
    face: &UnstructuredInteriorFace,
    ctx: UnstructuredLinearReconstructionCtxF32<'_>,
    owner_grad: InviscidPrimitiveGradientsT<f32>,
    neighbor_grad: InviscidPrimitiveGradientsT<f32>,
) -> Result<InterfacePrimitiveStatesF32> {
    let owner = face.owner;
    let neighbor = face.neighbor;
    let owner_prim = cell_primitive_f32(ctx.primitives, owner);
    let neighbor_prim = cell_primitive_f32(ctx.primitives, neighbor);
    let left =
        extrapolate_cell_primitive_f32(owner, &owner_prim, owner_grad, face.dr_owner_to_face, ctx)?;
    let right = extrapolate_cell_primitive_f32(
        neighbor,
        &neighbor_prim,
        neighbor_grad,
        face.dr_neighbor_to_face,
        ctx,
    )?;
    Ok(InterfacePrimitiveStatesF32 { left, right })
}

/// 非结构边界面 owner 侧外推（f32）；ghost 侧取 BC 原始变量。
pub fn reconstruct_unstructured_boundary_face_f32(
    face: &UnstructuredBoundaryFace,
    ctx: UnstructuredLinearReconstructionCtxF32<'_>,
    owner_grad: InviscidPrimitiveGradientsT<f32>,
) -> Result<InterfacePrimitiveStatesF32> {
    let owner = face.owner;
    let owner_prim = cell_primitive_f32(ctx.primitives, owner);
    let ghost = ctx.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "非结构边界面 FaceId({}) 缺少 ghost",
            face.face.index()
        ))
    })?;
    let ghost_prim_f64 =
        primitive_from_conserved_relaxed(ctx.eos, &ghost.conserved, ctx.min_pressure)?;
    let ghost_prim = primitive_state_f32_from_real(ghost_prim_f64);
    let left =
        extrapolate_cell_primitive_f32(owner, &owner_prim, owner_grad, face.dr_owner_to_face, ctx)?;
    Ok(InterfacePrimitiveStatesF32 {
        left,
        right: ghost_prim,
    })
}

#[must_use]
fn cell_primitive_f32(primitives: &PrimitiveFieldsT<f32>, cell: usize) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
        density: primitives.density.values()[cell],
        velocity: [
            primitives.velocity_x.values()[cell],
            primitives.velocity_y.values()[cell],
            primitives.velocity_z.values()[cell],
        ],
        pressure: primitives.pressure.values()[cell],
        temperature: 0.0,
    }
}

fn extrapolate_cell_primitive_f32(
    cell: usize,
    prim: &PrimitiveStateF32,
    grad: InviscidPrimitiveGradientsT<f32>,
    dr_to_face: Vector3,
    ctx: UnstructuredLinearReconstructionCtxF32<'_>,
) -> Result<PrimitiveStateF32> {
    let samples = sample_phi_extrema_and_list_f32(
        cell,
        ctx.mesh_cache,
        ctx.primitives,
        ctx.ghosts,
        ctx.eos,
        ctx.min_pressure,
    )?;
    let min_p = ctx.min_pressure as f32;
    let psi_rho = limit_cell_gradient_factor_f32(
        ctx.limiter,
        prim.density,
        samples.phi_min.density,
        samples.phi_max.density,
        grad.drho,
        &samples.density,
    );
    let psi_u = limit_cell_gradient_factor_f32(
        ctx.limiter,
        prim.velocity[0],
        samples.phi_min.velocity[0],
        samples.phi_max.velocity[0],
        grad.du,
        &samples.velocity_u,
    );
    let psi_v = limit_cell_gradient_factor_f32(
        ctx.limiter,
        prim.velocity[1],
        samples.phi_min.velocity[1],
        samples.phi_max.velocity[1],
        grad.dv,
        &samples.velocity_v,
    );
    let psi_w = limit_cell_gradient_factor_f32(
        ctx.limiter,
        prim.velocity[2],
        samples.phi_min.velocity[2],
        samples.phi_max.velocity[2],
        grad.dw,
        &samples.velocity_w,
    );
    let psi_p = limit_cell_gradient_factor_f32(
        ctx.limiter,
        prim.pressure,
        samples.phi_min.pressure,
        samples.phi_max.pressure,
        grad.dp,
        &samples.pressure,
    );
    Ok(PrimitiveStateF32 {
        density: (prim.density + psi_rho * dot_f32(grad.drho, dr_to_face)).max(1.0e-30_f32),
        velocity: [
            prim.velocity[0] + psi_u * dot_f32(grad.du, dr_to_face),
            prim.velocity[1] + psi_v * dot_f32(grad.dv, dr_to_face),
            prim.velocity[2] + psi_w * dot_f32(grad.dw, dr_to_face),
        ],
        pressure: (prim.pressure + psi_p * dot_f32(grad.dp, dr_to_face)).max(min_p),
        temperature: prim.temperature,
    })
}

struct SamplePhiDataF32 {
    phi_min: PrimitiveStateF32,
    phi_max: PrimitiveStateF32,
    density: Vec<([Real; 3], f32)>,
    pressure: Vec<([Real; 3], f32)>,
    velocity_u: Vec<([Real; 3], f32)>,
    velocity_v: Vec<([Real; 3], f32)>,
    velocity_w: Vec<([Real; 3], f32)>,
}

fn sample_phi_extrema_and_list_f32(
    cell: usize,
    mesh_cache: &UnstructuredSolverMeshCache,
    primitives: &PrimitiveFieldsT<f32>,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<SamplePhiDataF32> {
    let center = cell_primitive_f32(primitives, cell);
    let mut phi_min = center;
    let mut phi_max = center;
    let mut density = Vec::new();
    let mut pressure = Vec::new();
    let mut velocity_u = Vec::new();
    let mut velocity_v = Vec::new();
    let mut velocity_w = Vec::new();
    for sample in &mesh_cache.cell_gradient_samples[cell] {
        let phi = sample_phi_f32(sample, mesh_cache, primitives, ghosts, eos, min_pressure)?;
        phi_min = min_primitive_f32(&phi_min, &phi);
        phi_max = max_primitive_f32(&phi_max, &phi);
        let dr = [sample.dr.x, sample.dr.y, sample.dr.z];
        density.push((dr, phi.density));
        pressure.push((dr, phi.pressure));
        velocity_u.push((dr, phi.velocity[0]));
        velocity_v.push((dr, phi.velocity[1]));
        velocity_w.push((dr, phi.velocity[2]));
    }
    Ok(SamplePhiDataF32 {
        phi_min,
        phi_max,
        density,
        pressure,
        velocity_u,
        velocity_v,
        velocity_w,
    })
}

fn sample_phi_f32(
    sample: &crate::discretization::unstructured_face_cache::GradientLimiterSample,
    mesh_cache: &UnstructuredSolverMeshCache,
    primitives: &PrimitiveFieldsT<f32>,
    ghosts: &BoundaryGhostBuffer,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<PrimitiveStateF32> {
    match sample.kind {
        GradientLimiterSampleKind::NeighborCell(idx) => Ok(cell_primitive_f32(primitives, idx)),
        GradientLimiterSampleKind::Boundary(bidx) => {
            let bface = &mesh_cache.face_topology.boundary[bidx];
            let ghost = ghosts.get_face(bface.face).ok_or_else(|| {
                AsimuError::Boundary(format!(
                    "非结构限制器样本 FaceId({}) 缺少 ghost",
                    bface.face.index()
                ))
            })?;
            let prim = primitive_from_conserved_relaxed(eos, &ghost.conserved, min_pressure)?;
            Ok(primitive_state_f32_from_real(prim))
        }
    }
}

fn min_primitive_f32(a: &PrimitiveStateF32, b: &PrimitiveStateF32) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
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

fn max_primitive_f32(a: &PrimitiveStateF32, b: &PrimitiveStateF32) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
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

#[must_use]
fn dot_f32(grad: [f32; 3], dr: Vector3) -> f32 {
    grad[0] * dr.x as f32 + grad[1] * dr.y as f32 + grad[2] * dr.z as f32
}

#[must_use]
fn limit_cell_gradient_factor_f32(
    limiter: UnstructuredGradientLimiter,
    phi_i: f32,
    phi_min: f32,
    phi_max: f32,
    grad: [f32; 3],
    samples: &[([Real; 3], f32)],
) -> f32 {
    use crate::discretization::unstructured_limiter::{
        barth_jespersen_sample_factor, venkatakrishnan_sample_factor,
    };
    let phi_i_r = phi_i as Real;
    let grad_r = [grad[0] as Real, grad[1] as Real, grad[2] as Real];
    let mut psi = 1.0_f32;
    for &(dr, phi_m) in samples {
        let grad_dot_dr = grad_r[0] * dr[0] + grad_r[1] * dr[1] + grad_r[2] * dr[2];
        let sample_psi = match limiter {
            UnstructuredGradientLimiter::BarthJespersen => barth_jespersen_sample_factor(
                phi_i_r,
                phi_min as Real,
                phi_max as Real,
                grad_dot_dr,
            ),
            UnstructuredGradientLimiter::Venkatakrishnan => {
                venkatakrishnan_sample_factor(phi_i_r, phi_m as Real, grad_dot_dr)
            }
        };
        psi = psi.min(sample_psi as f32);
    }
    psi
}
