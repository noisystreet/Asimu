//! f32 无粘物理通量、scatter 与守恒态辅助。

use crate::core::{Real, Vector3};
use crate::discretization::inviscid::InviscidFlux;
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::error::{AsimuError, Result};
use crate::field::ConservedResidualT;
use crate::physics::IdealGasEoS;

/// f32 守恒态（面通量局部）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ConservedStateF32 {
    pub density: f32,
    pub momentum: [f32; 3],
    pub total_energy: f32,
}

/// f32 面法向无粘通量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InviscidFluxF32 {
    pub mass: f32,
    pub momentum: [f32; 3],
    pub energy: f32,
}

/// scatter 阶段内面几何（f32 预存 RHS 缩放）。
#[derive(Debug, Clone, Copy)]
pub struct InteriorInviscidScatterGeomF32 {
    pub owner: usize,
    pub neighbor: usize,
    pub owner_scale: f32,
    pub neighbor_scale: f32,
}

#[must_use]
pub(crate) fn inviscid_flux_f32_to_real(flux: InviscidFluxF32) -> InviscidFlux {
    InviscidFlux {
        mass: flux.mass as Real,
        momentum: [
            flux.momentum[0] as Real,
            flux.momentum[1] as Real,
            flux.momentum[2] as Real,
        ],
        energy: flux.energy as Real,
    }
}

/// 将 f32 无粘通量 scatter 到 typed 残差（全 f32，无 Real 桥接）。
#[inline(always)]
pub fn scatter_fused_interior_inviscid_face_f32(
    residual: &mut ConservedResidualT<f32>,
    geom: &InteriorInviscidScatterGeomF32,
    flux: &InviscidFluxF32,
) {
    let owner = geom.owner;
    let neighbor = geom.neighbor;
    let os = geom.owner_scale;
    let ns = geom.neighbor_scale;
    residual.density.values_mut()[owner] += os * flux.mass;
    residual.momentum_x.values_mut()[owner] += os * flux.momentum[0];
    residual.momentum_y.values_mut()[owner] += os * flux.momentum[1];
    residual.momentum_z.values_mut()[owner] += os * flux.momentum[2];
    residual.total_energy.values_mut()[owner] += os * flux.energy;
    residual.density.values_mut()[neighbor] += ns * flux.mass;
    residual.momentum_x.values_mut()[neighbor] += ns * flux.momentum[0];
    residual.momentum_y.values_mut()[neighbor] += ns * flux.momentum[1];
    residual.momentum_z.values_mut()[neighbor] += ns * flux.momentum[2];
    residual.total_energy.values_mut()[neighbor] += ns * flux.energy;
}

/// 边界面无粘 scatter（f32）。
#[inline(always)]
pub fn scatter_fused_boundary_inviscid_face_f32(
    residual: &mut ConservedResidualT<f32>,
    owner: usize,
    owner_scale: f32,
    flux: &InviscidFluxF32,
) {
    residual.density.values_mut()[owner] += owner_scale * flux.mass;
    residual.momentum_x.values_mut()[owner] += owner_scale * flux.momentum[0];
    residual.momentum_y.values_mut()[owner] += owner_scale * flux.momentum[1];
    residual.momentum_z.values_mut()[owner] += owner_scale * flux.momentum[2];
    residual.total_energy.values_mut()[owner] += owner_scale * flux.energy;
}

pub(crate) fn conserved_from_primitive_f32(
    eos: &IdealGasEoS,
    prim: &PrimitiveStateF32,
) -> Result<ConservedStateF32> {
    if prim.density <= 0.0 || prim.pressure <= 0.0 {
        return Err(AsimuError::Field(
            "f32 原始变量须为正密度与压力".to_string(),
        ));
    }
    let gamma = eos.gamma as f32;
    let rho = prim.density;
    let u2 = prim.velocity[0] * prim.velocity[0]
        + prim.velocity[1] * prim.velocity[1]
        + prim.velocity[2] * prim.velocity[2];
    let internal = prim.pressure / ((gamma - 1.0) * rho);
    Ok(ConservedStateF32 {
        density: rho,
        momentum: [
            rho * prim.velocity[0],
            rho * prim.velocity[1],
            rho * prim.velocity[2],
        ],
        total_energy: rho * internal + 0.5 * rho * u2,
    })
}

#[must_use]
pub(crate) fn physical_inviscid_flux_f32(
    cons: &ConservedStateF32,
    prim: &PrimitiveStateF32,
    normal: Vector3,
) -> InviscidFluxF32 {
    let nx = normal.x as f32;
    let ny = normal.y as f32;
    let nz = normal.z as f32;
    let un = prim.velocity[0] * nx + prim.velocity[1] * ny + prim.velocity[2] * nz;
    let p = prim.pressure;
    let rho = prim.density;
    let u = prim.velocity;
    InviscidFluxF32 {
        mass: rho * un,
        momentum: [
            rho * un * u[0] + p * nx,
            rho * un * u[1] + p * ny,
            rho * un * u[2] + p * nz,
        ],
        energy: (cons.total_energy + p) * un,
    }
}

#[inline]
pub(crate) fn normalize_face_normal_f32(normal: Vector3) -> Result<Vector3> {
    let mag = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
    if mag < Real::EPSILON {
        return Err(AsimuError::Mesh("面法向不能为零向量".to_string()));
    }
    Ok(Vector3::new(normal.x / mag, normal.y / mag, normal.z / mag))
}

#[must_use]
pub(crate) fn face_tangent_basis_f32(normal: Vector3) -> (Vector3, Vector3) {
    let reference = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let t1 = cross_f32(normal, reference);
    let t1 = normalize_unchecked_f32(t1);
    let t2 = cross_f32(normal, t1);
    (t1, normalize_unchecked_f32(t2))
}

fn cross_f32(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn normalize_unchecked_f32(v: Vector3) -> Vector3 {
    let mag = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    Vector3::new(v.x / mag, v.y / mag, v.z / mag)
}
