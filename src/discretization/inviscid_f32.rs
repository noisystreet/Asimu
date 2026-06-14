//! f32 无粘物理通量、scatter 与守恒态辅助。

use crate::core::Real;
use crate::discretization::inviscid::InviscidFlux;
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::error::{AsimuError, Result};
use crate::field::ConservedResidualT;
use crate::physics::IdealGasEoS;

/// f32 面法向（单位向量，与 `face_topology_f32` 预打包一致）。
pub type FaceNormalF32 = [f32; 3];

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
    normal: FaceNormalF32,
) -> InviscidFluxF32 {
    let [nx, ny, nz] = normal;
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
pub(crate) fn normalize_face_normal_f32(normal: FaceNormalF32) -> Result<FaceNormalF32> {
    let mag = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if mag < f32::EPSILON {
        return Err(AsimuError::Mesh("面法向不能为零向量".to_string()));
    }
    Ok([normal[0] / mag, normal[1] / mag, normal[2] / mag])
}

#[must_use]
pub(crate) fn face_tangent_basis_f32(normal: FaceNormalF32) -> (FaceNormalF32, FaceNormalF32) {
    let reference = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1 = cross_f32(normal, reference);
    let t1 = normalize_unchecked_f32(t1);
    let t2 = cross_f32(normal, t1);
    (t1, normalize_unchecked_f32(t2))
}

fn cross_f32(a: FaceNormalF32, b: FaceNormalF32) -> FaceNormalF32 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_unchecked_f32(v: FaceNormalF32) -> FaceNormalF32 {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / mag, v[1] / mag, v[2] / mag]
}
