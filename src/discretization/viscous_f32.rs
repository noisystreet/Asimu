//! f32 粘性面通量 compute + scatter（非 SIMD 串行路径）。

use crate::core::{ComputeFloat, Real};
use crate::discretization::gradient_typed::VelocityGradientSlicesT;
use crate::exec::ColoredViscousFaceGeom;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};

/// f32 单面粘性动量/能量通量。
#[derive(Debug, Clone, Copy, Default)]
pub struct ColoredViscousFaceFluxF32 {
    pub mx: f32,
    pub my: f32,
    pub mz: f32,
    pub energy: f32,
}

/// 面心预平均速度与梯度（f32）。
#[derive(Debug, Clone, Copy, Default)]
pub struct ViscousFaceAveragedLaneF32 {
    pub ux: f32,
    pub uy: f32,
    pub uz: f32,
    pub du_dx: f32,
    pub du_dy: f32,
    pub du_dz: f32,
    pub dv_dx: f32,
    pub dv_dy: f32,
    pub dv_dz: f32,
    pub dw_dx: f32,
    pub dw_dy: f32,
    pub dw_dz: f32,
    pub dt_dx: f32,
    pub dt_dy: f32,
    pub dt_dz: f32,
}

#[inline(always)]
pub fn fused_interior_viscous_face_flux_averaged_f32(
    avg: ViscousFaceAveragedLaneF32,
    nx: f32,
    ny: f32,
    nz: f32,
    mu: f32,
    lambda: f32,
) -> ColoredViscousFaceFluxF32 {
    let div_u = avg.du_dx + avg.dv_dy + avg.dw_dz;
    let two_thirds = 2.0_f32 / 3.0;
    let tau_xx = mu * (2.0 * avg.du_dx - two_thirds * div_u);
    let tau_yy = mu * (2.0 * avg.dv_dy - two_thirds * div_u);
    let tau_zz = mu * (2.0 * avg.dw_dz - two_thirds * div_u);
    let tau_xy = mu * (avg.du_dy + avg.dv_dx);
    let tau_xz = mu * (avg.du_dz + avg.dw_dx);
    let tau_yz = mu * (avg.dv_dz + avg.dw_dy);

    let tau_dot_n0 = tau_xx * nx + tau_xy * ny + tau_xz * nz;
    let tau_dot_n1 = tau_xy * nx + tau_yy * ny + tau_yz * nz;
    let tau_dot_n2 = tau_xz * nx + tau_yz * ny + tau_zz * nz;
    let heat_flux = lambda * (avg.dt_dx * nx + avg.dt_dy * ny + avg.dt_dz * nz);
    let energy_flux =
        -(heat_flux + tau_dot_n0 * avg.ux + tau_dot_n1 * avg.uy + tau_dot_n2 * avg.uz);

    ColoredViscousFaceFluxF32 {
        mx: -tau_dot_n0,
        my: -tau_dot_n1,
        mz: -tau_dot_n2,
        energy: energy_flux,
    }
}

#[inline(always)]
pub fn scatter_fused_interior_viscous_face_f32(
    residual: &mut ConservedResidualT<f32>,
    geom: &ColoredViscousFaceGeom,
    flux: &ColoredViscousFaceFluxF32,
) {
    let owner = geom.owner;
    let neighbor = geom.neighbor;
    let owner_scale = geom.owner_scale as f32;
    let neighbor_scale = geom.neighbor_scale as f32;
    residual.momentum_x.values_mut()[owner] =
        residual.momentum_x.values()[owner].add_mul_real(flux.mx, owner_scale as Real);
    residual.momentum_y.values_mut()[owner] =
        residual.momentum_y.values()[owner].add_mul_real(flux.my, owner_scale as Real);
    residual.momentum_z.values_mut()[owner] =
        residual.momentum_z.values()[owner].add_mul_real(flux.mz, owner_scale as Real);
    residual.total_energy.values_mut()[owner] =
        residual.total_energy.values()[owner].add_mul_real(flux.energy, owner_scale as Real);
    residual.momentum_x.values_mut()[neighbor] =
        residual.momentum_x.values()[neighbor].add_mul_real(flux.mx, neighbor_scale as Real);
    residual.momentum_y.values_mut()[neighbor] =
        residual.momentum_y.values()[neighbor].add_mul_real(flux.my, neighbor_scale as Real);
    residual.momentum_z.values_mut()[neighbor] =
        residual.momentum_z.values()[neighbor].add_mul_real(flux.mz, neighbor_scale as Real);
    residual.total_energy.values_mut()[neighbor] =
        residual.total_energy.values()[neighbor].add_mul_real(flux.energy, neighbor_scale as Real);
}

pub fn average_face_lane_f32(
    owner: usize,
    neighbor: usize,
    prim: &PrimitiveFieldsT<f32>,
    grad: &VelocityGradientSlicesT<'_, f32>,
) -> ViscousFaceAveragedLaneF32 {
    let half = 0.5_f32;
    let ux = prim.velocity_x.values();
    let uy = prim.velocity_y.values();
    let uz = prim.velocity_z.values();
    ViscousFaceAveragedLaneF32 {
        ux: half * (ux[owner] + ux[neighbor]),
        uy: half * (uy[owner] + uy[neighbor]),
        uz: half * (uz[owner] + uz[neighbor]),
        du_dx: half * (grad.du_dx[owner] + grad.du_dx[neighbor]),
        du_dy: half * (grad.du_dy[owner] + grad.du_dy[neighbor]),
        du_dz: half * (grad.du_dz[owner] + grad.du_dz[neighbor]),
        dv_dx: half * (grad.dv_dx[owner] + grad.dv_dx[neighbor]),
        dv_dy: half * (grad.dv_dy[owner] + grad.dv_dy[neighbor]),
        dv_dz: half * (grad.dv_dz[owner] + grad.dv_dz[neighbor]),
        dw_dx: half * (grad.dw_dx[owner] + grad.dw_dx[neighbor]),
        dw_dy: half * (grad.dw_dy[owner] + grad.dw_dy[neighbor]),
        dw_dz: half * (grad.dw_dz[owner] + grad.dw_dz[neighbor]),
        dt_dx: half * (grad.dt_dx[owner] + grad.dt_dx[neighbor]),
        dt_dy: half * (grad.dt_dy[owner] + grad.dt_dy[neighbor]),
        dt_dz: half * (grad.dt_dz[owner] + grad.dt_dz[neighbor]),
    }
}
