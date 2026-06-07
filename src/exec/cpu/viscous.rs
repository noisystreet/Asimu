//! 粘性内面通量：四路批处理 gather + f64x4 应力计算。

use crate::core::Real;

/// 四路 gather 后的面输入（SoA，lane = 面索引）。
#[derive(Debug, Clone, Copy, Default)]
pub struct ViscousFaceGather4 {
    pub ux_o: [Real; 4],
    pub ux_n: [Real; 4],
    pub uy_o: [Real; 4],
    pub uy_n: [Real; 4],
    pub uz_o: [Real; 4],
    pub uz_n: [Real; 4],
    pub du_dx_o: [Real; 4],
    pub du_dx_n: [Real; 4],
    pub du_dy_o: [Real; 4],
    pub du_dy_n: [Real; 4],
    pub du_dz_o: [Real; 4],
    pub du_dz_n: [Real; 4],
    pub dv_dx_o: [Real; 4],
    pub dv_dx_n: [Real; 4],
    pub dv_dy_o: [Real; 4],
    pub dv_dy_n: [Real; 4],
    pub dv_dz_o: [Real; 4],
    pub dv_dz_n: [Real; 4],
    pub dw_dx_o: [Real; 4],
    pub dw_dx_n: [Real; 4],
    pub dw_dy_o: [Real; 4],
    pub dw_dy_n: [Real; 4],
    pub dw_dz_o: [Real; 4],
    pub dw_dz_n: [Real; 4],
    pub dt_dx_o: [Real; 4],
    pub dt_dx_n: [Real; 4],
    pub dt_dy_o: [Real; 4],
    pub dt_dy_n: [Real; 4],
    pub dt_dz_o: [Real; 4],
    pub dt_dz_n: [Real; 4],
    pub nx: [Real; 4],
    pub ny: [Real; 4],
    pub nz: [Real; 4],
    pub mu: [Real; 4],
    pub lambda: [Real; 4],
}

/// 四路粘性通量输出。
#[derive(Debug, Clone, Copy, Default)]
pub struct ViscousFlux4 {
    pub mx: [Real; 4],
    pub my: [Real; 4],
    pub mz: [Real; 4],
    pub energy: [Real; 4],
}

/// 四内面几何与输运系数。
pub struct ViscousFaceBatchGeom {
    pub owners: [usize; 4],
    pub neighbors: [usize; 4],
    pub nx: [Real; 4],
    pub ny: [Real; 4],
    pub nz: [Real; 4],
    pub mu: [Real; 4],
    pub lambda: [Real; 4],
}

/// 速度与温度梯度 SoA 只读 slice。
pub struct VelocityGradientSoA<'a> {
    pub ux: &'a [Real],
    pub uy: &'a [Real],
    pub uz: &'a [Real],
    pub du_dx: &'a [Real],
    pub du_dy: &'a [Real],
    pub du_dz: &'a [Real],
    pub dv_dx: &'a [Real],
    pub dv_dy: &'a [Real],
    pub dv_dz: &'a [Real],
    pub dw_dx: &'a [Real],
    pub dw_dy: &'a [Real],
    pub dw_dz: &'a [Real],
    pub dt_dx: &'a [Real],
    pub dt_dy: &'a [Real],
    pub dt_dz: &'a [Real],
}

/// 由 SoA 梯度 slice 与单元/面数据 gather 四内面。
pub fn gather_viscous_face_batch4(
    geom: ViscousFaceBatchGeom,
    vel: VelocityGradientSoA<'_>,
) -> ViscousFaceGather4 {
    let mut g = ViscousFaceGather4::default();
    for lane in 0..4 {
        let o = geom.owners[lane];
        let n = geom.neighbors[lane];
        g.ux_o[lane] = vel.ux[o];
        g.ux_n[lane] = vel.ux[n];
        g.uy_o[lane] = vel.uy[o];
        g.uy_n[lane] = vel.uy[n];
        g.uz_o[lane] = vel.uz[o];
        g.uz_n[lane] = vel.uz[n];
        g.du_dx_o[lane] = vel.du_dx[o];
        g.du_dx_n[lane] = vel.du_dx[n];
        g.du_dy_o[lane] = vel.du_dy[o];
        g.du_dy_n[lane] = vel.du_dy[n];
        g.du_dz_o[lane] = vel.du_dz[o];
        g.du_dz_n[lane] = vel.du_dz[n];
        g.dv_dx_o[lane] = vel.dv_dx[o];
        g.dv_dx_n[lane] = vel.dv_dx[n];
        g.dv_dy_o[lane] = vel.dv_dy[o];
        g.dv_dy_n[lane] = vel.dv_dy[n];
        g.dv_dz_o[lane] = vel.dv_dz[o];
        g.dv_dz_n[lane] = vel.dv_dz[n];
        g.dw_dx_o[lane] = vel.dw_dx[o];
        g.dw_dx_n[lane] = vel.dw_dx[n];
        g.dw_dy_o[lane] = vel.dw_dy[o];
        g.dw_dy_n[lane] = vel.dw_dy[n];
        g.dw_dz_o[lane] = vel.dw_dz[o];
        g.dw_dz_n[lane] = vel.dw_dz[n];
        g.dt_dx_o[lane] = vel.dt_dx[o];
        g.dt_dx_n[lane] = vel.dt_dx[n];
        g.dt_dy_o[lane] = vel.dt_dy[o];
        g.dt_dy_n[lane] = vel.dt_dy[n];
        g.dt_dz_o[lane] = vel.dt_dz[o];
        g.dt_dz_n[lane] = vel.dt_dz[n];
        g.nx[lane] = geom.nx[lane];
        g.ny[lane] = geom.ny[lane];
        g.nz[lane] = geom.nz[lane];
        g.mu[lane] = geom.mu[lane];
        g.lambda[lane] = geom.lambda[lane];
    }
    g
}

/// 四路融合粘性通量（与 `fused_interior_viscous_face_flux` 数值一致）。
pub fn fused_interior_viscous_face_flux_batch4(g: &ViscousFaceGather4) -> ViscousFlux4 {
    #[cfg(feature = "simd-fvm")]
    {
        return fused_interior_viscous_face_flux_batch4_simd(g);
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        let mut out = ViscousFlux4::default();
        for lane in 0..4 {
            let f = fused_lane_scalar(g, lane);
            out.mx[lane] = f.0;
            out.my[lane] = f.1;
            out.mz[lane] = f.2;
            out.energy[lane] = f.3;
        }
        out
    }
}

#[cfg(feature = "simd-fvm")]
fn fused_interior_viscous_face_flux_batch4_simd(g: &ViscousFaceGather4) -> ViscousFlux4 {
    use wide::f64x4;

    let half = f64x4::splat(0.5);
    let two_thirds = f64x4::splat(2.0 / 3.0);
    let two = f64x4::splat(2.0);

    let lane = |arr: &[Real; 4]| f64x4::new(*arr);

    let u0 = half * (lane(&g.ux_o) + lane(&g.ux_n));
    let u1 = half * (lane(&g.uy_o) + lane(&g.uy_n));
    let u2 = half * (lane(&g.uz_o) + lane(&g.uz_n));

    let du0 = half * (lane(&g.du_dx_o) + lane(&g.du_dx_n));
    let du1 = half * (lane(&g.du_dy_o) + lane(&g.du_dy_n));
    let du2 = half * (lane(&g.du_dz_o) + lane(&g.du_dz_n));
    let dv0 = half * (lane(&g.dv_dx_o) + lane(&g.dv_dx_n));
    let dv1 = half * (lane(&g.dv_dy_o) + lane(&g.dv_dy_n));
    let dv2 = half * (lane(&g.dv_dz_o) + lane(&g.dv_dz_n));
    let dw0 = half * (lane(&g.dw_dx_o) + lane(&g.dw_dx_n));
    let dw1 = half * (lane(&g.dw_dy_o) + lane(&g.dw_dy_n));
    let dw2 = half * (lane(&g.dw_dz_o) + lane(&g.dw_dz_n));
    let dt0 = half * (lane(&g.dt_dx_o) + lane(&g.dt_dx_n));
    let dt1 = half * (lane(&g.dt_dy_o) + lane(&g.dt_dy_n));
    let dt2 = half * (lane(&g.dt_dz_o) + lane(&g.dt_dz_n));

    let mu = lane(&g.mu);
    let lambda = lane(&g.lambda);
    let nx = lane(&g.nx);
    let ny = lane(&g.ny);
    let nz = lane(&g.nz);

    let div_u = du0 + dv1 + dw2;
    let tau_xx = mu * (two * du0 - two_thirds * div_u);
    let tau_yy = mu * (two * dv1 - two_thirds * div_u);
    let tau_zz = mu * (two * dw2 - two_thirds * div_u);
    let tau_xy = mu * (du1 + dv0);
    let tau_xz = mu * (du2 + dw0);
    let tau_yz = mu * (dv2 + dw1);

    let tau_dot_n0 = tau_xx * nx + tau_xy * ny + tau_xz * nz;
    let tau_dot_n1 = tau_xy * nx + tau_yy * ny + tau_yz * nz;
    let tau_dot_n2 = tau_xz * nx + tau_yz * ny + tau_zz * nz;
    let heat_flux = lambda * (dt0 * nx + dt1 * ny + dt2 * nz);
    let energy_flux = -(heat_flux + tau_dot_n0 * u0 + tau_dot_n1 * u1 + tau_dot_n2 * u2);

    ViscousFlux4 {
        mx: (-tau_dot_n0).to_array(),
        my: (-tau_dot_n1).to_array(),
        mz: (-tau_dot_n2).to_array(),
        energy: energy_flux.to_array(),
    }
}

#[cfg(any(not(feature = "simd-fvm"), test))]
fn fused_lane_scalar(g: &ViscousFaceGather4, lane: usize) -> (Real, Real, Real, Real) {
    let half = 0.5;
    let u0 = half * (g.ux_o[lane] + g.ux_n[lane]);
    let u1 = half * (g.uy_o[lane] + g.uy_n[lane]);
    let u2 = half * (g.uz_o[lane] + g.uz_n[lane]);
    let du0 = half * (g.du_dx_o[lane] + g.du_dx_n[lane]);
    let du1 = half * (g.du_dy_o[lane] + g.du_dy_n[lane]);
    let du2 = half * (g.du_dz_o[lane] + g.du_dz_n[lane]);
    let dv0 = half * (g.dv_dx_o[lane] + g.dv_dx_n[lane]);
    let dv1 = half * (g.dv_dy_o[lane] + g.dv_dy_n[lane]);
    let dv2 = half * (g.dv_dz_o[lane] + g.dv_dz_n[lane]);
    let dw0 = half * (g.dw_dx_o[lane] + g.dw_dx_n[lane]);
    let dw1 = half * (g.dw_dy_o[lane] + g.dw_dy_n[lane]);
    let dw2 = half * (g.dw_dz_o[lane] + g.dw_dz_n[lane]);
    let dt0 = half * (g.dt_dx_o[lane] + g.dt_dx_n[lane]);
    let dt1 = half * (g.dt_dy_o[lane] + g.dt_dy_n[lane]);
    let dt2 = half * (g.dt_dz_o[lane] + g.dt_dz_n[lane]);
    let mu = g.mu[lane];
    let lambda = g.lambda[lane];
    let nx = g.nx[lane];
    let ny = g.ny[lane];
    let nz = g.nz[lane];
    let div_u = du0 + dv1 + dw2;
    let two_thirds = 2.0 / 3.0;
    let tau_xx = mu * (2.0 * du0 - two_thirds * div_u);
    let tau_yy = mu * (2.0 * dv1 - two_thirds * div_u);
    let tau_zz = mu * (2.0 * dw2 - two_thirds * div_u);
    let tau_xy = mu * (du1 + dv0);
    let tau_xz = mu * (du2 + dw0);
    let tau_yz = mu * (dv2 + dw1);
    let tau_dot_n0 = tau_xx * nx + tau_xy * ny + tau_xz * nz;
    let tau_dot_n1 = tau_xy * nx + tau_yy * ny + tau_yz * nz;
    let tau_dot_n2 = tau_xz * nx + tau_yz * ny + tau_zz * nz;
    let heat_flux = lambda * (dt0 * nx + dt1 * ny + dt2 * nz);
    let energy = -(heat_flux + tau_dot_n0 * u0 + tau_dot_n1 * u1 + tau_dot_n2 * u2);
    (-tau_dot_n0, -tau_dot_n1, -tau_dot_n2, energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn batch4_matches_lane_scalar() {
        let g = ViscousFaceGather4 {
            ux_o: [1.0, 2.0, 0.5, 1.1],
            ux_n: [0.8, 1.5, 0.6, 0.9],
            uy_o: [0.1; 4],
            uy_n: [0.2; 4],
            uz_o: [0.0; 4],
            uz_n: [0.0; 4],
            du_dx_o: [0.01; 4],
            du_dx_n: [0.02; 4],
            du_dy_o: [0.0; 4],
            du_dy_n: [0.0; 4],
            du_dz_o: [0.0; 4],
            du_dz_n: [0.0; 4],
            dv_dx_o: [0.0; 4],
            dv_dx_n: [0.0; 4],
            dv_dy_o: [0.01; 4],
            dv_dy_n: [0.01; 4],
            dv_dz_o: [0.0; 4],
            dv_dz_n: [0.0; 4],
            dw_dx_o: [0.0; 4],
            dw_dx_n: [0.0; 4],
            dw_dy_o: [0.0; 4],
            dw_dy_n: [0.0; 4],
            dw_dz_o: [0.0; 4],
            dw_dz_n: [0.0; 4],
            dt_dx_o: [0.001; 4],
            dt_dx_n: [0.001; 4],
            dt_dy_o: [0.0; 4],
            dt_dy_n: [0.0; 4],
            dt_dz_o: [0.0; 4],
            dt_dz_n: [0.0; 4],
            nx: [1.0, 0.0, 0.707, 0.6],
            ny: [0.0, 1.0, 0.707, 0.8],
            nz: [0.0, 0.0, 0.0, 0.0],
            mu: [1.8e-5; 4],
            lambda: [0.025; 4],
        };
        let batch = fused_interior_viscous_face_flux_batch4(&g);
        for lane in 0..4 {
            let (mx, my, mz, e) = fused_lane_scalar(&g, lane);
            assert!(approx_eq(batch.mx[lane], mx, 1.0e-12));
            assert!(approx_eq(batch.my[lane], my, 1.0e-12));
            assert!(approx_eq(batch.mz[lane], mz, 1.0e-12));
            assert!(approx_eq(batch.energy[lane], e, 1.0e-12));
        }
    }
}
