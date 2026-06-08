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
        fused_interior_viscous_face_flux_batch4_simd(g)
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

/// P9：cell SoA 直读 + 面平均 + f64x4 应力，跳过 `ViscousFaceGather4` 物化。
pub fn fused_interior_viscous_face_flux_batch4_from_soa(
    geom: &ViscousFaceBatchGeom,
    vel: &VelocityGradientSoA<'_>,
) -> ViscousFlux4 {
    #[cfg(feature = "simd-fvm")]
    {
        fused_interior_viscous_face_flux_batch4_from_soa_simd(geom, vel)
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        let mut out = ViscousFlux4::default();
        for lane in 0..4 {
            let o = geom.owners[lane];
            let n = geom.neighbors[lane];
            let half = 0.5;
            let avg = ViscousAveragedLaneScalar {
                u0: half * (vel.ux[o] + vel.ux[n]),
                u1: half * (vel.uy[o] + vel.uy[n]),
                u2: half * (vel.uz[o] + vel.uz[n]),
                du0: half * (vel.du_dx[o] + vel.du_dx[n]),
                du1: half * (vel.du_dy[o] + vel.du_dy[n]),
                du2: half * (vel.du_dz[o] + vel.du_dz[n]),
                dv0: half * (vel.dv_dx[o] + vel.dv_dx[n]),
                dv1: half * (vel.dv_dy[o] + vel.dv_dy[n]),
                dv2: half * (vel.dv_dz[o] + vel.dv_dz[n]),
                dw0: half * (vel.dw_dx[o] + vel.dw_dx[n]),
                dw1: half * (vel.dw_dy[o] + vel.dw_dy[n]),
                dw2: half * (vel.dw_dz[o] + vel.dw_dz[n]),
                dt0: half * (vel.dt_dx[o] + vel.dt_dx[n]),
                dt1: half * (vel.dt_dy[o] + vel.dt_dy[n]),
                dt2: half * (vel.dt_dz[o] + vel.dt_dz[n]),
            };
            let props = ViscousFacePropsScalar {
                mu: geom.mu[lane],
                lambda: geom.lambda[lane],
                nx: geom.nx[lane],
                ny: geom.ny[lane],
                nz: geom.nz[lane],
            };
            let (mx, my, mz, energy) = viscous_stress_flux_scalar(avg, props);
            out.mx[lane] = mx;
            out.my[lane] = my;
            out.mz[lane] = mz;
            out.energy[lane] = energy;
        }
        out
    }
}

#[cfg(feature = "simd-fvm")]
#[derive(Clone, Copy)]
struct ViscousAveragedLanes4 {
    u0: wide::f64x4,
    u1: wide::f64x4,
    u2: wide::f64x4,
    du0: wide::f64x4,
    du1: wide::f64x4,
    du2: wide::f64x4,
    dv0: wide::f64x4,
    dv1: wide::f64x4,
    dv2: wide::f64x4,
    dw0: wide::f64x4,
    dw1: wide::f64x4,
    dw2: wide::f64x4,
    dt0: wide::f64x4,
    dt1: wide::f64x4,
    dt2: wide::f64x4,
}

#[cfg(feature = "simd-fvm")]
#[derive(Clone, Copy)]
struct ViscousFaceProps4 {
    mu: wide::f64x4,
    lambda: wide::f64x4,
    nx: wide::f64x4,
    ny: wide::f64x4,
    nz: wide::f64x4,
}

#[cfg(any(not(feature = "simd-fvm"), test))]
#[derive(Clone, Copy)]
struct ViscousAveragedLaneScalar {
    u0: Real,
    u1: Real,
    u2: Real,
    du0: Real,
    du1: Real,
    du2: Real,
    dv0: Real,
    dv1: Real,
    dv2: Real,
    dw0: Real,
    dw1: Real,
    dw2: Real,
    dt0: Real,
    dt1: Real,
    dt2: Real,
}

#[cfg(any(not(feature = "simd-fvm"), test))]
#[derive(Clone, Copy)]
struct ViscousFacePropsScalar {
    mu: Real,
    lambda: Real,
    nx: Real,
    ny: Real,
    nz: Real,
}

#[cfg(feature = "simd-fvm")]
fn fused_interior_viscous_face_flux_batch4_from_soa_simd(
    geom: &ViscousFaceBatchGeom,
    vel: &VelocityGradientSoA<'_>,
) -> ViscousFlux4 {
    use wide::f64x4;

    let avg = ViscousAveragedLanes4 {
        u0: half_avg_batch4(vel.ux, &geom.owners, &geom.neighbors),
        u1: half_avg_batch4(vel.uy, &geom.owners, &geom.neighbors),
        u2: half_avg_batch4(vel.uz, &geom.owners, &geom.neighbors),
        du0: half_avg_batch4(vel.du_dx, &geom.owners, &geom.neighbors),
        du1: half_avg_batch4(vel.du_dy, &geom.owners, &geom.neighbors),
        du2: half_avg_batch4(vel.du_dz, &geom.owners, &geom.neighbors),
        dv0: half_avg_batch4(vel.dv_dx, &geom.owners, &geom.neighbors),
        dv1: half_avg_batch4(vel.dv_dy, &geom.owners, &geom.neighbors),
        dv2: half_avg_batch4(vel.dv_dz, &geom.owners, &geom.neighbors),
        dw0: half_avg_batch4(vel.dw_dx, &geom.owners, &geom.neighbors),
        dw1: half_avg_batch4(vel.dw_dy, &geom.owners, &geom.neighbors),
        dw2: half_avg_batch4(vel.dw_dz, &geom.owners, &geom.neighbors),
        dt0: half_avg_batch4(vel.dt_dx, &geom.owners, &geom.neighbors),
        dt1: half_avg_batch4(vel.dt_dy, &geom.owners, &geom.neighbors),
        dt2: half_avg_batch4(vel.dt_dz, &geom.owners, &geom.neighbors),
    };
    let props = ViscousFaceProps4 {
        mu: f64x4::new(geom.mu),
        lambda: f64x4::new(geom.lambda),
        nx: f64x4::new(geom.nx),
        ny: f64x4::new(geom.ny),
        nz: f64x4::new(geom.nz),
    };
    viscous_stress_flux_batch4_simd(avg, props)
}

#[cfg(feature = "simd-fvm")]
#[inline(always)]
fn half_avg_batch4(field: &[Real], owners: &[usize; 4], neighbors: &[usize; 4]) -> wide::f64x4 {
    use wide::f64x4;
    let half = 0.5;
    f64x4::new([
        half * (field[owners[0]] + field[neighbors[0]]),
        half * (field[owners[1]] + field[neighbors[1]]),
        half * (field[owners[2]] + field[neighbors[2]]),
        half * (field[owners[3]] + field[neighbors[3]]),
    ])
}

#[cfg(feature = "simd-fvm")]
fn fused_interior_viscous_face_flux_batch4_simd(g: &ViscousFaceGather4) -> ViscousFlux4 {
    use wide::f64x4;

    let half = f64x4::splat(0.5);
    let lane = |arr: &[Real; 4]| f64x4::new(*arr);

    let avg = ViscousAveragedLanes4 {
        u0: half * (lane(&g.ux_o) + lane(&g.ux_n)),
        u1: half * (lane(&g.uy_o) + lane(&g.uy_n)),
        u2: half * (lane(&g.uz_o) + lane(&g.uz_n)),
        du0: half * (lane(&g.du_dx_o) + lane(&g.du_dx_n)),
        du1: half * (lane(&g.du_dy_o) + lane(&g.du_dy_n)),
        du2: half * (lane(&g.du_dz_o) + lane(&g.du_dz_n)),
        dv0: half * (lane(&g.dv_dx_o) + lane(&g.dv_dx_n)),
        dv1: half * (lane(&g.dv_dy_o) + lane(&g.dv_dy_n)),
        dv2: half * (lane(&g.dv_dz_o) + lane(&g.dv_dz_n)),
        dw0: half * (lane(&g.dw_dx_o) + lane(&g.dw_dx_n)),
        dw1: half * (lane(&g.dw_dy_o) + lane(&g.dw_dy_n)),
        dw2: half * (lane(&g.dw_dz_o) + lane(&g.dw_dz_n)),
        dt0: half * (lane(&g.dt_dx_o) + lane(&g.dt_dx_n)),
        dt1: half * (lane(&g.dt_dy_o) + lane(&g.dt_dy_n)),
        dt2: half * (lane(&g.dt_dz_o) + lane(&g.dt_dz_n)),
    };
    let props = ViscousFaceProps4 {
        mu: lane(&g.mu),
        lambda: lane(&g.lambda),
        nx: lane(&g.nx),
        ny: lane(&g.ny),
        nz: lane(&g.nz),
    };

    viscous_stress_flux_batch4_simd(avg, props)
}

#[cfg(feature = "simd-fvm")]
fn viscous_stress_flux_batch4_simd(
    avg: ViscousAveragedLanes4,
    props: ViscousFaceProps4,
) -> ViscousFlux4 {
    use wide::f64x4;

    let two_thirds = f64x4::splat(2.0 / 3.0);
    let two = f64x4::splat(2.0);

    let div_u = avg.du0 + avg.dv1 + avg.dw2;
    let tau_xx = props.mu * (two * avg.du0 - two_thirds * div_u);
    let tau_yy = props.mu * (two * avg.dv1 - two_thirds * div_u);
    let tau_zz = props.mu * (two * avg.dw2 - two_thirds * div_u);
    let tau_xy = props.mu * (avg.du1 + avg.dv0);
    let tau_xz = props.mu * (avg.du2 + avg.dw0);
    let tau_yz = props.mu * (avg.dv2 + avg.dw1);

    let tau_dot_n0 = tau_xx * props.nx + tau_xy * props.ny + tau_xz * props.nz;
    let tau_dot_n1 = tau_xy * props.nx + tau_yy * props.ny + tau_yz * props.nz;
    let tau_dot_n2 = tau_xz * props.nx + tau_yz * props.ny + tau_zz * props.nz;
    let heat_flux = props.lambda * (avg.dt0 * props.nx + avg.dt1 * props.ny + avg.dt2 * props.nz);
    let energy_flux =
        -(heat_flux + tau_dot_n0 * avg.u0 + tau_dot_n1 * avg.u1 + tau_dot_n2 * avg.u2);

    ViscousFlux4 {
        mx: (-tau_dot_n0).to_array(),
        my: (-tau_dot_n1).to_array(),
        mz: (-tau_dot_n2).to_array(),
        energy: energy_flux.to_array(),
    }
}

#[cfg(any(not(feature = "simd-fvm"), test))]
fn viscous_stress_flux_scalar(
    avg: ViscousAveragedLaneScalar,
    props: ViscousFacePropsScalar,
) -> (Real, Real, Real, Real) {
    let div_u = avg.du0 + avg.dv1 + avg.dw2;
    let two_thirds = 2.0 / 3.0;
    let tau_xx = props.mu * (2.0 * avg.du0 - two_thirds * div_u);
    let tau_yy = props.mu * (2.0 * avg.dv1 - two_thirds * div_u);
    let tau_zz = props.mu * (2.0 * avg.dw2 - two_thirds * div_u);
    let tau_xy = props.mu * (avg.du1 + avg.dv0);
    let tau_xz = props.mu * (avg.du2 + avg.dw0);
    let tau_yz = props.mu * (avg.dv2 + avg.dw1);
    let tau_dot_n0 = tau_xx * props.nx + tau_xy * props.ny + tau_xz * props.nz;
    let tau_dot_n1 = tau_xy * props.nx + tau_yy * props.ny + tau_yz * props.nz;
    let tau_dot_n2 = tau_xz * props.nx + tau_yz * props.ny + tau_zz * props.nz;
    let heat_flux = props.lambda * (avg.dt0 * props.nx + avg.dt1 * props.ny + avg.dt2 * props.nz);
    let energy = -(heat_flux + tau_dot_n0 * avg.u0 + tau_dot_n1 * avg.u1 + tau_dot_n2 * avg.u2);
    (-tau_dot_n0, -tau_dot_n1, -tau_dot_n2, energy)
}

#[cfg(any(not(feature = "simd-fvm"), test))]
fn fused_lane_scalar(g: &ViscousFaceGather4, lane: usize) -> (Real, Real, Real, Real) {
    let half = 0.5;
    let avg = ViscousAveragedLaneScalar {
        u0: half * (g.ux_o[lane] + g.ux_n[lane]),
        u1: half * (g.uy_o[lane] + g.uy_n[lane]),
        u2: half * (g.uz_o[lane] + g.uz_n[lane]),
        du0: half * (g.du_dx_o[lane] + g.du_dx_n[lane]),
        du1: half * (g.du_dy_o[lane] + g.du_dy_n[lane]),
        du2: half * (g.du_dz_o[lane] + g.du_dz_n[lane]),
        dv0: half * (g.dv_dx_o[lane] + g.dv_dx_n[lane]),
        dv1: half * (g.dv_dy_o[lane] + g.dv_dy_n[lane]),
        dv2: half * (g.dv_dz_o[lane] + g.dv_dz_n[lane]),
        dw0: half * (g.dw_dx_o[lane] + g.dw_dx_n[lane]),
        dw1: half * (g.dw_dy_o[lane] + g.dw_dy_n[lane]),
        dw2: half * (g.dw_dz_o[lane] + g.dw_dz_n[lane]),
        dt0: half * (g.dt_dx_o[lane] + g.dt_dx_n[lane]),
        dt1: half * (g.dt_dy_o[lane] + g.dt_dy_n[lane]),
        dt2: half * (g.dt_dz_o[lane] + g.dt_dz_n[lane]),
    };
    let props = ViscousFacePropsScalar {
        mu: g.mu[lane],
        lambda: g.lambda[lane],
        nx: g.nx[lane],
        ny: g.ny[lane],
        nz: g.nz[lane],
    };
    viscous_stress_flux_scalar(avg, props)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    fn sample_gather() -> ViscousFaceGather4 {
        ViscousFaceGather4 {
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
        }
    }

    #[test]
    fn batch4_matches_lane_scalar() {
        let g = sample_gather();
        let batch = fused_interior_viscous_face_flux_batch4(&g);
        for lane in 0..4 {
            let (mx, my, mz, e) = fused_lane_scalar(&g, lane);
            assert!(approx_eq(batch.mx[lane], mx, 1.0e-12));
            assert!(approx_eq(batch.my[lane], my, 1.0e-12));
            assert!(approx_eq(batch.mz[lane], mz, 1.0e-12));
            assert!(approx_eq(batch.energy[lane], e, 1.0e-12));
        }
    }

    #[test]
    fn from_soa_matches_gather_batch4() {
        let g = sample_gather();
        let from_gather = fused_interior_viscous_face_flux_batch4(&g);
        let owners = [0usize, 1, 2, 3];
        let neighbors = [4, 5, 6, 7];
        let mut ux = [0.0; 8];
        let mut uy = [0.0; 8];
        let mut uz = [0.0; 8];
        let mut du_dx = [0.0; 8];
        let mut du_dy = [0.0; 8];
        let mut du_dz = [0.0; 8];
        let mut dv_dx = [0.0; 8];
        let mut dv_dy = [0.0; 8];
        let mut dv_dz = [0.0; 8];
        let mut dw_dx = [0.0; 8];
        let mut dw_dy = [0.0; 8];
        let mut dw_dz = [0.0; 8];
        let mut dt_dx = [0.0; 8];
        let mut dt_dy = [0.0; 8];
        let mut dt_dz = [0.0; 8];
        for lane in 0..4 {
            ux[owners[lane]] = g.ux_o[lane];
            ux[neighbors[lane]] = g.ux_n[lane];
            uy[owners[lane]] = g.uy_o[lane];
            uy[neighbors[lane]] = g.uy_n[lane];
            uz[owners[lane]] = g.uz_o[lane];
            uz[neighbors[lane]] = g.uz_n[lane];
            du_dx[owners[lane]] = g.du_dx_o[lane];
            du_dx[neighbors[lane]] = g.du_dx_n[lane];
            du_dy[owners[lane]] = g.du_dy_o[lane];
            du_dy[neighbors[lane]] = g.du_dy_n[lane];
            du_dz[owners[lane]] = g.du_dz_o[lane];
            du_dz[neighbors[lane]] = g.du_dz_n[lane];
            dv_dx[owners[lane]] = g.dv_dx_o[lane];
            dv_dx[neighbors[lane]] = g.dv_dx_n[lane];
            dv_dy[owners[lane]] = g.dv_dy_o[lane];
            dv_dy[neighbors[lane]] = g.dv_dy_n[lane];
            dv_dz[owners[lane]] = g.dv_dz_o[lane];
            dv_dz[neighbors[lane]] = g.dv_dz_n[lane];
            dw_dx[owners[lane]] = g.dw_dx_o[lane];
            dw_dx[neighbors[lane]] = g.dw_dx_n[lane];
            dw_dy[owners[lane]] = g.dw_dy_o[lane];
            dw_dy[neighbors[lane]] = g.dw_dy_n[lane];
            dw_dz[owners[lane]] = g.dw_dz_o[lane];
            dw_dz[neighbors[lane]] = g.dw_dz_n[lane];
            dt_dx[owners[lane]] = g.dt_dx_o[lane];
            dt_dx[neighbors[lane]] = g.dt_dx_n[lane];
            dt_dy[owners[lane]] = g.dt_dy_o[lane];
            dt_dy[neighbors[lane]] = g.dt_dy_n[lane];
            dt_dz[owners[lane]] = g.dt_dz_o[lane];
            dt_dz[neighbors[lane]] = g.dt_dz_n[lane];
        }
        let vel = VelocityGradientSoA {
            ux: &ux,
            uy: &uy,
            uz: &uz,
            du_dx: &du_dx,
            du_dy: &du_dy,
            du_dz: &du_dz,
            dv_dx: &dv_dx,
            dv_dy: &dv_dy,
            dv_dz: &dv_dz,
            dw_dx: &dw_dx,
            dw_dy: &dw_dy,
            dw_dz: &dw_dz,
            dt_dx: &dt_dx,
            dt_dy: &dt_dy,
            dt_dz: &dt_dz,
        };
        let geom = ViscousFaceBatchGeom {
            owners,
            neighbors,
            nx: g.nx,
            ny: g.ny,
            nz: g.nz,
            mu: g.mu,
            lambda: g.lambda,
        };
        let from_soa = fused_interior_viscous_face_flux_batch4_from_soa(&geom, &vel);
        for lane in 0..4 {
            assert!(approx_eq(from_soa.mx[lane], from_gather.mx[lane], 1.0e-12));
            assert!(approx_eq(from_soa.my[lane], from_gather.my[lane], 1.0e-12));
            assert!(approx_eq(from_soa.mz[lane], from_gather.mz[lane], 1.0e-12));
            assert!(approx_eq(
                from_soa.energy[lane],
                from_gather.energy[lane],
                1.0e-12
            ));
        }
    }
}
