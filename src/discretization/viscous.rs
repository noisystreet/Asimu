//! 可压缩 Navier-Stokes 粘性通量（Newtonian + Fourier 热传导）。

use crate::core::{Real, Vector3};
use crate::discretization::InviscidFlux;
use crate::discretization::gradient::{GradientFields, VelocityGradient, VelocityGradientSlices};
use crate::field::PrimitiveFields;
use crate::physics::{IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

/// 粘性面通量（与无粘通量相同的守恒分量布局）。
pub type ViscousFlux = InviscidFlux;

/// 由面两侧原始变量与梯度计算粘性通量 \(\mathbf{F}_v \cdot \mathbf{n}\)。
#[must_use]
pub fn viscous_face_flux(
    prim_l: &PrimitiveState,
    grad_l: &VelocityGradient,
    prim_r: &PrimitiveState,
    grad_r: &VelocityGradient,
    normal: Vector3,
    mu: Real,
    lambda: Real,
) -> ViscousFlux {
    let grad = average_gradient(grad_l, grad_r);
    let u = [
        0.5 * (prim_l.velocity[0] + prim_r.velocity[0]),
        0.5 * (prim_l.velocity[1] + prim_r.velocity[1]),
        0.5 * (prim_l.velocity[2] + prim_r.velocity[2]),
    ];
    viscous_face_flux_from_averaged(u, &grad, normal, mu, lambda)
}

/// 内面粘性通量：直接从 SoA 原始变量/梯度读取，避免 `PrimitiveState` 与 `VelocityGradient` 拷贝。
#[must_use]
pub fn viscous_interior_face_flux(
    prim: &PrimitiveFields,
    grad: &GradientFields,
    left: usize,
    right: usize,
    normal: Vector3,
    mu: Real,
    lambda: Real,
) -> ViscousFlux {
    let ux = prim.velocity_x.values();
    let uy = prim.velocity_y.values();
    let uz = prim.velocity_z.values();
    let u = [
        0.5 * (ux[left] + ux[right]),
        0.5 * (uy[left] + uy[right]),
        0.5 * (uz[left] + uz[right]),
    ];
    let avg =
        |field: &crate::field::ScalarField| 0.5 * (field.values()[left] + field.values()[right]);
    let averaged = VelocityGradient {
        du: [avg(&grad.du_dx), avg(&grad.du_dy), avg(&grad.du_dz)],
        dv: [avg(&grad.dv_dx), avg(&grad.dv_dy), avg(&grad.dv_dz)],
        dw: [avg(&grad.dw_dx), avg(&grad.dw_dy), avg(&grad.dw_dz)],
        dt: [avg(&grad.dt_dx), avg(&grad.dt_dy), avg(&grad.dt_dz)],
    };
    viscous_face_flux_from_averaged(u, &averaged, normal, mu, lambda)
}

/// 内面粘性 scatter 的可变残差切片。
pub(crate) struct InteriorViscousResidualMut<'a> {
    pub mx: &'a mut [Real],
    pub my: &'a mut [Real],
    pub mz: &'a mut [Real],
    pub energy: &'a mut [Real],
}

/// 内面粘性通量输入场切片。
pub(crate) struct InteriorViscousFaceInputs<'a> {
    pub grad: &'a VelocityGradientSlices<'a>,
    pub ux: &'a [Real],
    pub uy: &'a [Real],
    pub uz: &'a [Real],
}

/// 单面几何与物性。
#[derive(Clone, Copy)]
pub(crate) struct InteriorViscousFaceGeom {
    pub owner: usize,
    pub neighbor: usize,
    pub nx: Real,
    pub ny: Real,
    pub nz: Real,
    pub mu: Real,
    pub lambda: Real,
    pub owner_scale: Real,
    pub neighbor_scale: Real,
}

/// 单面粘性动量/能量通量（scatter 前）。
pub(crate) struct InteriorViscousFaceFlux {
    pub mx: Real,
    pub my: Real,
    pub mz: Real,
    pub energy: Real,
}

/// 内面心预平均速度与梯度 SoA（P7 非 SIMD：flux 顺序读）。
#[cfg(not(feature = "simd-fvm"))]
#[derive(Debug, Clone, Default)]
pub(crate) struct ViscousFaceAveragedSoA {
    pub lanes: Vec<ViscousFaceAveragedLane>,
}

#[cfg(not(feature = "simd-fvm"))]
impl ViscousFaceAveragedSoA {
    pub(crate) fn ensure(&mut self, num_faces: usize) {
        self.lanes.resize(
            num_faces,
            ViscousFaceAveragedLane {
                ux: 0.0,
                uy: 0.0,
                uz: 0.0,
                du_dx: 0.0,
                du_dy: 0.0,
                du_dz: 0.0,
                dv_dx: 0.0,
                dv_dy: 0.0,
                dv_dz: 0.0,
                dw_dx: 0.0,
                dw_dy: 0.0,
                dw_dz: 0.0,
                dt_dx: 0.0,
                dt_dy: 0.0,
                dt_dz: 0.0,
            },
        );
    }

    #[inline(always)]
    pub(crate) fn lane(&self, face: usize) -> ViscousFaceAveragedLane {
        self.lanes[face]
    }
}

/// 单内面预平均速度与梯度（面心值）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViscousFaceAveragedLane {
    pub ux: Real,
    pub uy: Real,
    pub uz: Real,
    pub du_dx: Real,
    pub du_dy: Real,
    pub du_dz: Real,
    pub dv_dx: Real,
    pub dv_dy: Real,
    pub dv_dz: Real,
    pub dw_dx: Real,
    pub dw_dy: Real,
    pub dw_dz: Real,
    pub dt_dx: Real,
    pub dt_dy: Real,
    pub dt_dz: Real,
}

/// 由面心预平均态计算粘性通量（与 cell gather 路径数值一致）。
#[inline(always)]
pub(crate) fn fused_interior_viscous_face_flux_averaged(
    avg: ViscousFaceAveragedLane,
    nx: Real,
    ny: Real,
    nz: Real,
    mu: Real,
    lambda: Real,
) -> InteriorViscousFaceFlux {
    let u0 = avg.ux;
    let u1 = avg.uy;
    let u2 = avg.uz;
    let du0 = avg.du_dx;
    let du1 = avg.du_dy;
    let du2 = avg.du_dz;
    let dv0 = avg.dv_dx;
    let dv1 = avg.dv_dy;
    let dv2 = avg.dv_dz;
    let dw0 = avg.dw_dx;
    let dw1 = avg.dw_dy;
    let dw2 = avg.dw_dz;
    let dt0 = avg.dt_dx;
    let dt1 = avg.dt_dy;
    let dt2 = avg.dt_dz;

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
    let energy_flux = -(heat_flux + tau_dot_n0 * u0 + tau_dot_n1 * u1 + tau_dot_n2 * u2);

    InteriorViscousFaceFlux {
        mx: -tau_dot_n0,
        my: -tau_dot_n1,
        mz: -tau_dot_n2,
        energy: energy_flux,
    }
}

/// 计算内面融合粘性通量（只读 inputs/geom，无 scatter）。
#[inline(always)]
pub(crate) fn fused_interior_viscous_face_flux(
    inputs: &InteriorViscousFaceInputs<'_>,
    geom: &InteriorViscousFaceGeom,
) -> InteriorViscousFaceFlux {
    let grad = inputs.grad;
    let ux = inputs.ux;
    let uy = inputs.uy;
    let uz = inputs.uz;
    let owner = geom.owner;
    let neighbor = geom.neighbor;
    let half = 0.5;
    fused_interior_viscous_face_flux_averaged(
        ViscousFaceAveragedLane {
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
        },
        geom.nx,
        geom.ny,
        geom.nz,
        geom.mu,
        geom.lambda,
    )
}

/// 将内面粘性通量 scatter 到 owner/neighbor 残差。
#[inline(always)]
pub(crate) fn scatter_fused_interior_viscous_face(
    residual: &mut InteriorViscousResidualMut<'_>,
    geom: &InteriorViscousFaceGeom,
    flux: &InteriorViscousFaceFlux,
) {
    let owner = geom.owner;
    let neighbor = geom.neighbor;
    let owner_scale = geom.owner_scale;
    let neighbor_scale = geom.neighbor_scale;
    residual.mx[owner] += owner_scale * flux.mx;
    residual.my[owner] += owner_scale * flux.my;
    residual.mz[owner] += owner_scale * flux.mz;
    residual.energy[owner] += owner_scale * flux.energy;
    residual.mx[neighbor] += neighbor_scale * flux.mx;
    residual.my[neighbor] += neighbor_scale * flux.my;
    residual.mz[neighbor] += neighbor_scale * flux.mz;
    residual.energy[neighbor] += neighbor_scale * flux.energy;
}

/// 内面粘性通量 + 残差 scatter 融合路径（无中间 `ViscousFlux`）。
#[inline(always)]
pub(crate) fn accumulate_fused_interior_viscous_face(
    inputs: &InteriorViscousFaceInputs<'_>,
    residual: &mut InteriorViscousResidualMut<'_>,
    geom: InteriorViscousFaceGeom,
) {
    let flux = fused_interior_viscous_face_flux(inputs, &geom);
    scatter_fused_interior_viscous_face(residual, &geom, &flux);
}

fn viscous_face_flux_from_averaged(
    u: [Real; 3],
    grad: &VelocityGradient,
    normal: Vector3,
    mu: Real,
    lambda: Real,
) -> ViscousFlux {
    let div_u = grad.du[0] + grad.dv[1] + grad.dw[2];
    let two_thirds = 2.0 / 3.0;

    let tau_xx = mu * (2.0 * grad.du[0] - two_thirds * div_u);
    let tau_yy = mu * (2.0 * grad.dv[1] - two_thirds * div_u);
    let tau_zz = mu * (2.0 * grad.dw[2] - two_thirds * div_u);
    let tau_xy = mu * (grad.du[1] + grad.dv[0]);
    let tau_xz = mu * (grad.du[2] + grad.dw[0]);
    let tau_yz = mu * (grad.dv[2] + grad.dw[1]);

    let nx = normal.x;
    let ny = normal.y;
    let nz = normal.z;

    let tau_dot_n = [
        tau_xx * nx + tau_xy * ny + tau_xz * nz,
        tau_xy * nx + tau_yy * ny + tau_yz * nz,
        tau_xz * nx + tau_yz * ny + tau_zz * nz,
    ];
    let heat_flux = lambda * (grad.dt[0] * nx + grad.dt[1] * ny + grad.dt[2] * nz);
    let viscous_work = tau_dot_n[0] * u[0] + tau_dot_n[1] * u[1] + tau_dot_n[2] * u[2];

    ViscousFlux {
        mass: 0.0,
        momentum: tau_dot_n,
        // 装配用 -F·A/V → dU/dt = +1/V·Σ(λ∇T·n + τ·u·n)·A，即 F = -(λ∇T·n + τ·u·n)
        energy: -(heat_flux + viscous_work),
    }
}

/// 面两侧粘度与热导率（算术平均）。
pub fn face_transport_coefficients(
    t_l: Real,
    t_r: Real,
    viscous: &ViscousPhysicsConfig,
    eos: &IdealGasEoS,
) -> crate::error::Result<(Real, Real)> {
    viscous.face_transport_coefficients(t_l, t_r, eos)
}

/// 面心梯度平均（壁面传导与内面共用）。
#[must_use]
pub fn average_gradient_for_wall(
    left: &VelocityGradient,
    right: &VelocityGradient,
) -> VelocityGradient {
    average_gradient(left, right)
}

fn average_gradient(left: &VelocityGradient, right: &VelocityGradient) -> VelocityGradient {
    VelocityGradient {
        du: [
            0.5 * (left.du[0] + right.du[0]),
            0.5 * (left.du[1] + right.du[1]),
            0.5 * (left.du[2] + right.du[2]),
        ],
        dv: [
            0.5 * (left.dv[0] + right.dv[0]),
            0.5 * (left.dv[1] + right.dv[1]),
            0.5 * (left.dv[2] + right.dv[2]),
        ],
        dw: [
            0.5 * (left.dw[0] + right.dw[0]),
            0.5 * (left.dw[1] + right.dw[1]),
            0.5 * (left.dw[2] + right.dw[2]),
        ],
        dt: [
            0.5 * (left.dt[0] + right.dt[0]),
            0.5 * (left.dt[1] + right.dt[1]),
            0.5 * (left.dt[2] + right.dt[2]),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::PrimitiveFields;
    use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

    #[test]
    fn interior_soa_flux_matches_structured_path() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(1.0e-5).expect("mu"), 0.72)
                .expect("cfg");
        let base = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let mut fast = base;
        fast.velocity[0] = 100.0;
        let grad_l = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let grad_r = VelocityGradient {
            du: [100.0, 0.0, 0.0],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let (mu, lambda) = face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc");
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let reference = viscous_face_flux(&base, &grad_l, &fast, &grad_r, normal, mu, lambda);

        let mut prim = PrimitiveFields::zeros(2).expect("prim");
        for (cell, state) in [(0, base), (1, fast)] {
            prim.density.values_mut()[cell] = state.density;
            prim.velocity_x.values_mut()[cell] = state.velocity[0];
            prim.velocity_y.values_mut()[cell] = state.velocity[1];
            prim.velocity_z.values_mut()[cell] = state.velocity[2];
            prim.pressure.values_mut()[cell] = state.pressure;
        }
        let mut grad = GradientFields::zeros(2).expect("grad");
        for (cell, g) in [(0, grad_l), (1, grad_r)] {
            grad.du_dx.values_mut()[cell] = g.du[0];
            grad.du_dy.values_mut()[cell] = g.du[1];
            grad.du_dz.values_mut()[cell] = g.du[2];
            grad.dv_dx.values_mut()[cell] = g.dv[0];
            grad.dv_dy.values_mut()[cell] = g.dv[1];
            grad.dv_dz.values_mut()[cell] = g.dv[2];
            grad.dw_dx.values_mut()[cell] = g.dw[0];
            grad.dw_dy.values_mut()[cell] = g.dw[1];
            grad.dw_dz.values_mut()[cell] = g.dw[2];
            grad.dt_dx.values_mut()[cell] = g.dt[0];
            grad.dt_dy.values_mut()[cell] = g.dt[1];
            grad.dt_dz.values_mut()[cell] = g.dt[2];
        }
        let soa = viscous_interior_face_flux(&prim, &grad, 0, 1, normal, mu, lambda);
        assert!((soa.mass - reference.mass).abs() < 1.0e-14);
        for i in 0..3 {
            assert!((soa.momentum[i] - reference.momentum[i]).abs() < 1.0e-14);
        }
        assert!((soa.energy - reference.energy).abs() < 1.0e-14);
    }

    #[test]
    fn fused_interior_flux_matches_soa_path() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(1.0e-5).expect("mu"), 0.72)
                .expect("cfg");
        let base = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let mut fast = base;
        fast.velocity[0] = 100.0;
        let grad_l = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let grad_r = VelocityGradient {
            du: [100.0, 0.0, 0.0],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let (mu, lambda) = face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc");
        let normal = Vector3::new(1.0, 0.0, 0.0);

        let mut prim = PrimitiveFields::zeros(2).expect("prim");
        for (cell, state) in [(0, base), (1, fast)] {
            prim.velocity_x.values_mut()[cell] = state.velocity[0];
            prim.velocity_y.values_mut()[cell] = state.velocity[1];
            prim.velocity_z.values_mut()[cell] = state.velocity[2];
        }
        let mut grad = GradientFields::zeros(2).expect("grad");
        for (cell, g) in [(0, grad_l), (1, grad_r)] {
            grad.du_dx.values_mut()[cell] = g.du[0];
            grad.du_dy.values_mut()[cell] = g.du[1];
            grad.du_dz.values_mut()[cell] = g.du[2];
            grad.dv_dx.values_mut()[cell] = g.dv[0];
            grad.dv_dy.values_mut()[cell] = g.dv[1];
            grad.dv_dz.values_mut()[cell] = g.dv[2];
            grad.dw_dx.values_mut()[cell] = g.dw[0];
            grad.dw_dy.values_mut()[cell] = g.dw[1];
            grad.dw_dz.values_mut()[cell] = g.dw[2];
            grad.dt_dx.values_mut()[cell] = g.dt[0];
            grad.dt_dy.values_mut()[cell] = g.dt[1];
            grad.dt_dz.values_mut()[cell] = g.dt[2];
        }

        let soa = viscous_interior_face_flux(&prim, &grad, 0, 1, normal, mu, lambda);
        let mut mx = [0.0; 2];
        let mut my = [0.0; 2];
        let mut mz = [0.0; 2];
        let mut energy = [0.0; 2];
        accumulate_fused_interior_viscous_face(
            &InteriorViscousFaceInputs {
                grad: &grad.velocity_gradient_slices(),
                ux: prim.velocity_x.values(),
                uy: prim.velocity_y.values(),
                uz: prim.velocity_z.values(),
            },
            &mut InteriorViscousResidualMut {
                mx: &mut mx,
                my: &mut my,
                mz: &mut mz,
                energy: &mut energy,
            },
            InteriorViscousFaceGeom {
                owner: 0,
                neighbor: 1,
                nx: normal.x,
                ny: normal.y,
                nz: normal.z,
                mu,
                lambda,
                owner_scale: -1.0,
                neighbor_scale: 1.0,
            },
        );
        assert!((mx[0] - soa.momentum[0]).abs() < 1.0e-14);
        assert!((mx[1] + soa.momentum[0]).abs() < 1.0e-14);
        assert!((energy[0] + soa.energy).abs() < 1.0e-14);
        assert!((energy[1] - soa.energy).abs() < 1.0e-14);
    }

    #[test]
    fn uniform_state_has_zero_viscous_flux() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = ViscousPhysicsConfig::default();
        let prim = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let grad = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let (mu, lambda) = face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc");
        let flux = viscous_face_flux(
            &prim,
            &grad,
            &prim,
            &grad,
            Vector3::new(1.0, 0.0, 0.0),
            mu,
            lambda,
        );
        assert!(flux.mass.abs() < 1.0e-14);
        assert!(flux.momentum.iter().all(|&m| m.abs() < 1.0e-14));
        assert!(flux.energy.abs() < 1.0e-14);
    }

    #[test]
    fn shear_layer_has_nonzero_momentum_flux() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(1.0e-5).expect("mu"), 0.72)
                .expect("cfg");
        let base = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let mut fast = base;
        fast.velocity[0] = 100.0;
        let grad_l = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let grad_r = VelocityGradient {
            du: [100.0, 0.0, 0.0],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let (mu, lambda) = face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc");
        let flux = viscous_face_flux(
            &base,
            &grad_l,
            &fast,
            &grad_r,
            Vector3::new(1.0, 0.0, 0.0),
            mu,
            lambda,
        );
        assert!(flux.momentum[0].abs() > 0.0);
        assert!(flux.mass.abs() < 1.0e-14);
        let _ = lambda;
    }
}
