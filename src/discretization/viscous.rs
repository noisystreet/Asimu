//! 可压缩 Navier-Stokes 粘性通量（Newtonian + Fourier 热传导）。

use crate::core::{Real, Vector3};
use crate::discretization::InviscidFlux;
use crate::discretization::gradient::VelocityGradient;
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
        energy: viscous_work + heat_flux,
    }
}

/// 面两侧粘度与热导率（算术平均）。
pub fn face_transport_coefficients(
    t_l: Real,
    t_r: Real,
    viscous: &ViscousPhysicsConfig,
    eos: &IdealGasEoS,
) -> crate::error::Result<(Real, Real)> {
    let mu_l = viscous.model.dynamic_viscosity(t_l)?;
    let mu_r = viscous.model.dynamic_viscosity(t_r)?;
    let mu = 0.5 * (mu_l + mu_r);
    let lambda_l = viscous
        .model
        .thermal_conductivity(t_l, eos, viscous.prandtl)?;
    let lambda_r = viscous
        .model
        .thermal_conductivity(t_r, eos, viscous.prandtl)?;
    Ok((mu, 0.5 * (lambda_l + lambda_r)))
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
    use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

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
