//! 谱半径 f32 热路径（非结构 \(\sigma_i\) 原生 f32 输出）。

use crate::error::Result;
use crate::field::PrimitiveFieldsT;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// 面一侧 f32 原始变量（谱半径用）。
pub struct FacePrimitiveLaneF32 {
    pub rho: f32,
    pub pressure: f32,
    pub velocity: [f32; 3],
}

/// 内界面法向谱半径 \((|u_n|+a)_L+(|u_n|+a)_R)/2\)（f32 计算）。
#[must_use]
pub fn face_spectral_radius_f32(
    left: FacePrimitiveLaneF32,
    right: FacePrimitiveLaneF32,
    normal: [f32; 3],
    gamma: f32,
) -> f32 {
    let lam_l = normal_speed_plus_sound_f32(left.rho, left.pressure, left.velocity, normal, gamma);
    let lam_r =
        normal_speed_plus_sound_f32(right.rho, right.pressure, right.velocity, normal, gamma);
    0.5 * (lam_l + lam_r)
}

fn normal_speed_plus_sound_f32(
    rho: f32,
    pressure: f32,
    velocity: [f32; 3],
    normal: [f32; 3],
    gamma: f32,
) -> f32 {
    let rho = rho.max(1.0e-30_f32);
    let u_n = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
    let a = (gamma * pressure.max(1.0e-30_f32) / rho).sqrt();
    u_n.abs() + a
}

/// 每单元 \(\max(\nu,\alpha)\)（f32 原始变量；输运系数边界转 Real 一次）。
pub fn cell_viscous_diffusivity_max_f32(
    primitives: &PrimitiveFieldsT<f32>,
    eos: &IdealGasEoS,
    viscous: &ViscousPhysicsConfig,
) -> Result<Vec<f32>> {
    let n = primitives.num_cells();
    let mut diff = Vec::with_capacity(n);
    for i in 0..n {
        let rho = primitives.density.values()[i].max(1.0e-30_f32);
        let pressure = primitives.pressure.values()[i].max(1.0e-30_f32);
        let t_star = viscous.static_temperature_f32(pressure, rho, eos);
        let (mu_eff, _lambda) = viscous.face_transport_coefficients_f32(t_star, t_star, eos)?;
        let nu = mu_eff / rho;
        let alpha = mu_eff / (rho * viscous.prandtl as f32);
        diff.push(nu.max(alpha));
    }
    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Vector3, approx_eq};
    use crate::physics::PrimitiveState;

    #[test]
    fn face_spectral_radius_f32_matches_f64_reference() {
        let prim_l = PrimitiveState {
            density: 1.2,
            velocity: [100.0, 0.0, 0.0],
            pressure: 101_325.0,
            temperature: 300.0,
        };
        let prim_r = PrimitiveState {
            density: 1.0,
            velocity: [80.0, 0.0, 0.0],
            pressure: 100_000.0,
            temperature: 290.0,
        };
        let normal_f64 = Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let gamma = 1.4_f32;
        let f32_val = face_spectral_radius_f32(
            FacePrimitiveLaneF32 {
                rho: prim_l.density as f32,
                pressure: prim_l.pressure as f32,
                velocity: [
                    prim_l.velocity[0] as f32,
                    prim_l.velocity[1] as f32,
                    prim_l.velocity[2] as f32,
                ],
            },
            FacePrimitiveLaneF32 {
                rho: prim_r.density as f32,
                pressure: prim_r.pressure as f32,
                velocity: [
                    prim_r.velocity[0] as f32,
                    prim_r.velocity[1] as f32,
                    prim_r.velocity[2] as f32,
                ],
            },
            [1.0_f32, 0.0, 0.0],
            gamma,
        );
        let f64_val = crate::solver::compressible::spectral_radius::face_spectral_radius(
            &prim_l, &prim_r, normal_f64, 1.4,
        );
        assert!(approx_eq(f32_val as f64, f64_val, 1.0e-3));
    }
}
