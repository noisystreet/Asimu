//! 低马赫预处理面谱半径（f64；LU-SGS 扫掠与单元 \(\sigma\) 共用）。

use crate::core::{Real, Vector3};
use crate::physics::PrimitiveState;

use super::spectral_radius::face_spectral_radius;

/// 低马赫预处理法向谱半径：声速项按 \(\beta=\max(M, M_{\text{cut}})\) 缩放。
#[must_use]
pub(crate) fn face_spectral_radius_preconditioned(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
    mach_cutoff: Real,
) -> Real {
    let lam_l = normal_speed_plus_scaled_sound(prim_l, normal, gamma, mach_cutoff);
    let lam_r = normal_speed_plus_scaled_sound(prim_r, normal, gamma, mach_cutoff);
    0.5 * (lam_l + lam_r)
}

/// 按配置选择常规或低马赫预处理面谱半径（P2：与对角 \(\sigma^\text{LM}\) 一致）。
#[must_use]
pub(crate) fn face_spectral_radius_with_low_mach(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
    low_mach: Option<crate::solver::time::LowMachPreconditioningConfig>,
) -> Real {
    match low_mach {
        Some(cfg) => {
            face_spectral_radius_preconditioned(prim_l, prim_r, normal, gamma, cfg.mach_cutoff)
        }
        None => face_spectral_radius(prim_l, prim_r, normal, gamma),
    }
}

fn normal_speed_plus_scaled_sound(
    prim: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
    mach_cutoff: Real,
) -> Real {
    let rho = prim.density.max(1.0e-30);
    let u = prim.velocity;
    let speed = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let u_n = u[0] * normal.x + u[1] * normal.y + u[2] * normal.z;
    let a = (gamma * prim.pressure.max(1.0e-30) / rho).sqrt();
    let mach = if a > 0.0 { speed / a } else { 0.0 };
    let beta = mach.max(mach_cutoff).min(1.0);
    u_n.abs() + beta * a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::time::LowMachPreconditioningConfig;

    #[test]
    fn face_spectral_radius_with_low_mach_reduces_hyperbolic_lambda() {
        let gamma = 1.4;
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let prim = PrimitiveState {
            density: 1.0,
            pressure: 1.0 / gamma,
            velocity: [0.05, 0.0, 0.0],
            temperature: 300.0,
        };
        let base = face_spectral_radius(&prim, &prim, normal, gamma);
        let lm = face_spectral_radius_with_low_mach(
            &prim,
            &prim,
            normal,
            gamma,
            Some(LowMachPreconditioningConfig { mach_cutoff: 0.1 }),
        );
        assert!(lm < base);
        assert!(lm > prim.velocity[0].abs());
    }
}
