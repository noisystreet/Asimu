//! 低马赫预处理面谱半径（f64/f32；LU-SGS 扫掠与单元 \(\sigma\) 共用）。

use crate::core::{Real, Vector3};
use crate::physics::PrimitiveState;
use crate::solver::time::LowMachPreconditioningConfig;

use super::spectral_radius::face_spectral_radius;
use super::spectral_radius_f32::FacePrimitiveLaneF32;

const PHYSICAL_LAMBDA_EPS: Real = 1.0e-12;

/// 单侧预处理双曲谱半径：\(\max(|\lambda_-|,|\lambda_+|,|u_n|)\)，\(\lambda_\pm=\tfrac12(u_n\pm\sqrt{u_n^2+\beta^2 a^2})\)。
#[must_use]
pub(crate) fn side_preconditioned_hyperbolic_lambda(
    prim: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
    cfg: Option<LowMachPreconditioningConfig>,
) -> Real {
    let rho = prim.density.max(1.0e-30);
    let u = prim.velocity;
    let speed = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let u_n = u[0] * normal.x + u[1] * normal.y + u[2] * normal.z;
    let a = (gamma * prim.pressure.max(1.0e-30) / rho).sqrt();
    let physical = u_n.abs() + a;
    let Some(cfg) = cfg else {
        return physical;
    };
    let mach = if a > 0.0 { speed / a } else { 0.0 };
    let beta = cfg.sound_speed_multiplier(mach);
    if beta >= 1.0 - PHYSICAL_LAMBDA_EPS {
        return physical;
    }
    let uc = (u_n * u_n + beta * beta * a * a).sqrt();
    let lambda_plus = 0.5 * (u_n + uc);
    let lambda_minus = 0.5 * (u_n - uc);
    lambda_plus.abs().max(lambda_minus.abs()).max(u_n.abs())
}

/// f32 面心 lane 版本（与 f64 语义一致）。
#[must_use]
pub(crate) fn side_preconditioned_hyperbolic_lambda_f32(
    prim: FacePrimitiveLaneF32,
    normal: [f32; 3],
    gamma: f32,
    cfg: Option<LowMachPreconditioningConfig>,
) -> f32 {
    let rho = prim.rho.max(1.0e-30_f32);
    let vel = prim.velocity;
    let speed = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    let u_n = vel[0] * normal[0] + vel[1] * normal[1] + vel[2] * normal[2];
    let a = (gamma * prim.pressure.max(1.0e-30_f32) / rho).sqrt();
    let physical = u_n.abs() + a;
    let Some(cfg) = cfg else {
        return physical;
    };
    let mach = if a > 0.0 { speed / a } else { 0.0 };
    let beta = cfg.sound_speed_multiplier_f32(mach);
    if beta >= 1.0 - PHYSICAL_LAMBDA_EPS as f32 {
        return physical;
    }
    let uc = (u_n * u_n + beta * beta * a * a).sqrt();
    let lambda_plus = 0.5 * (u_n + uc);
    let lambda_minus = 0.5 * (u_n - uc);
    lambda_plus.abs().max(lambda_minus.abs()).max(u_n.abs())
}

#[must_use]
pub(crate) fn face_spectral_radius_preconditioned(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
    cfg: LowMachPreconditioningConfig,
) -> Real {
    let lam_l = side_preconditioned_hyperbolic_lambda(prim_l, normal, gamma, Some(cfg));
    let lam_r = side_preconditioned_hyperbolic_lambda(prim_r, normal, gamma, Some(cfg));
    0.5 * (lam_l + lam_r)
}

#[must_use]
pub(crate) fn face_spectral_radius_with_low_mach(
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
    low_mach: Option<LowMachPreconditioningConfig>,
) -> Real {
    match low_mach {
        Some(cfg) => face_spectral_radius_preconditioned(prim_l, prim_r, normal, gamma, cfg),
        None => face_spectral_radius(prim_l, prim_r, normal, gamma),
    }
}

#[must_use]
pub(crate) fn face_spectral_radius_f32_with_low_mach_unified(
    left: FacePrimitiveLaneF32,
    right: FacePrimitiveLaneF32,
    normal: [f32; 3],
    gamma: f32,
    low_mach: Option<LowMachPreconditioningConfig>,
) -> f32 {
    let lam_l = side_preconditioned_hyperbolic_lambda_f32(left, normal, gamma, low_mach);
    let lam_r = side_preconditioned_hyperbolic_lambda_f32(right, normal, gamma, low_mach);
    0.5 * (lam_l + lam_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::time::{LowMachBlend, LowMachPreconditioningConfig};

    fn lm_cfg() -> LowMachPreconditioningConfig {
        LowMachPreconditioningConfig {
            mach_cutoff: 0.1,
            max_mach: 0.3,
            blend: LowMachBlend::Smooth,
            jacobian: false,
        }
    }

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
        let lm = face_spectral_radius_with_low_mach(&prim, &prim, normal, gamma, Some(lm_cfg()));
        assert!(lm < base);
        assert!(lm > prim.velocity[0].abs());
    }

    #[test]
    fn smooth_blend_at_high_mach_matches_standard_radius() {
        let gamma = 1.4;
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let prim = PrimitiveState {
            density: 1.0,
            pressure: 1.0 / gamma,
            velocity: [0.5, 0.0, 0.0],
            temperature: 300.0,
        };
        let base = face_spectral_radius(&prim, &prim, normal, gamma);
        let lm = face_spectral_radius_with_low_mach(&prim, &prim, normal, gamma, Some(lm_cfg()));
        assert!((lm - base).abs() < 1.0e-12);
    }

    #[test]
    fn preconditioned_eigenvalue_lambda_below_linear_beta_a_at_rest() {
        let gamma = 1.4;
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let prim = PrimitiveState {
            density: 1.0,
            pressure: 1.0 / gamma,
            velocity: [0.0, 0.0, 0.0],
            temperature: 300.0,
        };
        let cfg = lm_cfg();
        let lm = side_preconditioned_hyperbolic_lambda(&prim, normal, gamma, Some(cfg));
        let a = 1.0;
        let linear = cfg.sound_speed_multiplier(0.0) * a;
        assert!(lm <= linear + 1.0e-12);
    }
}
