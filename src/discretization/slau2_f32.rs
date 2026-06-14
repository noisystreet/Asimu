//! SLAU2 通量矢量分裂 f32 热路径（语义对齐 `slau2.rs`）。
//!
//! 参考：Shima & Kitamura, AIAA J. 49 (2011)；Kitamura & Shima, J. Comput. Phys. 245 (2013)。

use crate::discretization::inviscid_f32::{
    FaceNormalF32, InviscidFluxF32, face_tangent_basis_f32, normalize_face_normal_f32,
};
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::error::{AsimuError, Result};
use crate::physics::IdealGasEoS;

/// f32 SLAU2 数值通量（理想气体 Euler）。
pub fn slau2_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: FaceNormalF32,
    eos: &IdealGasEoS,
) -> Result<InviscidFluxF32> {
    let n = normalize_face_normal_f32(normal)?;
    let (t1, t2) = face_tangent_basis_f32(n);
    let gamma = eos.gamma as f32;
    let frame_l = face_frame_from_primitive_f32(prim_l, n, t1, t2)?;
    let frame_r = face_frame_from_primitive_f32(prim_r, n, t1, t2)?;
    validate_face_state_f32(&frame_l)?;
    validate_face_state_f32(&frame_r)?;
    let face_flux = slau2_face_flux_f32(&frame_l, &frame_r, gamma)?;
    Ok(to_global_flux_f32(face_flux, n, t1, t2))
}

#[derive(Clone, Copy)]
struct FaceFrameStateF32 {
    rho: f32,
    un: f32,
    ut: [f32; 2],
    p: f32,
}

#[derive(Clone, Copy)]
struct FaceFrameFluxF32 {
    mass: f32,
    normal_momentum: f32,
    tangential_momentum: [f32; 2],
    energy: f32,
}

fn face_frame_from_primitive_f32(
    prim: &PrimitiveStateF32,
    normal: FaceNormalF32,
    t1: FaceNormalF32,
    t2: FaceNormalF32,
) -> Result<FaceFrameStateF32> {
    if prim.density <= 0.0 || prim.pressure <= 0.0 {
        return Err(AsimuError::Field(
            "SLAU2 f32 状态须为正密度与压力".to_string(),
        ));
    }
    let [nx, ny, nz] = normal;
    let [t1x, t1y, t1z] = t1;
    let [t2x, t2y, t2z] = t2;
    let u = prim.velocity;
    Ok(FaceFrameStateF32 {
        rho: prim.density,
        un: u[0] * nx + u[1] * ny + u[2] * nz,
        ut: [
            u[0] * t1x + u[1] * t1y + u[2] * t1z,
            u[0] * t2x + u[1] * t2y + u[2] * t2z,
        ],
        p: prim.pressure,
    })
}

fn validate_face_state_f32(state: &FaceFrameStateF32) -> Result<()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(AsimuError::Field(
            "SLAU2 f32 状态须为正密度与压力".to_string(),
        ));
    }
    Ok(())
}

fn sound_speed_f32(rho: f32, pressure: f32, gamma: f32) -> f32 {
    (gamma * pressure / rho).sqrt().max(1.0e-12_f32)
}

fn speed_magnitude_f32(state: &FaceFrameStateF32) -> f32 {
    let speed_sq = state.un * state.un + state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1];
    speed_sq.sqrt()
}

fn specific_enthalpy_f32(state: &FaceFrameStateF32, gamma: f32) -> f32 {
    let speed_sq = state.un * state.un + state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1];
    gamma / (gamma - 1.0) * state.p / state.rho + 0.5 * speed_sq
}

fn supersonic_alpha_f32(mach: f32) -> f32 {
    if mach.abs() >= 1.0 { 0.0 } else { 1.0 }
}

fn pressure_beta_plus_f32(mach: f32, alpha: f32) -> f32 {
    (1.0 - alpha) * 0.5 * (1.0 + mach.signum()) + alpha * 0.25 * (2.0 - mach) * (mach + 1.0).powi(2)
}

fn pressure_beta_minus_f32(mach: f32, alpha: f32) -> f32 {
    pressure_beta_plus_f32(-mach, alpha)
}

fn mass_coupling_g_f32(ml: f32, mr: f32) -> f32 {
    let left = ml.clamp(-1.0, 0.0);
    let right = mr.clamp(0.0, 1.0);
    -left * right
}

fn mass_pressure_xi_f32(speed_l: f32, speed_r: f32, c: f32) -> f32 {
    let speed = (0.5 * (speed_l * speed_l + speed_r * speed_r)).sqrt();
    let m_cap = (speed / c).min(1.0);
    let one_minus = 1.0 - m_cap;
    one_minus * one_minus
}

fn slau2_pressure_dissipation_f32(speed_l: f32, speed_r: f32, c: f32) -> f32 {
    let speed = (0.5 * (speed_l * speed_l + speed_r * speed_r)).sqrt();
    (speed / c).min(1.0)
}

fn interface_pressure_slau2_f32(
    left: &FaceFrameStateF32,
    right: &FaceFrameStateF32,
    c: f32,
) -> f32 {
    let ml = left.un / c;
    let mr = right.un / c;
    let alpha_l = supersonic_alpha_f32(ml);
    let alpha_r = supersonic_alpha_f32(mr);
    let p_plus_l = pressure_beta_plus_f32(ml, alpha_l);
    let p_minus_r = pressure_beta_minus_f32(mr, alpha_r);
    let dp = right.p - left.p;
    let p_bar = 0.5 * (left.p + right.p);
    let diss =
        slau2_pressure_dissipation_f32(speed_magnitude_f32(left), speed_magnitude_f32(right), c)
            * (p_plus_l + p_minus_r - 1.0)
            * p_bar;
    p_bar - 0.5 * (p_plus_l - p_minus_r) * dp + diss
}

fn slau2_face_flux_f32(
    left: &FaceFrameStateF32,
    right: &FaceFrameStateF32,
    gamma: f32,
) -> Result<FaceFrameFluxF32> {
    let c_l = sound_speed_f32(left.rho, left.p, gamma);
    let c_r = sound_speed_f32(right.rho, right.p, gamma);
    let c = 0.5 * (c_l + c_r);
    let ml = left.un / c;
    let mr = right.un / c;
    let dp = right.p - left.p;
    let g = mass_coupling_g_f32(ml, mr);
    let vn_abs = (left.rho * left.un.abs() + right.rho * right.un.abs()) / (left.rho + right.rho);
    let vn_abs_l = (1.0 - g) * vn_abs + g * left.un.abs();
    let vn_abs_r = (1.0 - g) * vn_abs + g * right.un.abs();
    let xi = mass_pressure_xi_f32(speed_magnitude_f32(left), speed_magnitude_f32(right), c);
    let mass =
        0.5 * (left.rho * (left.un + vn_abs_l) + right.rho * (right.un - vn_abs_r) - xi * dp / c);
    let p_face = interface_pressure_slau2_f32(left, right, c);
    let hl = specific_enthalpy_f32(left, gamma);
    let hr = specific_enthalpy_f32(right, gamma);
    let mass_plus = 0.5 * (mass + mass.abs());
    let mass_minus = 0.5 * (mass - mass.abs());
    Ok(FaceFrameFluxF32 {
        mass,
        normal_momentum: mass_plus * left.un + mass_minus * right.un + p_face,
        tangential_momentum: [
            mass_plus * left.ut[0] + mass_minus * right.ut[0],
            mass_plus * left.ut[1] + mass_minus * right.ut[1],
        ],
        energy: mass_plus * hl + mass_minus * hr,
    })
}

fn to_global_flux_f32(
    face: FaceFrameFluxF32,
    normal: FaceNormalF32,
    t1: FaceNormalF32,
    t2: FaceNormalF32,
) -> crate::discretization::inviscid_f32::InviscidFluxF32 {
    let [nx, ny, nz] = normal;
    let [t1x, t1y, t1z] = t1;
    let [t2x, t2y, t2z] = t2;
    crate::discretization::inviscid_f32::InviscidFluxF32 {
        mass: face.mass,
        momentum: [
            face.normal_momentum * nx
                + face.tangential_momentum[0] * t1x
                + face.tangential_momentum[1] * t2x,
            face.normal_momentum * ny
                + face.tangential_momentum[0] * t1y
                + face.tangential_momentum[1] * t2y,
            face.normal_momentum * nz
                + face.tangential_momentum[0] * t1z
                + face.tangential_momentum[1] * t2z,
        ],
        energy: face.energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Real, Vector3, approx_eq};
    use crate::discretization::slau2::slau2_flux;
    use crate::discretization::viscous_boundary_f32::primitive_state_f32_from_real;
    use crate::physics::{ConservedState, PrimitiveState};

    #[test]
    fn slau2_f32_matches_f64_on_primitive_pair() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let left = PrimitiveState {
            density: 1.0,
            velocity: [0.5, 0.0, 0.0],
            pressure: 1.0,
            temperature: 1.0,
        };
        let right = PrimitiveState {
            density: 0.9,
            velocity: [0.3, 0.0, 0.0],
            pressure: 0.95,
            temperature: 1.0,
        };
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let normal_f64 = Vector3::new(1.0, 0.0, 0.0);
        let normal = [1.0_f32, 0.0, 0.0];
        let f64_flux = slau2_flux(&cons_l, &cons_r, normal_f64, &eos).expect("f64");
        let f32_flux = slau2_flux_with_primitives_f32(
            &primitive_state_f32_from_real(left),
            &primitive_state_f32_from_real(right),
            normal,
            &eos,
        )
        .expect("f32");
        assert!(approx_eq(f32_flux.mass as Real, f64_flux.mass, 1.0e-3));
        assert!(approx_eq(f32_flux.energy as Real, f64_flux.energy, 1.0e-2));
    }

    #[test]
    fn uniform_states_match_f64_slau2_f32() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let n_f64 = Vector3::new(1.0, 0.0, 0.0);
        let n = [1.0_f32, 0.0, 0.0];
        for mach in [0.0_f64, 0.5, 2.0] {
            let prim = eos
                .freestream_primitive(mach, 1.0, 1.0, [1.0, 0.0, 0.0])
                .expect("prim");
            let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
            let f64_flux = slau2_flux(&cons, &cons, n_f64, &eos).expect("f64");
            let f32_flux = slau2_flux_with_primitives_f32(
                &primitive_state_f32_from_real(prim),
                &primitive_state_f32_from_real(prim),
                n,
                &eos,
            )
            .expect("f32");
            assert!(approx_eq(f32_flux.mass as Real, f64_flux.mass, 1.0e-3));
            assert!(approx_eq(f32_flux.energy as Real, f64_flux.energy, 1.0e-2));
        }
    }

    #[test]
    fn sod_interface_slau2_f32_finite_mass_flux() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left_prim = PrimitiveState {
            density: 1.0,
            velocity: [0.0, 0.0, 0.0],
            pressure: 1.0,
            temperature: 1.0,
        };
        let right_prim = PrimitiveState {
            density: 0.125,
            velocity: [0.0, 0.0, 0.0],
            pressure: 0.1,
            temperature: 0.25,
        };
        let flux = slau2_flux_with_primitives_f32(
            &primitive_state_f32_from_real(left_prim),
            &primitive_state_f32_from_real(right_prim),
            [1.0_f32, 0.0, 0.0],
            &eos,
        )
        .expect("flux");
        assert!(flux.mass.is_finite());
        assert!(flux.momentum[0].is_finite());
        assert!(flux.energy.is_finite());
        assert!(flux.mass > 0.0);
    }
}
