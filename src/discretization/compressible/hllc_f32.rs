//! HLLC 近似 Riemann 求解器 f32 热路径（语义对齐 `hllc.rs`）。

use crate::discretization::inviscid_f32::{
    FaceNormalF32, InviscidFluxF32, conserved_from_primitive_f32, face_tangent_basis_f32,
    normalize_face_normal_f32,
};
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::error::{AsimuError, Result};
use crate::physics::IdealGasEoS;

/// f32 HLLC 数值通量（理想气体 Euler）。
pub fn hllc_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: FaceNormalF32,
    eos: &IdealGasEoS,
) -> Result<InviscidFluxF32> {
    let n = normalize_face_normal_f32(normal)?;
    let (t1, t2) = face_tangent_basis_f32(n);
    let gamma = eos.gamma as f32;
    let frame_l = to_face_frame_f32(prim_l, n, t1, t2, eos)?;
    let frame_r = to_face_frame_f32(prim_r, n, t1, t2, eos)?;
    let face_flux = hllc_face_frame_flux_f32(&frame_l, &frame_r, gamma)?;
    Ok(to_global_flux_f32(face_flux, n, t1, t2))
}

#[derive(Clone, Copy)]
struct FaceFrameStateF32 {
    rho: f32,
    un: f32,
    ut: [f32; 2],
    p: f32,
    rho_e: f32,
}

#[derive(Clone, Copy)]
struct FaceFrameFluxF32 {
    mass: f32,
    normal_momentum: f32,
    tangential_momentum: [f32; 2],
    energy: f32,
}

#[derive(Clone, Copy)]
struct FaceConservedF32 {
    mass: f32,
    normal_momentum: f32,
    tangential_momentum: [f32; 2],
    energy: f32,
}

fn to_face_frame_f32(
    prim: &PrimitiveStateF32,
    normal: FaceNormalF32,
    t1: FaceNormalF32,
    t2: FaceNormalF32,
    eos: &IdealGasEoS,
) -> Result<FaceFrameStateF32> {
    let cons = conserved_from_primitive_f32(eos, prim)?;
    let [nx, ny, nz] = normal;
    let [t1x, t1y, t1z] = t1;
    let [t2x, t2y, t2z] = t2;
    let rho = prim.density;
    let u = prim.velocity;
    let un = u[0] * nx + u[1] * ny + u[2] * nz;
    let ut0 = u[0] * t1x + u[1] * t1y + u[2] * t1z;
    let ut1 = u[0] * t2x + u[1] * t2y + u[2] * t2z;
    Ok(FaceFrameStateF32 {
        rho,
        un,
        ut: [ut0, ut1],
        p: prim.pressure,
        rho_e: cons.total_energy,
    })
}

fn hllc_face_frame_flux_f32(
    left: &FaceFrameStateF32,
    right: &FaceFrameStateF32,
    gamma: f32,
) -> Result<FaceFrameFluxF32> {
    validate_face_state_f32(left)?;
    validate_face_state_f32(right)?;
    let (p_star, u_star) = solve_star_pressure_velocity_f32(left, right, gamma)?;
    let s_l = wave_speed_left_f32(p_star, left, gamma);
    let s_r = wave_speed_right_f32(p_star, right, gamma);
    let flux_l = physical_face_flux_f32(left);
    let flux_r = physical_face_flux_f32(right);
    if s_l >= 0.0 {
        return Ok(flux_l);
    }
    if u_star >= 0.0 {
        let u_l = face_conserved_f32(left);
        let u_star_l = star_state_f32(left, p_star, u_star, s_l);
        return Ok(add_fluxes_f32(
            flux_l,
            scale_conserved_f32(sub_conserved_f32(u_star_l, u_l), s_l),
        ));
    }
    if s_r >= 0.0 {
        let u_r = face_conserved_f32(right);
        let u_star_r = star_state_f32(right, p_star, u_star, s_r);
        return Ok(add_fluxes_f32(
            flux_r,
            scale_conserved_f32(sub_conserved_f32(u_star_r, u_r), s_r),
        ));
    }
    Ok(flux_r)
}

fn validate_face_state_f32(state: &FaceFrameStateF32) -> Result<()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(AsimuError::Field(
            "HLLC f32 状态须为正密度与压力".to_string(),
        ));
    }
    Ok(())
}

fn solve_star_pressure_velocity_f32(
    left: &FaceFrameStateF32,
    right: &FaceFrameStateF32,
    gamma: f32,
) -> Result<(f32, f32)> {
    let a_l = sound_speed_f32(left.rho, left.p, gamma);
    let a_r = sound_speed_f32(right.rho, right.p, gamma);
    let mut p = pvrs_initial_pressure_f32(left, right, a_l, a_r);
    for _ in 0..64 {
        let f_l = pressure_function_f32(p, left, gamma);
        let f_r = pressure_function_f32(p, right, gamma);
        let g_l = pressure_derivative_f32(p, left, gamma);
        let g_r = pressure_derivative_f32(p, right, gamma);
        let denom = g_l + g_r;
        if denom.abs() < 1.0e-30 {
            break;
        }
        let dp = -(f_l + f_r + right.un - left.un) / denom;
        p = (p + dp).max(1.0e-12);
        if dp.abs() < 1.0e-6 * (p + 1.0) {
            break;
        }
    }
    let f_l = pressure_function_f32(p, left, gamma);
    let f_r = pressure_function_f32(p, right, gamma);
    let velocity = 0.5 * (left.un + right.un) + 0.5 * (f_r - f_l);
    Ok((p, velocity))
}

fn pvrs_initial_pressure_f32(
    left: &FaceFrameStateF32,
    right: &FaceFrameStateF32,
    a_l: f32,
    a_r: f32,
) -> f32 {
    let rho_bar = 0.5 * (left.rho + right.rho);
    let a_bar = 0.5 * (a_l + a_r);
    let p_pvrs = 0.5 * (left.p + right.p) - 0.125 * (right.un - left.un) * rho_bar * a_bar;
    let p_min = 1.0e-6;
    let p_max = 2.0 * left.p.max(right.p);
    p_pvrs.clamp(p_min, p_max.max(p_min))
}

fn pressure_function_f32(p: f32, state: &FaceFrameStateF32, gamma: f32) -> f32 {
    let a = sound_speed_f32(state.rho, state.p, gamma);
    if p > state.p {
        let ratio = p / state.p;
        let term = (gamma + 1.0) / (2.0 * gamma) * (ratio - 1.0) + 1.0;
        (p - state.p) / (state.rho * a * term.sqrt())
    } else {
        2.0 * a / (gamma - 1.0) * ((p / state.p).powf((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    }
}

fn pressure_derivative_f32(p: f32, state: &FaceFrameStateF32, gamma: f32) -> f32 {
    let a = sound_speed_f32(state.rho, state.p, gamma);
    if p > state.p {
        let ratio = p / state.p;
        let term = (gamma + 1.0) / (2.0 * gamma) * (ratio - 1.0) + 1.0;
        term.sqrt() / (state.rho * a)
            + (p - state.p) / (2.0 * state.rho * a * term.powf(1.5)) * (gamma + 1.0)
                / (2.0 * gamma * state.p)
    } else {
        1.0 / (state.rho * a) * (p / state.p).powf(-(gamma + 1.0) / (2.0 * gamma))
    }
}

fn sound_speed_f32(rho: f32, pressure: f32, gamma: f32) -> f32 {
    (gamma * pressure / rho).sqrt()
}

fn wave_speed_left_f32(p_star: f32, left: &FaceFrameStateF32, gamma: f32) -> f32 {
    if p_star <= left.p {
        left.un - sound_speed_f32(left.rho, left.p, gamma)
    } else {
        left.un
            - sound_speed_f32(left.rho, left.p, gamma)
                * ((gamma + 1.0) / (2.0 * gamma) * (p_star / left.p - 1.0) + 1.0).sqrt()
    }
}

fn wave_speed_right_f32(p_star: f32, right: &FaceFrameStateF32, gamma: f32) -> f32 {
    if p_star <= right.p {
        right.un + sound_speed_f32(right.rho, right.p, gamma)
    } else {
        right.un
            + sound_speed_f32(right.rho, right.p, gamma)
                * ((gamma + 1.0) / (2.0 * gamma) * (p_star / right.p - 1.0) + 1.0).sqrt()
    }
}

fn star_state_f32(
    state: &FaceFrameStateF32,
    _p_star: f32,
    u_star: f32,
    s_k: f32,
) -> FaceConservedF32 {
    let rho_star = state.rho * (s_k - state.un) / (s_k - u_star);
    let specific_e = state.rho_e / state.rho
        + (u_star - state.un) * (u_star + state.p / (state.rho * (s_k - state.un)));
    FaceConservedF32 {
        mass: rho_star,
        normal_momentum: rho_star * u_star,
        tangential_momentum: [rho_star * state.ut[0], rho_star * state.ut[1]],
        energy: rho_star * specific_e,
    }
}

fn face_conserved_f32(state: &FaceFrameStateF32) -> FaceConservedF32 {
    FaceConservedF32 {
        mass: state.rho,
        normal_momentum: state.rho * state.un,
        tangential_momentum: [state.rho * state.ut[0], state.rho * state.ut[1]],
        energy: state.rho_e,
    }
}

fn physical_face_flux_f32(state: &FaceFrameStateF32) -> FaceFrameFluxF32 {
    FaceFrameFluxF32 {
        mass: state.rho * state.un,
        normal_momentum: state.rho * state.un * state.un + state.p,
        tangential_momentum: [
            state.rho * state.un * state.ut[0],
            state.rho * state.un * state.ut[1],
        ],
        energy: (state.rho_e + state.p) * state.un,
    }
}

fn sub_conserved_f32(a: FaceConservedF32, b: FaceConservedF32) -> FaceConservedF32 {
    FaceConservedF32 {
        mass: a.mass - b.mass,
        normal_momentum: a.normal_momentum - b.normal_momentum,
        tangential_momentum: [
            a.tangential_momentum[0] - b.tangential_momentum[0],
            a.tangential_momentum[1] - b.tangential_momentum[1],
        ],
        energy: a.energy - b.energy,
    }
}

fn scale_conserved_f32(state: FaceConservedF32, scale: f32) -> FaceConservedF32 {
    FaceConservedF32 {
        mass: scale * state.mass,
        normal_momentum: scale * state.normal_momentum,
        tangential_momentum: [
            scale * state.tangential_momentum[0],
            scale * state.tangential_momentum[1],
        ],
        energy: scale * state.energy,
    }
}

fn add_fluxes_f32(base: FaceFrameFluxF32, correction: FaceConservedF32) -> FaceFrameFluxF32 {
    FaceFrameFluxF32 {
        mass: base.mass + correction.mass,
        normal_momentum: base.normal_momentum + correction.normal_momentum,
        tangential_momentum: [
            base.tangential_momentum[0] + correction.tangential_momentum[0],
            base.tangential_momentum[1] + correction.tangential_momentum[1],
        ],
        energy: base.energy + correction.energy,
    }
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
    use crate::discretization::hllc::hllc_flux_with_primitives;
    use crate::discretization::viscous_boundary_f32::primitive_state_f32_from_real;
    use crate::physics::{ConservedState, PrimitiveState};

    #[test]
    fn hllc_f32_matches_f64_on_sod_interface() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let left = PrimitiveState {
            density: 1.0,
            velocity: [0.0, 0.0, 0.0],
            pressure: 1.0,
            temperature: 1.0,
        };
        let right = PrimitiveState {
            density: 0.125,
            velocity: [0.0, 0.0, 0.0],
            pressure: 0.1,
            temperature: 1.0,
        };
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let normal_f64 = Vector3::new(1.0, 0.0, 0.0);
        let normal = [1.0_f32, 0.0, 0.0];
        let f64_flux = hllc_flux_with_primitives(&cons_l, &cons_r, &left, &right, normal_f64, &eos)
            .expect("f64");
        let f32_flux = hllc_flux_with_primitives_f32(
            &primitive_state_f32_from_real(left),
            &primitive_state_f32_from_real(right),
            normal,
            &eos,
        )
        .expect("f32");
        assert!(approx_eq(f32_flux.mass as Real, f64_flux.mass, 1.0e-3));
        assert!(approx_eq(f32_flux.energy as Real, f64_flux.energy, 1.0e-2));
    }
}
