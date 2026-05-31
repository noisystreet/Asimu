//! HLLC 近似 Riemann 求解器（Toro 2009 §10）。
//!
//! 理论：[`docs/theory/inviscid_flux.md`](../../docs/theory/inviscid_flux.md) §5

use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::field::primitive_from_conserved;
use crate::physics::{
    ConservedState, IdealGasEoS, RiemannPrimitive1d, solve_star_pressure_velocity,
};

use super::flux_common::{face_tangent_basis, normalize_face_normal};
use super::inviscid::InviscidFlux;

/// HLLC 数值通量 \(\hat{\mathbf{F}} \cdot \mathbf{n}\)（理想气体 Euler）。
pub fn hllc_flux(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<InviscidFlux> {
    let prim_l = primitive_from_conserved(eos, left)?;
    let prim_r = primitive_from_conserved(eos, right)?;
    hllc_flux_with_primitives(left, right, &prim_l, &prim_r, normal, eos)
}

/// 使用已恢复的单元原始变量（跳过守恒→原始）。
pub fn hllc_flux_with_primitives(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &crate::physics::PrimitiveState,
    prim_r: &crate::physics::PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<InviscidFlux> {
    let n = normalize_face_normal(normal)?;
    let (t1, t2) = face_tangent_basis(n);
    let frame_l = to_face_frame(left, prim_l, n, t1, t2);
    let frame_r = to_face_frame(right, prim_r, n, t1, t2);
    let face_flux = hllc_face_frame_flux(&frame_l, &frame_r, eos.gamma)?;
    Ok(to_global_flux(face_flux, n, t1, t2))
}

#[derive(Clone, Copy)]
struct FaceFrameState {
    rho: Real,
    un: Real,
    ut: [Real; 2],
    p: Real,
    rho_e: Real,
}

#[derive(Clone, Copy)]
struct FaceFrameFlux {
    mass: Real,
    normal_momentum: Real,
    tangential_momentum: [Real; 2],
    energy: Real,
}

fn to_face_frame(
    cons: &ConservedState,
    prim: &crate::physics::PrimitiveState,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> FaceFrameState {
    let inv_rho = 1.0 / cons.density;
    let ux = cons.momentum[0] * inv_rho;
    let uy = cons.momentum[1] * inv_rho;
    let uz = cons.momentum[2] * inv_rho;
    FaceFrameState {
        rho: cons.density,
        un: ux * normal.x + uy * normal.y + uz * normal.z,
        ut: [
            ux * t1.x + uy * t1.y + uz * t1.z,
            ux * t2.x + uy * t2.y + uz * t2.z,
        ],
        p: prim.pressure,
        rho_e: cons.total_energy,
    }
}

fn hllc_face_frame_flux(
    left: &FaceFrameState,
    right: &FaceFrameState,
    gamma: Real,
) -> Result<FaceFrameFlux> {
    validate_face_state(left)?;
    validate_face_state(right)?;
    let left_1d = RiemannPrimitive1d {
        density: left.rho,
        velocity: left.un,
        pressure: left.p,
    };
    let right_1d = RiemannPrimitive1d {
        density: right.rho,
        velocity: right.un,
        pressure: right.p,
    };
    let (p_star, u_star) = solve_star_pressure_velocity(left_1d, right_1d, gamma)?;
    let s_l = wave_speed_left(p_star, left, gamma);
    let s_r = wave_speed_right(p_star, right, gamma);
    let flux_l = physical_face_flux(left);
    let flux_r = physical_face_flux(right);
    if s_l >= 0.0 {
        return Ok(flux_l);
    }
    if u_star >= 0.0 {
        let u_l = face_conserved(left);
        let u_star_l = star_state(left, p_star, u_star, s_l);
        return Ok(add_fluxes(
            flux_l,
            scale_conserved(sub_conserved(u_star_l, u_l), s_l),
        ));
    }
    if s_r >= 0.0 {
        let u_r = face_conserved(right);
        let u_star_r = star_state(right, p_star, u_star, s_r);
        return Ok(add_fluxes(
            flux_r,
            scale_conserved(sub_conserved(u_star_r, u_r), s_r),
        ));
    }
    Ok(flux_r)
}

fn validate_face_state(state: &FaceFrameState) -> Result<()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(AsimuError::Field("HLLC 状态须为正密度与压力".to_string()));
    }
    Ok(())
}

fn sound_speed(rho: Real, pressure: Real, gamma: Real) -> Real {
    (gamma * pressure / rho).sqrt()
}

fn wave_speed_left(p_star: Real, left: &FaceFrameState, gamma: Real) -> Real {
    if p_star <= left.p {
        left.un - sound_speed(left.rho, left.p, gamma)
    } else {
        left.un
            - sound_speed(left.rho, left.p, gamma)
                * ((gamma + 1.0) / (2.0 * gamma) * (p_star / left.p - 1.0) + 1.0).sqrt()
    }
}

fn wave_speed_right(p_star: Real, right: &FaceFrameState, gamma: Real) -> Real {
    if p_star <= right.p {
        right.un + sound_speed(right.rho, right.p, gamma)
    } else {
        right.un
            + sound_speed(right.rho, right.p, gamma)
                * ((gamma + 1.0) / (2.0 * gamma) * (p_star / right.p - 1.0) + 1.0).sqrt()
    }
}

fn star_state(state: &FaceFrameState, _p_star: Real, u_star: Real, s_k: Real) -> FaceConserved {
    let rho_star = state.rho * (s_k - state.un) / (s_k - u_star);
    let specific_e = state.rho_e / state.rho
        + (u_star - state.un) * (u_star + state.p / (state.rho * (s_k - state.un)));
    FaceConserved {
        mass: rho_star,
        normal_momentum: rho_star * u_star,
        tangential_momentum: [rho_star * state.ut[0], rho_star * state.ut[1]],
        energy: rho_star * specific_e,
    }
}

#[derive(Clone, Copy)]
struct FaceConserved {
    mass: Real,
    normal_momentum: Real,
    tangential_momentum: [Real; 2],
    energy: Real,
}

fn face_conserved(state: &FaceFrameState) -> FaceConserved {
    FaceConserved {
        mass: state.rho,
        normal_momentum: state.rho * state.un,
        tangential_momentum: [state.rho * state.ut[0], state.rho * state.ut[1]],
        energy: state.rho_e,
    }
}

fn physical_face_flux(state: &FaceFrameState) -> FaceFrameFlux {
    FaceFrameFlux {
        mass: state.rho * state.un,
        normal_momentum: state.rho * state.un * state.un + state.p,
        tangential_momentum: [
            state.rho * state.un * state.ut[0],
            state.rho * state.un * state.ut[1],
        ],
        energy: (state.rho_e + state.p) * state.un,
    }
}

fn sub_conserved(a: FaceConserved, b: FaceConserved) -> FaceConserved {
    FaceConserved {
        mass: a.mass - b.mass,
        normal_momentum: a.normal_momentum - b.normal_momentum,
        tangential_momentum: [
            a.tangential_momentum[0] - b.tangential_momentum[0],
            a.tangential_momentum[1] - b.tangential_momentum[1],
        ],
        energy: a.energy - b.energy,
    }
}

fn scale_conserved(state: FaceConserved, scale: Real) -> FaceConserved {
    FaceConserved {
        mass: scale * state.mass,
        normal_momentum: scale * state.normal_momentum,
        tangential_momentum: [
            scale * state.tangential_momentum[0],
            scale * state.tangential_momentum[1],
        ],
        energy: scale * state.energy,
    }
}

fn add_fluxes(base: FaceFrameFlux, correction: FaceConserved) -> FaceFrameFlux {
    FaceFrameFlux {
        mass: base.mass + correction.mass,
        normal_momentum: base.normal_momentum + correction.normal_momentum,
        tangential_momentum: [
            base.tangential_momentum[0] + correction.tangential_momentum[0],
            base.tangential_momentum[1] + correction.tangential_momentum[1],
        ],
        energy: base.energy + correction.energy,
    }
}

fn to_global_flux(face: FaceFrameFlux, normal: Vector3, t1: Vector3, t2: Vector3) -> InviscidFlux {
    InviscidFlux {
        mass: face.mass,
        momentum: [
            face.normal_momentum * normal.x
                + face.tangential_momentum[0] * t1.x
                + face.tangential_momentum[1] * t2.x,
            face.normal_momentum * normal.y
                + face.tangential_momentum[0] * t1.y
                + face.tangential_momentum[1] * t2.y,
            face.normal_momentum * normal.z
                + face.tangential_momentum[0] * t1.z
                + face.tangential_momentum[1] * t2.z,
        ],
        energy: face.energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::inviscid::physical_inviscid_flux;
    use crate::physics::PrimitiveState;

    fn sod_left(eos: &IdealGasEoS) -> ConservedState {
        ConservedState::from_primitive(
            eos,
            &PrimitiveState {
                density: 1.0,
                velocity: [0.0, 0.0, 0.0],
                pressure: 1.0,
                temperature: 1.0,
            },
        )
        .expect("left")
    }

    fn sod_right(eos: &IdealGasEoS) -> ConservedState {
        ConservedState::from_primitive(
            eos,
            &PrimitiveState {
                density: 0.125,
                velocity: [0.0, 0.0, 0.0],
                pressure: 0.1,
                temperature: 1.0,
            },
        )
        .expect("right")
    }

    #[test]
    fn identical_states_match_physical_flux() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let prim = eos
            .freestream_primitive(0.3, 1.0, 1.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let n = Vector3::new(1.0, 0.0, 0.0);
        let hllc = hllc_flux(&cons, &cons, n, &eos).expect("hllc");
        let phys = physical_inviscid_flux(&cons, &prim, n);
        assert!(approx_eq(hllc.mass, phys.mass, 1.0e-10));
        assert!(approx_eq(hllc.momentum[0], phys.momentum[0], 1.0e-10));
        assert!(approx_eq(hllc.energy, phys.energy, 1.0e-10));
    }

    #[test]
    fn sod_interface_hllc_flux_matches_reference_values() {
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left = sod_left(&eos);
        let right = sod_right(&eos);
        let n = Vector3::new(1.0, 0.0, 0.0);
        let flux = hllc_flux(&left, &right, n, &eos).expect("flux");
        assert!(approx_eq(
            flux.momentum[0],
            0.384_823_518_755_483_26,
            1.0e-8
        ));
        assert!(approx_eq(flux.mass, 0.519_919_020_532_720_8, 1.0e-8));
        assert!(approx_eq(flux.energy, 1.249_169_532_541_802_2, 1.0e-8));
    }
}
