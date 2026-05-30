//! SLAU2 通量矢量分裂（Shima & Kitamura SLAU 质量通量 + JCP 2013 压力耗散修正）。
//!
//! 参考：Shima & Kitamura, AIAA J. 49 (2011)；Kitamura & Shima, J. Comput. Phys. 245 (2013)。

use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::physics::{ConservedState, IdealGasEoS};

use super::flux_common::{face_tangent_basis, normalize_face_normal};
use super::inviscid::InviscidFlux;

/// SLAU2 数值通量 \(\hat{\mathbf{F}} \cdot \mathbf{n}\)（理想气体 Euler）。
pub fn slau2_flux(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<InviscidFlux> {
    let n = normalize_face_normal(normal)?;
    let (t1, t2) = face_tangent_basis(n);
    let frame_l = face_frame_from_conserved(left, eos.gamma, n, t1, t2)?;
    let frame_r = face_frame_from_conserved(right, eos.gamma, n, t1, t2)?;
    validate_face_state(&frame_l)?;
    validate_face_state(&frame_r)?;
    let face_flux = slau2_face_flux(&frame_l, &frame_r, eos.gamma)?;
    Ok(to_global_flux(face_flux, n, t1, t2))
}

#[derive(Clone, Copy)]
struct FaceFrameState {
    rho: Real,
    un: Real,
    ut: [Real; 2],
    p: Real,
}

#[derive(Clone, Copy)]
struct FaceFrameFlux {
    mass: Real,
    normal_momentum: Real,
    tangential_momentum: [Real; 2],
    energy: Real,
}

fn face_frame_from_conserved(
    cons: &ConservedState,
    gamma: Real,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Result<FaceFrameState> {
    let rho = cons.density;
    if rho <= Real::EPSILON {
        return Err(AsimuError::Field("SLAU2 状态须为正密度".to_string()));
    }
    let inv_rho = 1.0 / rho;
    let ux = cons.momentum[0] * inv_rho;
    let uy = cons.momentum[1] * inv_rho;
    let uz = cons.momentum[2] * inv_rho;
    let ke = 0.5 * rho * (ux * ux + uy * uy + uz * uz);
    let pressure = ((gamma - 1.0) * (cons.total_energy - ke)).max(1.0e-6);
    Ok(FaceFrameState {
        rho,
        un: ux * normal.x + uy * normal.y + uz * normal.z,
        ut: [
            ux * t1.x + uy * t1.y + uz * t1.z,
            ux * t2.x + uy * t2.y + uz * t2.z,
        ],
        p: pressure,
    })
}

fn validate_face_state(state: &FaceFrameState) -> Result<()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(AsimuError::Field("SLAU2 状态须为正密度与压力".to_string()));
    }
    Ok(())
}

fn sound_speed(rho: Real, pressure: Real, gamma: Real) -> Real {
    (gamma * pressure / rho).sqrt().max(1.0e-12)
}

fn specific_enthalpy(state: &FaceFrameState, gamma: Real) -> Real {
    let speed_sq = state.un * state.un + state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1];
    gamma / (gamma - 1.0) * state.p / state.rho + 0.5 * speed_sq
}

/// 亚声速开关：\(|M|\ge 1\) 时为 0，否则为 1。
fn supersonic_alpha(mach: Real) -> Real {
    if mach.abs() >= 1.0 { 0.0 } else { 1.0 }
}

/// SLAU 压力因子 \(\beta(M)\)。
fn pressure_beta(mach: Real, alpha: Real) -> Real {
    (1.0 - alpha) * 0.5 * (1.0 + mach.signum()) + alpha * 0.25 * (2.0 - mach) * (mach + 1.0).powi(2)
}

/// SLAU 质量通量中的 \(g(M_L,M_R)\in[0,1]\)。
fn mass_coupling_g(ml: Real, mr: Real) -> Real {
    let left = ml.clamp(-1.0, 0.0);
    let right = mr.clamp(0.0, 1.0);
    -left * right
}

/// SLAU 质量通量压力扩散系数 \(\xi=(1-M_t)^2\)，\(M_t=\min(1,|u_t|/c)\)。
fn mass_pressure_xi(ut_mag_l: Real, ut_mag_r: Real, c: Real) -> Real {
    let vt = (0.5 * (ut_mag_l * ut_mag_l + ut_mag_r * ut_mag_r)).sqrt();
    let m_cap = (vt / c).min(1.0);
    let one_minus = 1.0 - m_cap;
    one_minus * one_minus
}

/// SLAU2 压力第三耗散项系数：\(M_n(2-M_n)\)，\(M_n=\min(1,\tfrac12(|M_L|+|M_R|))\)（高马赫耗散 \(\propto M\)）。
fn slau2_pressure_dissipation(ml: Real, mr: Real) -> Real {
    let m_n = (0.5 * (ml.abs() + mr.abs())).min(1.0);
    m_n * (2.0 - m_n)
}

fn interface_pressure_slau2(left: &FaceFrameState, right: &FaceFrameState, c: Real) -> Real {
    let ml = left.un / c;
    let mr = right.un / c;
    let alpha_l = supersonic_alpha(ml);
    let alpha_r = supersonic_alpha(mr);
    let beta_l = pressure_beta(ml, alpha_l);
    let beta_r = pressure_beta(mr, alpha_r);
    let dp = right.p - left.p;
    let diss = slau2_pressure_dissipation(ml, mr) * (beta_l + beta_r - 1.0) * (left.p + right.p);
    0.5 * (left.p + right.p + (beta_l - beta_r) * dp + diss)
}

fn slau2_face_flux(
    left: &FaceFrameState,
    right: &FaceFrameState,
    gamma: Real,
) -> Result<FaceFrameFlux> {
    let c_l = sound_speed(left.rho, left.p, gamma);
    let c_r = sound_speed(right.rho, right.p, gamma);
    let c = 0.5 * (c_l + c_r);
    let ml = left.un / c;
    let mr = right.un / c;
    let dp = right.p - left.p;
    let g = mass_coupling_g(ml, mr);
    let vn_abs = (left.rho * left.un.abs() + right.rho * right.un.abs()) / (left.rho + right.rho);
    let vn_abs_l = (1.0 - g) * vn_abs + g * left.un.abs();
    let vn_abs_r = (1.0 - g) * vn_abs + g * right.un.abs();
    let ut_l = (left.ut[0] * left.ut[0] + left.ut[1] * left.ut[1]).sqrt();
    let ut_r = (right.ut[0] * right.ut[0] + right.ut[1] * right.ut[1]).sqrt();
    let xi = mass_pressure_xi(ut_l, ut_r, c);
    let mass =
        0.5 * (left.rho * (left.un + vn_abs_l) + right.rho * (right.un - vn_abs_r) - xi * dp / c);
    let p_face = interface_pressure_slau2(left, right, c);
    let hl = specific_enthalpy(left, gamma);
    let hr = specific_enthalpy(right, gamma);
    let mass_plus = 0.5 * (mass + mass.abs());
    let mass_minus = 0.5 * (mass - mass.abs());
    Ok(FaceFrameFlux {
        mass,
        normal_momentum: mass_plus * left.un + mass_minus * right.un + p_face,
        tangential_momentum: [
            mass_plus * left.ut[0] + mass_minus * right.ut[0],
            mass_plus * left.ut[1] + mass_minus * right.ut[1],
        ],
        energy: mass_plus * hl + mass_minus * hr,
    })
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

    #[test]
    fn uniform_supersonic_mass_flux_matches_physical() {
        let eos = IdealGasEoS::new(1.4, 287.0).expect("eos");
        let prim = eos
            .freestream_primitive(2.0, 1.0e5, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let n = Vector3::new(1.0, 0.0, 0.0);
        let slau2 = slau2_flux(&cons, &cons, n, &eos).expect("slau2");
        let phys = physical_inviscid_flux(&cons, &prim, n);
        assert!(approx_eq(slau2.mass, phys.mass, 1.0e-8));
        assert!(slau2.mass > 0.0);
        assert!(slau2.energy.is_finite());
    }

    #[test]
    fn rest_state_has_zero_mass_flux() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let flux = slau2_flux(&cons, &cons, Vector3::new(1.0, 0.0, 0.0), &eos).expect("flux");
        assert!(flux.mass.abs() < 1.0e-10);
    }

    #[test]
    fn sod_interface_slau2_finite_mass_flux() {
        use crate::physics::PrimitiveState;

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
        let cl = ConservedState::from_primitive(&eos, &left_prim).expect("cl");
        let cr = ConservedState::from_primitive(&eos, &right_prim).expect("cr");
        let flux = slau2_flux(&cl, &cr, Vector3::new(1.0, 0.0, 0.0), &eos).expect("flux");
        assert!(flux.mass.is_finite());
        assert!(flux.momentum[0].is_finite());
        assert!(flux.energy.is_finite());
        assert!(flux.mass > 0.0);
    }
}
