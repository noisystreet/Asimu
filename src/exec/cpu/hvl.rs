//! Hanel–Van Leer FVS 四路批处理（`simd-fvm`：亚音速能量分裂可用 f64x4）。

use crate::core::{Real, Vector3};
use crate::physics::ConservedState;

use super::roe::InviscidFlux5;

/// 四路一阶 Hanel–Van Leer 面通量（守恒态输入）。
pub fn face_inviscid_flux_first_order_hanel_batch4(
    left: [&ConservedState; 4],
    right: [&ConservedState; 4],
    normal: [Vector3; 4],
    gamma: Real,
) -> [Option<InviscidFlux5>; 4] {
    #[cfg(feature = "simd-fvm")]
    {
        hanel_batch4_simd(left, right, normal, gamma)
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        let mut out = [None; 4];
        for i in 0..4 {
            out[i] = hanel_lane(left[i], right[i], normal[i], gamma);
        }
        out
    }
}

#[cfg(not(feature = "simd-fvm"))]
fn hanel_lane(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    gamma: Real,
) -> Option<InviscidFlux5> {
    let n = normalize(normal)?;
    let (t1, t2) = face_tangent_basis(n);
    let frame_l = face_frame_from_conserved(left, gamma, n, t1, t2)?;
    let frame_r = face_frame_from_conserved(right, gamma, n, t1, t2)?;
    validate_face_state(&frame_l).ok()?;
    validate_face_state(&frame_r).ok()?;
    let flux_l_plus = fvs_positive_hanel(&frame_l, gamma);
    let flux_r_minus = fvs_negative_hanel(&frame_r, gamma);
    let face_flux = add_face_fluxes(flux_l_plus, flux_r_minus);
    Some(to_global_flux(face_flux, n, t1, t2))
}

#[cfg(feature = "simd-fvm")]
fn hanel_batch4_simd(
    left: [&ConservedState; 4],
    right: [&ConservedState; 4],
    normal: [Vector3; 4],
    gamma: Real,
) -> [Option<InviscidFlux5>; 4] {
    let mut out = [None; 4];
    let mut frame_l = [None; 4];
    let mut frame_r = [None; 4];
    let mut n_arr = [Vector3::new(0.0, 0.0, 0.0); 4];
    let mut t1_arr = [Vector3::new(0.0, 0.0, 0.0); 4];
    let mut t2_arr = [Vector3::new(0.0, 0.0, 0.0); 4];
    for i in 0..4 {
        let Some(n) = normalize(normal[i]) else {
            return out;
        };
        let (t1, t2) = face_tangent_basis(n);
        let Some(fl) = face_frame_from_conserved(left[i], gamma, n, t1, t2) else {
            return out;
        };
        let Some(fr) = face_frame_from_conserved(right[i], gamma, n, t1, t2) else {
            return out;
        };
        if validate_face_state(&fl).is_err() || validate_face_state(&fr).is_err() {
            return out;
        }
        n_arr[i] = n;
        t1_arr[i] = t1;
        t2_arr[i] = t2;
        frame_l[i] = Some(fl);
        frame_r[i] = Some(fr);
    }
    let fl0 = frame_l[0].expect("frame_l");
    let fl1 = frame_l[1].expect("frame_l");
    let fl2 = frame_l[2].expect("frame_l");
    let fl3 = frame_l[3].expect("frame_l");
    let fr0 = frame_r[0].expect("frame_r");
    let fr1 = frame_r[1].expect("frame_r");
    let fr2 = frame_r[2].expect("frame_r");
    let fr3 = frame_r[3].expect("frame_r");
    let plus = fvs_positive_hanel_batch4([fl0, fl1, fl2, fl3], gamma);
    let minus = fvs_negative_hanel_batch4([fr0, fr1, fr2, fr3], gamma);
    for i in 0..4 {
        let face_flux = add_face_fluxes(plus[i], minus[i]);
        out[i] = Some(to_global_flux(face_flux, n_arr[i], t1_arr[i], t2_arr[i]));
    }
    out
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

fn normalize(n: Vector3) -> Option<Vector3> {
    let mag = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
    if mag < Real::EPSILON {
        return None;
    }
    Some(Vector3::new(n.x / mag, n.y / mag, n.z / mag))
}

fn face_tangent_basis(normal: Vector3) -> (Vector3, Vector3) {
    let ref_vec = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let t1 = cross(normal, ref_vec);
    let t1 = normalize_unchecked(t1);
    let t2 = cross(normal, t1);
    (t1, normalize_unchecked(t2))
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn normalize_unchecked(v: Vector3) -> Vector3 {
    let mag = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    Vector3::new(v.x / mag, v.y / mag, v.z / mag)
}

fn face_frame_from_conserved(
    cons: &ConservedState,
    gamma: Real,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> Option<FaceFrameState> {
    let rho = cons.density;
    if rho <= Real::EPSILON {
        return None;
    }
    let inv_rho = 1.0 / rho;
    let ux = cons.momentum[0] * inv_rho;
    let uy = cons.momentum[1] * inv_rho;
    let uz = cons.momentum[2] * inv_rho;
    let ke = 0.5 * rho * (ux * ux + uy * uy + uz * uz);
    let pressure = ((gamma - 1.0) * (cons.total_energy - ke)).max(1.0e-6);
    let internal = pressure / (gamma - 1.0);
    let rho_e = ke + internal;
    Some(FaceFrameState {
        rho,
        un: ux * normal.x + uy * normal.y + uz * normal.z,
        ut: [
            ux * t1.x + uy * t1.y + uz * t1.z,
            ux * t2.x + uy * t2.y + uz * t2.z,
        ],
        p: pressure,
        rho_e,
    })
}

fn validate_face_state(state: &FaceFrameState) -> Result<(), ()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(());
    }
    Ok(())
}

fn sound_speed(rho: Real, pressure: Real, gamma: Real) -> Real {
    (gamma * pressure / rho).sqrt()
}

fn specific_enthalpy(state: &FaceFrameState, gamma: Real) -> Real {
    let a = sound_speed(state.rho, state.p, gamma);
    a * a / (gamma - 1.0)
        + 0.5 * (state.un * state.un + state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1])
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

fn fvs_positive_hanel(state: &FaceFrameState, gamma: Real) -> FaceFrameFlux {
    fvs_positive_flux(state, gamma, true)
}

fn fvs_negative_hanel(state: &FaceFrameState, gamma: Real) -> FaceFrameFlux {
    let full = physical_face_flux(state);
    let plus = fvs_positive_hanel(state, gamma);
    FaceFrameFlux {
        mass: full.mass - plus.mass,
        normal_momentum: full.normal_momentum - plus.normal_momentum,
        tangential_momentum: [
            full.tangential_momentum[0] - plus.tangential_momentum[0],
            full.tangential_momentum[1] - plus.tangential_momentum[1],
        ],
        energy: full.energy - plus.energy,
    }
}

fn fvs_positive_flux(state: &FaceFrameState, gamma: Real, hanel_energy: bool) -> FaceFrameFlux {
    let full = physical_face_flux(state);
    let a = sound_speed(state.rho, state.p, gamma);
    let mach = state.un / a;
    if mach <= -1.0 {
        return FaceFrameFlux {
            mass: 0.0,
            normal_momentum: 0.0,
            tangential_momentum: [0.0, 0.0],
            energy: 0.0,
        };
    }
    if mach >= 1.0 {
        return full;
    }
    let mach_plus = mach + 1.0;
    let mass_plus = 0.25 * state.rho * a * mach_plus * mach_plus;
    let normal_velocity_plus = ((gamma - 1.0) * state.un + 2.0 * a) / gamma;
    let tangential_ke = 0.5 * (state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1]);
    let energy = if hanel_energy {
        mass_plus * specific_enthalpy(state, gamma)
    } else {
        let acoustic_energy =
            ((gamma - 1.0) * state.un + 2.0 * a).powi(2) / (2.0 * (gamma * gamma - 1.0));
        mass_plus * (acoustic_energy + tangential_ke)
    };
    FaceFrameFlux {
        mass: mass_plus,
        normal_momentum: mass_plus * normal_velocity_plus,
        tangential_momentum: [mass_plus * state.ut[0], mass_plus * state.ut[1]],
        energy,
    }
}

#[cfg(feature = "simd-fvm")]
fn fvs_positive_hanel_batch4(states: [FaceFrameState; 4], gamma: Real) -> [FaceFrameFlux; 4] {
    use wide::f64x4;

    let lane = |f: fn(FaceFrameState) -> Real, s: [FaceFrameState; 4]| {
        f64x4::new([f(s[0]), f(s[1]), f(s[2]), f(s[3])])
    };
    let rho = lane(|s| s.rho, states);
    let un = lane(|s| s.un, states);
    let ut0 = lane(|s| s.ut[0], states);
    let ut1 = lane(|s| s.ut[1], states);
    let p = lane(|s| s.p, states);
    let rho_e = lane(|s| s.rho_e, states);

    let g = f64x4::splat(gamma);
    let gm1 = g - f64x4::splat(1.0);
    let a = (g * p / rho).sqrt();
    let mach = un / a;

    let full_mass = rho * un;
    let full_normal = rho * un * un + p;
    let full_t0 = rho * un * ut0;
    let full_t1 = rho * un * ut1;
    let full_energy = (rho_e + p) * un;

    let zero = FaceFrameFlux {
        mass: 0.0,
        normal_momentum: 0.0,
        tangential_momentum: [0.0, 0.0],
        energy: 0.0,
    };
    let full = |i: usize| FaceFrameFlux {
        mass: full_mass.to_array()[i],
        normal_momentum: full_normal.to_array()[i],
        tangential_momentum: [full_t0.to_array()[i], full_t1.to_array()[i]],
        energy: full_energy.to_array()[i],
    };

    let mach_plus = mach + f64x4::splat(1.0);
    let mass_plus = f64x4::splat(0.25) * rho * a * mach_plus * mach_plus;
    let normal_velocity_plus = (gm1 * un + f64x4::splat(2.0) * a) / g;
    let tangential_ke = f64x4::splat(0.5) * (ut0 * ut0 + ut1 * ut1);
    let enthalpy = a * a / gm1 + tangential_ke + f64x4::splat(0.5) * un * un;
    let subsonic_energy = mass_plus * enthalpy;

    let mut out = [zero; 4];
    for (i, out_i) in out.iter_mut().enumerate() {
        let m = mach.to_array()[i];
        *out_i = if m <= -1.0 {
            zero
        } else if m >= 1.0 {
            full(i)
        } else {
            let mp = mass_plus.to_array()[i];
            FaceFrameFlux {
                mass: mp,
                normal_momentum: mp * normal_velocity_plus.to_array()[i],
                tangential_momentum: [mp * ut0.to_array()[i], mp * ut1.to_array()[i]],
                energy: subsonic_energy.to_array()[i],
            }
        };
    }
    out
}

#[cfg(feature = "simd-fvm")]
fn fvs_negative_hanel_batch4(states: [FaceFrameState; 4], gamma: Real) -> [FaceFrameFlux; 4] {
    let mut out = [FaceFrameFlux {
        mass: 0.0,
        normal_momentum: 0.0,
        tangential_momentum: [0.0, 0.0],
        energy: 0.0,
    }; 4];
    for i in 0..4 {
        out[i] = fvs_negative_hanel(&states[i], gamma);
    }
    out
}

fn add_face_fluxes(left: FaceFrameFlux, right: FaceFrameFlux) -> FaceFrameFlux {
    FaceFrameFlux {
        mass: left.mass + right.mass,
        normal_momentum: left.normal_momentum + right.normal_momentum,
        tangential_momentum: [
            left.tangential_momentum[0] + right.tangential_momentum[0],
            left.tangential_momentum[1] + right.tangential_momentum[1],
        ],
        energy: left.energy + right.energy,
    }
}

fn to_global_flux(face: FaceFrameFlux, normal: Vector3, t1: Vector3, t2: Vector3) -> InviscidFlux5 {
    InviscidFlux5 {
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
    use crate::discretization::hanel_van_leer_flux;
    use crate::physics::{IdealGasEoS, PrimitiveState};

    fn sample_states(gamma: Real) -> ([ConservedState; 4], [ConservedState; 4], [Vector3; 4]) {
        let eos = IdealGasEoS::new(gamma, 1.0).expect("eos");
        let pairs = [
            (
                PrimitiveState {
                    density: 1.0,
                    velocity: [0.3, 0.0, 0.0],
                    pressure: 1.0,
                    temperature: 1.0,
                },
                PrimitiveState {
                    density: 0.8,
                    velocity: [0.2, 0.0, 0.0],
                    pressure: 0.9,
                    temperature: 1.0,
                },
                Vector3::new(1.0, 0.0, 0.0),
            ),
            (
                PrimitiveState {
                    density: 1.2,
                    velocity: [0.1, 0.2, 0.0],
                    pressure: 1.1,
                    temperature: 1.0,
                },
                PrimitiveState {
                    density: 1.0,
                    velocity: [0.0, 0.15, 0.0],
                    pressure: 1.0,
                    temperature: 1.0,
                },
                Vector3::new(0.6, 0.8, 0.0),
            ),
            (
                PrimitiveState {
                    density: 0.9,
                    velocity: [0.5, 0.0, 0.1],
                    pressure: 0.95,
                    temperature: 1.0,
                },
                PrimitiveState {
                    density: 0.85,
                    velocity: [0.4, 0.05, 0.0],
                    pressure: 0.9,
                    temperature: 1.0,
                },
                Vector3::new(0.0, 1.0, 0.0),
            ),
            (
                PrimitiveState {
                    density: 1.05,
                    velocity: [0.2, -0.1, 0.05],
                    pressure: 1.05,
                    temperature: 1.0,
                },
                PrimitiveState {
                    density: 0.95,
                    velocity: [0.15, 0.0, -0.05],
                    pressure: 1.0,
                    temperature: 1.0,
                },
                Vector3::new(0.707, 0.707, 0.0),
            ),
        ];
        let mut left = [ConservedState {
            density: 0.0,
            momentum: [0.0; 3],
            total_energy: 0.0,
        }; 4];
        let mut right = left;
        let mut normal = [Vector3::new(0.0, 0.0, 0.0); 4];
        for (i, (pl, pr, n)) in pairs.iter().enumerate() {
            left[i] = ConservedState::from_primitive(&eos, pl).expect("left");
            right[i] = ConservedState::from_primitive(&eos, pr).expect("right");
            normal[i] = *n;
        }
        (left, right, normal)
    }

    #[test]
    fn hanel_batch4_matches_scalar_reference() {
        let gamma = 1.4;
        let (left, right, normal) = sample_states(gamma);
        let left_ref = [&left[0], &left[1], &left[2], &left[3]];
        let right_ref = [&right[0], &right[1], &right[2], &right[3]];
        let batch = face_inviscid_flux_first_order_hanel_batch4(left_ref, right_ref, normal, gamma);
        let eos = IdealGasEoS::new(gamma, 1.0).expect("eos");
        for i in 0..4 {
            let ref_flux = hanel_van_leer_flux(&left[i], &right[i], normal[i], &eos).expect("ref");
            let Some(got) = batch[i] else {
                panic!("batch lane {i} returned None");
            };
            assert!(approx_eq(got.mass, ref_flux.mass, 1.0e-10));
            assert!(approx_eq(got.momentum[0], ref_flux.momentum[0], 1.0e-10));
            assert!(approx_eq(got.momentum[1], ref_flux.momentum[1], 1.0e-10));
            assert!(approx_eq(got.momentum[2], ref_flux.momentum[2], 1.0e-10));
            assert!(approx_eq(got.energy, ref_flux.energy, 1.0e-10));
        }
    }
}
