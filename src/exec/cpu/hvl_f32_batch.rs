//! Hanel–Van Leer FVS 四路批处理 f32（`simd-fvm`：亚音速能量分裂用 `f32x4`）。

use crate::discretization::inviscid_f32::{FaceNormalF32, InviscidFluxF32};
#[cfg(not(feature = "simd-fvm"))]
use crate::discretization::van_leer_f32::hanel_van_leer_flux_with_primitives_f32;
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::physics::IdealGasEoS;

/// 四路一阶 Hanel–Van Leer 面通量（f32 原变量）。
pub fn face_inviscid_flux_first_order_hanel_batch4_f32(
    left: [&PrimitiveStateF32; 4],
    right: [&PrimitiveStateF32; 4],
    normals: [FaceNormalF32; 4],
    eos: &IdealGasEoS,
) -> [Option<InviscidFluxF32>; 4] {
    #[cfg(feature = "simd-fvm")]
    {
        hanel_batch4_f32_simd(left, right, normals, eos)
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        let mut out = [None; 4];
        for i in 0..4 {
            out[i] =
                hanel_van_leer_flux_with_primitives_f32(left[i], right[i], normals[i], eos).ok();
        }
        out
    }
}

#[cfg(feature = "simd-fvm")]
fn hanel_batch4_f32_simd(
    left: [&PrimitiveStateF32; 4],
    right: [&PrimitiveStateF32; 4],
    normals: [FaceNormalF32; 4],
    eos: &IdealGasEoS,
) -> [Option<InviscidFluxF32>; 4] {
    let mut out = [None; 4];
    let gamma = eos.gamma as f32;
    let mut frame_l = [None; 4];
    let mut frame_r = [None; 4];
    let mut n_arr = [[0.0f32; 3]; 4];
    let mut t1_arr = [[0.0f32; 3]; 4];
    let mut t2_arr = [[0.0f32; 3]; 4];
    for i in 0..4 {
        let Some((n, t1, t2, fl, fr)) = build_face_frames_f32(left[i], right[i], normals[i], gamma)
        else {
            return out;
        };
        n_arr[i] = n;
        t1_arr[i] = t1;
        t2_arr[i] = t2;
        frame_l[i] = Some(fl);
        frame_r[i] = Some(fr);
    }
    let fl_arr = [
        frame_l[0].expect("frame"),
        frame_l[1].expect("frame"),
        frame_l[2].expect("frame"),
        frame_l[3].expect("frame"),
    ];
    let fr_arr = [
        frame_r[0].expect("frame"),
        frame_r[1].expect("frame"),
        frame_r[2].expect("frame"),
        frame_r[3].expect("frame"),
    ];
    let plus_l = fvs_positive_hanel_batch4_f32(fl_arr, gamma);
    let minus_r = fvs_negative_hanel_batch4_f32(fr_arr, gamma);
    for i in 0..4 {
        let face = add_face_fluxes_f32(plus_l[i], minus_r[i]);
        out[i] = Some(to_global_flux_f32(face, n_arr[i], t1_arr[i], t2_arr[i]));
    }
    out
}

#[cfg(feature = "simd-fvm")]
#[derive(Clone, Copy)]
struct FaceFrameStateF32 {
    rho: f32,
    un: f32,
    ut: [f32; 2],
    p: f32,
    rho_e: f32,
}

#[cfg(feature = "simd-fvm")]
#[derive(Clone, Copy)]
struct FaceFrameFluxF32 {
    mass: f32,
    normal_momentum: f32,
    tangential_momentum: [f32; 2],
    energy: f32,
}

#[cfg(feature = "simd-fvm")]
type FaceFramesF32 = (
    [f32; 3],
    [f32; 3],
    [f32; 3],
    FaceFrameStateF32,
    FaceFrameStateF32,
);

#[cfg(feature = "simd-fvm")]
fn build_face_frames_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: FaceNormalF32,
    gamma: f32,
) -> Option<FaceFramesF32> {
    if prim_l.density <= 0.0
        || prim_r.density <= 0.0
        || prim_l.pressure <= 0.0
        || prim_r.pressure <= 0.0
    {
        return None;
    }
    let n = normalize_f32(normal)?;
    let (t1, t2) = face_tangent_basis_f32(n);
    let fl = face_frame_from_primitive_f32(prim_l, gamma, n, t1, t2)?;
    let fr = face_frame_from_primitive_f32(prim_r, gamma, n, t1, t2)?;
    if fl.rho <= 0.0 || fl.p <= 0.0 || fr.rho <= 0.0 || fr.p <= 0.0 {
        return None;
    }
    Some((n, t1, t2, fl, fr))
}

#[cfg(feature = "simd-fvm")]
fn normalize_f32(n: FaceNormalF32) -> Option<[f32; 3]> {
    let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if mag < f32::EPSILON {
        return None;
    }
    Some([n[0] / mag, n[1] / mag, n[2] / mag])
}

#[cfg(feature = "simd-fvm")]
fn face_tangent_basis_f32(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let ref_vec = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1 = cross_f32(normal, ref_vec);
    let t1 = normalize_unchecked_f32(t1);
    let t2 = cross_f32(normal, t1);
    (t1, normalize_unchecked_f32(t2))
}

#[cfg(feature = "simd-fvm")]
fn cross_f32(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(feature = "simd-fvm")]
fn normalize_unchecked_f32(v: [f32; 3]) -> [f32; 3] {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / mag, v[1] / mag, v[2] / mag]
}

#[cfg(feature = "simd-fvm")]
fn face_frame_from_primitive_f32(
    prim: &PrimitiveStateF32,
    gamma: f32,
    normal: [f32; 3],
    t1: [f32; 3],
    t2: [f32; 3],
) -> Option<FaceFrameStateF32> {
    let rho = prim.density;
    let u = prim.velocity;
    let un = u[0] * normal[0] + u[1] * normal[1] + u[2] * normal[2];
    let ut0 = u[0] * t1[0] + u[1] * t1[1] + u[2] * t1[2];
    let ut1 = u[0] * t2[0] + u[1] * t2[1] + u[2] * t2[2];
    let u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let internal = prim.pressure / (gamma - 1.0);
    Some(FaceFrameStateF32 {
        rho,
        un,
        ut: [ut0, ut1],
        p: prim.pressure,
        rho_e: 0.5 * rho * u2 + internal,
    })
}

#[cfg(feature = "simd-fvm")]
fn fvs_positive_hanel_batch4_f32(
    states: [FaceFrameStateF32; 4],
    gamma: f32,
) -> [FaceFrameFluxF32; 4] {
    use wide::f32x4;

    let lane = |f: fn(&FaceFrameStateF32) -> f32, s: [FaceFrameStateF32; 4]| {
        f32x4::new([f(&s[0]), f(&s[1]), f(&s[2]), f(&s[3])])
    };
    let rho = lane(|s| s.rho, states);
    let un = lane(|s| s.un, states);
    let ut0 = lane(|s| s.ut[0], states);
    let ut1 = lane(|s| s.ut[1], states);
    let p = lane(|s| s.p, states);
    let rho_e = lane(|s| s.rho_e, states);

    let g = f32x4::splat(gamma);
    let gm1 = g - f32x4::splat(1.0);
    let a = (g * p / rho).sqrt();
    let mach = un / a;

    let full_mass = rho * un;
    let full_normal = rho * un * un + p;
    let full_t0 = rho * un * ut0;
    let full_t1 = rho * un * ut1;
    let full_energy = (rho_e + p) * un;

    let zero = FaceFrameFluxF32 {
        mass: 0.0,
        normal_momentum: 0.0,
        tangential_momentum: [0.0, 0.0],
        energy: 0.0,
    };
    let full = |i: usize| FaceFrameFluxF32 {
        mass: full_mass.to_array()[i],
        normal_momentum: full_normal.to_array()[i],
        tangential_momentum: [full_t0.to_array()[i], full_t1.to_array()[i]],
        energy: full_energy.to_array()[i],
    };

    let mach_plus = mach + f32x4::splat(1.0);
    let mass_plus = f32x4::splat(0.25) * rho * a * mach_plus * mach_plus;
    let normal_velocity_plus = (gm1 * un + f32x4::splat(2.0) * a) / g;
    let tangential_ke = f32x4::splat(0.5) * (ut0 * ut0 + ut1 * ut1);
    let enthalpy = a * a / gm1 + tangential_ke + f32x4::splat(0.5) * un * un;
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
            FaceFrameFluxF32 {
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
fn fvs_negative_hanel_batch4_f32(
    states: [FaceFrameStateF32; 4],
    gamma: f32,
) -> [FaceFrameFluxF32; 4] {
    let mut out = [FaceFrameFluxF32 {
        mass: 0.0,
        normal_momentum: 0.0,
        tangential_momentum: [0.0, 0.0],
        energy: 0.0,
    }; 4];
    for i in 0..4 {
        let full = physical_face_flux_f32(&states[i]);
        let plus = fvs_positive_hanel_batch4_f32([states[i]; 4], gamma)[0];
        out[i] = FaceFrameFluxF32 {
            mass: full.mass - plus.mass,
            normal_momentum: full.normal_momentum - plus.normal_momentum,
            tangential_momentum: [
                full.tangential_momentum[0] - plus.tangential_momentum[0],
                full.tangential_momentum[1] - plus.tangential_momentum[1],
            ],
            energy: full.energy - plus.energy,
        };
    }
    out
}

#[cfg(feature = "simd-fvm")]
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

#[cfg(feature = "simd-fvm")]
fn add_face_fluxes_f32(left: FaceFrameFluxF32, right: FaceFrameFluxF32) -> FaceFrameFluxF32 {
    FaceFrameFluxF32 {
        mass: left.mass + right.mass,
        normal_momentum: left.normal_momentum + right.normal_momentum,
        tangential_momentum: [
            left.tangential_momentum[0] + right.tangential_momentum[0],
            left.tangential_momentum[1] + right.tangential_momentum[1],
        ],
        energy: left.energy + right.energy,
    }
}

#[cfg(feature = "simd-fvm")]
fn to_global_flux_f32(
    face: FaceFrameFluxF32,
    normal: [f32; 3],
    t1: [f32; 3],
    t2: [f32; 3],
) -> InviscidFluxF32 {
    InviscidFluxF32 {
        mass: face.mass,
        momentum: [
            face.normal_momentum * normal[0]
                + face.tangential_momentum[0] * t1[0]
                + face.tangential_momentum[1] * t2[0],
            face.normal_momentum * normal[1]
                + face.tangential_momentum[0] * t1[1]
                + face.tangential_momentum[1] * t2[1],
            face.normal_momentum * normal[2]
                + face.tangential_momentum[0] * t1[2]
                + face.tangential_momentum[1] * t2[2],
        ],
        energy: face.energy,
    }
}

#[cfg(all(test, feature = "simd-fvm"))]
mod tests {
    use super::*;
    use crate::discretization::van_leer_f32::hanel_van_leer_flux_with_primitives_f32;
    use crate::physics::IdealGasEoS;

    fn uniform_prim(mach: f32) -> PrimitiveStateF32 {
        let eos = IdealGasEoS::AIR_STANDARD;
        let a = (eos.gamma as f32 * eos.gas_constant as f32 * 288.15f32).sqrt();
        let u = mach * a;
        PrimitiveStateF32 {
            density: 1.2,
            velocity: [u, 0.0, 0.0],
            pressure: 101_325.0,
            temperature: 288.15,
        }
    }

    #[test]
    fn hanel_batch4_f32_matches_scalar_lanes() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = uniform_prim(0.3);
        let normal = [1.0f32, 0.0, 0.0];
        let left = [&prim; 4];
        let right = [&prim; 4];
        let normals = [normal; 4];
        let batch = face_inviscid_flux_first_order_hanel_batch4_f32(left, right, normals, &eos);
        for lane in &batch {
            let scalar =
                hanel_van_leer_flux_with_primitives_f32(&prim, &prim, normal, &eos).expect("lane");
            let batched = lane.expect("batch lane");
            assert!((scalar.mass - batched.mass).abs() < 1.0e-4);
            assert!((scalar.energy - batched.energy).abs() < 1.0e-2);
        }
    }
}
