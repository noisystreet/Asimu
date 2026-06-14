//! Van Leer / Hanel FVS f32 热路径（语义对齐 `van_leer.rs` / CUDA HVL kernel）。

use crate::discretization::inviscid_f32::{
    FaceNormalF32, InviscidFluxF32, face_tangent_basis_f32, normalize_face_normal_f32,
};
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::error::{AsimuError, Result};
use crate::physics::IdealGasEoS;

#[derive(Clone, Copy)]
enum EnergyFluxSplitF32 {
    VanLeer,
    Hanel,
}

/// f32 Van Leer FVS 数值通量。
pub fn van_leer_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: FaceNormalF32,
    eos: &IdealGasEoS,
) -> Result<InviscidFluxF32> {
    fvs_flux_with_primitives_f32(prim_l, prim_r, normal, eos, EnergyFluxSplitF32::VanLeer)
}

/// f32 Hanel 修正 Van Leer FVS 数值通量。
pub fn hanel_van_leer_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: FaceNormalF32,
    eos: &IdealGasEoS,
) -> Result<InviscidFluxF32> {
    fvs_flux_with_primitives_f32(prim_l, prim_r, normal, eos, EnergyFluxSplitF32::Hanel)
}

fn fvs_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: FaceNormalF32,
    eos: &IdealGasEoS,
    energy_split: EnergyFluxSplitF32,
) -> Result<InviscidFluxF32> {
    let n = normalize_face_normal_f32(normal)?;
    let (t1, t2) = face_tangent_basis_f32(n);
    let gamma = eos.gamma as f32;
    let frame_l = face_frame_from_primitive_f32(prim_l, gamma, n, t1, t2)?;
    let frame_r = face_frame_from_primitive_f32(prim_r, gamma, n, t1, t2)?;
    validate_face_state_f32(&frame_l)?;
    validate_face_state_f32(&frame_r)?;
    let flux_l_plus = fvs_positive_flux_f32(&frame_l, gamma, energy_split);
    let flux_r_minus = fvs_negative_flux_f32(&frame_r, gamma, energy_split);
    let face_flux = add_face_fluxes_f32(flux_l_plus, flux_r_minus);
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

fn face_frame_from_primitive_f32(
    prim: &PrimitiveStateF32,
    gamma: f32,
    normal: FaceNormalF32,
    t1: FaceNormalF32,
    t2: FaceNormalF32,
) -> Result<FaceFrameStateF32> {
    if prim.density <= 0.0 || prim.pressure <= 0.0 {
        return Err(AsimuError::Field(
            "Van Leer f32 状态须为正密度与压力".to_string(),
        ));
    }
    let [nx, ny, nz] = normal;
    let [t1x, t1y, t1z] = t1;
    let [t2x, t2y, t2z] = t2;
    let rho = prim.density;
    let u = prim.velocity;
    let un = u[0] * nx + u[1] * ny + u[2] * nz;
    let ut0 = u[0] * t1x + u[1] * t1y + u[2] * t1z;
    let ut1 = u[0] * t2x + u[1] * t2y + u[2] * t2z;
    let u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let internal = prim.pressure / (gamma - 1.0);
    Ok(FaceFrameStateF32 {
        rho,
        un,
        ut: [ut0, ut1],
        p: prim.pressure,
        rho_e: 0.5 * rho * u2 + internal,
    })
}

fn validate_face_state_f32(state: &FaceFrameStateF32) -> Result<()> {
    if state.rho <= 0.0 || state.p <= 0.0 {
        return Err(AsimuError::Field(
            "Van Leer f32 状态须为正密度与压力".to_string(),
        ));
    }
    Ok(())
}

fn sound_speed_f32(rho: f32, pressure: f32, gamma: f32) -> f32 {
    (gamma * pressure / rho).sqrt()
}

fn specific_enthalpy_f32(state: &FaceFrameStateF32, gamma: f32) -> f32 {
    let a = sound_speed_f32(state.rho, state.p, gamma);
    a * a / (gamma - 1.0)
        + 0.5 * (state.un * state.un + state.ut[0] * state.ut[0] + state.ut[1] * state.ut[1])
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

fn fvs_positive_flux_f32(
    state: &FaceFrameStateF32,
    gamma: f32,
    energy_split: EnergyFluxSplitF32,
) -> FaceFrameFluxF32 {
    let full = physical_face_flux_f32(state);
    let a = sound_speed_f32(state.rho, state.p, gamma);
    let mach = state.un / a;
    if mach <= -1.0 {
        return FaceFrameFluxF32 {
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
    let energy = match energy_split {
        EnergyFluxSplitF32::VanLeer => {
            let acoustic_energy =
                ((gamma - 1.0) * state.un + 2.0 * a).powi(2) / (2.0 * (gamma * gamma - 1.0));
            mass_plus * (acoustic_energy + tangential_ke)
        }
        EnergyFluxSplitF32::Hanel => mass_plus * specific_enthalpy_f32(state, gamma),
    };
    FaceFrameFluxF32 {
        mass: mass_plus,
        normal_momentum: mass_plus * normal_velocity_plus,
        tangential_momentum: [mass_plus * state.ut[0], mass_plus * state.ut[1]],
        energy,
    }
}

fn fvs_negative_flux_f32(
    state: &FaceFrameStateF32,
    gamma: f32,
    energy_split: EnergyFluxSplitF32,
) -> FaceFrameFluxF32 {
    let full = physical_face_flux_f32(state);
    let plus = fvs_positive_flux_f32(state, gamma, energy_split);
    FaceFrameFluxF32 {
        mass: full.mass - plus.mass,
        normal_momentum: full.normal_momentum - plus.normal_momentum,
        tangential_momentum: [
            full.tangential_momentum[0] - plus.tangential_momentum[0],
            full.tangential_momentum[1] - plus.tangential_momentum[1],
        ],
        energy: full.energy - plus.energy,
    }
}

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
    use crate::discretization::van_leer::{hanel_van_leer_flux, van_leer_flux};
    use crate::discretization::viscous_boundary_f32::primitive_state_f32_from_real;
    use crate::physics::{ConservedState, PrimitiveState};

    fn freestream_pair() -> (PrimitiveState, PrimitiveState) {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(0.8, 1.0, 1.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let shifted = PrimitiveState {
            velocity: [0.9, 0.0, 0.0],
            ..prim
        };
        (prim, shifted)
    }

    #[test]
    fn van_leer_f32_matches_f64_on_primitive_pair() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let (left, right) = freestream_pair();
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let normal_f64 = Vector3::new(0.6, 0.8, 0.0);
        let normal = [0.6_f32, 0.8, 0.0];
        let f64_flux = van_leer_flux(&cons_l, &cons_r, normal_f64, &eos).expect("f64");
        let f32_flux = van_leer_flux_with_primitives_f32(
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
    fn hanel_f32_matches_f64_on_primitive_pair() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let (left, right) = freestream_pair();
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let normal_f64 = Vector3::new(0.6, 0.8, 0.0);
        let normal = [0.6_f32, 0.8, 0.0];
        let f64_flux = hanel_van_leer_flux(&cons_l, &cons_r, normal_f64, &eos).expect("f64");
        let f32_flux = hanel_van_leer_flux_with_primitives_f32(
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
