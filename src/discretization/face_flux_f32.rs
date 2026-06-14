//! 无粘面通量 f32 入口：界面原始变量 → 原生 Riemann 求值。

use crate::core::{Real, Vector3};
use crate::discretization::face_flux::{FaceFluxInput, face_inviscid_flux};
use crate::discretization::flux_config::{FluxScheme, InviscidFluxConfig};
use crate::discretization::hllc_f32::hllc_flux_with_primitives_f32;
use crate::discretization::inviscid::InviscidFlux;
use crate::discretization::reconstruction_unstructured_f32::InterfacePrimitiveStatesF32;
use crate::discretization::roe_f32::roe_flux_with_primitives_f32;
use crate::discretization::viscous_boundary_f32::{
    PrimitiveStateF32, primitive_state_f32_from_real, primitive_state_f32_to_real,
};
use crate::error::Result;
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed};
use crate::physics::{ConservedState, IdealGasEoS};

/// 由 f32 界面原始变量计算数值通量（非结构二阶 / 边界面路径）。
pub fn face_inviscid_flux_from_interface_f32(
    iface: InterfacePrimitiveStatesF32,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    dispatch_inviscid_flux_with_primitives_f32(&iface.left, &iface.right, normal, eos, config)
}

/// 一阶内面通量：直接从 `PrimitiveFieldsT<f32>` SoA 读取。
pub fn face_inviscid_flux_first_order_interior_soa_f32(
    owner: usize,
    neighbor: usize,
    primitives: &PrimitiveFieldsT<f32>,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    let prim_l = primitive_state_f32_from_soa(primitives, owner);
    let prim_r = primitive_state_f32_from_soa(primitives, neighbor);
    dispatch_inviscid_flux_with_primitives_f32(&prim_l, &prim_r, normal, eos, config)
}

/// 一阶边界面通量：owner SoA + ghost 守恒态。
pub fn face_inviscid_flux_first_order_boundary_soa_f32(
    owner: usize,
    primitives: &PrimitiveFieldsT<f32>,
    ghost: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
    min_pressure: Real,
) -> Result<InviscidFlux> {
    let prim_l = primitive_state_f32_from_soa(primitives, owner);
    let ghost_prim = primitive_from_conserved_relaxed(eos, ghost, min_pressure)?;
    let prim_r = primitive_state_f32_from_real(ghost_prim);
    dispatch_inviscid_flux_with_primitives_f32(&prim_l, &prim_r, normal, eos, config)
}

fn dispatch_inviscid_flux_with_primitives_f32(
    prim_l: &PrimitiveStateF32,
    prim_r: &PrimitiveStateF32,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => {
            roe_flux_with_primitives_f32(prim_l, prim_r, normal, eos, &roe_cfg)
        }
        FluxScheme::Hllc => hllc_flux_with_primitives_f32(prim_l, prim_r, normal, eos),
        FluxScheme::VanLeer | FluxScheme::HanelVanLeer | FluxScheme::Slau2 => {
            let owner = primitive_state_f32_to_real(*prim_l);
            let neighbor = primitive_state_f32_to_real(*prim_r);
            face_inviscid_flux(
                FaceFluxInput::first_order(&owner, &neighbor),
                normal,
                eos,
                config,
            )
        }
    }
}

#[inline]
fn primitive_state_f32_from_soa(
    primitives: &PrimitiveFieldsT<f32>,
    cell: usize,
) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
        density: primitives.density.values()[cell],
        velocity: [
            primitives.velocity_x.values()[cell],
            primitives.velocity_y.values()[cell],
            primitives.velocity_z.values()[cell],
        ],
        pressure: primitives.pressure.values()[cell],
        temperature: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::reconstruction::InterfacePrimitiveStates;
    use crate::discretization::reconstruction_unstructured_f32::InterfacePrimitiveStatesF32;
    use crate::discretization::viscous_boundary_f32::primitive_state_f32_from_real;
    use crate::discretization::{InviscidFluxConfig, face_inviscid_flux_from_interface};
    use crate::physics::PrimitiveState;

    #[test]
    fn interface_f32_hllc_matches_f64_reference() {
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
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let config = InviscidFluxConfig::muscl_hllc();
        let f64_flux = face_inviscid_flux_from_interface(
            InterfacePrimitiveStates { left, right },
            normal,
            &eos,
            &config,
        )
        .expect("f64");
        let f32_flux = face_inviscid_flux_from_interface_f32(
            InterfacePrimitiveStatesF32 {
                left: primitive_state_f32_from_real(left),
                right: primitive_state_f32_from_real(right),
            },
            normal,
            &eos,
            &config,
        )
        .expect("f32");
        assert!(approx_eq(f32_flux.mass, f64_flux.mass, 1.0e-3));
        assert!(approx_eq(f32_flux.energy, f64_flux.energy, 1.0e-2));
    }
}
