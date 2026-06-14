//! 无粘面通量 typed 分发（f32/f64 共用入口；scatter 仍输出 `InviscidFlux`）。

use crate::core::{ComputeFloat, Real, Vector3};
use crate::discretization::GhostCellState;
use crate::discretization::face_flux::{FaceFluxInput, face_inviscid_flux};
use crate::discretization::flux_config::{FluxScheme, InviscidFluxConfig};
use crate::discretization::hllc_f32::hllc_flux_with_primitives_f32;
use crate::discretization::inviscid::InviscidFlux;
use crate::discretization::reconstruction_unstructured_f32::InterfacePrimitiveStatesF32;
use crate::discretization::roe_f32::roe_flux_with_primitives_f32;
use crate::discretization::slau2_f32::slau2_flux_with_primitives_f32;
use crate::discretization::van_leer_f32::{
    hanel_van_leer_flux_with_primitives_f32, van_leer_flux_with_primitives_f32,
};
use crate::discretization::viscous_boundary_f32::{
    PrimitiveStateF32, primitive_state_f32_from_real,
};
use crate::error::Result;
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed};
use crate::physics::{ConservedState, IdealGasEoS};

/// 一阶 / 边界面无粘通量 typed 分发。
pub trait InviscidFaceFluxTyped: ComputeFloat {
    fn first_order_interior_soa(
        primitives: &PrimitiveFieldsT<Self>,
        owner: usize,
        neighbor: usize,
        normal: Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
    ) -> Result<InviscidFlux>;

    fn first_order_boundary_soa(
        primitives: &PrimitiveFieldsT<Self>,
        owner: usize,
        ghost: &GhostCellState,
        normal: Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
        min_pressure: Real,
    ) -> Result<InviscidFlux>;
}

impl InviscidFaceFluxTyped for f32 {
    fn first_order_interior_soa(
        primitives: &PrimitiveFieldsT<f32>,
        owner: usize,
        neighbor: usize,
        normal: Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
    ) -> Result<InviscidFlux> {
        let prim_l = primitive_lane_f32(primitives, owner);
        let prim_r = primitive_lane_f32(primitives, neighbor);
        dispatch_inviscid_flux_primitives_f32(&prim_l, &prim_r, normal, eos, config)
    }

    fn first_order_boundary_soa(
        primitives: &PrimitiveFieldsT<f32>,
        owner: usize,
        ghost: &GhostCellState,
        normal: Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
        min_pressure: Real,
    ) -> Result<InviscidFlux> {
        let prim_l = primitive_lane_f32(primitives, owner);
        let ghost_prim = primitive_from_conserved_relaxed(eos, &ghost.conserved, min_pressure)?;
        let prim_r = primitive_state_f32_from_real(ghost_prim);
        dispatch_inviscid_flux_primitives_f32(&prim_l, &prim_r, normal, eos, config)
    }
}

impl InviscidFaceFluxTyped for f64 {
    fn first_order_interior_soa(
        primitives: &PrimitiveFieldsT<f64>,
        owner: usize,
        neighbor: usize,
        normal: Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
    ) -> Result<InviscidFlux> {
        let owner_prim = primitives.cell_primitive(owner);
        let neighbor_prim = primitives.cell_primitive(neighbor);
        face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &neighbor_prim),
            normal,
            eos,
            config,
        )
    }

    fn first_order_boundary_soa(
        primitives: &PrimitiveFieldsT<f64>,
        owner: usize,
        ghost: &GhostCellState,
        normal: Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
        min_pressure: Real,
    ) -> Result<InviscidFlux> {
        let owner_prim = primitives.cell_primitive(owner);
        let ghost_prim = primitive_from_conserved_relaxed(eos, &ghost.conserved, min_pressure)?;
        face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &ghost_prim),
            normal,
            eos,
            config,
        )
    }
}

/// f32 界面原始变量数值通量（非结构 MUSCL / 边界面）。
pub fn face_inviscid_flux_from_interface_f32(
    iface: InterfacePrimitiveStatesF32,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    dispatch_inviscid_flux_primitives_f32(&iface.left, &iface.right, normal, eos, config)
}

/// f32 一阶内面 SoA 通量（兼容旧 API）。
pub fn face_inviscid_flux_first_order_interior_soa_f32(
    owner: usize,
    neighbor: usize,
    primitives: &PrimitiveFieldsT<f32>,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    f32::first_order_interior_soa(primitives, owner, neighbor, normal, eos, config)
}

/// f32 一阶边界面 SoA 通量（兼容旧 API）。
pub fn face_inviscid_flux_first_order_boundary_soa_f32(
    owner: usize,
    primitives: &PrimitiveFieldsT<f32>,
    ghost: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
    min_pressure: Real,
) -> Result<InviscidFlux> {
    let prim_l = primitive_lane_f32(primitives, owner);
    let ghost_prim = primitive_from_conserved_relaxed(eos, ghost, min_pressure)?;
    let prim_r = primitive_state_f32_from_real(ghost_prim);
    dispatch_inviscid_flux_primitives_f32(&prim_l, &prim_r, normal, eos, config)
}

pub(crate) fn dispatch_inviscid_flux_primitives_f32(
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
        FluxScheme::VanLeer => van_leer_flux_with_primitives_f32(prim_l, prim_r, normal, eos),
        FluxScheme::HanelVanLeer => {
            hanel_van_leer_flux_with_primitives_f32(prim_l, prim_r, normal, eos)
        }
        FluxScheme::Slau2 => slau2_flux_with_primitives_f32(prim_l, prim_r, normal, eos),
    }
}

#[inline]
fn primitive_lane_f32(primitives: &PrimitiveFieldsT<f32>, cell: usize) -> PrimitiveStateF32 {
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
    use crate::discretization::face_inviscid_flux_from_interface;
    use crate::discretization::reconstruction::InterfacePrimitiveStates;
    use crate::discretization::slau2::slau2_flux;
    use crate::discretization::van_leer::{hanel_van_leer_flux, van_leer_flux};
    use crate::physics::{ConservedState, PrimitiveState};

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

    #[test]
    fn interface_f32_hanel_matches_f64_reference() {
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
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let config = InviscidFluxConfig::hanel_van_leer_first_order();
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let f64_flux = hanel_van_leer_flux(&cons_l, &cons_r, normal, &eos).expect("f64");
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

    #[test]
    fn interface_f32_van_leer_matches_f64_reference() {
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
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let config = InviscidFluxConfig::van_leer_first_order();
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let f64_flux = van_leer_flux(&cons_l, &cons_r, normal, &eos).expect("f64");
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

    #[test]
    fn interface_f32_slau2_matches_f64_reference() {
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
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let config = InviscidFluxConfig::muscl_slau2();
        let cons_l = ConservedState::from_primitive(&eos, &left).expect("left");
        let cons_r = ConservedState::from_primitive(&eos, &right).expect("right");
        let f64_flux = slau2_flux(&cons_l, &cons_r, normal, &eos).expect("f64");
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
