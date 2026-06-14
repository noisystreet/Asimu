//! 非结构一阶无粘面通量 typed 分发（f32 原生 Riemann）。

use crate::core::{ComputeFloat, Real};
use crate::discretization::{
    FaceFluxInput, GhostCellState, InviscidFluxConfig, face_inviscid_flux,
    face_inviscid_flux_first_order_boundary_soa_f32,
    face_inviscid_flux_first_order_interior_soa_f32,
};
use crate::error::Result;
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed};
use crate::physics::IdealGasEoS;

/// 一阶内/边界面通量分发（f32 原生 Riemann / f64 既有路径）。
pub(super) trait InviscidFirstOrderFaceFlux: ComputeFloat {
    fn first_order_interior_flux(
        primitives: &PrimitiveFieldsT<Self>,
        owner: usize,
        neighbor: usize,
        normal: crate::core::Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
    ) -> Result<crate::discretization::InviscidFlux>;

    fn first_order_boundary_flux(
        primitives: &PrimitiveFieldsT<Self>,
        owner: usize,
        ghost: &GhostCellState,
        normal: crate::core::Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
        min_pressure: Real,
    ) -> Result<crate::discretization::InviscidFlux>;
}

impl InviscidFirstOrderFaceFlux for f32 {
    fn first_order_interior_flux(
        primitives: &PrimitiveFieldsT<f32>,
        owner: usize,
        neighbor: usize,
        normal: crate::core::Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
    ) -> Result<crate::discretization::InviscidFlux> {
        face_inviscid_flux_first_order_interior_soa_f32(
            owner, neighbor, primitives, normal, eos, config,
        )
    }

    fn first_order_boundary_flux(
        primitives: &PrimitiveFieldsT<f32>,
        owner: usize,
        ghost: &GhostCellState,
        normal: crate::core::Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
        min_pressure: Real,
    ) -> Result<crate::discretization::InviscidFlux> {
        face_inviscid_flux_first_order_boundary_soa_f32(
            owner,
            primitives,
            &ghost.conserved,
            normal,
            eos,
            config,
            min_pressure,
        )
    }
}

impl InviscidFirstOrderFaceFlux for f64 {
    fn first_order_interior_flux(
        primitives: &PrimitiveFieldsT<f64>,
        owner: usize,
        neighbor: usize,
        normal: crate::core::Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
    ) -> Result<crate::discretization::InviscidFlux> {
        let owner_prim = primitives.cell_primitive(owner);
        let neighbor_prim = primitives.cell_primitive(neighbor);
        face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &neighbor_prim),
            normal,
            eos,
            config,
        )
    }

    fn first_order_boundary_flux(
        primitives: &PrimitiveFieldsT<f64>,
        owner: usize,
        ghost: &GhostCellState,
        normal: crate::core::Vector3,
        eos: &IdealGasEoS,
        config: &InviscidFluxConfig,
        min_pressure: Real,
    ) -> Result<crate::discretization::InviscidFlux> {
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

pub(super) fn first_order_interior_flux<T: InviscidFirstOrderFaceFlux>(
    primitives: &PrimitiveFieldsT<T>,
    owner: usize,
    neighbor: usize,
    normal: crate::core::Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<crate::discretization::InviscidFlux> {
    T::first_order_interior_flux(primitives, owner, neighbor, normal, eos, config)
}
