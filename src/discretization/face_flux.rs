//! 无粘面通量入口：原始变量界面重构 + 数值 Riemann 求解 dispatch。

use crate::core::{Real, Vector3};
use crate::error::Result;
use crate::field::PrimitiveFields;
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use super::flux_config::{FluxScheme, InviscidFluxConfig};
use super::hllc::hllc_flux_with_primitives;
use super::inviscid::InviscidFlux;
use super::reconstruction::{
    InterfacePrimitiveStates, PrimitiveMusclStencil1d, interface_conserved_pair,
    reconstruct_face_primitives,
};
use super::roe::roe_flux_with_primitives;
use super::slau2::slau2_flux;
use super::van_leer::{hanel_van_leer_flux, van_leer_flux};

/// 面通量输入：owner/neighbor 及可选 MUSCL 原始变量模板点。
#[derive(Debug, Clone, Copy)]
pub struct FaceFluxInput<'a> {
    pub owner: &'a PrimitiveState,
    pub neighbor: &'a PrimitiveState,
    pub left_of_owner: Option<&'a PrimitiveState>,
    pub right_of_neighbor: Option<&'a PrimitiveState>,
}

impl<'a> FaceFluxInput<'a> {
    #[must_use]
    pub const fn first_order(owner: &'a PrimitiveState, neighbor: &'a PrimitiveState) -> Self {
        Self {
            owner,
            neighbor,
            left_of_owner: None,
            right_of_neighbor: None,
        }
    }

    #[must_use]
    pub const fn from_stencil(stencil: PrimitiveMusclStencil1d<'a>) -> Self {
        Self {
            owner: stencil.owner,
            neighbor: stencil.neighbor,
            left_of_owner: stencil.left_of_owner,
            right_of_neighbor: stencil.right_of_neighbor,
        }
    }
}

/// 一阶内面通量：直接从 `PrimitiveFields` SoA 读取，跳过 `FaceFluxInput` 与界面 struct 拷贝。
pub fn face_inviscid_flux_first_order_interior_soa(
    owner: usize,
    neighbor: usize,
    primitives: &PrimitiveFields,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    if matches!(
        config.scheme,
        FluxScheme::VanLeer | FluxScheme::HanelVanLeer | FluxScheme::Slau2
    ) {
        let left = conserved_from_primitive_soa(primitives, owner, eos)?;
        let right = conserved_from_primitive_soa(primitives, neighbor, eos)?;
        return dispatch_inviscid_flux_conserved(&left, &right, normal, eos, config);
    }
    let prim_l = primitive_state_from_soa(primitives, owner);
    let prim_r = primitive_state_from_soa(primitives, neighbor);
    let left = ConservedState::from_primitive(eos, &prim_l)?;
    let right = ConservedState::from_primitive(eos, &prim_r)?;
    dispatch_inviscid_flux_with_primitives(&left, &right, &prim_l, &prim_r, normal, eos, config)
}

/// 一阶边界面通量：owner 侧 SoA，ghost 侧守恒态（FVS 免 ghost 原始变量解码）。
pub fn face_inviscid_flux_first_order_boundary_soa(
    owner: usize,
    primitives: &PrimitiveFields,
    ghost: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
    min_pressure: Real,
) -> Result<InviscidFlux> {
    if matches!(
        config.scheme,
        FluxScheme::VanLeer | FluxScheme::HanelVanLeer | FluxScheme::Slau2
    ) {
        let left = conserved_from_primitive_soa(primitives, owner, eos)?;
        return dispatch_inviscid_flux_conserved(&left, ghost, normal, eos, config);
    }
    let prim_l = primitive_state_from_soa(primitives, owner);
    let left = ConservedState::from_primitive(eos, &prim_l)?;
    let prim_r = crate::field::primitive_from_conserved_relaxed(eos, ghost, min_pressure)?;
    dispatch_inviscid_flux_with_primitives(&left, ghost, &prim_l, &prim_r, normal, eos, config)
}

/// 面数值通量：原始变量重构后转守恒态并调用 Roe / HLLC / FVS。
pub fn face_inviscid_flux(
    input: FaceFluxInput<'_>,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    let stencil = PrimitiveMusclStencil1d {
        left_of_owner: input.left_of_owner,
        owner: input.owner,
        neighbor: input.neighbor,
        right_of_neighbor: input.right_of_neighbor,
    };
    let iface = reconstruct_face_primitives(stencil, config.reconstruction, config.limiter);
    face_inviscid_flux_from_interface(iface, normal, eos, config)
}

/// 由已重构界面原始变量直接计算数值通量（非结构二阶路径）。
pub fn face_inviscid_flux_from_interface(
    iface: InterfacePrimitiveStates,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    let (left, right) = interface_conserved_pair(eos, &iface)?;
    dispatch_inviscid_flux_with_primitives(
        &left,
        &right,
        &iface.left,
        &iface.right,
        normal,
        eos,
        config,
    )
}

fn dispatch_inviscid_flux_with_primitives(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => {
            roe_flux_with_primitives(left, right, prim_l, prim_r, normal, eos, &roe_cfg)
        }
        FluxScheme::Hllc => hllc_flux_with_primitives(left, right, prim_l, prim_r, normal, eos),
        FluxScheme::VanLeer => van_leer_flux(left, right, normal, eos),
        FluxScheme::HanelVanLeer => hanel_van_leer_flux(left, right, normal, eos),
        FluxScheme::Slau2 => slau2_flux(left, right, normal, eos),
    }
}

fn dispatch_inviscid_flux_conserved(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    match config.scheme {
        FluxScheme::VanLeer => van_leer_flux(left, right, normal, eos),
        FluxScheme::HanelVanLeer => hanel_van_leer_flux(left, right, normal, eos),
        FluxScheme::Slau2 => slau2_flux(left, right, normal, eos),
        FluxScheme::Roe(_) | FluxScheme::Hllc => Err(crate::error::AsimuError::Config(
            "dispatch_inviscid_flux_conserved 仅用于 FVS 格式".to_string(),
        )),
    }
}

#[inline]
fn primitive_state_from_soa(primitives: &PrimitiveFields, cell: usize) -> PrimitiveState {
    PrimitiveState {
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

fn conserved_from_primitive_soa(
    primitives: &PrimitiveFields,
    cell: usize,
    eos: &IdealGasEoS,
) -> Result<ConservedState> {
    let prim = primitive_state_from_soa(primitives, cell);
    ConservedState::from_primitive(eos, &prim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::physics::IdealGasEoS;

    #[test]
    fn first_order_interior_soa_matches_face_flux_input() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let mut prim = PrimitiveFields::zeros(2).expect("prim");
        prim.density.values_mut()[0] = 1.0;
        prim.density.values_mut()[1] = 0.9;
        prim.pressure.values_mut()[0] = 1.0;
        prim.pressure.values_mut()[1] = 0.95;
        prim.velocity_x.values_mut()[0] = 0.3;
        prim.velocity_x.values_mut()[1] = 0.2;
        let config = InviscidFluxConfig::hanel_van_leer_first_order();
        let normal = Vector3::new(1.0, 0.0, 0.0);
        let soa = face_inviscid_flux_first_order_interior_soa(0, 1, &prim, normal, &eos, &config)
            .expect("soa");
        let owner = prim.cell_primitive(0);
        let neighbor = prim.cell_primitive(1);
        let legacy = face_inviscid_flux(
            FaceFluxInput::first_order(&owner, &neighbor),
            normal,
            &eos,
            &config,
        )
        .expect("legacy");
        assert!(approx_eq(soa.mass, legacy.mass, 1.0e-12));
        assert!(approx_eq(soa.energy, legacy.energy, 1.0e-12));
    }
}
