//! 无粘面通量入口：原始变量界面重构 + 数值 Riemann 求解 dispatch。

use crate::core::Vector3;
use crate::error::Result;
use crate::physics::{IdealGasEoS, PrimitiveState};

use super::flux_config::{FluxScheme, InviscidFluxConfig};
use super::hllc::hllc_flux_with_primitives;
use super::inviscid::InviscidFlux;
use super::reconstruction::{
    PrimitiveMusclStencil1d, interface_conserved_pair, reconstruct_face_primitives,
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
    let (left, right) = interface_conserved_pair(eos, &iface)?;
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => roe_flux_with_primitives(
            &left,
            &right,
            &iface.left,
            &iface.right,
            normal,
            eos,
            &roe_cfg,
        ),
        FluxScheme::Hllc => {
            hllc_flux_with_primitives(&left, &right, &iface.left, &iface.right, normal, eos)
        }
        FluxScheme::VanLeer => van_leer_flux(&left, &right, normal, eos),
        FluxScheme::HanelVanLeer => hanel_van_leer_flux(&left, &right, normal, eos),
        FluxScheme::Slau2 => slau2_flux(&left, &right, normal, eos),
    }
}
