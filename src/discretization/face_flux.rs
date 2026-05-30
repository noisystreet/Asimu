//! 无粘面通量入口：界面重构 + 数值 Riemann 求解 dispatch。

use crate::core::Vector3;
use crate::error::Result;
use crate::physics::{ConservedState, IdealGasEoS};

use super::flux_config::{FluxScheme, InviscidFluxConfig};
use super::hllc::hllc_flux;
use super::inviscid::InviscidFlux;
use super::reconstruction::{MusclStencil1d, reconstruct_face_states};
use super::roe::roe_flux;
use super::slau2::slau2_flux;
use super::van_leer::{hanel_van_leer_flux, van_leer_flux};

/// 面通量输入：owner/neighbor 及可选 MUSCL 模板点。
#[derive(Debug, Clone, Copy)]
pub struct FaceFluxInput<'a> {
    pub owner: &'a ConservedState,
    pub neighbor: &'a ConservedState,
    pub left_of_owner: Option<&'a ConservedState>,
    pub right_of_neighbor: Option<&'a ConservedState>,
}

impl<'a> FaceFluxInput<'a> {
    #[must_use]
    pub const fn first_order(owner: &'a ConservedState, neighbor: &'a ConservedState) -> Self {
        Self {
            owner,
            neighbor,
            left_of_owner: None,
            right_of_neighbor: None,
        }
    }

    #[must_use]
    pub const fn from_stencil(stencil: MusclStencil1d<'a>) -> Self {
        Self {
            owner: stencil.owner,
            neighbor: stencil.neighbor,
            left_of_owner: stencil.left_of_owner,
            right_of_neighbor: stencil.right_of_neighbor,
        }
    }
}

/// 面数值通量：按配置做界面重构并调用 Roe / HLLC。
pub fn face_inviscid_flux(
    input: FaceFluxInput<'_>,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<InviscidFlux> {
    let stencil = MusclStencil1d {
        left_of_owner: input.left_of_owner,
        owner: input.owner,
        neighbor: input.neighbor,
        right_of_neighbor: input.right_of_neighbor,
    };
    let iface = reconstruct_face_states(stencil, config.reconstruction, config.limiter);
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => roe_flux(&iface.left, &iface.right, normal, eos, &roe_cfg),
        FluxScheme::Hllc => hllc_flux(&iface.left, &iface.right, normal, eos),
        FluxScheme::VanLeer => van_leer_flux(&iface.left, &iface.right, normal, eos),
        FluxScheme::HanelVanLeer => hanel_van_leer_flux(&iface.left, &iface.right, normal, eos),
        FluxScheme::Slau2 => slau2_flux(&iface.left, &iface.right, normal, eos),
    }
}
