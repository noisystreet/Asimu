//! 一阶无粘面通量对守恒变量的解析 Jacobian（Roe / Hanel–Van Leer / 物理通量）。
//!
//! 理论：[`docs/theory/inviscid_flux.md`](../../../docs/theory/inviscid_flux.md) §4–§5；
//! 用于 `block_lusgs` 预条件器面块装配，替代 off-diagonal / 对角无粘项的有限差分。

#![allow(clippy::too_many_arguments)]

use crate::core::{Real, Vector3};
use crate::discretization::flux_config::{FluxScheme, InviscidFluxConfig};
use crate::error::{AsimuError, Result};
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use super::inviscid::velocity_dot_normal;
use super::roe::{
    RoeFluxConfig, roe_absolute_flux_jacobian_frozen, roe_absolute_flux_jacobian_frozen_precond,
};
use super::van_leer_jacobian::fvs_flux_jacobian_wrt_conserved;

/// \(\partial \hat{\mathbf{F}} / \partial \mathbf{U}\)（行：通量分量，列：守恒分量）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservedFluxJacobian {
    pub data: [[Real; 5]; 5],
}

impl ConservedFluxJacobian {
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            data: [[0.0; 5]; 5],
        }
    }

    #[must_use]
    pub fn scale(self, factor: Real) -> Self {
        let mut out = Self::zero();
        for row in 0..5 {
            for col in 0..5 {
                out.data[row][col] = factor * self.data[row][col];
            }
        }
        out
    }

    #[must_use]
    pub fn add_jacobian(self, other: Self) -> Self {
        let mut out = Self::zero();
        for row in 0..5 {
            for col in 0..5 {
                out.data[row][col] = self.data[row][col] + other.data[row][col];
            }
        }
        out
    }
}

/// 当前配置是否支持解析面块 Jacobian（一阶 Roe / Hanel–Van Leer）。
#[must_use]
pub fn first_order_face_flux_jacobian_supported(config: &InviscidFluxConfig) -> bool {
    config.reconstruction == crate::discretization::ReconstructionKind::FirstOrder
        && matches!(config.scheme, FluxScheme::Roe(_) | FluxScheme::HanelVanLeer)
}

/// 一阶内面通量 Jacobian；`low_mach` 且 `jacobian=true` 时用预处理 Roe \(|A|\)。
pub fn first_order_interior_flux_jacobian_with_low_mach(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
    low_mach: Option<crate::solver::time::LowMachPreconditioningConfig>,
) -> Result<(ConservedFluxJacobian, ConservedFluxJacobian)> {
    if let Some(cfg) = low_mach.filter(|c| c.jacobian) {
        let beta = cfg.face_average_sound_speed_multiplier(prim_l, prim_r, eos.gamma);
        return first_order_interior_flux_jacobian_precond(
            left, right, prim_l, prim_r, normal, eos, config, beta,
        );
    }
    first_order_interior_flux_jacobian(left, right, prim_l, prim_r, normal, eos, config)
}

/// 一阶内面通量：\(\partial \hat{\mathbf{F}} / \partial \mathbf{U}_L\) 与 \(\partial \hat{\mathbf{F}} / \partial \mathbf{U}_R\)。
pub fn first_order_interior_flux_jacobian(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<(ConservedFluxJacobian, ConservedFluxJacobian)> {
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => {
            roe_interior_flux_jacobian(left, right, prim_l, prim_r, normal, eos, &roe_cfg)
        }
        FluxScheme::HanelVanLeer => fvs_flux_jacobian_wrt_conserved(
            left,
            right,
            normal,
            eos,
            super::van_leer_jacobian::FvsEnergySplit::Hanel,
        ),
        _ => Err(AsimuError::Config(
            "解析面块 Jacobian 暂仅支持 first_order Roe / HanelVanLeer".to_string(),
        )),
    }
}

fn first_order_interior_flux_jacobian_precond(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
    beta: Real,
) -> Result<(ConservedFluxJacobian, ConservedFluxJacobian)> {
    match config.scheme {
        FluxScheme::Roe(roe_cfg) => roe_interior_flux_jacobian_precond(
            left, right, prim_l, prim_r, normal, eos, &roe_cfg, beta,
        ),
        FluxScheme::HanelVanLeer => {
            first_order_interior_flux_jacobian(left, right, prim_l, prim_r, normal, eos, config)
        }
        _ => Err(AsimuError::Config(
            "预处理面块 Jacobian 暂仅支持 first_order Roe".to_string(),
        )),
    }
}

fn roe_interior_flux_jacobian_precond(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
    roe_cfg: &RoeFluxConfig,
    beta: Real,
) -> Result<(ConservedFluxJacobian, ConservedFluxJacobian)> {
    let abs_a = roe_absolute_flux_jacobian_frozen_precond(
        left, right, prim_l, prim_r, normal, eos, roe_cfg, beta,
    )?;
    let a_l = physical_inviscid_flux_jacobian_conserved(left, prim_l, normal, eos.gamma);
    let a_r = physical_inviscid_flux_jacobian_conserved(right, prim_r, normal, eos.gamma);
    let abs_j = matrix_to_jacobian(abs_a);
    Ok((
        a_l.add_jacobian(abs_j).scale(0.5),
        a_r.add_jacobian(abs_j.scale(-1.0)).scale(0.5),
    ))
}

fn roe_interior_flux_jacobian(
    left: &ConservedState,
    right: &ConservedState,
    prim_l: &PrimitiveState,
    prim_r: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
    roe_cfg: &RoeFluxConfig,
) -> Result<(ConservedFluxJacobian, ConservedFluxJacobian)> {
    let abs_a =
        roe_absolute_flux_jacobian_frozen(left, right, prim_l, prim_r, normal, eos, roe_cfg)?;
    let a_l = physical_inviscid_flux_jacobian_conserved(left, prim_l, normal, eos.gamma);
    let a_r = physical_inviscid_flux_jacobian_conserved(right, prim_r, normal, eos.gamma);
    let abs_j = matrix_to_jacobian(abs_a);
    Ok((
        a_l.add_jacobian(abs_j).scale(0.5),
        a_r.add_jacobian(abs_j.scale(-1.0)).scale(0.5),
    ))
}

/// 物理通量 \(\mathbf{F}\cdot\mathbf{n}\) 对守恒变量 \(\mathbf{U}=[\rho,\rho\mathbf{u},\rho E]\) 的 Jacobian。
#[must_use]
pub fn physical_inviscid_flux_jacobian_conserved(
    cons: &ConservedState,
    prim: &PrimitiveState,
    normal: Vector3,
    gamma: Real,
) -> ConservedFluxJacobian {
    let rho = cons.density;
    let mx = cons.momentum[0];
    let my = cons.momentum[1];
    let mz = cons.momentum[2];
    let energy = cons.total_energy;
    let nx = normal.x;
    let ny = normal.y;
    let nz = normal.z;
    let s = mx * nx + my * ny + mz * nz;
    let inv_rho = 1.0 / rho;
    let un = s * inv_rho;
    let ke = 0.5 * (mx * mx + my * my + mz * mz) * inv_rho;
    let pressure = (gamma - 1.0) * (energy - ke);
    let gm1 = gamma - 1.0;

    let mut jac = ConservedFluxJacobian::zero();
    // F_mass = m·n
    jac.data[0][1] = nx;
    jac.data[0][2] = ny;
    jac.data[0][3] = nz;

    let dp_drho = gm1 * ke * inv_rho;
    let dp_dmx = -gm1 * mx * inv_rho;
    let dp_dmy = -gm1 * my * inv_rho;
    let dp_dmz = -gm1 * mz * inv_rho;
    let dp_denergy = gm1;

    let dun_drho = -s * inv_rho * inv_rho;
    let dun_dmx = nx * inv_rho;
    let dun_dmy = ny * inv_rho;
    let dun_dmz = nz * inv_rho;

    for (flux_row, mi, ni) in [(1, mx, nx), (2, my, ny), (3, mz, nz)] {
        jac.data[flux_row][0] = dun_drho * mi + dp_drho * ni;
        jac.data[flux_row][1] = dun_dmx * mi + dp_dmx * ni;
        jac.data[flux_row][2] = dun_dmy * mi + dp_dmy * ni;
        jac.data[flux_row][3] = dun_dmz * mi + dp_dmz * ni;
        jac.data[flux_row][4] = dp_denergy * ni;
        jac.data[flux_row][flux_row] += un;
    }

    // F_energy = (E + p) * un
    jac.data[4][0] = (energy + pressure) * dun_drho + un * dp_drho;
    jac.data[4][1] = (energy + pressure) * dun_dmx + un * dp_dmx;
    jac.data[4][2] = (energy + pressure) * dun_dmy + un * dp_dmy;
    jac.data[4][3] = (energy + pressure) * dun_dmz + un * dp_dmz;
    jac.data[4][4] = un * (1.0 + dp_denergy);

    let _ = velocity_dot_normal(prim.velocity, normal);
    jac
}

fn matrix_to_jacobian(matrix: [[Real; 5]; 5]) -> ConservedFluxJacobian {
    ConservedFluxJacobian { data: matrix }
}

#[cfg(test)]
#[path = "face_flux_jacobian_tests.rs"]
mod tests;
