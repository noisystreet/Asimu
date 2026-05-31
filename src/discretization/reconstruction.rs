//! 界面状态重构（一阶 / MUSCL，原始变量 \(\rho,\mathbf{u},p\)）。
//!
//! 理论：[`docs/theory/interface_reconstruction.md`](../../docs/theory/interface_reconstruction.md)

use crate::core::Real;
use crate::error::Result;
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use super::flux_common::limited_slope;
use super::flux_config::{ReconstructionKind, SlopeLimiter};

/// 面左右原始变量态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfacePrimitiveStates {
    /// 面内侧（owner 单元一侧）。
    pub left: PrimitiveState,
    /// 面外侧（neighbor / ghost 一侧）。
    pub right: PrimitiveState,
}

/// 1D 面通量重构用的四点原始变量模板。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimitiveMusclStencil1d<'a> {
    pub left_of_owner: Option<&'a PrimitiveState>,
    pub owner: &'a PrimitiveState,
    pub neighbor: &'a PrimitiveState,
    pub right_of_neighbor: Option<&'a PrimitiveState>,
}

/// 一阶分段常数重构（原始变量）。
#[must_use]
pub fn reconstruct_first_order(
    owner: PrimitiveState,
    neighbor: PrimitiveState,
) -> InterfacePrimitiveStates {
    InterfacePrimitiveStates {
        left: owner,
        right: neighbor,
    }
}

/// 按配置重构界面原始变量。
pub fn reconstruct_face_primitives(
    stencil: PrimitiveMusclStencil1d<'_>,
    kind: ReconstructionKind,
    limiter: SlopeLimiter,
) -> InterfacePrimitiveStates {
    match kind {
        ReconstructionKind::FirstOrder => InterfacePrimitiveStates {
            left: *stencil.owner,
            right: *stencil.neighbor,
        },
        ReconstructionKind::Muscl => reconstruct_muscl_1d(stencil, limiter),
    }
}

/// 界面原始变量 → 守恒变量（供 Riemann / FVS 使用）。
pub fn interface_conserved_pair(
    eos: &IdealGasEoS,
    iface: &InterfacePrimitiveStates,
) -> Result<(ConservedState, ConservedState)> {
    Ok((
        ConservedState::from_primitive(eos, &iface.left)?,
        ConservedState::from_primitive(eos, &iface.right)?,
    ))
}

fn reconstruct_muscl_1d(
    stencil: PrimitiveMusclStencil1d<'_>,
    limiter: SlopeLimiter,
) -> InterfacePrimitiveStates {
    let left = extrapolate_left(
        stencil.owner,
        stencil.left_of_owner,
        stencil.neighbor,
        limiter,
    );
    let right = extrapolate_right(
        stencil.owner,
        stencil.neighbor,
        stencil.right_of_neighbor,
        limiter,
    );
    InterfacePrimitiveStates { left, right }
}

fn extrapolate_left(
    owner: &PrimitiveState,
    left_of_owner: Option<&PrimitiveState>,
    neighbor: &PrimitiveState,
    limiter: SlopeLimiter,
) -> PrimitiveState {
    let d_plus = component_delta(neighbor, owner);
    let slope = match left_of_owner {
        Some(l) => limited_delta(&component_delta(owner, l), &d_plus, limiter),
        None => zero_delta(),
    };
    add_scaled(owner, &slope, 0.5)
}

fn extrapolate_right(
    owner: &PrimitiveState,
    neighbor: &PrimitiveState,
    right_of_neighbor: Option<&PrimitiveState>,
    limiter: SlopeLimiter,
) -> PrimitiveState {
    let d_minus = component_delta(neighbor, owner);
    let slope = match right_of_neighbor {
        Some(r) => limited_delta(&d_minus, &component_delta(r, neighbor), limiter),
        None => zero_delta(),
    };
    add_scaled(neighbor, &slope, -0.5)
}

fn limited_delta(
    d_minus: &PrimitiveDelta,
    d_plus: &PrimitiveDelta,
    limiter: SlopeLimiter,
) -> PrimitiveDelta {
    PrimitiveDelta {
        density: limited_slope(d_minus.density, d_plus.density, limiter),
        velocity: [
            limited_slope(d_minus.velocity[0], d_plus.velocity[0], limiter),
            limited_slope(d_minus.velocity[1], d_plus.velocity[1], limiter),
            limited_slope(d_minus.velocity[2], d_plus.velocity[2], limiter),
        ],
        pressure: limited_slope(d_minus.pressure, d_plus.pressure, limiter),
    }
}

fn zero_delta() -> PrimitiveDelta {
    PrimitiveDelta {
        density: 0.0,
        velocity: [0.0, 0.0, 0.0],
        pressure: 0.0,
    }
}

#[derive(Clone, Copy)]
struct PrimitiveDelta {
    density: Real,
    velocity: [Real; 3],
    pressure: Real,
}

fn component_delta(a: &PrimitiveState, b: &PrimitiveState) -> PrimitiveDelta {
    PrimitiveDelta {
        density: a.density - b.density,
        velocity: [
            a.velocity[0] - b.velocity[0],
            a.velocity[1] - b.velocity[1],
            a.velocity[2] - b.velocity[2],
        ],
        pressure: a.pressure - b.pressure,
    }
}

fn add_scaled(base: &PrimitiveState, delta: &PrimitiveDelta, scale: Real) -> PrimitiveState {
    PrimitiveState {
        density: (base.density + scale * delta.density).max(1.0e-30),
        velocity: [
            base.velocity[0] + scale * delta.velocity[0],
            base.velocity[1] + scale * delta.velocity[1],
            base.velocity[2] + scale * delta.velocity[2],
        ],
        pressure: (base.pressure + scale * delta.pressure).max(1.0e-30),
        temperature: base.temperature,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    fn prim(density: Real, pressure: Real) -> PrimitiveState {
        PrimitiveState {
            density,
            velocity: [0.0, 0.0, 0.0],
            pressure,
            temperature: 1.0,
        }
    }

    #[test]
    fn first_order_passes_cell_values_unchanged() {
        let owner = prim(1.2, 1.0);
        let neighbor = prim(0.8, 0.5);
        let iface = reconstruct_first_order(owner, neighbor);
        assert_eq!(iface.left, owner);
        assert_eq!(iface.right, neighbor);
    }

    #[test]
    fn muscl_reduces_to_first_order_without_stencil_neighbors() {
        let owner = prim(1.0, 1.0);
        let neighbor = prim(0.5, 0.5);
        let stencil = PrimitiveMusclStencil1d {
            left_of_owner: None,
            owner: &owner,
            neighbor: &neighbor,
            right_of_neighbor: None,
        };
        let iface = reconstruct_muscl_1d(stencil, SlopeLimiter::Minmod);
        assert!(approx_eq(iface.left.density, owner.density, 1.0e-12));
        assert!(approx_eq(iface.right.density, neighbor.density, 1.0e-12));
    }

    #[test]
    fn muscl_limits_slope_at_discontinuity() {
        let left = prim(1.0, 1.0);
        let owner = prim(1.0, 1.0);
        let neighbor = prim(0.125, 0.1);
        let right = prim(0.125, 0.1);
        let stencil = PrimitiveMusclStencil1d {
            left_of_owner: Some(&left),
            owner: &owner,
            neighbor: &neighbor,
            right_of_neighbor: Some(&right),
        };
        let iface = reconstruct_muscl_1d(stencil, SlopeLimiter::Minmod);
        assert!(approx_eq(iface.left.density, owner.density, 1.0e-12));
        assert!(approx_eq(iface.right.density, neighbor.density, 1.0e-12));
    }
}
