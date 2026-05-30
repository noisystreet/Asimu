//! 界面状态重构（一阶 / MUSCL）。
//!
//! 理论：[`docs/theory/interface_reconstruction.md`](../../docs/theory/interface_reconstruction.md)

use crate::physics::ConservedState;

use super::flux_common::limited_slope;
use super::flux_config::{ReconstructionKind, SlopeLimiter};

/// 面左右守恒态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceStates {
    /// 面内侧（owner 单元一侧）。
    pub left: ConservedState,
    /// 面外侧（neighbor / ghost 一侧）。
    pub right: ConservedState,
}

/// 1D 面通量重构用的四点模板（owner 左邻、owner、neighbor、neighbor 右邻）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MusclStencil1d<'a> {
    pub left_of_owner: Option<&'a ConservedState>,
    pub owner: &'a ConservedState,
    pub neighbor: &'a ConservedState,
    pub right_of_neighbor: Option<&'a ConservedState>,
}

/// 一阶分段常数重构。
#[must_use]
pub fn reconstruct_first_order(owner: ConservedState, neighbor: ConservedState) -> InterfaceStates {
    InterfaceStates {
        left: owner,
        right: neighbor,
    }
}

/// 按配置重构界面态。
pub fn reconstruct_face_states(
    stencil: MusclStencil1d<'_>,
    kind: ReconstructionKind,
    limiter: SlopeLimiter,
) -> InterfaceStates {
    match kind {
        ReconstructionKind::FirstOrder => InterfaceStates {
            left: *stencil.owner,
            right: *stencil.neighbor,
        },
        ReconstructionKind::Muscl => reconstruct_muscl_1d(stencil, limiter),
    }
}

fn reconstruct_muscl_1d(stencil: MusclStencil1d<'_>, limiter: SlopeLimiter) -> InterfaceStates {
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
    InterfaceStates { left, right }
}

fn extrapolate_left(
    owner: &ConservedState,
    left_of_owner: Option<&ConservedState>,
    neighbor: &ConservedState,
    limiter: SlopeLimiter,
) -> ConservedState {
    let d_plus = component_delta(neighbor, owner);
    let slope = match left_of_owner {
        Some(l) => limited_delta(&component_delta(owner, l), &d_plus, limiter),
        None => zero_delta(),
    };
    add_scaled(owner, &slope, 0.5)
}

fn extrapolate_right(
    owner: &ConservedState,
    neighbor: &ConservedState,
    right_of_neighbor: Option<&ConservedState>,
    limiter: SlopeLimiter,
) -> ConservedState {
    let d_minus = component_delta(neighbor, owner);
    let slope = match right_of_neighbor {
        Some(r) => limited_delta(&d_minus, &component_delta(r, neighbor), limiter),
        None => zero_delta(),
    };
    add_scaled(neighbor, &slope, -0.5)
}

fn limited_delta(
    d_minus: &ConservedDelta,
    d_plus: &ConservedDelta,
    limiter: SlopeLimiter,
) -> ConservedDelta {
    ConservedDelta {
        density: limited_slope(d_minus.density, d_plus.density, limiter),
        momentum: [
            limited_slope(d_minus.momentum[0], d_plus.momentum[0], limiter),
            limited_slope(d_minus.momentum[1], d_plus.momentum[1], limiter),
            limited_slope(d_minus.momentum[2], d_plus.momentum[2], limiter),
        ],
        total_energy: limited_slope(d_minus.total_energy, d_plus.total_energy, limiter),
    }
}

fn zero_delta() -> ConservedDelta {
    ConservedDelta {
        density: 0.0,
        momentum: [0.0, 0.0, 0.0],
        total_energy: 0.0,
    }
}

#[derive(Clone, Copy)]
struct ConservedDelta {
    density: crate::core::Real,
    momentum: [crate::core::Real; 3],
    total_energy: crate::core::Real,
}

fn component_delta(a: &ConservedState, b: &ConservedState) -> ConservedDelta {
    ConservedDelta {
        density: a.density - b.density,
        momentum: [
            a.momentum[0] - b.momentum[0],
            a.momentum[1] - b.momentum[1],
            a.momentum[2] - b.momentum[2],
        ],
        total_energy: a.total_energy - b.total_energy,
    }
}

fn add_scaled(
    base: &ConservedState,
    delta: &ConservedDelta,
    scale: crate::core::Real,
) -> ConservedState {
    ConservedState {
        density: base.density + scale * delta.density,
        momentum: [
            base.momentum[0] + scale * delta.momentum[0],
            base.momentum[1] + scale * delta.momentum[1],
            base.momentum[2] + scale * delta.momentum[2],
        ],
        total_energy: base.total_energy + scale * delta.total_energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn first_order_passes_cell_values_unchanged() {
        let owner = ConservedState {
            density: 1.2,
            momentum: [0.5, 0.0, 0.0],
            total_energy: 2.5,
        };
        let neighbor = ConservedState {
            density: 0.8,
            momentum: [0.1, 0.0, 0.0],
            total_energy: 1.8,
        };
        let iface = reconstruct_first_order(owner, neighbor);
        assert_eq!(iface.left, owner);
        assert_eq!(iface.right, neighbor);
    }

    #[test]
    fn muscl_reduces_to_first_order_without_stencil_neighbors() {
        let owner = ConservedState {
            density: 1.0,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 2.5,
        };
        let neighbor = ConservedState {
            density: 0.5,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 1.2,
        };
        let stencil = MusclStencil1d {
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
        let left = ConservedState {
            density: 1.0,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 2.5,
        };
        let owner = ConservedState {
            density: 1.0,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 2.5,
        };
        let neighbor = ConservedState {
            density: 0.125,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 0.25,
        };
        let right = ConservedState {
            density: 0.125,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 0.25,
        };
        let stencil = MusclStencil1d {
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
