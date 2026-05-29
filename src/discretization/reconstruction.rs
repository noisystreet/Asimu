//! 界面状态重构（v0.3 首版：一阶分段常数）。

use crate::physics::ConservedState;

/// 面左右守恒态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceStates {
    /// 面内侧（owner 单元中心值）。
    pub left: ConservedState,
    /// 面外侧（neighbor 或 ghost 单元中心值）。
    pub right: ConservedState,
}

/// 一阶分段常数重构：\(U_L = U_i\)，\(U_R = U_{i+1}\)（或边界 ghost）。
///
/// 法向由 mesh 约定为 owner → neighbor；左右态与法向无关，通量求解器负责投影。
#[must_use]
pub fn reconstruct_first_order(owner: ConservedState, neighbor: ConservedState) -> InterfaceStates {
    InterfaceStates {
        left: owner,
        right: neighbor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
