//! 结构化 3D 网格周期拓扑标志。

use crate::boundary::BoundarySet;

/// 结构化六面体网格各方向的周期配对状态。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuredPeriodic3d {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl StructuredPeriodic3d {
    #[must_use]
    pub fn from_boundary(boundary: &BoundarySet) -> Self {
        Self {
            x: boundary.has_periodic_pair("i_min", "i_max"),
            y: boundary.has_periodic_pair("j_min", "j_max"),
            z: boundary.has_periodic_pair("k_min", "k_max"),
        }
    }
}
