//! 结构化不可压缩内部面速度插值。

use crate::core::Real;
use crate::discretization::incompressible_face_boundary::cell_velocity;
use crate::field::IncompressibleFields;

/// 内部面速度分量：两侧均为真实 cell，边界 ghost 只在边界面通量函数中使用。
#[must_use]
pub(crate) fn interior_face_velocity(
    fields: &IncompressibleFields,
    left: usize,
    right: usize,
    component: usize,
) -> Real {
    0.5 * (cell_velocity(fields, left)[component] + cell_velocity(fields, right)[component])
}
