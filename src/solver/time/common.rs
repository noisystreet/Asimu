//! 显式时间推进共用辅助（正性钳制等）。

use crate::field::ConservedFields;
use crate::physics::IdealGasEoS;

pub(crate) fn maybe_enforce_positivity(
    _fields: &mut ConservedFields,
    _eos: Option<&IdealGasEoS>,
    _min_pressure: crate::core::Real,
) {
    // 已禁用正性钳制——不做任何操作。
}
