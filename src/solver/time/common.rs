//! 显式时间推进共用辅助（正性钳制等）。

use crate::core::ComputeFloat;
use crate::field::ConservedFieldsT;
use crate::physics::IdealGasEoS;

pub(crate) fn maybe_enforce_positivity<T: ComputeFloat>(
    _fields: &mut ConservedFieldsT<T>,
    _eos: Option<&IdealGasEoS>,
    _min_pressure: crate::core::Real,
) {
    // 已禁用正性钳制——不做任何操作。
}
