//! 显式时间推进共用辅助（正性钳制等）。

use crate::field::ConservedFields;
use crate::physics::IdealGasEoS;

pub(crate) fn maybe_enforce_positivity(
    fields: &mut ConservedFields,
    eos: Option<&IdealGasEoS>,
    min_pressure: crate::core::Real,
) {
    if let Some(eos) = eos {
        fields.enforce_positivity(eos, min_pressure);
    }
}
