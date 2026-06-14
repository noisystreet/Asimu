//! 非结构一阶无粘面通量 typed 分发。

pub use crate::discretization::face_flux_typed::InviscidFaceFluxTyped;

use crate::discretization::InviscidFluxConfig;
use crate::error::Result;
use crate::field::PrimitiveFieldsT;
use crate::physics::IdealGasEoS;

/// 兼容别名：一阶装配 trait 与 [`InviscidFaceFluxTyped`] 相同。
pub(super) trait InviscidFirstOrderFaceFlux: InviscidFaceFluxTyped {}

impl<T: InviscidFaceFluxTyped> InviscidFirstOrderFaceFlux for T {}

#[allow(dead_code)]
#[cfg_attr(not(feature = "parallel-fvm"), allow(dead_code))]
pub(super) fn first_order_interior_flux<T: InviscidFaceFluxTyped>(
    primitives: &PrimitiveFieldsT<T>,
    owner: usize,
    neighbor: usize,
    normal: crate::core::Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<crate::discretization::InviscidFlux> {
    T::first_order_interior_soa(primitives, owner, neighbor, normal, eos, config)
}
