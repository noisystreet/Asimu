//! 可压缩场的波速估计工具。

use crate::core::Real;
use crate::error::Result;
use crate::field::{ConservedFields, primitive_from_conserved_relaxed};
use crate::physics::{IdealGasEoS, PrimitiveState};

/// 全场最大波速 \(|u| + a\)（CFL 估计）。
pub fn max_wave_speed(
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<Real> {
    let mut max_speed = Real::EPSILON;
    for i in 0..fields.num_cells() {
        let prim = primitive_from_conserved_relaxed(eos, &fields.cell_state(i)?, min_pressure)?;
        max_speed = max_speed.max(wave_speed_primitive(&prim, eos)?);
    }
    Ok(max_speed)
}

fn wave_speed_primitive(prim: &PrimitiveState, eos: &IdealGasEoS) -> Result<Real> {
    let rho = prim.density.max(1.0e-12);
    let pressure = prim.pressure.max(1.0e-6);
    let speed = (prim.velocity[0] * prim.velocity[0]
        + prim.velocity[1] * prim.velocity[1]
        + prim.velocity[2] * prim.velocity[2])
        .sqrt();
    Ok(speed + (eos.gamma * pressure / rho).sqrt())
}
