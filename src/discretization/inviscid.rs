//! 无粘 Euler 物理通量 \( \mathbf{F}(\mathbf{U}) \cdot \mathbf{n} \)。

use crate::core::{Real, Vector3};
use crate::physics::{ConservedState, PrimitiveState};

/// 面法向无粘通量（质量、动量、能量）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InviscidFlux {
    pub mass: Real,
    pub momentum: [Real; 3],
    pub energy: Real,
}

/// 由守恒态与原始变量计算 \( \mathbf{F} \cdot \mathbf{n} \)。
#[must_use]
pub fn physical_inviscid_flux(
    cons: &ConservedState,
    prim: &PrimitiveState,
    normal: Vector3,
) -> InviscidFlux {
    let un = velocity_dot_normal(prim.velocity, normal);
    let p = prim.pressure;
    let rho = prim.density;
    let u = prim.velocity;
    InviscidFlux {
        mass: rho * un,
        momentum: [
            rho * un * u[0] + p * normal.x,
            rho * un * u[1] + p * normal.y,
            rho * un * u[2] + p * normal.z,
        ],
        energy: (cons.total_energy + p) * un,
    }
}

#[must_use]
pub fn velocity_dot_normal(velocity: [Real; 3], normal: Vector3) -> Real {
    velocity[0] * normal.x + velocity[1] * normal.y + velocity[2] * normal.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{ConservedState, IdealGasEoS};

    #[test]
    fn uniform_rest_state_has_zero_mass_flux() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let prim = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [1.0, 0.0, 0.0])
            .expect("prim");
        let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
        let flux = physical_inviscid_flux(&cons, &prim, Vector3::new(1.0, 0.0, 0.0));
        assert!(flux.mass.abs() < 1.0e-12);
        assert!(flux.energy.abs() < 1.0e-12);
    }
}
