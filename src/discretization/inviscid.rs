//! 无粘 Euler 物理通量 \( \mathbf{F}(\mathbf{U}) \cdot \mathbf{n} \)。
//!
//! 理论：[`docs/theory/inviscid_flux.md`](../../docs/theory/inviscid_flux.md) §2

use crate::core::{ComputeFloat, Real, Vector3};
use crate::field::ConservedResidual;
use crate::field::ConservedResidualT;
use crate::physics::{ConservedState, PrimitiveState};

/// 面法向无粘通量（质量、动量、能量）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InviscidFlux {
    pub mass: Real,
    pub momentum: [Real; 3],
    pub energy: Real,
}

/// 内面无粘 scatter 的可变残差切片。
pub(crate) struct InteriorInviscidResidualMut<'a> {
    pub density: &'a mut [Real],
    pub mx: &'a mut [Real],
    pub my: &'a mut [Real],
    pub mz: &'a mut [Real],
    pub energy: &'a mut [Real],
}

/// scatter 阶段内面几何（预存 RHS 缩放，避免热路径除法）。
#[derive(Debug, Clone, Copy)]
pub struct InteriorInviscidScatterGeom {
    pub owner: usize,
    pub neighbor: usize,
    pub owner_scale: Real,
    pub neighbor_scale: Real,
}

#[must_use]
pub(crate) fn interior_inviscid_residual_mut(
    residual: &mut ConservedResidual,
) -> InteriorInviscidResidualMut<'_> {
    InteriorInviscidResidualMut {
        density: residual.density.values_mut(),
        mx: residual.momentum_x.values_mut(),
        my: residual.momentum_y.values_mut(),
        mz: residual.momentum_z.values_mut(),
        energy: residual.total_energy.values_mut(),
    }
}

/// 将内面无粘通量 scatter 到 owner/neighbor 残差（预存 scale，无 `Result` 分支）。
#[inline(always)]
#[cfg_attr(all(feature = "parallel-fvm", not(test)), allow(dead_code))]
pub(crate) fn scatter_fused_interior_inviscid_face(
    residual: &mut InteriorInviscidResidualMut<'_>,
    geom: &InteriorInviscidScatterGeom,
    flux: &InviscidFlux,
) {
    let owner = geom.owner;
    let neighbor = geom.neighbor;
    let owner_scale = geom.owner_scale;
    let neighbor_scale = geom.neighbor_scale;
    residual.density[owner] += owner_scale * flux.mass;
    residual.mx[owner] += owner_scale * flux.momentum[0];
    residual.my[owner] += owner_scale * flux.momentum[1];
    residual.mz[owner] += owner_scale * flux.momentum[2];
    residual.energy[owner] += owner_scale * flux.energy;
    residual.density[neighbor] += neighbor_scale * flux.mass;
    residual.mx[neighbor] += neighbor_scale * flux.momentum[0];
    residual.my[neighbor] += neighbor_scale * flux.momentum[1];
    residual.mz[neighbor] += neighbor_scale * flux.momentum[2];
    residual.energy[neighbor] += neighbor_scale * flux.energy;
}

/// 将内面无粘通量 scatter 到 typed 残差（通量/scale 仍为 `f64`）。
#[inline(always)]
#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
pub(crate) fn scatter_fused_interior_inviscid_face_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    geom: &InteriorInviscidScatterGeom,
    flux: &InviscidFlux,
) {
    let owner = geom.owner;
    let neighbor = geom.neighbor;
    residual.density.values_mut()[owner] =
        residual.density.values()[owner].add_mul_real(T::from_real(flux.mass), geom.owner_scale);
    residual.momentum_x.values_mut()[owner] = residual.momentum_x.values()[owner]
        .add_mul_real(T::from_real(flux.momentum[0]), geom.owner_scale);
    residual.momentum_y.values_mut()[owner] = residual.momentum_y.values()[owner]
        .add_mul_real(T::from_real(flux.momentum[1]), geom.owner_scale);
    residual.momentum_z.values_mut()[owner] = residual.momentum_z.values()[owner]
        .add_mul_real(T::from_real(flux.momentum[2]), geom.owner_scale);
    residual.total_energy.values_mut()[owner] = residual.total_energy.values()[owner]
        .add_mul_real(T::from_real(flux.energy), geom.owner_scale);
    residual.density.values_mut()[neighbor] = residual.density.values()[neighbor]
        .add_mul_real(T::from_real(flux.mass), geom.neighbor_scale);
    residual.momentum_x.values_mut()[neighbor] = residual.momentum_x.values()[neighbor]
        .add_mul_real(T::from_real(flux.momentum[0]), geom.neighbor_scale);
    residual.momentum_y.values_mut()[neighbor] = residual.momentum_y.values()[neighbor]
        .add_mul_real(T::from_real(flux.momentum[1]), geom.neighbor_scale);
    residual.momentum_z.values_mut()[neighbor] = residual.momentum_z.values()[neighbor]
        .add_mul_real(T::from_real(flux.momentum[2]), geom.neighbor_scale);
    residual.total_energy.values_mut()[neighbor] = residual.total_energy.values()[neighbor]
        .add_mul_real(T::from_real(flux.energy), geom.neighbor_scale);
}

/// 边界面无粘 scatter（typed 残差）。
#[inline(always)]
pub(crate) fn scatter_fused_boundary_inviscid_face_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    owner: usize,
    owner_scale: Real,
    flux: &InviscidFlux,
) {
    residual.density.values_mut()[owner] =
        residual.density.values()[owner].add_mul_real(T::from_real(flux.mass), owner_scale);
    residual.momentum_x.values_mut()[owner] = residual.momentum_x.values()[owner]
        .add_mul_real(T::from_real(flux.momentum[0]), owner_scale);
    residual.momentum_y.values_mut()[owner] = residual.momentum_y.values()[owner]
        .add_mul_real(T::from_real(flux.momentum[1]), owner_scale);
    residual.momentum_z.values_mut()[owner] = residual.momentum_z.values()[owner]
        .add_mul_real(T::from_real(flux.momentum[2]), owner_scale);
    residual.total_energy.values_mut()[owner] =
        residual.total_energy.values()[owner].add_mul_real(T::from_real(flux.energy), owner_scale);
}

/// 边界面无粘 scatter（预存 owner scale）。
#[inline(always)]
pub(crate) fn scatter_fused_boundary_inviscid_face(
    residual: &mut InteriorInviscidResidualMut<'_>,
    owner: usize,
    owner_scale: Real,
    flux: &InviscidFlux,
) {
    residual.density[owner] += owner_scale * flux.mass;
    residual.mx[owner] += owner_scale * flux.momentum[0];
    residual.my[owner] += owner_scale * flux.momentum[1];
    residual.mz[owner] += owner_scale * flux.momentum[2];
    residual.energy[owner] += owner_scale * flux.energy;
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
    use crate::core::approx_eq;
    use crate::discretization::residual::accumulate_interior_face;
    use crate::field::ConservedResidual;
    use crate::physics::{ConservedState, IdealGasEoS};

    #[test]
    fn fused_scatter_matches_accumulate_interior_face() {
        let flux = InviscidFlux {
            mass: 1.0,
            momentum: [2.0, 3.0, 4.0],
            energy: 5.0,
        };
        let mut fused = ConservedResidual::zeros(3).expect("fused");
        let mut legacy = ConservedResidual::zeros(3).expect("legacy");
        let geom = InteriorInviscidScatterGeom {
            owner: 0,
            neighbor: 1,
            owner_scale: -2.0,
            neighbor_scale: 0.5,
        };
        scatter_fused_interior_inviscid_face(
            &mut interior_inviscid_residual_mut(&mut fused),
            &geom,
            &flux,
        );
        accumulate_interior_face(&mut legacy, 0, 1, &flux, 2.0, 1.0, 4.0).expect("legacy");
        assert!(approx_eq(
            fused.density.values()[0],
            legacy.density.values()[0],
            1.0e-15
        ));
        assert!(approx_eq(
            fused.momentum_x.values()[1],
            legacy.momentum_x.values()[1],
            1.0e-15
        ));
        assert!(approx_eq(
            fused.total_energy.values()[0],
            legacy.total_energy.values()[0],
            1.0e-15
        ));
    }

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
