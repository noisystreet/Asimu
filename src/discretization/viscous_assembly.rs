//! 结构/非结构粘性残差装配共用逻辑（壁面通量、scatter 符号约定）。

use crate::boundary::WallHeat;
use crate::core::{Real, Vector3};
use crate::discretization::InviscidFlux;
use crate::discretization::gradient::{GradientFields, VelocityGradient};
use crate::discretization::residual::{accumulate_boundary_face, accumulate_interior_face};
use crate::discretization::viscous::{
    ViscousFlux, average_gradient_for_wall, face_transport_coefficients, viscous_face_flux,
};
use crate::discretization::wall_thermal::wall_heat_flux_into_fluid;
use crate::error::Result;
use crate::field::{ConservedResidual, PrimitiveFields};
use crate::physics::{IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

/// 粘性边界面 BC 语义（结构/非结构统一）。
#[derive(Debug, Clone, Copy)]
pub struct ViscousBoundaryFaceKind {
    pub is_wall: bool,
    pub no_slip: bool,
    pub wall_heat: Option<WallHeat>,
}

/// 粘性边界面通量输入（不含 ghost 构造）。
pub struct ViscousBoundaryFluxParams<'a> {
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub primitives: &'a PrimitiveFields,
    pub gradients: &'a GradientFields,
}

/// 由 SoA 原始变量与预计算温度构造 `PrimitiveState`。
#[must_use]
pub fn viscous_primitive_at(
    primitives: &PrimitiveFields,
    temperatures: &[Real],
    cell: usize,
) -> PrimitiveState {
    PrimitiveState {
        density: primitives.density.values()[cell],
        velocity: [
            primitives.velocity_x.values()[cell],
            primitives.velocity_y.values()[cell],
            primitives.velocity_z.values()[cell],
        ],
        pressure: primitives.pressure.values()[cell],
        temperature: temperatures[cell],
    }
}

/// 壁面 ghost：法向分量用 \((\phi_g-\phi_o)/(2\delta)\)，切向保留单元差分梯度。
#[must_use]
pub fn wall_extrapolated_gradient(
    grad_cell: &VelocityGradient,
    prim_owner: &PrimitiveState,
    prim_ghost: &PrimitiveState,
    normal: Vector3,
    spacing: Real,
) -> VelocityGradient {
    if spacing <= Real::EPSILON {
        return *grad_cell;
    }
    let inv_two_delta = 1.0 / (2.0 * spacing);
    let mut grad = *grad_cell;
    for (grad_comp, u_o, u_g) in [
        (&mut grad.du, prim_owner.velocity[0], prim_ghost.velocity[0]),
        (&mut grad.dv, prim_owner.velocity[1], prim_ghost.velocity[1]),
        (&mut grad.dw, prim_owner.velocity[2], prim_ghost.velocity[2]),
    ] {
        let dudn = (u_g - u_o) * inv_two_delta;
        let grad_n = grad_comp[0] * normal.x + grad_comp[1] * normal.y + grad_comp[2] * normal.z;
        let corr = dudn - grad_n;
        grad_comp[0] += corr * normal.x;
        grad_comp[1] += corr * normal.y;
        grad_comp[2] += corr * normal.z;
    }
    let dtdn = (prim_ghost.temperature - prim_owner.temperature) * inv_two_delta;
    let grad_t_n = grad.dt[0] * normal.x + grad.dt[1] * normal.y + grad.dt[2] * normal.z;
    let corr_t = dtdn - grad_t_n;
    grad.dt[0] += corr_t * normal.x;
    grad.dt[1] += corr_t * normal.y;
    grad.dt[2] += corr_t * normal.z;
    grad
}

/// 边界面粘性通量（Newtonian + 壁面热边界）。
pub fn viscous_flux_at_boundary(
    params: &ViscousBoundaryFluxParams<'_>,
    owner: usize,
    ghost_prim: PrimitiveState,
    normal: Vector3,
    spacing: Real,
    kind: ViscousBoundaryFaceKind,
    temperatures: &[Real],
) -> Result<ViscousFlux> {
    let prim_o = viscous_primitive_at(params.primitives, temperatures, owner);
    let t_ghost = params.viscous.static_temperature(
        ghost_prim.pressure,
        ghost_prim.density.max(1.0e-30),
        params.eos,
    );
    let mut ghost = ghost_prim;
    ghost.temperature = t_ghost;
    let grad_o = params.gradients.velocity_grad_at(owner);
    let grad_g = if kind.is_wall {
        wall_extrapolated_gradient(&grad_o, &prim_o, &ghost, normal, spacing)
    } else {
        grad_o
    };
    let (mu, lambda) =
        face_transport_coefficients(temperatures[owner], t_ghost, params.viscous, params.eos)?;
    let mut flux = viscous_face_flux(&prim_o, &grad_o, &ghost, &grad_g, normal, mu, lambda);
    if kind.no_slip {
        let grad = average_gradient_for_wall(&grad_o, &grad_g);
        flux.energy =
            lambda * (grad.dt[0] * normal.x + grad.dt[1] * normal.y + grad.dt[2] * normal.z);
    }
    if let Some(heat) = kind.wall_heat {
        flux.energy =
            wall_heat_flux_into_fluid(prim_o.temperature, ghost.temperature, spacing, lambda, heat);
    }
    Ok(flux)
}

/// NS 动量式符号约定 → FVM scatter 用无粘通量布局。
#[must_use]
pub fn viscous_flux_for_accumulation(flux: &ViscousFlux) -> InviscidFlux {
    InviscidFlux {
        mass: flux.mass,
        momentum: [-flux.momentum[0], -flux.momentum[1], -flux.momentum[2]],
        energy: flux.energy,
    }
}

/// 内面粘性通量 scatter。
pub fn accumulate_viscous_interior(
    residual: &mut ConservedResidual,
    owner: usize,
    neighbor: usize,
    flux: &ViscousFlux,
    area: Real,
    owner_volume: Real,
    neighbor_volume: Real,
) -> Result<()> {
    let inv = viscous_flux_for_accumulation(flux);
    accumulate_interior_face(
        residual,
        owner,
        neighbor,
        &inv,
        area,
        owner_volume,
        neighbor_volume,
    )
}

/// 边界面粘性通量 scatter。
pub fn accumulate_viscous_boundary(
    residual: &mut ConservedResidual,
    owner: usize,
    flux: &ViscousFlux,
    area: Real,
    owner_volume: Real,
) -> Result<()> {
    let inv = viscous_flux_for_accumulation(flux);
    accumulate_boundary_face(residual, owner, &inv, area, owner_volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::gradient::VelocityGradient;
    use crate::field::ConservedResidual;
    use crate::physics::ViscosityModel;

    #[test]
    fn shear_layer_viscous_work_heats_slow_side() {
        use crate::core::Vector3;
        use crate::physics::IdealGasEoS;

        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(1.0e-5).expect("mu"), 0.72)
                .expect("cfg");
        let slow = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [10.0, 0.0, 0.0])
            .expect("prim");
        let mut fast = slow;
        fast.velocity[0] = 110.0;
        let grad_l = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let grad_r = VelocityGradient {
            du: [100.0, 0.0, 0.0],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let (mu, lambda) = face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc");
        let flux = viscous_face_flux(
            &slow,
            &grad_l,
            &fast,
            &grad_r,
            Vector3::new(1.0, 0.0, 0.0),
            mu,
            lambda,
        );
        let mut rhs = ConservedResidual::zeros(2).expect("rhs");
        accumulate_viscous_interior(&mut rhs, 0, 1, &flux, 1.0, 1.0, 1.0).expect("acc");
        assert!(
            rhs.total_energy.values()[0] > 1.0e-12,
            "shear dissipation should heat slower owner cell, got {}",
            rhs.total_energy.values()[0]
        );
    }
}
