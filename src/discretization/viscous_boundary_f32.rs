//! f32 粘性边界面通量 compute + scatter（几何仍 f64）。

use crate::boundary::WallHeat;
use crate::core::{Real, Vector3};
use crate::discretization::gradient_typed::{GradientFieldsT, VelocityGradientT};
use crate::discretization::viscous::face_transport_coefficients;
use crate::discretization::viscous_assembly::ViscousBoundaryFaceKind;
use crate::discretization::viscous_f32::{
    ColoredViscousFaceFluxF32, ViscousFaceAveragedLaneF32,
    fused_interior_viscous_face_flux_averaged_f32,
};
use crate::error::Result;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};
use crate::physics::{IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

/// f32 原始变量（边界面通量局部态；定义见 [`PrimitiveStateF32`]）。
pub use crate::physics::PrimitiveStateF32;

/// f32 粘性边界面通量输入。
pub struct ViscousBoundaryFluxParamsF32<'a> {
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub gradients: &'a GradientFieldsT<f32>,
}

#[must_use]
pub fn primitive_state_f32_to_real(prim: PrimitiveStateF32) -> PrimitiveState {
    PrimitiveState {
        density: prim.density as Real,
        velocity: [
            prim.velocity[0] as Real,
            prim.velocity[1] as Real,
            prim.velocity[2] as Real,
        ],
        pressure: prim.pressure as Real,
        temperature: prim.temperature as Real,
    }
}

#[must_use]
pub fn primitive_state_f32_from_real(prim: PrimitiveState) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
        density: prim.density as f32,
        velocity: [
            prim.velocity[0] as f32,
            prim.velocity[1] as f32,
            prim.velocity[2] as f32,
        ],
        pressure: prim.pressure as f32,
        temperature: prim.temperature as f32,
    }
}

#[must_use]
pub fn viscous_primitive_at_f32(
    primitives: &PrimitiveFieldsT<f32>,
    temperatures: &[f32],
    cell: usize,
) -> PrimitiveStateF32 {
    PrimitiveStateF32 {
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
pub fn wall_extrapolated_gradient_f32(
    grad_cell: &VelocityGradientT<f32>,
    prim_owner: &PrimitiveStateF32,
    prim_ghost: &PrimitiveStateF32,
    normal: Vector3,
    spacing: Real,
) -> VelocityGradientT<f32> {
    if spacing <= Real::EPSILON {
        return *grad_cell;
    }
    let inv_two_delta = 1.0_f32 / (2.0 * spacing as f32);
    let nx = normal.x as f32;
    let ny = normal.y as f32;
    let nz = normal.z as f32;
    let mut grad = *grad_cell;
    for (grad_comp, u_o, u_g) in [
        (&mut grad.du, prim_owner.velocity[0], prim_ghost.velocity[0]),
        (&mut grad.dv, prim_owner.velocity[1], prim_ghost.velocity[1]),
        (&mut grad.dw, prim_owner.velocity[2], prim_ghost.velocity[2]),
    ] {
        let dudn = (u_g - u_o) * inv_two_delta;
        let grad_n = grad_comp[0] * nx + grad_comp[1] * ny + grad_comp[2] * nz;
        let corr = dudn - grad_n;
        grad_comp[0] += corr * nx;
        grad_comp[1] += corr * ny;
        grad_comp[2] += corr * nz;
    }
    let dtdn = (prim_ghost.temperature - prim_owner.temperature) * inv_two_delta;
    let grad_t_n = grad.dt[0] * nx + grad.dt[1] * ny + grad.dt[2] * nz;
    let corr_t = dtdn - grad_t_n;
    grad.dt[0] += corr_t * nx;
    grad.dt[1] += corr_t * ny;
    grad.dt[2] += corr_t * nz;
    grad
}

#[must_use]
fn average_gradient_f32(
    left: &VelocityGradientT<f32>,
    right: &VelocityGradientT<f32>,
) -> VelocityGradientT<f32> {
    let half = 0.5_f32;
    VelocityGradientT {
        du: [
            half * (left.du[0] + right.du[0]),
            half * (left.du[1] + right.du[1]),
            half * (left.du[2] + right.du[2]),
        ],
        dv: [
            half * (left.dv[0] + right.dv[0]),
            half * (left.dv[1] + right.dv[1]),
            half * (left.dv[2] + right.dv[2]),
        ],
        dw: [
            half * (left.dw[0] + right.dw[0]),
            half * (left.dw[1] + right.dw[1]),
            half * (left.dw[2] + right.dw[2]),
        ],
        dt: [
            half * (left.dt[0] + right.dt[0]),
            half * (left.dt[1] + right.dt[1]),
            half * (left.dt[2] + right.dt[2]),
        ],
    }
}

#[must_use]
fn viscous_face_flux_f32(
    prim_l: &PrimitiveStateF32,
    grad_l: &VelocityGradientT<f32>,
    prim_r: &PrimitiveStateF32,
    grad_r: &VelocityGradientT<f32>,
    normal: Vector3,
    mu: f32,
    lambda: f32,
) -> ColoredViscousFaceFluxF32 {
    let grad = average_gradient_f32(grad_l, grad_r);
    let half = 0.5_f32;
    let lane = ViscousFaceAveragedLaneF32 {
        ux: half * (prim_l.velocity[0] + prim_r.velocity[0]),
        uy: half * (prim_l.velocity[1] + prim_r.velocity[1]),
        uz: half * (prim_l.velocity[2] + prim_r.velocity[2]),
        du_dx: grad.du[0],
        du_dy: grad.du[1],
        du_dz: grad.du[2],
        dv_dx: grad.dv[0],
        dv_dy: grad.dv[1],
        dv_dz: grad.dv[2],
        dw_dx: grad.dw[0],
        dw_dy: grad.dw[1],
        dw_dz: grad.dw[2],
        dt_dx: grad.dt[0],
        dt_dy: grad.dt[1],
        dt_dz: grad.dt[2],
    };
    fused_interior_viscous_face_flux_averaged_f32(
        lane,
        normal.x as f32,
        normal.y as f32,
        normal.z as f32,
        mu,
        lambda,
    )
}

#[must_use]
fn wall_heat_flux_into_fluid_f32(
    t_owner: f32,
    t_ghost: f32,
    spacing: Real,
    lambda: f32,
    wall_heat: WallHeat,
) -> f32 {
    match wall_heat {
        WallHeat::Adiabatic => 0.0,
        WallHeat::HeatFlux { flux } => flux as f32,
        WallHeat::Isothermal { temperature } => {
            if spacing <= Real::EPSILON {
                0.0
            } else {
                let _ = t_ghost;
                lambda * ((temperature as f32) - t_owner) / (spacing as f32)
            }
        }
    }
}

/// 边界面粘性通量（f32 compute；物性/几何仍 f64）。
pub fn viscous_flux_at_boundary_f32(
    params: &ViscousBoundaryFluxParamsF32<'_>,
    owner: usize,
    ghost_prim: PrimitiveStateF32,
    normal: Vector3,
    spacing: Real,
    kind: ViscousBoundaryFaceKind,
    temperatures: &[f32],
) -> Result<ColoredViscousFaceFluxF32> {
    let prim_o = viscous_primitive_at_f32(params.primitives, temperatures, owner);
    let t_ghost = params.viscous.static_temperature(
        ghost_prim.pressure as Real,
        ghost_prim.density.max(1.0e-30_f32) as Real,
        params.eos,
    );
    let mut ghost = ghost_prim;
    ghost.temperature = t_ghost as f32;
    let grad_o = params.gradients.velocity_grad_at(owner);
    let grad_g = if kind.is_wall {
        wall_extrapolated_gradient_f32(&grad_o, &prim_o, &ghost, normal, spacing)
    } else {
        grad_o
    };
    let (mu, lambda) = face_transport_coefficients(
        temperatures[owner] as Real,
        ghost.temperature as Real,
        params.viscous,
        params.eos,
    )?;
    let mu = mu as f32;
    let lambda = lambda as f32;
    let mut flux = viscous_face_flux_f32(&prim_o, &grad_o, &ghost, &grad_g, normal, mu, lambda);
    if kind.no_slip {
        let grad = average_gradient_f32(&grad_o, &grad_g);
        flux.energy = lambda
            * (grad.dt[0] * normal.x as f32
                + grad.dt[1] * normal.y as f32
                + grad.dt[2] * normal.z as f32);
    }
    if let Some(heat) = kind.wall_heat {
        flux.energy = wall_heat_flux_into_fluid_f32(
            prim_o.temperature,
            ghost.temperature,
            spacing,
            lambda,
            heat,
        );
    }
    Ok(flux)
}

#[inline(always)]
pub fn scatter_viscous_boundary_f32(
    residual: &mut ConservedResidualT<f32>,
    owner: usize,
    flux: &ColoredViscousFaceFluxF32,
    area: f32,
    owner_volume: f32,
) {
    let scale = -area / owner_volume;
    residual.momentum_x.values_mut()[owner] += scale * flux.mx;
    residual.momentum_y.values_mut()[owner] += scale * flux.my;
    residual.momentum_z.values_mut()[owner] += scale * flux.mz;
    residual.total_energy.values_mut()[owner] += scale * flux.energy;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::gradient_typed::GradientFieldsT;
    use crate::field::PrimitiveFieldsT;

    #[test]
    fn f32_wall_extrapolation_adjusts_normal_gradient_component() {
        let grad_o = VelocityGradientT {
            du: [0.0, 1.0, 0.0],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let owner = PrimitiveStateF32 {
            density: 1.0,
            velocity: [0.0, 0.0, 0.0],
            pressure: 1.0,
            temperature: 300.0,
        };
        let ghost = PrimitiveStateF32 {
            velocity: [2.0, 0.0, 0.0],
            ..owner
        };
        let grad_g = wall_extrapolated_gradient_f32(
            &grad_o,
            &owner,
            &ghost,
            Vector3::new(1.0, 0.0, 0.0),
            0.1,
        );
        assert!((grad_g.du[0] - 10.0).abs() < 1.0e-4);
        assert!((grad_g.du[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn f32_uniform_farfield_boundary_flux_is_near_zero() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous = ViscousPhysicsConfig::default();
        let n = 1;
        let prim = {
            let mut p = PrimitiveFieldsT::<f32>::zeros(n).expect("p");
            p.density.values_mut()[0] = 1.0;
            p.pressure.values_mut()[0] = 101_325.0;
            p.velocity_x.values_mut()[0] = 10.0;
            p
        };
        let mut grad = GradientFieldsT::<f32>::zeros(n).expect("g");
        grad.dt_dx.values_mut()[0] = 0.0;
        let params = ViscousBoundaryFluxParamsF32 {
            eos: &eos,
            viscous: &viscous,
            primitives: &prim,
            gradients: &grad,
        };
        let ghost = viscous_primitive_at_f32(&prim, &[300.0_f32], 0);
        let flux = viscous_flux_at_boundary_f32(
            &params,
            0,
            ghost,
            Vector3::new(1.0, 0.0, 0.0),
            0.2,
            ViscousBoundaryFaceKind {
                is_wall: false,
                no_slip: false,
                wall_heat: None,
            },
            &[300.0_f32],
        )
        .expect("flux");
        assert!(flux.mx.abs() < 1.0e-5);
        assert!(flux.my.abs() < 1.0e-5);
        assert!(flux.mz.abs() < 1.0e-5);
        assert!(flux.energy.abs() < 1.0e-5);
    }
}
