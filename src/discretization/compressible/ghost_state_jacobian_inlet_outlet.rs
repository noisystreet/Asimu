//! 入口/出口 ghost 守恒 Jacobian（自 `ghost_state_jacobian` 拆分）。

use crate::core::{Real, Vector3};
use crate::discretization::compressible::bc_compressible::{InletGhostParams, inlet_ghost};
use crate::error::{AsimuError, Result};
use crate::field::{max_physical_increment_scale, state_after_increment};
use crate::mesh::FaceGeometry3d;
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
};

use super::{
    StateJacobian, conserved_to_primitive_jacobian, dr_plus_wrt_primitive, entropy_constant,
    entropy_gradient, farfield_ghost_state_jacobian_wrt_owner, identity_jacobian,
    multiply_state_jacobian, normal_component, normal_velocity, primitive_to_conserved_jacobian,
    scale_row, sound_speed, tangential_velocity, tangential_velocity_jacobian,
    velocity_from_normal_tangent,
};

/// 亚声速/超声速入口 ghost Jacobian 参数。
pub struct InletGhostJacobianParams<'a> {
    pub normal: Vector3,
    pub supersonic: bool,
    pub total_pressure: Real,
    pub total_temperature: Real,
    pub velocity_direction: [Real; 3],
    pub freestream: &'a FreestreamParams,
}

/// \(\partial \mathbf{U}_g / \partial \mathbf{U}_o\)（超声速入口复用远场解析链；亚声速入口用守恒 FD）。
pub fn inlet_ghost_state_jacobian_wrt_owner(
    owner: &ConservedState,
    params: &InletGhostJacobianParams<'_>,
    eos: &IdealGasEoS,
    p_floor: Real,
    epsilon_rel: Real,
) -> Result<StateJacobian> {
    if params.supersonic {
        return farfield_ghost_state_jacobian_wrt_owner(
            owner,
            params.normal,
            params.freestream,
            eos,
            p_floor,
        );
    }
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, p_floor)?;
    if prim.pressure < p_floor {
        return Err(AsimuError::Solver(
            "inlet ghost Jacobian：压力钳制区需数值路径".to_string(),
        ));
    }
    let geom = FaceGeometry3d {
        normal: params.normal,
        spacing: 1.0,
        area: 1.0,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let fs_ctx = FreestreamContext::new(eos, None, None);
    numeric_ghost_jacobian(
        owner,
        |state| {
            inlet_ghost(
                state,
                &geom,
                &InletGhostParams {
                    supersonic: false,
                    velocity_direction: params.velocity_direction,
                    freestream: params.freestream,
                    fs_ctx: &fs_ctx,
                    total_pressure: params.total_pressure,
                    total_temperature: params.total_temperature,
                },
            )
            .map(|g| g.conserved)
        },
        epsilon_rel,
        eos.gamma,
        p_floor,
    )
}

/// \(\partial \mathbf{U}_g / \partial \mathbf{U}_o\)（出口；超声速为零梯度 \(\mathbf I\)）。
pub fn outlet_ghost_state_jacobian_wrt_owner(
    owner: &ConservedState,
    normal: Vector3,
    static_pressure: Real,
    supersonic: bool,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<StateJacobian> {
    if supersonic {
        return Ok(identity_jacobian());
    }
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, p_floor)?;
    if prim.pressure < p_floor {
        return Err(AsimuError::Solver(
            "outlet ghost Jacobian：压力钳制区需数值路径".to_string(),
        ));
    }
    let j_prim = subsonic_outlet_ghost_primitive_jacobian(&prim, normal, static_pressure, eos)?;
    let j_u_to_prim = conserved_to_primitive_jacobian(owner, eos.gamma);
    let ghost_prim = subsonic_outlet_ghost_primitive(&prim, normal, static_pressure, eos)?;
    let j_ghost_prim_to_u = primitive_to_conserved_jacobian(&ghost_prim, eos.gamma);
    Ok(multiply_state_jacobian(
        &j_ghost_prim_to_u,
        &multiply_state_jacobian(&j_prim, &j_u_to_prim),
    ))
}

fn numeric_ghost_jacobian<F>(
    base: &ConservedState,
    mut ghost_at: F,
    epsilon_rel: Real,
    gamma: Real,
    p_floor: Real,
) -> Result<StateJacobian>
where
    F: FnMut(&ConservedState) -> Result<ConservedState>,
{
    let base_ghost = ghost_at(base)?;
    let mut g = [[0.0; 5]; 5];
    for (col, g_col) in g.iter_mut().enumerate() {
        let Some(perturbation) =
            finite_difference_perturbation(base, col, epsilon_rel, gamma, p_floor)
        else {
            continue;
        };
        let perturbed_owner =
            state_after_increment(base, perturbation.increment, perturbation.epsilon);
        let ghost = ghost_at(&perturbed_owner)?;
        let diff = if perturbation.increment[col] >= 0.0 {
            conserved_difference(&ghost, &base_ghost, perturbation.epsilon)
        } else {
            conserved_difference(&base_ghost, &ghost, perturbation.epsilon)
        };
        for (row, entry) in g_col.iter_mut().enumerate() {
            *entry = diff[row];
        }
    }
    Ok(g)
}

struct FiniteDifferencePerturbation {
    increment: [Real; 5],
    epsilon: Real,
}

fn finite_difference_perturbation(
    base_state: &ConservedState,
    col: usize,
    epsilon_rel: Real,
    gamma: Real,
    p_floor: Real,
) -> Option<FiniteDifferencePerturbation> {
    let scales = conserved_component_scales(base_state);
    let requested_eps = epsilon_rel.sqrt() * scales[col];
    if requested_eps <= 0.0 || !requested_eps.is_finite() {
        return None;
    }
    for sign in [1.0, -1.0] {
        let mut increment = component_basis_increment(col);
        if sign < 0.0 {
            for entry in &mut increment {
                *entry *= -1.0;
            }
        }
        let eps =
            max_physical_increment_scale(base_state, increment, requested_eps, gamma, p_floor);
        if eps > 0.0 {
            return Some(FiniteDifferencePerturbation {
                increment,
                epsilon: eps,
            });
        }
    }
    None
}

fn component_basis_increment(component: usize) -> [Real; 5] {
    let mut increment = [0.0; 5];
    increment[component] = 1.0;
    increment
}

fn conserved_component_scales(state: &ConservedState) -> [Real; 5] {
    [
        state.density.abs().max(1.0),
        state.momentum[0].abs().max(1.0),
        state.momentum[1].abs().max(1.0),
        state.momentum[2].abs().max(1.0),
        state.total_energy.abs().max(1.0),
    ]
}

fn conserved_difference(perturbed: &ConservedState, base: &ConservedState, eps: Real) -> [Real; 5] {
    [
        (perturbed.density - base.density) / eps,
        (perturbed.momentum[0] - base.momentum[0]) / eps,
        (perturbed.momentum[1] - base.momentum[1]) / eps,
        (perturbed.momentum[2] - base.momentum[2]) / eps,
        (perturbed.total_energy - base.total_energy) / eps,
    ]
}

fn subsonic_outlet_ghost_primitive(
    owner: &PrimitiveState,
    normal: Vector3,
    static_pressure: Real,
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let owner_sound = sound_speed(owner, eos);
    let outgoing = normal_velocity(owner.velocity, normal) + 2.0 * owner_sound / (eos.gamma - 1.0);
    let entropy = entropy_constant(owner, eos.gamma);
    let density = (static_pressure / entropy).powf(1.0 / eos.gamma);
    let sound = (eos.gamma * static_pressure / density).sqrt();
    let un = outgoing - 2.0 * sound / (eos.gamma - 1.0);
    let tangent = tangential_velocity(owner.velocity, normal);
    primitive_from_pressure_entropy_velocity(
        static_pressure,
        entropy,
        velocity_from_normal_tangent(un, tangent, normal),
        eos,
    )
}

fn subsonic_outlet_ghost_primitive_jacobian(
    owner: &PrimitiveState,
    normal: Vector3,
    static_pressure: Real,
    eos: &IdealGasEoS,
) -> Result<StateJacobian> {
    let entropy = entropy_constant(owner, eos.gamma);
    let density = (static_pressure / entropy).powf(1.0 / eos.gamma);
    let sound = (eos.gamma * static_pressure / density).sqrt();
    let doutgoing = dr_plus_wrt_primitive(owner, normal, eos);
    let dentropy = entropy_gradient(owner, eos.gamma);
    let ddensity_dprim = scale_row(dentropy, -density / (eos.gamma * entropy.max(1.0e-30)));
    let dsound_dprim = scale_row(ddensity_dprim, -0.5 * sound / density.max(1.0e-30));
    let mut dun_g = doutgoing;
    for col in 0..5 {
        dun_g[col] -= 2.0 / (eos.gamma - 1.0) * dsound_dprim[col];
    }
    let tangent_jac = tangential_velocity_jacobian(normal);
    let mut dvelocity = [[0.0; 5]; 3];
    for i in 0..3 {
        for col in 0..5 {
            dvelocity[i][col] = normal_component(i, normal) * dun_g[col] + tangent_jac[i][col];
        }
    }
    let mut jac = [[0.0; 5]; 5];
    for col in 0..5 {
        jac[0][col] = ddensity_dprim[col];
        for i in 0..3 {
            jac[1 + i][col] = dvelocity[i][col];
        }
    }
    let _ = (static_pressure, sound);
    Ok(jac)
}

fn primitive_from_pressure_entropy_velocity(
    pressure: Real,
    entropy: Real,
    velocity: [Real; 3],
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let density = (pressure / entropy).powf(1.0 / eos.gamma);
    Ok(PrimitiveState {
        density,
        velocity,
        pressure,
        temperature: pressure / (density * eos.gas_constant),
    })
}

#[cfg(test)]
#[path = "ghost_state_jacobian_inlet_outlet_tests.rs"]
mod tests;
