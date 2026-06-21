//! 一阶边界面通量对 owner 守恒变量的 Jacobian（\(\partial\hat F/\partial U + (\partial\hat F/\partial U_g)(\partial U_g/\partial U)\)）。
//!
//! 内面通量项用 `face_flux_jacobian::first_order_interior_flux_jacobian`；ghost 项对 wall/symmetry/farfield
//! 经原语 BC 解析链式（`ghost_state_jacobian`），热流壁与钳制区仍数值回退。

use crate::core::{Real, Vector3};
use crate::discretization::flux_config::InviscidFluxConfig;
use crate::discretization::unstructured_face_cache::UnstructuredBoundaryInviscidKind;
use crate::error::{AsimuError, Result};
use crate::field::{
    max_physical_increment_scale, primitive_from_conserved_relaxed, state_after_increment,
};
use crate::mesh::FaceGeometry3d;
use crate::physics::{
    ConservedState, FreestreamContext, IdealGasEoS, PrimitiveState, ViscousPhysicsConfig,
};

use super::bc_compressible::{farfield_ghost, outlet_ghost, symmetry_ghost, wall_ghost};
use super::face_flux_jacobian::{ConservedFluxJacobian, first_order_interior_flux_jacobian};
use super::ghost_state_jacobian::{
    InletGhostJacobianParams, StateJacobian, farfield_ghost_state_jacobian_wrt_owner,
    inlet_ghost_state_jacobian_wrt_owner, outlet_ghost_state_jacobian_wrt_owner,
    symmetry_ghost_state_jacobian_wrt_owner, wall_ghost_state_jacobian_wrt_owner,
};

/// 边界面通量 Jacobian 装配上下文（BC 类型已编码在 `inviscid_kind` 中）。
pub struct BoundaryFluxJacobianContext<'a> {
    pub normal: Vector3,
    pub inviscid_kind: UnstructuredBoundaryInviscidKind,
    pub geom: &'a FaceGeometry3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub p_floor: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub epsilon_rel: Real,
}

/// \(\partial \hat{\mathbf{F}} / \partial \mathbf{U}_{\mathrm{owner}}\)（边界面：ghost 随 owner 变化）。
pub fn first_order_boundary_flux_jacobian_wrt_owner(
    owner: &ConservedState,
    ghost: &ConservedState,
    prim_owner: &PrimitiveState,
    ctx: &BoundaryFluxJacobianContext<'_>,
) -> Result<ConservedFluxJacobian> {
    let (d_fl, d_fr) = first_order_interior_flux_jacobian(
        owner,
        ghost,
        prim_owner,
        &primitive_from_conserved_relaxed(ctx.eos, ghost, ctx.p_floor)?,
        ctx.normal,
        ctx.eos,
        ctx.config,
    )?;
    let ghost_jacobian = boundary_ghost_jacobian_wrt_owner(
        owner,
        ctx.inviscid_kind,
        ctx.geom,
        ctx.eos,
        ctx.p_floor,
        ctx.viscous,
        ctx.epsilon_rel,
    )?;
    Ok(add_flux_jacobians(
        d_fl,
        multiply_flux_jacobian(&d_fr, &ghost_jacobian),
    ))
}

pub fn add_flux_jacobians(
    a: ConservedFluxJacobian,
    b: ConservedFluxJacobian,
) -> ConservedFluxJacobian {
    a.add_jacobian(b)
}

pub fn multiply_flux_jacobian(
    flux: &ConservedFluxJacobian,
    state: &StateJacobian,
) -> ConservedFluxJacobian {
    let mut out = ConservedFluxJacobian::zero();
    for row in 0..5 {
        for (col, out_cell) in out.data[row].iter_mut().enumerate() {
            *out_cell = flux.data[row]
                .iter()
                .zip(state.iter())
                .map(|(f, state_row)| f * state_row[col])
                .sum();
        }
    }
    out
}

fn boundary_ghost_jacobian_wrt_owner(
    owner: &ConservedState,
    kind: UnstructuredBoundaryInviscidKind,
    geom: &FaceGeometry3d,
    eos: &IdealGasEoS,
    p_floor: Real,
    viscous: Option<&ViscousPhysicsConfig>,
    epsilon_rel: Real,
) -> Result<StateJacobian> {
    match kind {
        UnstructuredBoundaryInviscidKind::Wall { no_slip, heat } => {
            match wall_ghost_state_jacobian_wrt_owner(
                owner,
                geom.normal,
                no_slip,
                heat,
                eos,
                p_floor,
            ) {
                Ok(jacobian) => Ok(jacobian),
                Err(_) => ghost_jacobian_numeric(
                    owner,
                    |state| boundary_ghost_conserved(state, kind, geom, eos, p_floor, viscous),
                    epsilon_rel,
                    eos.gamma,
                    p_floor,
                ),
            }
        }
        UnstructuredBoundaryInviscidKind::Symmetry => {
            symmetry_ghost_state_jacobian_wrt_owner(owner, geom.normal, eos, p_floor)
        }
        UnstructuredBoundaryInviscidKind::Farfield(params) => {
            match farfield_ghost_state_jacobian_wrt_owner(owner, geom.normal, &params, eos, p_floor)
            {
                Ok(jacobian) => Ok(jacobian),
                Err(_) => ghost_jacobian_numeric(
                    owner,
                    |state| {
                        farfield_ghost(
                            state,
                            geom,
                            &params,
                            &FreestreamContext::new(eos, None, viscous),
                        )
                        .map(|g| g.conserved)
                    },
                    epsilon_rel,
                    eos.gamma,
                    p_floor,
                ),
            }
        }
        UnstructuredBoundaryInviscidKind::Inlet {
            supersonic,
            total_pressure,
            total_temperature,
            velocity_direction,
            freestream: inlet_fs,
        } => inlet_ghost_state_jacobian_wrt_owner(
            owner,
            &InletGhostJacobianParams {
                normal: geom.normal,
                supersonic,
                total_pressure,
                total_temperature,
                velocity_direction,
                freestream: &inlet_fs,
            },
            eos,
            p_floor,
            epsilon_rel,
        ),
        UnstructuredBoundaryInviscidKind::Outlet {
            supersonic,
            static_pressure,
        } => match outlet_ghost_state_jacobian_wrt_owner(
            owner,
            geom.normal,
            static_pressure,
            supersonic,
            eos,
            p_floor,
        ) {
            Ok(jacobian) => Ok(jacobian),
            Err(_) => ghost_jacobian_numeric(
                owner,
                |state| {
                    outlet_ghost(state, geom, static_pressure, supersonic, eos, p_floor)
                        .map(|g| g.conserved)
                },
                epsilon_rel,
                eos.gamma,
                p_floor,
            ),
        },
    }
}

fn boundary_ghost_conserved(
    owner: &ConservedState,
    kind: UnstructuredBoundaryInviscidKind,
    geom: &FaceGeometry3d,
    eos: &IdealGasEoS,
    p_floor: Real,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<ConservedState> {
    let fs_ctx = FreestreamContext::new(eos, None, viscous);
    match kind {
        UnstructuredBoundaryInviscidKind::Wall { no_slip, heat } => {
            Ok(wall_ghost(owner, geom, no_slip, heat, &fs_ctx, p_floor, viscous)?.conserved)
        }
        UnstructuredBoundaryInviscidKind::Symmetry => {
            Ok(symmetry_ghost(owner, geom, &fs_ctx, p_floor, viscous)?.conserved)
        }
        _ => Err(AsimuError::Boundary(
            "boundary_ghost_conserved 仅用于 Wall / Symmetry".to_string(),
        )),
    }
}

fn ghost_jacobian_numeric<F>(
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

#[cfg(test)]
#[path = "boundary_flux_jacobian_tests.rs"]
mod tests;
