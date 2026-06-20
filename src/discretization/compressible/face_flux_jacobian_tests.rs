use super::*;
use crate::core::{Real, Vector3};
use crate::discretization::flux_config::InviscidFluxConfig;
use crate::discretization::reconstruction::{interface_conserved_pair, reconstruct_first_order};
use crate::discretization::{RoeFluxConfig, hanel_van_leer_flux, roe_flux};
use crate::error::Result;
use crate::field::{max_physical_increment_scale, primitive_from_conserved, state_after_increment};
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

fn component_basis_increment(component: usize) -> [Real; 5] {
    let mut out = [0.0; 5];
    out[component] = 1.0;
    out
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

fn flux_fd_jacobian(
    flux_fn: impl Fn(&ConservedState) -> Result<crate::discretization::InviscidFlux>,
    state: &ConservedState,
    eos: &IdealGasEoS,
    p_floor: Real,
    epsilon_rel: Real,
) -> [[Real; 5]; 5] {
    let base = flux_fn(state).expect("base flux");
    let base_vec = [
        base.mass,
        base.momentum[0],
        base.momentum[1],
        base.momentum[2],
        base.energy,
    ];
    let scales = conserved_component_scales(state);
    let mut jac = [[0.0; 5]; 5];
    for col in 0..5 {
        let requested_eps = epsilon_rel.sqrt() * scales[col];
        let unit = component_basis_increment(col);
        let eps = max_physical_increment_scale(state, unit, requested_eps, eos.gamma, p_floor);
        let perturbed = state_after_increment(state, unit, eps);
        let flux = flux_fn(&perturbed).expect("pert flux");
        let pert_vec = [
            flux.mass,
            flux.momentum[0],
            flux.momentum[1],
            flux.momentum[2],
            flux.energy,
        ];
        for row in 0..5 {
            jac[row][col] = (pert_vec[row] - base_vec[row]) / eps;
        }
    }
    jac
}

fn assert_jacobian_close(analytic: [[Real; 5]; 5], fd: [[Real; 5]; 5], rtol: Real) {
    for row in 0..5 {
        for col in 0..5 {
            let a = analytic[row][col];
            let f = fd[row][col];
            let scale = a.abs().max(f.abs()).max(1.0);
            assert!(
                (a - f).abs() <= rtol * scale,
                "row={row} col={col} analytic={a} fd={f} rtol={rtol}"
            );
        }
    }
}

#[test]
fn physical_flux_jacobian_matches_finite_difference() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let prim = eos
        .freestream_primitive(0.35, 101_325.0, 300.0, [0.9, 0.1, 0.05])
        .expect("prim");
    let cons = ConservedState::from_primitive(&eos, &prim).expect("cons");
    let normal = Vector3::new(0.6, 0.8, 0.0);
    let prim = primitive_from_conserved(&eos, &cons).expect("prim");
    let analytic = physical_inviscid_flux_jacobian_conserved(&cons, &prim, normal, eos.gamma);
    let fd = flux_fd_jacobian(
        |state| {
            let prim = primitive_from_conserved(&eos, state)?;
            Ok(crate::discretization::physical_inviscid_flux(
                state, &prim, normal,
            ))
        },
        &cons,
        &eos,
        1.0,
        1.0e-8,
    );
    assert_jacobian_close(analytic.data, fd, 2.0e-2);
}

#[test]
fn roe_flux_jacobian_matches_finite_difference() {
    let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
    let left_prim = PrimitiveState {
        density: 1.0,
        velocity: [0.25, 0.05, 0.0],
        pressure: 1.0,
        temperature: 1.0,
    };
    let right_prim = PrimitiveState {
        density: 0.85,
        velocity: [0.15, -0.02, 0.01],
        pressure: 0.92,
        temperature: 1.0,
    };
    let iface = reconstruct_first_order(left_prim, right_prim);
    let (left, right) = interface_conserved_pair(&eos, &iface).expect("cons");
    let normal = Vector3::new(1.0, 0.0, 0.0);
    let config = InviscidFluxConfig::roe_first_order();
    let (d_fl, d_fr) = first_order_interior_flux_jacobian(
        &left,
        &right,
        &iface.left,
        &iface.right,
        normal,
        &eos,
        &config,
    )
    .expect("jac");
    let roe_cfg = RoeFluxConfig::default();
    let fd_left = flux_fd_jacobian(
        |state| roe_flux(state, &right, normal, &eos, &roe_cfg),
        &left,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    let fd_right = flux_fd_jacobian(
        |state| roe_flux(&left, state, normal, &eos, &roe_cfg),
        &right,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    assert_jacobian_close(d_fl.data, fd_left, 0.15);
    assert_jacobian_close(d_fr.data, fd_right, 0.15);
}

#[test]
fn hanel_flux_jacobian_matches_finite_difference() {
    let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
    let left_prim = PrimitiveState {
        density: 1.0,
        velocity: [0.35, 0.1, 0.0],
        pressure: 1.0,
        temperature: 1.0,
    };
    let right_prim = PrimitiveState {
        density: 0.9,
        velocity: [0.2, 0.05, 0.0],
        pressure: 0.95,
        temperature: 1.0,
    };
    let iface = reconstruct_first_order(left_prim, right_prim);
    let (left, right) = interface_conserved_pair(&eos, &iface).expect("cons");
    let normal = Vector3::new(0.707, 0.707, 0.0);
    let config = InviscidFluxConfig::hanel_van_leer_first_order();
    let (d_fl, d_fr) = first_order_interior_flux_jacobian(
        &left,
        &right,
        &iface.left,
        &iface.right,
        normal,
        &eos,
        &config,
    )
    .expect("jac");
    let fd_left = flux_fd_jacobian(
        |state| hanel_van_leer_flux(state, &right, normal, &eos),
        &left,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    let fd_right = flux_fd_jacobian(
        |state| hanel_van_leer_flux(&left, state, normal, &eos),
        &right,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    assert_jacobian_close(d_fl.data, fd_left, 0.15);
    assert_jacobian_close(d_fr.data, fd_right, 0.15);
}
