use super::*;
use crate::boundary::WallHeat;
use crate::core::Vector3;
use crate::field::{max_physical_increment_scale, state_after_increment};
use crate::mesh::FaceGeometry3d;
use crate::physics::{ConservedState, FreestreamParams, IdealGasEoS, PrimitiveState};

fn consistent_prim(
    eos: &IdealGasEoS,
    density: Real,
    velocity: [Real; 3],
    pressure: Real,
) -> PrimitiveState {
    PrimitiveState {
        density,
        velocity,
        pressure,
        temperature: pressure / (density * eos.gas_constant),
    }
}

fn ghost_fd_jacobian<F>(
    ghost_fn: F,
    owner: &ConservedState,
    eos: &IdealGasEoS,
    p_floor: Real,
    epsilon_rel: Real,
) -> StateJacobian
where
    F: Fn(&ConservedState) -> crate::error::Result<ConservedState>,
{
    let base = ghost_fn(owner).expect("base ghost");
    let scales = [
        owner.density.abs().max(1.0),
        owner.momentum[0].abs().max(1.0),
        owner.momentum[1].abs().max(1.0),
        owner.momentum[2].abs().max(1.0),
        owner.total_energy.abs().max(1.0),
    ];
    let mut jac = [[0.0; 5]; 5];
    for col in 0..5 {
        let requested_eps = epsilon_rel.sqrt() * scales[col];
        let mut increment = [0.0; 5];
        increment[col] = 1.0;
        let eps = max_physical_increment_scale(owner, increment, requested_eps, eos.gamma, p_floor);
        if eps <= 0.0 {
            continue;
        }
        let perturbed = state_after_increment(owner, increment, eps);
        let ghost = ghost_fn(&perturbed).expect("pert ghost");
        jac[0][col] = (ghost.density - base.density) / eps;
        for i in 0..3 {
            jac[1 + i][col] = (ghost.momentum[i] - base.momentum[i]) / eps;
        }
        jac[4][col] = (ghost.total_energy - base.total_energy) / eps;
    }
    jac
}

fn assert_jacobian_close(analytic: StateJacobian, fd: StateJacobian, rtol: Real) {
    for row in 0..5 {
        for col in 0..5 {
            let a = analytic[row][col];
            let f = fd[row][col];
            let scale = a.abs().max(f.abs()).max(1.0);
            assert!(
                (a - f).abs() <= rtol * scale,
                "row={row} col={col} analytic={a} fd={f}"
            );
        }
    }
}

#[test]
fn wall_no_slip_ghost_jacobian_matches_finite_difference() {
    let eos = IdealGasEoS::new(1.4, 287.0).expect("eos");
    let prim = consistent_prim(&eos, 1.0, [0.35, 0.08, 0.02], 101_325.0);
    let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
    let normal = Vector3::new(1.0, 0.0, 0.0);
    let geom = FaceGeometry3d {
        normal,
        spacing: 0.1,
        area: 0.2,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let analytic = wall_ghost_state_jacobian_wrt_owner(
        &owner,
        normal,
        true,
        WallHeat::Adiabatic,
        &eos,
        1.0e-6,
    )
    .expect("analytic");
    let fd = ghost_fd_jacobian(
        |state| {
            Ok(crate::discretization::wall_ghost(
                state,
                &geom,
                true,
                WallHeat::Adiabatic,
                &crate::physics::FreestreamContext::new(&eos, None, None),
                1.0e-6,
                None,
            )?
            .conserved)
        },
        &owner,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    assert_jacobian_close(analytic, fd, 0.05);
}

#[test]
fn symmetry_ghost_jacobian_matches_finite_difference() {
    let eos = IdealGasEoS::new(1.4, 287.0).expect("eos");
    let prim = consistent_prim(&eos, 1.0, [0.2, 0.15, 0.05], 101_325.0);
    let owner = ConservedState::from_primitive(&eos, &prim).expect("owner");
    let normal = Vector3::new(0.0, 1.0, 0.0);
    let geom = FaceGeometry3d {
        normal,
        spacing: 0.1,
        area: 0.2,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let analytic =
        symmetry_ghost_state_jacobian_wrt_owner(&owner, normal, &eos, 1.0e-6).expect("analytic");
    let fd = ghost_fd_jacobian(
        |state| {
            Ok(crate::discretization::symmetry_ghost(
                state,
                &geom,
                &crate::physics::FreestreamContext::new(&eos, None, None),
                1.0e-6,
                None,
            )?
            .conserved)
        },
        &owner,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    assert_jacobian_close(analytic, fd, 0.05);
}

#[test]
fn farfield_subsonic_mixed_ghost_jacobian_matches_finite_difference() {
    let eos = IdealGasEoS::AIR_STANDARD;
    let fs = FreestreamParams {
        mach: 0.1,
        pressure: 101_325.0,
        temperature: 288.15,
        velocity_direction: [1.0, 0.0, 0.0],
        alpha: 0.0,
        beta: 0.0,
    };
    let farfield = eos
        .freestream_primitive(fs.mach, fs.pressure, fs.temperature, fs.velocity_direction)
        .expect("farfield");
    let owner_prim = PrimitiveState {
        density: farfield.density * 1.05,
        velocity: [farfield.velocity[0] * 0.9, 0.02, 0.01],
        pressure: farfield.pressure * 1.02,
        temperature: farfield.temperature,
    };
    let owner_prim = consistent_prim(
        &eos,
        owner_prim.density,
        owner_prim.velocity,
        owner_prim.pressure,
    );
    let owner = ConservedState::from_primitive(&eos, &owner_prim).expect("owner");
    let normal = Vector3::new(1.0, 0.0, 0.0);
    let geom = FaceGeometry3d {
        normal,
        spacing: 0.1,
        area: 0.2,
        center: Vector3::new(0.0, 0.0, 0.0),
    };
    let analytic = farfield_ghost_state_jacobian_wrt_owner(&owner, normal, &fs, &eos, 1.0e-6)
        .expect("analytic");
    let fd = ghost_fd_jacobian(
        |state| {
            Ok(crate::discretization::farfield_ghost(
                state,
                &geom,
                &fs,
                &crate::physics::FreestreamContext::new(&eos, None, None),
            )?
            .conserved)
        },
        &owner,
        &eos,
        1.0e-6,
        1.0e-8,
    );
    assert_jacobian_close(analytic, fd, 0.08);
}
