//! Hanel–Van Leer 面通量解析 Jacobian（面坐标链式法则）。

use crate::core::{Real, Vector3};
use crate::error::Result;
use crate::physics::{ConservedState, IdealGasEoS};

use crate::discretization::flux_common::{face_tangent_basis, normalize_face_normal};

use super::van_leer::{
    FaceFrameState, face_frame_from_conserved, sound_speed, specific_enthalpy, validate_face_state,
};

/// FVS 能量分裂（供解析 Jacobian 与通量共用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum FvsEnergySplit {
    VanLeer,
    Hanel,
}

type FluxJac5 = [[Real; 5]; 5];

/// Hanel / Van Leer 一阶面通量对左右守恒变量的解析 Jacobian。
pub(crate) fn fvs_flux_jacobian_wrt_conserved(
    left: &ConservedState,
    right: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    energy_split: FvsEnergySplit,
) -> Result<(
    super::face_flux_jacobian::ConservedFluxJacobian,
    super::face_flux_jacobian::ConservedFluxJacobian,
)> {
    use super::face_flux_jacobian::ConservedFluxJacobian;
    let n = normalize_face_normal(normal)?;
    let (t1, t2) = face_tangent_basis(n);
    let frame_l = face_frame_from_conserved(left, eos.gamma, n, t1, t2)?;
    let frame_r = face_frame_from_conserved(right, eos.gamma, n, t1, t2)?;
    validate_face_state(&frame_l)?;
    validate_face_state(&frame_r)?;
    let plus_l = fvs_positive_face_flux_jacobian(&frame_l, eos.gamma, energy_split);
    let plus_r = fvs_positive_face_flux_jacobian(&frame_r, eos.gamma, energy_split);
    let phys_l = physical_face_flux_jacobian(&frame_l, eos.gamma);
    let phys_r = physical_face_flux_jacobian(&frame_r, eos.gamma);
    let minus_r = subtract_jacobian(phys_r, plus_r);
    let _minus_l = subtract_jacobian(phys_l, plus_l);
    let state_l = face_state_jacobian_wrt_conserved(left, eos.gamma, n, t1, t2);
    let state_r = face_state_jacobian_wrt_conserved(right, eos.gamma, n, t1, t2);
    let rot = global_flux_jacobian_from_face(n, t1, t2);
    Ok((
        ConservedFluxJacobian {
            data: multiply_jacobian_chain(rot, multiply_jacobian_chain(plus_l, state_l)),
        },
        ConservedFluxJacobian {
            data: multiply_jacobian_chain(rot, multiply_jacobian_chain(minus_r, state_r)),
        },
    ))
}

fn subtract_jacobian(a: FluxJac5, b: FluxJac5) -> FluxJac5 {
    let mut out = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn multiply_jacobian_chain(a: FluxJac5, b: FluxJac5) -> FluxJac5 {
    let mut out = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            let mut sum = 0.0;
            for k in 0..5 {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }
    out
}

fn global_flux_jacobian_from_face(normal: Vector3, t1: Vector3, t2: Vector3) -> FluxJac5 {
    let mut jac = [[0.0; 5]; 5];
    jac[0][0] = 1.0;
    jac[1][1] = normal.x;
    jac[1][2] = t1.x;
    jac[1][3] = t2.x;
    jac[2][1] = normal.y;
    jac[2][2] = t1.y;
    jac[2][3] = t2.y;
    jac[3][1] = normal.z;
    jac[3][2] = t1.z;
    jac[3][3] = t2.z;
    jac[4][4] = 1.0;
    jac
}

fn face_state_jacobian_wrt_conserved(
    cons: &ConservedState,
    gamma: Real,
    normal: Vector3,
    t1: Vector3,
    t2: Vector3,
) -> FluxJac5 {
    let rho = cons.density;
    let mx = cons.momentum[0];
    let my = cons.momentum[1];
    let mz = cons.momentum[2];
    let inv_rho = 1.0 / rho;
    let ke = 0.5 * (mx * mx + my * my + mz * mz) * inv_rho;
    let gm1 = gamma - 1.0;
    let nx = normal.x;
    let ny = normal.y;
    let nz = normal.z;
    let t1x = t1.x;
    let t1y = t1.y;
    let t1z = t1.z;
    let t2x = t2.x;
    let t2y = t2.y;
    let t2z = t2.z;

    let dun_drho = -(mx * nx + my * ny + mz * nz) * inv_rho * inv_rho;
    let dut1_drho = -(mx * t1x + my * t1y + mz * t1z) * inv_rho * inv_rho;
    let dut2_drho = -(mx * t2x + my * t2y + mz * t2z) * inv_rho * inv_rho;
    let dp_drho = gm1 * ke * inv_rho;

    let mut jac = [[0.0; 5]; 5];
    jac[0][0] = 1.0;
    jac[1][0] = dun_drho;
    jac[1][1] = nx * inv_rho;
    jac[1][2] = ny * inv_rho;
    jac[1][3] = nz * inv_rho;
    jac[2][0] = dut1_drho;
    jac[2][1] = t1x * inv_rho;
    jac[2][2] = t1y * inv_rho;
    jac[2][3] = t1z * inv_rho;
    jac[3][0] = dut2_drho;
    jac[3][1] = t2x * inv_rho;
    jac[3][2] = t2y * inv_rho;
    jac[3][3] = t2z * inv_rho;
    jac[4][0] = dp_drho;
    jac[4][4] = gm1;
    jac[4][1] = -gm1 * mx * inv_rho;
    jac[4][2] = -gm1 * my * inv_rho;
    jac[4][3] = -gm1 * mz * inv_rho;
    jac
}

fn physical_face_flux_jacobian(state: &FaceFrameState, gamma: Real) -> FluxJac5 {
    let rho = state.rho;
    let un = state.un;
    let ut1 = state.ut[0];
    let ut2 = state.ut[1];
    let p = state.p;
    let gm1 = gamma - 1.0;
    let vel2 = un * un + ut1 * ut1 + ut2 * ut2;
    let ke = 0.5 * rho * vel2;
    let rho_e = ke + p / gm1;
    let mut jac = [[0.0; 5]; 5];
    jac[0][0] = un;
    jac[0][1] = rho;
    jac[1][0] = un * un;
    jac[1][1] = 2.0 * rho * un;
    jac[1][4] = 1.0;
    jac[2][0] = un * ut1;
    jac[2][1] = rho * ut1;
    jac[2][2] = rho * un;
    jac[3][0] = un * ut2;
    jac[3][1] = rho * ut2;
    jac[3][3] = rho * un;
    let drho_e_drho = 0.5 * vel2;
    let drho_e_dun = rho * un;
    let drho_e_dut1 = rho * ut1;
    let drho_e_dut2 = rho * ut2;
    let drho_e_dp = 1.0 / gm1;
    jac[4][0] = un * drho_e_drho;
    jac[4][1] = (rho_e + p) + un * drho_e_dun;
    jac[4][2] = un * drho_e_dut1;
    jac[4][3] = un * drho_e_dut2;
    jac[4][4] = un * (1.0 + drho_e_dp);
    jac
}

fn fvs_positive_face_flux_jacobian(
    state: &FaceFrameState,
    gamma: Real,
    energy_split: FvsEnergySplit,
) -> FluxJac5 {
    let a = sound_speed(state.rho, state.p, gamma);
    let mach = state.un / a;
    if mach <= -1.0 {
        return [[0.0; 5]; 5];
    }
    if mach >= 1.0 {
        return physical_face_flux_jacobian(state, gamma);
    }
    fvs_positive_subsonic_face_flux_jacobian(state, gamma, energy_split, a, mach)
}

fn fvs_positive_subsonic_face_flux_jacobian(
    state: &FaceFrameState,
    gamma: Real,
    energy_split: FvsEnergySplit,
    a: Real,
    mach: Real,
) -> FluxJac5 {
    let rho = state.rho;
    let un = state.un;
    let ut1 = state.ut[0];
    let ut2 = state.ut[1];
    let mach_plus = mach + 1.0;
    let mass = 0.25 * rho * a * mach_plus * mach_plus;
    let da_drho = -0.5 * a / rho;
    let da_dp = 0.5 * gamma / (rho * a);
    let dm_drho = 0.25 * mach_plus * mach_plus * (a + rho * da_drho)
        - 0.5 * rho * mach_plus * un * da_drho / a;
    let dm_dun = 0.5 * rho * mach_plus;
    let dm_dp = 0.25 * rho * da_dp * mach_plus * mach_plus;
    let normal_velocity = ((gamma - 1.0) * un + 2.0 * a) / gamma;
    let dnv_dun = (gamma - 1.0) / gamma;
    let dnv_da = 2.0 / gamma;
    let dnv_drho = dnv_da * da_drho;
    let dnv_dp = dnv_da * da_dp;
    let tangential_ke = 0.5 * (ut1 * ut1 + ut2 * ut2);
    let mut jac = [[0.0; 5]; 5];
    jac[0][0] = dm_drho;
    jac[0][1] = dm_dun;
    jac[0][4] = dm_dp;
    jac[1][0] = dm_drho * normal_velocity + mass * dnv_drho;
    jac[1][1] = dm_dun * normal_velocity + mass * dnv_dun;
    jac[1][4] = dm_dp * normal_velocity + mass * dnv_dp;
    jac[2][0] = dm_drho * ut1;
    jac[2][1] = dm_dun * ut1;
    jac[2][2] = mass;
    jac[2][4] = dm_dp * ut1;
    jac[3][0] = dm_drho * ut2;
    jac[3][1] = dm_dun * ut2;
    jac[3][3] = mass;
    jac[3][4] = dm_dp * ut2;
    match energy_split {
        FvsEnergySplit::VanLeer => {
            let acoustic = ((gamma - 1.0) * un + 2.0 * a).powi(2) / (2.0 * (gamma * gamma - 1.0));
            let d_acoustic_dun =
                (gamma - 1.0) * ((gamma - 1.0) * un + 2.0 * a) / (gamma * gamma - 1.0);
            let d_acoustic_da =
                2.0 * ((gamma - 1.0) * un + 2.0 * a) * 2.0 / (2.0 * (gamma * gamma - 1.0));
            let d_acoustic_drho = d_acoustic_da * da_drho;
            let d_acoustic_dp = d_acoustic_da * da_dp;
            jac[4][0] = dm_drho * (acoustic + tangential_ke) + mass * d_acoustic_drho;
            jac[4][1] = dm_dun * (acoustic + tangential_ke) + mass * d_acoustic_dun;
            jac[4][4] = dm_dp * (acoustic + tangential_ke) + mass * d_acoustic_dp;
        }
        FvsEnergySplit::Hanel => {
            let h = specific_enthalpy(state, gamma);
            let dh = specific_enthalpy_jacobian(state, gamma, a, da_drho, da_dp);
            jac[4][0] = dm_drho * h + mass * dh[0];
            jac[4][1] = dm_dun * h + mass * dh[1];
            jac[4][2] = mass * dh[2];
            jac[4][3] = mass * dh[3];
            jac[4][4] = dm_dp * h + mass * dh[4];
        }
    }
    jac
}

fn specific_enthalpy_jacobian(
    state: &FaceFrameState,
    gamma: Real,
    a: Real,
    da_drho: Real,
    da_dp: Real,
) -> [Real; 5] {
    let gm1 = gamma - 1.0;
    let mut dh = [0.0; 5];
    dh[0] = da_drho * 2.0 * a / gm1;
    dh[1] = state.un;
    dh[2] = state.ut[0];
    dh[3] = state.ut[1];
    dh[4] = da_dp * 2.0 * a / gm1;
    dh
}
