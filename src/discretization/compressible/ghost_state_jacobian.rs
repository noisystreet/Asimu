//! 边界面 ghost 守恒状态对 owner 的解析 Jacobian（原语 BC 链式法则）。
//!
//! 变量顺序：守恒 \([\rho,m_x,m_y,m_z,E]\)，原语 \([\rho,u,v,w,p]\)。

use crate::boundary::WallHeat;
use crate::core::{Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
};

pub type StateJacobian = [[Real; 5]; 5];

pub fn multiply_state_jacobian(a: &StateJacobian, b: &StateJacobian) -> StateJacobian {
    let mut out = [[0.0; 5]; 5];
    for row in 0..5 {
        for col in 0..5 {
            out[row][col] = a[row]
                .iter()
                .zip(b.iter())
                .map(|(a_row_k, b_k_col)| a_row_k * b_k_col[col])
                .sum();
        }
    }
    out
}

/// \(\partial \mathbf{U}_g / \partial \mathbf{U}_o\)（绝热/等温 wall 或 symmetry；`HeatFlux` 返回 `Err`）。
pub fn wall_ghost_state_jacobian_wrt_owner(
    owner: &ConservedState,
    normal: Vector3,
    no_slip: bool,
    heat: WallHeat,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<StateJacobian> {
    let prim = crate::field::primitive_from_conserved_relaxed(eos, owner, p_floor)?;
    if prim.pressure < p_floor {
        return Err(AsimuError::Solver(
            "wall ghost Jacobian：压力钳制区需数值路径".to_string(),
        ));
    }
    match heat {
        WallHeat::Adiabatic | WallHeat::Isothermal { .. } => {
            let j_prim = wall_ghost_primitive_jacobian(&prim, normal, no_slip, heat, eos);
            let j_u_to_prim = conserved_to_primitive_jacobian(owner, eos.gamma);
            let j_ghost_prim_to_u = primitive_to_conserved_jacobian(
                &wall_ghost_primitive(&prim, normal, no_slip, heat, eos),
                eos.gamma,
            );
            Ok(multiply_state_jacobian(
                &j_ghost_prim_to_u,
                &multiply_state_jacobian(&j_prim, &j_u_to_prim),
            ))
        }
        WallHeat::HeatFlux { .. } => Err(AsimuError::Solver(
            "热流壁 ghost Jacobian 需数值路径".to_string(),
        )),
    }
}

pub fn symmetry_ghost_state_jacobian_wrt_owner(
    owner: &ConservedState,
    normal: Vector3,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<StateJacobian> {
    wall_ghost_state_jacobian_wrt_owner(owner, normal, false, WallHeat::Adiabatic, eos, p_floor)
}

/// \(\partial \mathbf{U}_g / \partial \mathbf{U}_o\)（远场特征 BC；含 Riemann 混合区）。
pub fn farfield_ghost_state_jacobian_wrt_owner(
    owner: &ConservedState,
    normal: Vector3,
    params: &FreestreamParams,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<StateJacobian> {
    let owner_prim = crate::field::primitive_from_conserved_relaxed(eos, owner, p_floor)?;
    if owner_prim.pressure < p_floor {
        return Err(AsimuError::Solver(
            "farfield ghost Jacobian：压力钳制区需数值路径".to_string(),
        ));
    }
    let farfield = FreestreamContext::new(eos, None, None).primitive(params)?;
    let j_prim = farfield_ghost_primitive_jacobian(&owner_prim, &farfield, normal, eos)?;
    let j_u_to_prim = conserved_to_primitive_jacobian(owner, eos.gamma);
    let ghost_prim = farfield_ghost_primitive(&owner_prim, &farfield, normal, eos);
    let j_ghost_prim_to_u = primitive_to_conserved_jacobian(&ghost_prim, eos.gamma);
    Ok(multiply_state_jacobian(
        &j_ghost_prim_to_u,
        &multiply_state_jacobian(&j_prim, &j_u_to_prim),
    ))
}

#[path = "ghost_state_jacobian_inlet_outlet.rs"]
mod inlet_outlet;
pub use inlet_outlet::{
    InletGhostJacobianParams, inlet_ghost_state_jacobian_wrt_owner,
    outlet_ghost_state_jacobian_wrt_owner,
};

fn wall_ghost_primitive(
    owner: &PrimitiveState,
    normal: Vector3,
    no_slip: bool,
    heat: WallHeat,
    eos: &IdealGasEoS,
) -> PrimitiveState {
    let velocity = wall_ghost_velocity(owner.velocity, normal, no_slip);
    let t_owner = owner.pressure / (owner.density.max(1.0e-30) * eos.gas_constant);
    let t_ghost = match heat {
        WallHeat::Adiabatic => t_owner,
        WallHeat::Isothermal { temperature } => temperature,
        WallHeat::HeatFlux { .. } => t_owner,
    };
    let density = owner.pressure * eos.gamma / t_ghost;
    PrimitiveState {
        density,
        velocity,
        pressure: owner.pressure,
        temperature: t_ghost,
    }
}

fn wall_ghost_velocity(owner: [Real; 3], normal: Vector3, no_slip: bool) -> [Real; 3] {
    let un = dot(owner, normal);
    let tangent = [
        owner[0] - un * normal.x,
        owner[1] - un * normal.y,
        owner[2] - un * normal.z,
    ];
    let un_g = -un;
    if no_slip {
        [
            -tangent[0] + un_g * normal.x,
            -tangent[1] + un_g * normal.y,
            -tangent[2] + un_g * normal.z,
        ]
    } else {
        [
            tangent[0] + un_g * normal.x,
            tangent[1] + un_g * normal.y,
            tangent[2] + un_g * normal.z,
        ]
    }
}

fn wall_ghost_primitive_jacobian(
    owner: &PrimitiveState,
    normal: Vector3,
    no_slip: bool,
    heat: WallHeat,
    eos: &IdealGasEoS,
) -> StateJacobian {
    let t_ghost = match heat {
        WallHeat::Adiabatic => owner.pressure / (owner.density.max(1.0e-30) * eos.gas_constant),
        WallHeat::Isothermal { temperature } => temperature,
        WallHeat::HeatFlux { .. } => {
            owner.pressure / (owner.density.max(1.0e-30) * eos.gas_constant)
        }
    };
    let mut jac = [[0.0; 5]; 5];
    match heat {
        WallHeat::Adiabatic => {
            // \(\rho_g^*=\gamma p^*/T^*\)，\(T^*=p^*/(\rho^* R)\Rightarrow \rho_g^*=\gamma R\rho^*\)。
            jac[0][0] = eos.gamma * eos.gas_constant;
            jac[4][4] = 1.0;
        }
        WallHeat::Isothermal { temperature } => {
            jac[4][4] = 1.0;
            jac[0][4] = eos.gamma / temperature;
        }
        WallHeat::HeatFlux { .. } => {}
    }
    let vel_jac = velocity_mirror_jacobian(normal, no_slip);
    for i in 0..3 {
        for j in 0..3 {
            jac[1 + i][1 + j] = vel_jac[i][j];
        }
    }
    let _ = (owner, t_ghost);
    jac
}

fn velocity_mirror_jacobian(normal: Vector3, no_slip: bool) -> [[Real; 3]; 3] {
    if no_slip {
        return [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }
    let nx = normal.x;
    let ny = normal.y;
    let nz = normal.z;
    [
        [1.0 - 2.0 * nx * nx, -2.0 * nx * ny, -2.0 * nx * nz],
        [-2.0 * ny * nx, 1.0 - 2.0 * ny * ny, -2.0 * ny * nz],
        [-2.0 * nz * nx, -2.0 * nz * ny, 1.0 - 2.0 * nz * nz],
    ]
}

fn farfield_ghost_primitive(
    owner: &PrimitiveState,
    farfield: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> PrimitiveState {
    characteristic_farfield_primitive(owner, farfield, normal, eos).expect("farfield ghost prim")
}

fn farfield_ghost_primitive_jacobian(
    owner: &PrimitiveState,
    farfield: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<StateJacobian> {
    let a_owner = sound_speed(owner, eos);
    let a_far = sound_speed(farfield, eos);
    let un_owner = normal_velocity(owner.velocity, normal);
    let un_far = normal_velocity(farfield.velocity, normal);
    if un_far <= -a_far {
        return Ok([[0.0; 5]; 5]);
    }
    if un_owner >= a_owner {
        return Ok(identity_jacobian());
    }
    farfield_mixed_ghost_primitive_jacobian(owner, farfield, normal, eos)
}

fn farfield_mixed_ghost_primitive_jacobian(
    owner: &PrimitiveState,
    farfield: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<StateJacobian> {
    let gm1 = eos.gamma - 1.0;
    let a_owner = sound_speed(owner, eos);
    let a_far = sound_speed(farfield, eos);
    let un_owner = normal_velocity(owner.velocity, normal);
    let un_far = normal_velocity(farfield.velocity, normal);
    let r_plus = un_owner + 2.0 * a_owner / gm1;
    let r_minus = un_far - 2.0 * a_far / gm1;
    let un_g = 0.5 * (r_plus + r_minus);
    let sound = (0.25 * gm1 * (r_plus - r_minus)).max(1.0e-30);
    let use_farfield_entropy = un_g < 0.0;

    let dr_plus = dr_plus_wrt_primitive(owner, normal, eos);
    let dsound_dr_plus = if r_plus - r_minus > 0.0 {
        0.25 * gm1
    } else {
        0.0
    };
    let dsound = scale_row(dr_plus, dsound_dr_plus);
    let dun_g = scale_row(dr_plus, 0.5);

    let (entropy, dentropy, tangent_jac) = if use_farfield_entropy {
        (
            entropy_constant(farfield, eos.gamma),
            [0.0; 5],
            [[0.0; 5]; 3],
        )
    } else {
        (
            entropy_constant(owner, eos.gamma),
            entropy_gradient(owner, eos.gamma),
            tangential_velocity_jacobian(normal),
        )
    };

    let mut dvelocity = [[0.0; 5]; 3];
    for (i, row) in dvelocity.iter_mut().enumerate() {
        for j in 0..5 {
            row[j] = normal_component(i, normal) * dun_g[j];
        }
    }
    for (i, row) in dvelocity.iter_mut().enumerate() {
        for j in 0..5 {
            row[j] += tangent_jac[i][j];
        }
    }

    prim_from_sound_entropy_jacobian(sound, entropy, eos, dsound, dentropy, dvelocity)
}

fn prim_from_sound_entropy_jacobian(
    sound: Real,
    entropy: Real,
    eos: &IdealGasEoS,
    dsound: [Real; 5],
    dentropy: [Real; 5],
    dvelocity: [[Real; 5]; 3],
) -> Result<StateJacobian> {
    let gm1 = eos.gamma - 1.0;
    let inv_gm1 = 1.0 / gm1;
    let density = (sound * sound / (eos.gamma * entropy)).powf(inv_gm1);
    let pressure = entropy * density.powf(eos.gamma);
    let mut jac = [[0.0; 5]; 5];

    let drho_dsound = density * inv_gm1 * 2.0 / sound;
    let drho_dentropy = -density * inv_gm1 / entropy;
    let dp_dsound = pressure * eos.gamma / density * drho_dsound;
    let dp_dentropy = density.powf(eos.gamma) + pressure * eos.gamma / density * drho_dentropy;

    for col in 0..5 {
        jac[0][col] = drho_dsound * dsound[col] + drho_dentropy * dentropy[col];
        jac[4][col] = dp_dsound * dsound[col] + dp_dentropy * dentropy[col];
        for i in 0..3 {
            jac[1 + i][col] = dvelocity[i][col];
        }
    }
    Ok(jac)
}

pub(super) fn dr_plus_wrt_primitive(
    owner: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> [Real; 5] {
    let gm1 = eos.gamma - 1.0;
    let rho = owner.density;
    let p = owner.pressure;
    let a = sound_speed(owner, eos);
    let inv_rho_a = 1.0 / (rho * a);
    let da_drho = -0.5 * eos.gamma * p * inv_rho_a / rho;
    let da_dp = 0.5 * eos.gamma * inv_rho_a;
    let mut row = [0.0; 5];
    row[0] = 2.0 / gm1 * da_drho;
    row[1] = normal.x;
    row[2] = normal.y;
    row[3] = normal.z;
    row[4] = 2.0 / gm1 * da_dp;
    row
}

pub(super) fn entropy_gradient(owner: &PrimitiveState, gamma: Real) -> [Real; 5] {
    let rho = owner.density.max(1.0e-30);
    let mut grad = [0.0; 5];
    grad[0] = -gamma * owner.pressure * rho.powf(-gamma - 1.0);
    grad[4] = rho.powf(-gamma);
    grad
}

pub(super) fn tangential_velocity_jacobian(normal: Vector3) -> [[Real; 5]; 3] {
    let n = [normal.x, normal.y, normal.z];
    let mut jac = [[0.0; 5]; 3];
    for i in 0..3 {
        for j in 0..3 {
            jac[i][1 + j] = if i == j { 1.0 } else { 0.0 } - n[i] * n[j];
        }
    }
    jac
}

pub(super) fn scale_row(row: [Real; 5], scale: Real) -> [Real; 5] {
    row.map(|v| v * scale)
}

pub(super) fn normal_component(index: usize, normal: Vector3) -> Real {
    match index {
        0 => normal.x,
        1 => normal.y,
        _ => normal.z,
    }
}

pub fn conserved_to_primitive_jacobian(cons: &ConservedState, gamma: Real) -> StateJacobian {
    let rho = cons.density.max(1.0e-30);
    let mx = cons.momentum[0];
    let my = cons.momentum[1];
    let mz = cons.momentum[2];
    let inv_rho = 1.0 / rho;
    let ke = 0.5 * (mx * mx + my * my + mz * mz) * inv_rho * inv_rho;
    let gm1 = gamma - 1.0;
    let mut jac = [[0.0; 5]; 5];
    jac[0][0] = 1.0;
    jac[1][0] = -mx * inv_rho * inv_rho;
    jac[1][1] = inv_rho;
    jac[2][0] = -my * inv_rho * inv_rho;
    jac[2][2] = inv_rho;
    jac[3][0] = -mz * inv_rho * inv_rho;
    jac[3][3] = inv_rho;
    jac[4][0] = gm1 * ke / rho;
    jac[4][1] = -gm1 * mx * inv_rho;
    jac[4][2] = -gm1 * my * inv_rho;
    jac[4][3] = -gm1 * mz * inv_rho;
    jac[4][4] = gm1;
    jac
}

pub fn primitive_to_conserved_jacobian(prim: &PrimitiveState, gamma: Real) -> StateJacobian {
    let rho = prim.density;
    let u = prim.velocity[0];
    let v = prim.velocity[1];
    let w = prim.velocity[2];
    let gm1 = gamma - 1.0;
    let mut jac = [[0.0; 5]; 5];
    jac[0][0] = 1.0;
    jac[1][0] = u;
    jac[1][1] = rho;
    jac[2][0] = v;
    jac[2][2] = rho;
    jac[3][0] = w;
    jac[3][3] = rho;
    jac[4][0] = 0.5 * (u * u + v * v + w * w);
    jac[4][1] = rho * u;
    jac[4][2] = rho * v;
    jac[4][3] = rho * w;
    jac[4][4] = 1.0 / gm1;
    jac
}

pub(super) fn identity_jacobian() -> StateJacobian {
    let mut g = [[0.0; 5]; 5];
    for (i, row) in g.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    g
}

fn characteristic_farfield_primitive(
    owner: &PrimitiveState,
    farfield: &PrimitiveState,
    normal: Vector3,
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let a_owner = sound_speed(owner, eos);
    let a_farfield = sound_speed(farfield, eos);
    let un_owner = normal_velocity(owner.velocity, normal);
    let un_farfield = normal_velocity(farfield.velocity, normal);
    if un_farfield <= -a_farfield {
        return Ok(*farfield);
    }
    if un_owner >= a_owner {
        return Ok(*owner);
    }
    let gm1 = eos.gamma - 1.0;
    let r_plus = un_owner + 2.0 * a_owner / gm1;
    let r_minus = un_farfield - 2.0 * a_farfield / gm1;
    let un = 0.5 * (r_plus + r_minus);
    let sound = (0.25 * gm1 * (r_plus - r_minus)).max(1.0e-30);
    let use_farfield_entropy = un < 0.0;
    let entropy_source = if use_farfield_entropy {
        farfield
    } else {
        owner
    };
    let velocity_source = if use_farfield_entropy {
        farfield
    } else {
        owner
    };
    let entropy = entropy_constant(entropy_source, eos.gamma);
    let tangent = tangential_velocity(velocity_source.velocity, normal);
    primitive_from_sound_entropy_velocity(
        sound,
        entropy,
        velocity_from_normal_tangent(un, tangent, normal),
        eos,
    )
}

fn primitive_from_sound_entropy_velocity(
    sound: Real,
    entropy: Real,
    velocity: [Real; 3],
    eos: &IdealGasEoS,
) -> Result<PrimitiveState> {
    let density = (sound * sound / (eos.gamma * entropy)).powf(1.0 / (eos.gamma - 1.0));
    let pressure = entropy * density.powf(eos.gamma);
    Ok(PrimitiveState {
        density,
        velocity,
        pressure,
        temperature: pressure / (density * eos.gas_constant),
    })
}

pub(super) fn sound_speed(prim: &PrimitiveState, eos: &IdealGasEoS) -> Real {
    (eos.gamma * prim.pressure / prim.density).sqrt()
}

pub(super) fn entropy_constant(prim: &PrimitiveState, gamma: Real) -> Real {
    prim.pressure / prim.density.powf(gamma)
}

pub(super) fn normal_velocity(velocity: [Real; 3], normal: Vector3) -> Real {
    dot(velocity, normal)
}

pub(super) fn tangential_velocity(velocity: [Real; 3], normal: Vector3) -> [Real; 3] {
    let un = normal_velocity(velocity, normal);
    [
        velocity[0] - un * normal.x,
        velocity[1] - un * normal.y,
        velocity[2] - un * normal.z,
    ]
}

pub(super) fn velocity_from_normal_tangent(
    un: Real,
    tangent: [Real; 3],
    normal: Vector3,
) -> [Real; 3] {
    [
        tangent[0] + un * normal.x,
        tangent[1] + un * normal.y,
        tangent[2] + un * normal.z,
    ]
}

pub(super) fn dot(v: [Real; 3], normal: Vector3) -> Real {
    v[0] * normal.x + v[1] * normal.y + v[2] * normal.z
}

#[cfg(test)]
#[path = "ghost_state_jacobian_tests.rs"]
mod tests;
