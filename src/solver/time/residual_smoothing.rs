//! 方向分裂隐式残差光顺。
//!
//! 用于稳态伪时间推进的 RHS 预处理，不改变最终稳态方程。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, ScalarField};
use crate::mesh::StructuredMesh3d;
use crate::physics::{ConservedState, IdealGasEoS};

/// 方向分裂隐式残差光顺配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualSmoothingConfig {
    pub enabled: bool,
    pub epsilon: Real,
    pub sweeps: usize,
}

impl ResidualSmoothingConfig {
    pub const DEFAULT_EPSILON: Real = 0.5;

    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            epsilon: Self::DEFAULT_EPSILON,
            sweeps: 1,
        }
    }

    pub fn parse(enabled: bool, epsilon: Option<Real>, sweeps: Option<usize>) -> Result<Self> {
        let epsilon = epsilon.unwrap_or(Self::DEFAULT_EPSILON);
        let sweeps = sweeps.unwrap_or(1);
        if enabled && (!epsilon.is_finite() || epsilon < 0.0 || sweeps == 0) {
            return Err(AsimuError::Config(
                "residual_smoothing 参数无效：epsilon 须非负且 sweeps 须大于 0".to_string(),
            ));
        }
        Ok(Self {
            enabled,
            epsilon,
            sweeps,
        })
    }
}

impl Default for ResidualSmoothingConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// 对 3D 结构网格残差执行 i→j→k 方向分裂隐式光顺。
pub fn smooth_residual_3d(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    config: ResidualSmoothingConfig,
) -> Result<()> {
    if !config.enabled || config.epsilon <= 0.0 {
        return Ok(());
    }
    if residual.num_cells() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "残差尺寸 {} 与网格单元数 {} 不一致",
            residual.num_cells(),
            mesh.num_cells()
        )));
    }
    for _ in 0..config.sweeps {
        smooth_field_all_directions(&mut residual.density, mesh, config.epsilon);
        smooth_field_all_directions(&mut residual.momentum_x, mesh, config.epsilon);
        smooth_field_all_directions(&mut residual.momentum_y, mesh, config.epsilon);
        smooth_field_all_directions(&mut residual.momentum_z, mesh, config.epsilon);
        smooth_field_all_directions(&mut residual.total_energy, mesh, config.epsilon);
    }
    Ok(())
}

/// 光顺残差并按单元限制更新方向，避免光顺后的 RHS 破坏密度/内能正性。
pub fn smooth_residual_3d_limited(
    residual: &mut ConservedResidual,
    base: &ConservedFields,
    update_scales: &[Real],
    mesh: &StructuredMesh3d,
    eos: &IdealGasEoS,
    min_pressure: Real,
    config: ResidualSmoothingConfig,
) -> Result<()> {
    if !config.enabled || config.epsilon <= 0.0 {
        return Ok(());
    }
    if base.num_cells() != mesh.num_cells() || update_scales.len() != mesh.num_cells() {
        return Err(AsimuError::Field(
            "残差光顺正性限制的场/步长尺寸与网格不一致".to_string(),
        ));
    }
    let original = residual.clone();
    smooth_residual_3d(residual, mesh, config)?;
    for (cell, &update_scale) in update_scales.iter().enumerate().take(mesh.num_cells()) {
        limit_cell_residual(
            residual,
            &original,
            base,
            cell,
            update_scale,
            eos.gamma,
            min_pressure,
        )?;
    }
    Ok(())
}

fn smooth_field_all_directions(field: &mut ScalarField, mesh: &StructuredMesh3d, epsilon: Real) {
    smooth_i(field.values_mut(), mesh, epsilon);
    smooth_j(field.values_mut(), mesh, epsilon);
    smooth_k(field.values_mut(), mesh, epsilon);
}

fn smooth_i(values: &mut [Real], mesh: &StructuredMesh3d, epsilon: Real) {
    let mut line = vec![0.0; mesh.nx];
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for (i, value) in line.iter_mut().enumerate() {
                *value = values[mesh.cell_index(i, j, k)];
            }
            solve_zero_gradient_line(&mut line, epsilon);
            for (i, &value) in line.iter().enumerate() {
                values[mesh.cell_index(i, j, k)] = value;
            }
        }
    }
}

fn smooth_j(values: &mut [Real], mesh: &StructuredMesh3d, epsilon: Real) {
    let mut line = vec![0.0; mesh.ny];
    for k in 0..mesh.nz {
        for i in 0..mesh.nx {
            for (j, value) in line.iter_mut().enumerate() {
                *value = values[mesh.cell_index(i, j, k)];
            }
            solve_zero_gradient_line(&mut line, epsilon);
            for (j, &value) in line.iter().enumerate() {
                values[mesh.cell_index(i, j, k)] = value;
            }
        }
    }
}

fn smooth_k(values: &mut [Real], mesh: &StructuredMesh3d, epsilon: Real) {
    let mut line = vec![0.0; mesh.nz];
    for j in 0..mesh.ny {
        for i in 0..mesh.nx {
            for (k, value) in line.iter_mut().enumerate() {
                *value = values[mesh.cell_index(i, j, k)];
            }
            solve_zero_gradient_line(&mut line, epsilon);
            for (k, &value) in line.iter().enumerate() {
                values[mesh.cell_index(i, j, k)] = value;
            }
        }
    }
}

fn solve_zero_gradient_line(rhs: &mut [Real], epsilon: Real) {
    let n = rhs.len();
    if n <= 1 || epsilon <= 0.0 {
        return;
    }
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let first_diag = 1.0 + epsilon;
    c_prime[0] = -epsilon / first_diag;
    d_prime[0] = rhs[0] / first_diag;

    for i in 1..n {
        let diag = if i + 1 == n {
            1.0 + epsilon
        } else {
            1.0 + 2.0 * epsilon
        };
        let denom = diag + epsilon * c_prime[i - 1];
        c_prime[i] = if i + 1 == n { 0.0 } else { -epsilon / denom };
        d_prime[i] = (rhs[i] + epsilon * d_prime[i - 1]) / denom;
    }

    rhs[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = d_prime[i] - c_prime[i] * rhs[i + 1];
    }
}

fn limit_cell_residual(
    residual: &mut ConservedResidual,
    original: &ConservedResidual,
    base: &ConservedFields,
    cell: usize,
    scale: Real,
    gamma: Real,
    min_pressure: Real,
) -> Result<()> {
    if scale <= 0.0 {
        return Ok(());
    }
    let base_state = base.cell_state(cell)?;
    let smooth = residual_cell(residual, cell);
    if updated_state_is_physical(&base_state, smooth, scale, gamma, min_pressure) {
        return Ok(());
    }
    let raw = residual_cell(original, cell);
    if let Some(blended) = find_positive_blend(&base_state, raw, smooth, scale, gamma, min_pressure)
    {
        write_residual_cell(residual, cell, blended);
        return Ok(());
    }
    if let Some(limited_raw) = find_positive_scale(&base_state, raw, scale, gamma, min_pressure) {
        write_residual_cell(residual, cell, limited_raw);
        return Ok(());
    }
    write_residual_cell(residual, cell, [0.0; 5]);
    Ok(())
}

fn find_positive_blend(
    base: &ConservedState,
    raw: [Real; 5],
    smooth: [Real; 5],
    scale: Real,
    gamma: Real,
    min_pressure: Real,
) -> Option<[Real; 5]> {
    let mut alpha = 1.0;
    for _ in 0..12 {
        let candidate = blend_residual(raw, smooth, alpha);
        if updated_state_is_physical(base, candidate, scale, gamma, min_pressure) {
            return Some(candidate);
        }
        alpha *= 0.5;
    }
    if updated_state_is_physical(base, raw, scale, gamma, min_pressure) {
        Some(raw)
    } else {
        None
    }
}

fn find_positive_scale(
    base: &ConservedState,
    residual: [Real; 5],
    scale: Real,
    gamma: Real,
    min_pressure: Real,
) -> Option<[Real; 5]> {
    let mut alpha = 0.5;
    for _ in 0..12 {
        let candidate = residual.map(|value| alpha * value);
        if updated_state_is_physical(base, candidate, scale, gamma, min_pressure) {
            return Some(candidate);
        }
        alpha *= 0.5;
    }
    None
}

fn updated_state_is_physical(
    base: &ConservedState,
    residual: [Real; 5],
    scale: Real,
    gamma: Real,
    min_pressure: Real,
) -> bool {
    let rho = base.density + scale * residual[0];
    let momentum = [
        base.momentum[0] + scale * residual[1],
        base.momentum[1] + scale * residual[2],
        base.momentum[2] + scale * residual[3],
    ];
    let total_energy = base.total_energy + scale * residual[4];
    if rho <= 0.0 || !rho.is_finite() || !total_energy.is_finite() {
        return false;
    }
    let ke = 0.5
        * (momentum[0] * momentum[0] + momentum[1] * momentum[1] + momentum[2] * momentum[2])
        / rho;
    let min_internal = min_pressure.max(0.0) / (gamma - 1.0);
    let internal = total_energy - ke;
    internal.is_finite() && internal > min_internal
}

fn blend_residual(raw: [Real; 5], smooth: [Real; 5], alpha: Real) -> [Real; 5] {
    [
        raw[0] + alpha * (smooth[0] - raw[0]),
        raw[1] + alpha * (smooth[1] - raw[1]),
        raw[2] + alpha * (smooth[2] - raw[2]),
        raw[3] + alpha * (smooth[3] - raw[3]),
        raw[4] + alpha * (smooth[4] - raw[4]),
    ]
}

fn residual_cell(residual: &ConservedResidual, cell: usize) -> [Real; 5] {
    [
        residual.density.values()[cell],
        residual.momentum_x.values()[cell],
        residual.momentum_y.values()[cell],
        residual.momentum_z.values()[cell],
        residual.total_energy.values()[cell],
    ]
}

fn write_residual_cell(residual: &mut ConservedResidual, cell: usize, values: [Real; 5]) {
    residual.density.values_mut()[cell] = values[0];
    residual.momentum_x.values_mut()[cell] = values[1];
    residual.momentum_y.values_mut()[cell] = values[2];
    residual.momentum_z.values_mut()[cell] = values[3];
    residual.total_energy.values_mut()[cell] = values[4];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::StructuredMesh3d;
    use crate::physics::{ConservedState, IdealGasEoS};

    #[test]
    fn constant_residual_is_preserved() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        for value in residual.density.values_mut() {
            *value = 2.0;
        }
        smooth_residual_3d(
            &mut residual,
            &mesh,
            ResidualSmoothingConfig {
                enabled: true,
                epsilon: 0.5,
                sweeps: 2,
            },
        )
        .expect("smooth");
        assert!(
            residual
                .density
                .values()
                .iter()
                .all(|&value| (value - 2.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn impulse_is_smoothed_along_all_directions() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 3, 3, 1.0, 1.0, 1.0).expect("mesh");
        let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let center = mesh.cell_index(1, 1, 1);
        residual.density.values_mut()[center] = 1.0;
        smooth_residual_3d(
            &mut residual,
            &mesh,
            ResidualSmoothingConfig {
                enabled: true,
                epsilon: 0.5,
                sweeps: 1,
            },
        )
        .expect("smooth");
        assert!(residual.density.values()[center] < 1.0);
        assert!(residual.density.values()[mesh.cell_index(0, 1, 1)] > 0.0);
        assert!(residual.density.values()[mesh.cell_index(1, 0, 1)] > 0.0);
        assert!(residual.density.values()[mesh.cell_index(1, 1, 0)] > 0.0);
    }

    #[test]
    fn limiter_rejects_smoothed_residual_that_breaks_internal_energy() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let base_state = ConservedState {
            density: 1.0,
            momentum: [1.0, 0.0, 0.0],
            total_energy: 1.0,
        };
        let base = ConservedFields::uniform(mesh.num_cells(), base_state).expect("base");
        let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let center = mesh.cell_index(1, 0, 0);
        residual.momentum_x.values_mut()[center] = 10.0;
        smooth_residual_3d_limited(
            &mut residual,
            &base,
            &[1.0; 3],
            &mesh,
            &IdealGasEoS::AIR_STANDARD,
            0.0,
            ResidualSmoothingConfig {
                enabled: true,
                epsilon: 0.5,
                sweeps: 1,
            },
        )
        .expect("smooth");
        let updated = [
            base_state.density + residual.density.values()[center],
            base_state.momentum[0] + residual.momentum_x.values()[center],
            base_state.total_energy + residual.total_energy.values()[center],
        ];
        let internal = updated[2] - 0.5 * updated[1] * updated[1] / updated[0];
        assert!(internal > 0.0);
    }
}
