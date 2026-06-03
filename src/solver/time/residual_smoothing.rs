//! 方向分裂隐式残差光顺。
//!
//! 用于稳态伪时间推进的 RHS 预处理，不改变最终稳态方程。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, ScalarField};
use crate::mesh::StructuredMesh3d;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::StructuredMesh3d;

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
}
