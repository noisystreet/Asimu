//! 不可压缩初场散度投影（Rhie-Chow Poisson 迭代，供 V&V 初场使用）。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    IncompressiblePressureCorrectionConfig, apply_rhie_chow_pressure_projection_to_fields,
    assemble_incompressible_pressure_correction_3d, compute_incompressible_rhie_chow_divergence_3d,
    incompressible_pressure_correction_dirichlet,
};
use crate::error::Result;
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::StructuredMesh3d;

use super::diagnostics::max_abs_scalar_field;
use super::linear::{IncompressiblePressureLinearSolverConfig, solve_pressure_correction};
use super::pressure_reference::volume_weighted_pressure_mean;

/// 初场散度投影模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressibleProjectionMode {
    /// 固定速度，迭代调整压力以满足 Rhie-Chow 散度（Taylor–Green 等解析初场）。
    RhieChowPressureOnly,
}

/// 初场散度投影配置。
#[derive(Debug, Clone, Copy)]
pub struct IncompressibleProjectionConfig<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub boundary: &'a BoundarySet,
    pub density: Real,
    pub linear: IncompressiblePressureLinearSolverConfig,
    pub max_iterations: usize,
    pub tolerance: Real,
    pub mode: IncompressibleProjectionMode,
}

impl<'a> IncompressibleProjectionConfig<'a> {
    #[must_use]
    pub const fn rhie_chow_pressure_only(
        mesh: &'a StructuredMesh3d,
        boundary: &'a BoundarySet,
        density: Real,
        linear: IncompressiblePressureLinearSolverConfig,
        max_iterations: usize,
        tolerance: Real,
    ) -> Self {
        Self {
            mesh,
            boundary,
            density,
            linear,
            max_iterations,
            tolerance,
            mode: IncompressibleProjectionMode::RhieChowPressureOnly,
        }
    }
}

/// 散度投影结果统计。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleProjectionStats {
    pub iterations: usize,
    pub max_abs_divergence_before: Real,
    pub max_abs_divergence_after: Real,
    pub pressure_solve_converged: bool,
}

/// 将 collocated 初场投影至 Rhie-Chow 面通量散度低于 `tolerance`。
#[must_use = "散度投影失败须向上传播"]
pub fn project_incompressible_fields_divergence_free_3d(
    fields: IncompressibleFields,
    config: IncompressibleProjectionConfig<'_>,
) -> Result<(IncompressibleFields, IncompressibleProjectionStats)> {
    let d = ScalarField::uniform(config.mesh.num_cells(), 1.0)?;
    project_incompressible_fields_divergence_free_with_d_3d(fields, &d, config)
}

/// 将 collocated 初场在给定 \(d_P\) 下投影至 Rhie-Chow 面通量散度低于 `tolerance`。
#[must_use = "散度投影失败须向上传播"]
pub fn project_incompressible_fields_divergence_free_with_d_3d(
    mut fields: IncompressibleFields,
    d_coefficient: &ScalarField,
    config: IncompressibleProjectionConfig<'_>,
) -> Result<(IncompressibleFields, IncompressibleProjectionStats)> {
    match config.mode {
        IncompressibleProjectionMode::RhieChowPressureOnly => {
            let stats = project_rhie_chow_pressure_only(&mut fields, d_coefficient, config)?;
            Ok((fields, stats))
        }
    }
}

/// 固定速度，迭代调整压力使 Rhie-Chow 面通量散度低于容差；返回 \(p_{\mathrm{old}}-p_{\mathrm{new}}\) 以便并入 PISO 压力更新。
#[must_use = "动量预测后压力对齐失败须向上传播"]
pub fn reconcile_rhie_chow_pressure_with_fixed_velocity_3d(
    fields: &mut IncompressibleFields,
    d_coefficient: &ScalarField,
    config: IncompressibleProjectionConfig<'_>,
) -> Result<Vec<Real>> {
    let before = fields.pressure.values().to_vec();
    project_rhie_chow_pressure_only(fields, d_coefficient, config)?;
    Ok(before
        .iter()
        .zip(fields.pressure.values())
        .map(|(old, new)| old - new)
        .collect())
}

fn project_rhie_chow_pressure_only(
    fields: &mut IncompressibleFields,
    d_coefficient: &ScalarField,
    config: IncompressibleProjectionConfig<'_>,
) -> Result<IncompressibleProjectionStats> {
    let initial_div = compute_incompressible_rhie_chow_divergence_3d(
        config.mesh,
        fields,
        d_coefficient,
        config.boundary,
    )?;
    let max_abs_divergence_before = max_abs_scalar_field(&initial_div);
    if max_abs_divergence_before <= config.tolerance {
        return Ok(IncompressibleProjectionStats {
            iterations: 0,
            max_abs_divergence_before,
            max_abs_divergence_after: max_abs_divergence_before,
            pressure_solve_converged: true,
        });
    }

    let mut pressure_solve_converged = true;
    let mut iterations = 0usize;
    for _ in 0..config.max_iterations {
        let divergence = compute_incompressible_rhie_chow_divergence_3d(
            config.mesh,
            fields,
            d_coefficient,
            config.boundary,
        )?;
        let max_div = max_abs_scalar_field(&divergence);
        if max_div <= config.tolerance {
            break;
        }
        let system = assemble_incompressible_pressure_correction_3d(
            config.mesh,
            &divergence,
            d_coefficient,
            config.boundary,
            IncompressiblePressureCorrectionConfig::new(config.density, 0, 0.0)?,
        )?;
        let mut solution = solve_pressure_correction(&system, config.linear)?;
        pressure_solve_converged &= solution.converged;
        normalize_pressure_correction_mean(&mut solution.correction, config.mesh, config.boundary);
        apply_rhie_chow_pressure_projection_to_fields(
            config.mesh,
            fields,
            d_coefficient,
            &solution.correction,
            config.boundary,
        )?;
        iterations += 1;
    }

    let final_div = compute_incompressible_rhie_chow_divergence_3d(
        config.mesh,
        fields,
        d_coefficient,
        config.boundary,
    )?;
    let max_abs_divergence_after = max_abs_scalar_field(&final_div);
    Ok(IncompressibleProjectionStats {
        iterations,
        max_abs_divergence_before,
        max_abs_divergence_after,
        pressure_solve_converged,
    })
}

fn normalize_pressure_correction_mean(
    pressure_correction: &mut [Real],
    mesh: &StructuredMesh3d,
    boundary: &BoundarySet,
) {
    if has_pressure_correction_dirichlet(boundary) || pressure_correction.is_empty() {
        return;
    }
    let reference = volume_weighted_pressure_mean(pressure_correction, mesh);
    for value in pressure_correction {
        *value -= reference;
    }
}

fn has_pressure_correction_dirichlet(boundary: &BoundarySet) -> bool {
    boundary
        .patches()
        .iter()
        .any(|patch| incompressible_pressure_correction_dirichlet(&patch.kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn uniform_field_projection_is_noop() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 0.1).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::default();
        let (projected, stats) = project_incompressible_fields_divergence_free_3d(
            fields.clone(),
            IncompressibleProjectionConfig::rhie_chow_pressure_only(
                &mesh,
                &boundary,
                1.0,
                IncompressiblePressureLinearSolverConfig::default(),
                4,
                1.0e-8,
            ),
        )
        .expect("project");
        assert_eq!(stats.iterations, 0);
        assert!(approx_eq(
            projected.velocity_x.values()[0],
            fields.velocity_x.values()[0],
            1.0e-12
        ));
    }
}
