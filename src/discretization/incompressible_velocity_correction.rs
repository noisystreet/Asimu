//! 压力校正后的不可压缩场更新。
//!
//! 显式 face flux `phi` 是 pressure-velocity coupling 的守恒状态。压力校正阶段直接
//! 修正 `phi` 并欠松弛更新 cell-centered 压力；cell-centered 速度保留动量预测解，
//! 下一轮动量方程再通过更新后的压力梯度响应压力校正，避免把封闭腔体的大 \(p'\)
//! 直接反投影成过大的 cell 速度。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::StructuredMesh3d;

/// Rhie-Chow 速度重构参数。
pub struct RhieChowVelocityCorrectionConfig<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub current: &'a IncompressibleFields,
    pub predicted: &'a IncompressibleFields,
    pub pressure_correction: &'a [Real],
    pub d_coefficient: &'a [Real],
    pub pressure_under_relaxation: Real,
    pub boundary: &'a BoundarySet,
    pub periodic_x: bool,
}

/// 由 \(u^*\) 与压力校正更新 cell-centered 场。
pub fn corrected_incompressible_fields_rhie_chow_3d(
    config: RhieChowVelocityCorrectionConfig<'_>,
) -> Result<IncompressibleFields> {
    let RhieChowVelocityCorrectionConfig {
        mesh,
        current,
        predicted,
        pressure_correction,
        d_coefficient,
        pressure_under_relaxation,
        boundary: _boundary,
        periodic_x: _periodic_x,
    } = config;
    let n = mesh.num_cells();
    if pressure_correction.len() != n || d_coefficient.len() != n {
        return Err(AsimuError::Field(
            "Rhie-Chow 速度重构长度与网格单元数不一致".to_string(),
        ));
    }
    predicted.validate_len(n)?;
    current.validate_len(n)?;
    let pressure = build_updated_pressure(current, pressure_correction, pressure_under_relaxation)?;
    Ok(IncompressibleFields {
        pressure,
        velocity_x: predicted.velocity_x.clone(),
        velocity_y: predicted.velocity_y.clone(),
        velocity_z: predicted.velocity_z.clone(),
    })
}

fn build_updated_pressure(
    current: &IncompressibleFields,
    pressure_correction: &[Real],
    pressure_under_relaxation: Real,
) -> Result<ScalarField> {
    let values = current
        .pressure
        .values()
        .iter()
        .zip(pressure_correction.iter())
        .map(|(value, correction)| value + pressure_under_relaxation * correction)
        .collect::<Vec<_>>();
    ScalarField::from_values(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn uniform_pressure_preserves_predicted_velocity() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 1.0).expect("mesh");
        let current =
            IncompressibleFields::uniform(mesh.num_cells(), 1.0, [0.1, 0.2, 0.0]).expect("cur");
        let predicted =
            IncompressibleFields::uniform(mesh.num_cells(), 1.0, [0.3, 0.4, 0.0]).expect("pred");
        let d = vec![0.01; mesh.num_cells()];
        let p_corr = vec![0.0; mesh.num_cells()];
        let boundary = BoundarySet::default();

        let corrected =
            corrected_incompressible_fields_rhie_chow_3d(RhieChowVelocityCorrectionConfig {
                mesh: &mesh,
                current: &current,
                predicted: &predicted,
                pressure_correction: &p_corr,
                d_coefficient: &d,
                pressure_under_relaxation: 0.01,
                boundary: &boundary,
                periodic_x: false,
            })
            .expect("corrected");

        assert!(approx_eq(corrected.velocity_x.values()[0], 0.3, 1.0e-12));
        assert!(approx_eq(corrected.velocity_y.values()[0], 0.4, 1.0e-12));
    }

    #[test]
    fn pressure_correction_keeps_predicted_velocity_field() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 1, 1, 3.0, 1.0, 1.0).expect("mesh");
        let current =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("cur");
        let predicted =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, 0.0, 0.0]).expect("pred");
        let d = vec![0.1; mesh.num_cells()];
        let p_corr = vec![0.0, 1.0, 2.0];
        let boundary = BoundarySet::default();

        let corrected =
            corrected_incompressible_fields_rhie_chow_3d(RhieChowVelocityCorrectionConfig {
                mesh: &mesh,
                current: &current,
                predicted: &predicted,
                pressure_correction: &p_corr,
                d_coefficient: &d,
                pressure_under_relaxation: 0.5,
                boundary: &boundary,
                periodic_x: false,
            })
            .expect("corrected");

        assert!(approx_eq(corrected.velocity_x.values()[1], 1.0, 1.0e-12));
        assert!(approx_eq(corrected.velocity_y.values()[1], 0.0, 1.0e-12));
    }
}
