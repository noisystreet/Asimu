//! 不可压缩结构化 3D Rhie-Chow 面质量通量。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::incompressible_phi::IncompressibleFaceFluxField;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::StructuredMesh3d;

/// 用 Rhie-Chow 面通量计算连续性残差。
///
/// 内部面通量使用中心插值速度，并减去 \(d_f \nabla p\) 的面法向贡献；
/// 边界面按不可压缩边界类型给定法向通量。返回值为每个单元的净体积通量除以体积。
pub fn compute_incompressible_rhie_chow_divergence_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    d_coefficient: &ScalarField,
    boundary: &BoundarySet,
) -> Result<ScalarField> {
    fields.validate_len(mesh.num_cells())?;
    if d_coefficient.len() != mesh.num_cells() {
        return Err(AsimuError::Field(format!(
            "Rhie-Chow d_P 长度 {} 与网格单元数 {} 不一致",
            d_coefficient.len(),
            mesh.num_cells()
        )));
    }
    IncompressibleFaceFluxField::from_rhie_chow(mesh, fields, d_coefficient, boundary)?
        .divergence(mesh)
}

/// 压力校正后用同一套 Rhie-Chow 面通量计算连续性残差。
pub fn compute_pressure_corrected_rhie_chow_divergence_3d(
    config: PressureCorrectedRhieChowDivergenceConfig<'_>,
) -> Result<ScalarField> {
    let PressureCorrectedRhieChowDivergenceConfig {
        mesh,
        fields,
        d_coefficient,
        pressure_correction,
        correction_scale,
        boundary,
    } = config;
    fields.validate_len(mesh.num_cells())?;
    if d_coefficient.len() != mesh.num_cells() || pressure_correction.len() != mesh.num_cells() {
        return Err(AsimuError::Field(
            "Rhie-Chow 压力校正通量长度与网格单元数不一致".to_string(),
        ));
    }
    let corrected_fields =
        pressure_corrected_fields(fields, pressure_correction, correction_scale)?;
    let d = ScalarField::from_values(d_coefficient.to_vec())?;
    IncompressibleFaceFluxField::from_rhie_chow(mesh, &corrected_fields, &d, boundary)?
        .divergence(mesh)
}

pub struct PressureCorrectedRhieChowDivergenceConfig<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub fields: &'a IncompressibleFields,
    pub d_coefficient: &'a [Real],
    pub pressure_correction: &'a [Real],
    pub correction_scale: Real,
    pub boundary: &'a BoundarySet,
}

fn pressure_corrected_fields(
    fields: &IncompressibleFields,
    pressure_correction: &[Real],
    correction_scale: Real,
) -> Result<IncompressibleFields> {
    let pressure = fields
        .pressure
        .values()
        .iter()
        .zip(pressure_correction.iter())
        .map(|(p, correction)| p - correction_scale * correction)
        .collect::<Vec<_>>();
    Ok(IncompressibleFields {
        pressure: ScalarField::from_values(pressure)?,
        velocity_x: fields.velocity_x.clone(),
        velocity_y: fields.velocity_y.clone(),
        velocity_z: fields.velocity_z.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::core::approx_eq;
    use crate::mesh::BoundaryMesh;

    #[test]
    fn uniform_field_has_zero_rhie_chow_divergence() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");

        let div = compute_incompressible_rhie_chow_divergence_3d(
            &mesh,
            &fields,
            &d,
            &BoundarySet::default(),
        )
        .expect("div");

        assert!(
            div.values()
                .iter()
                .all(|value| approx_eq(*value, 0.0, 1.0e-12))
        );
    }

    #[test]
    fn pressure_gradient_drives_rhie_chow_flux() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 1, 1, 2.0, 1.0, 1.0).expect("mesh");
        let fields = IncompressibleFields {
            pressure: ScalarField::from_values(vec![0.0, 2.0]).expect("p"),
            velocity_x: ScalarField::uniform(mesh.num_cells(), 0.0).expect("u"),
            velocity_y: ScalarField::uniform(mesh.num_cells(), 0.0).expect("v"),
            velocity_z: ScalarField::uniform(mesh.num_cells(), 0.0).expect("w"),
        };
        let d = ScalarField::uniform(mesh.num_cells(), 0.5).expect("d");

        let div = compute_incompressible_rhie_chow_divergence_3d(
            &mesh,
            &fields,
            &d,
            &BoundarySet::default(),
        )
        .expect("div");

        assert!(approx_eq(div.values()[0], -1.0, 1.0e-12));
        assert!(approx_eq(div.values()[1], 1.0, 1.0e-12));
    }

    #[test]
    fn velocity_inlet_boundary_contributes_flux() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("faces"),
            BoundaryKind::IncompressibleVelocityInlet {
                velocity: [1.0, 0.0, 0.0],
            },
        )]);

        let div = compute_incompressible_rhie_chow_divergence_3d(&mesh, &fields, &d, &boundary)
            .expect("div");

        assert!(approx_eq(div.values()[0], -1.0, 1.0e-12));
    }
}
