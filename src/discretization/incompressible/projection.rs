//! 不可压缩初场散度投影 helper（Rhie-Chow 压力 Poisson + 速度校正）。
//!
//! 理论：[`docs/theory/incompressible_simplec_piso.md`](../../../docs/theory/incompressible_simplec_piso.md) §2–§3。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::gradient::compute_structured_scalar_gradients_3d;
use crate::discretization::periodic::StructuredPeriodic3d;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::StructuredMesh3d;

/// 将压力校正写入 cell-centered 压力：\(p \leftarrow p - \alpha p'\)。
pub fn apply_pressure_correction_to_fields(
    fields: &mut IncompressibleFields,
    pressure_correction: &[Real],
    scale: Real,
) -> Result<()> {
    let n = fields.pressure.len();
    if pressure_correction.len() != n {
        return Err(AsimuError::Field(format!(
            "压力校正长度 {} 与压力场 {} 不一致",
            pressure_correction.len(),
            n
        )));
    }
    let pressure = fields
        .pressure
        .values()
        .iter()
        .zip(pressure_correction.iter())
        .map(|(value, correction)| value - scale * correction)
        .collect::<Vec<_>>();
    fields.pressure = ScalarField::from_values(pressure)?;
    Ok(())
}

/// 从 cell-centered 速度减去 \(d \nabla p'\)（结构化梯度，支持周期边界）。
pub fn subtract_d_pressure_gradient_from_velocity_3d(
    mesh: &StructuredMesh3d,
    fields: &mut IncompressibleFields,
    d_coefficient: &ScalarField,
    pressure_correction: &[Real],
    boundary: &BoundarySet,
) -> Result<()> {
    fields.validate_len(mesh.num_cells())?;
    if d_coefficient.len() != mesh.num_cells() || pressure_correction.len() != mesh.num_cells() {
        return Err(AsimuError::Field(
            "散度投影 d 或压力校正长度与网格单元数不一致".to_string(),
        ));
    }
    let periodic = StructuredPeriodic3d::from_boundary(boundary);
    let gradients = compute_structured_scalar_gradients_3d(mesh, pressure_correction, periodic);
    let mut ux = fields.velocity_x.values().to_vec();
    let mut uy = fields.velocity_y.values().to_vec();
    let mut uz = fields.velocity_z.values().to_vec();
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let d = d_coefficient.values()[cell];
                ux[cell] -= d * gradients[cell].x;
                uy[cell] -= d * gradients[cell].y;
                uz[cell] -= d * gradients[cell].z;
            }
        }
    }
    fields.velocity_x = ScalarField::from_values(ux)?;
    fields.velocity_y = ScalarField::from_values(uy)?;
    fields.velocity_z = ScalarField::from_values(uz)?;
    Ok(())
}

/// 仅调整压力以满足 Rhie-Chow 面通量散度（保持速度不变）。
pub fn apply_rhie_chow_pressure_projection_to_fields(
    mesh: &StructuredMesh3d,
    fields: &mut IncompressibleFields,
    d_coefficient: &ScalarField,
    pressure_correction: &[Real],
    boundary: &BoundarySet,
) -> Result<()> {
    let _ = (mesh, d_coefficient, boundary);
    apply_pressure_correction_to_fields(fields, pressure_correction, 1.0)
}
