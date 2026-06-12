//! 压力校正后的 Rhie-Chow 一致速度重构。
//!
//! colocated 网格上 `u* - d∇p'` 的中心差分与 Rhie-Chow 面差分不一致，会导致
//! 压力方程残差很小但 face-flux / 下一步 Rhie-Chow 散度仍大。此处用
//! `u_face = ū* - d_f Δp'/Δn` 重构 cell 速度，使修正场与压力 Poisson 同一套面算子。

use crate::boundary::{BoundaryPatch, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible_boundary_flux::{
    IncompressibleBoundaryOwnerMap, interior_face_velocity,
};
use crate::discretization::incompressible_face_boundary::incompressible_boundary_face_velocity;
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

/// 由 \(u^*\) 与更新后的压力，经 Rhie-Chow 面速度重构 cell-centered 速度。
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
        boundary,
        periodic_x,
    } = config;
    let n = mesh.num_cells();
    if pressure_correction.len() != n || d_coefficient.len() != n {
        return Err(AsimuError::Field(
            "Rhie-Chow 速度重构长度与网格单元数不一致".to_string(),
        ));
    }
    predicted.validate_len(n)?;
    current.validate_len(n)?;
    let ctx = RhieChowVelocityCorrectionCtx {
        mesh,
        predicted,
        pressure_correction,
        d_coefficient,
        pressure_under_relaxation,
        boundary,
        periodic_x,
        spacing: CartesianSpacing::from_mesh(mesh)?,
        boundary_map: IncompressibleBoundaryOwnerMap::build(mesh, boundary),
    };
    let pressure = build_updated_pressure(current, pressure_correction, pressure_under_relaxation)?;
    let (velocity_x, velocity_y, velocity_z) = reconstruct_velocity_components(&ctx)?;
    Ok(IncompressibleFields {
        pressure,
        velocity_x: ScalarField::from_values(velocity_x)?,
        velocity_y: ScalarField::from_values(velocity_y)?,
        velocity_z: ScalarField::from_values(velocity_z)?,
    })
}

fn reconstruct_velocity_components(
    ctx: &RhieChowVelocityCorrectionCtx<'_>,
) -> Result<(Vec<Real>, Vec<Real>, Vec<Real>)> {
    let n = ctx.mesh.num_cells();
    let mut velocity_x = vec![0.0; n];
    let mut velocity_y = vec![0.0; n];
    let mut velocity_z = vec![0.0; n];
    for k in 0..ctx.mesh.nz {
        for j in 0..ctx.mesh.ny {
            for i in 0..ctx.mesh.nx {
                let cell = ctx.mesh.cell_index(i, j, k);
                velocity_x[cell] = reconstruct_component(ctx, 0, i, j, k)?;
                velocity_y[cell] = reconstruct_component(ctx, 1, i, j, k)?;
                velocity_z[cell] = reconstruct_component(ctx, 2, i, j, k)?;
            }
        }
    }
    Ok((velocity_x, velocity_y, velocity_z))
}

struct RhieChowVelocityCorrectionCtx<'a> {
    mesh: &'a StructuredMesh3d,
    predicted: &'a IncompressibleFields,
    pressure_correction: &'a [Real],
    d_coefficient: &'a [Real],
    pressure_under_relaxation: Real,
    boundary: &'a BoundarySet,
    periodic_x: bool,
    spacing: CartesianSpacing,
    boundary_map: IncompressibleBoundaryOwnerMap,
}

impl RhieChowVelocityCorrectionCtx<'_> {
    fn axis_spacing(&self, component: usize) -> Real {
        match component {
            0 => self.spacing.dx,
            1 => self.spacing.dy,
            _ => self.spacing.dz,
        }
    }
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

fn reconstruct_component(
    ctx: &RhieChowVelocityCorrectionCtx<'_>,
    component: usize,
    i: usize,
    j: usize,
    k: usize,
) -> Result<Real> {
    let owner = ctx.mesh.cell_index(i, j, k);
    let spacing = ctx.axis_spacing(component);
    let mut sum = 0.0;
    let mut count = 0usize;
    for spec in axis_face_specs(ctx.mesh, component, i, j, k, ctx.periodic_x, ctx.boundary) {
        if let Some((left, right)) = spec.left.zip(spec.right) {
            sum += rhie_chow_face_velocity(ctx, left, right, component, spacing)?;
            count += 1;
        } else if let Some(patch) = spec.lower_patch.or(spec.upper_patch) {
            sum += boundary_face_component(ctx.predicted, patch, owner, component);
            count += 1;
        }
    }
    if count == 0 {
        return Ok(predicted_cell_component(ctx.predicted, owner, component));
    }
    Ok(sum / count as Real)
}

fn predicted_cell_component(
    predicted: &IncompressibleFields,
    cell: usize,
    component: usize,
) -> Real {
    match component {
        0 => predicted.velocity_x.values()[cell],
        1 => predicted.velocity_y.values()[cell],
        _ => predicted.velocity_z.values()[cell],
    }
}

struct AxisFaceSpec<'a> {
    left: Option<usize>,
    right: Option<usize>,
    lower_patch: Option<&'a BoundaryPatch>,
    upper_patch: Option<&'a BoundaryPatch>,
}

fn axis_face_specs<'a>(
    mesh: &StructuredMesh3d,
    component: usize,
    i: usize,
    j: usize,
    k: usize,
    periodic_x: bool,
    boundary: &'a BoundarySet,
) -> Vec<AxisFaceSpec<'a>> {
    let mut specs = Vec::new();
    match component {
        0 => {
            if i > 0 {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(i - 1, j, k)),
                    right: Some(mesh.cell_index(i, j, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else if periodic_x && mesh.nx > 1 {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(mesh.nx - 1, j, k)),
                    right: Some(mesh.cell_index(0, j, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else {
                specs.push(AxisFaceSpec {
                    left: None,
                    right: None,
                    lower_patch: find_patch(boundary, "i_min"),
                    upper_patch: None,
                });
            }
            if i + 1 < mesh.nx {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(i, j, k)),
                    right: Some(mesh.cell_index(i + 1, j, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else if periodic_x && mesh.nx > 1 {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(mesh.nx - 1, j, k)),
                    right: Some(mesh.cell_index(0, j, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else {
                specs.push(AxisFaceSpec {
                    left: None,
                    right: None,
                    lower_patch: None,
                    upper_patch: find_patch(boundary, "i_max"),
                });
            }
        }
        1 => {
            if j > 0 {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(i, j - 1, k)),
                    right: Some(mesh.cell_index(i, j, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else {
                specs.push(AxisFaceSpec {
                    left: None,
                    right: None,
                    lower_patch: find_patch(boundary, "j_min"),
                    upper_patch: None,
                });
            }
            if j + 1 < mesh.ny {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(i, j, k)),
                    right: Some(mesh.cell_index(i, j + 1, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else {
                specs.push(AxisFaceSpec {
                    left: None,
                    right: None,
                    lower_patch: None,
                    upper_patch: find_patch(boundary, "j_max"),
                });
            }
        }
        _ => {
            if k > 0 {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(i, j, k - 1)),
                    right: Some(mesh.cell_index(i, j, k)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else {
                specs.push(AxisFaceSpec {
                    left: None,
                    right: None,
                    lower_patch: find_patch(boundary, "k_min"),
                    upper_patch: None,
                });
            }
            if k + 1 < mesh.nz {
                specs.push(AxisFaceSpec {
                    left: Some(mesh.cell_index(i, j, k)),
                    right: Some(mesh.cell_index(i, j, k + 1)),
                    lower_patch: None,
                    upper_patch: None,
                });
            } else {
                specs.push(AxisFaceSpec {
                    left: None,
                    right: None,
                    lower_patch: None,
                    upper_patch: find_patch(boundary, "k_max"),
                });
            }
        }
    }
    specs
}

fn find_patch<'a>(boundary: &'a BoundarySet, name: &str) -> Option<&'a BoundaryPatch> {
    boundary.patches().iter().find(|patch| patch.name == name)
}

fn boundary_face_component(
    predicted: &IncompressibleFields,
    patch: &BoundaryPatch,
    owner: usize,
    component: usize,
) -> Real {
    incompressible_boundary_face_velocity(owner, &patch.kind, predicted)[component]
}

fn rhie_chow_face_velocity(
    ctx: &RhieChowVelocityCorrectionCtx<'_>,
    left: usize,
    right: usize,
    component: usize,
    spacing: Real,
) -> Result<Real> {
    let u_face = interior_face_velocity(ctx.predicted, left, right, component, &ctx.boundary_map);
    let d_face = 0.5 * (ctx.d_coefficient[left] + ctx.d_coefficient[right]);
    let dp = ctx.pressure_correction[right] - ctx.pressure_correction[left];
    Ok(u_face - ctx.pressure_under_relaxation * d_face * dp / spacing)
}

#[derive(Debug, Clone, Copy)]
struct CartesianSpacing {
    dx: Real,
    dy: Real,
    dz: Real,
}

impl CartesianSpacing {
    fn from_mesh(mesh: &StructuredMesh3d) -> Result<Self> {
        let dx = mesh.node_x(1, 0, 0) - mesh.node_x(0, 0, 0);
        let dy = mesh.node_y(0, 1, 0) - mesh.node_y(0, 0, 0);
        let dz = mesh.node_z(0, 0, 1) - mesh.node_z(0, 0, 0);
        if dx.abs() <= Real::EPSILON || dy.abs() <= Real::EPSILON || dz.abs() <= Real::EPSILON {
            return Err(AsimuError::Mesh(
                "Rhie-Chow 速度重构要求正的 Cartesian 网格间距".to_string(),
            ));
        }
        Ok(Self {
            dx: dx.abs(),
            dy: dy.abs(),
            dz: dz.abs(),
        })
    }
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
}
