//! 不可压缩结构化 3D 基础算子。
//!
//! 理论映射：[`docs/theory/incompressible_simplec_piso.md`](../../docs/theory/incompressible_simplec_piso.md)
//! §1–§3。I1 阶段仅覆盖 cell-centered、Cartesian 结构化网格上的连续性残差与速度
//! Laplacian 骨架。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::mesh::StructuredMesh3d;

/// 速度三分量的 cell-centered Laplacian。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleVelocityLaplacian {
    pub velocity_x: ScalarField,
    pub velocity_y: ScalarField,
    pub velocity_z: ScalarField,
}

/// 计算不可压缩连续性残差 \(\nabla\cdot\mathbf{u}\)。
///
/// 前置：`fields` 长度等于 `mesh.num_cells()`。I1 仅支持 Cartesian 均匀结构化网格；
/// 边界缺失邻居按零法向梯度 ghost 处理。
pub fn compute_incompressible_divergence_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> Result<ScalarField> {
    fields.validate_len(mesh.num_cells())?;
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let mut values = Vec::with_capacity(mesh.num_cells());
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let du_dx = central_diff_x(mesh, fields.velocity_x.values(), i, j, k, spacing.dx);
                let dv_dy = central_diff_y(mesh, fields.velocity_y.values(), i, j, k, spacing.dy);
                let dw_dz = central_diff_z(mesh, fields.velocity_z.values(), i, j, k, spacing.dz);
                values.push(du_dx + dv_dy + dw_dz);
            }
        }
    }
    ScalarField::from_values(values)
}

/// 计算速度三分量的 Cartesian Laplacian \(\nabla^2 u_i\)。
///
/// 前置：`fields` 长度等于 `mesh.num_cells()`。边界缺失邻居按零法向梯度 ghost 处理；
/// 后续 SIMPLEC/PISO 装配会用显式边界通量替代该 I1 skeleton。
pub fn compute_incompressible_velocity_laplacian_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
) -> Result<IncompressibleVelocityLaplacian> {
    fields.validate_len(mesh.num_cells())?;
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    Ok(IncompressibleVelocityLaplacian {
        velocity_x: scalar_laplacian(mesh, &fields.velocity_x, spacing)?,
        velocity_y: scalar_laplacian(mesh, &fields.velocity_y, spacing)?,
        velocity_z: scalar_laplacian(mesh, &fields.velocity_z, spacing)?,
    })
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
                "不可压缩 Cartesian 算子要求正的网格间距".to_string(),
            ));
        }
        Ok(Self {
            dx: dx.abs(),
            dy: dy.abs(),
            dz: dz.abs(),
        })
    }
}

fn scalar_laplacian(
    mesh: &StructuredMesh3d,
    field: &ScalarField,
    spacing: CartesianSpacing,
) -> Result<ScalarField> {
    let mut values = Vec::with_capacity(mesh.num_cells());
    let inv_dx2 = 1.0 / (spacing.dx * spacing.dx);
    let inv_dy2 = 1.0 / (spacing.dy * spacing.dy);
    let inv_dz2 = 1.0 / (spacing.dz * spacing.dz);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let center = cell_value(mesh, field.values(), i, j, k);
                let lap = (cell_value(mesh, field.values(), east(i, mesh.nx), j, k) - 2.0 * center
                    + cell_value(mesh, field.values(), west(i), j, k))
                    * inv_dx2
                    + (cell_value(mesh, field.values(), i, north(j, mesh.ny), k) - 2.0 * center
                        + cell_value(mesh, field.values(), i, south(j), k))
                        * inv_dy2
                    + (cell_value(mesh, field.values(), i, j, top(k, mesh.nz)) - 2.0 * center
                        + cell_value(mesh, field.values(), i, j, bottom(k)))
                        * inv_dz2;
                values.push(lap);
            }
        }
    }
    ScalarField::from_values(values)
}

fn central_diff_x(
    mesh: &StructuredMesh3d,
    values: &[Real],
    i: usize,
    j: usize,
    k: usize,
    dx: Real,
) -> Real {
    (cell_value(mesh, values, east(i, mesh.nx), j, k) - cell_value(mesh, values, west(i), j, k))
        / (2.0 * dx)
}

fn central_diff_y(
    mesh: &StructuredMesh3d,
    values: &[Real],
    i: usize,
    j: usize,
    k: usize,
    dy: Real,
) -> Real {
    (cell_value(mesh, values, i, north(j, mesh.ny), k) - cell_value(mesh, values, i, south(j), k))
        / (2.0 * dy)
}

fn central_diff_z(
    mesh: &StructuredMesh3d,
    values: &[Real],
    i: usize,
    j: usize,
    k: usize,
    dz: Real,
) -> Real {
    (cell_value(mesh, values, i, j, top(k, mesh.nz)) - cell_value(mesh, values, i, j, bottom(k)))
        / (2.0 * dz)
}

fn cell_value(mesh: &StructuredMesh3d, values: &[Real], i: usize, j: usize, k: usize) -> Real {
    values[mesh.cell_index(i, j, k)]
}

fn west(i: usize) -> usize {
    i.saturating_sub(1)
}

fn east(i: usize, nx: usize) -> usize {
    (i + 1).min(nx - 1)
}

fn south(j: usize) -> usize {
    j.saturating_sub(1)
}

fn north(j: usize, ny: usize) -> usize {
    (j + 1).min(ny - 1)
}

fn bottom(k: usize) -> usize {
    k.saturating_sub(1)
}

fn top(k: usize, nz: usize) -> usize {
    (k + 1).min(nz - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    fn mesh_3x3x3() -> StructuredMesh3d {
        StructuredMesh3d::uniform_box("inc", 3, 3, 3, 3.0, 3.0, 3.0).expect("mesh")
    }

    fn fields_from_components(
        mesh: &StructuredMesh3d,
        pressure: Real,
        u: impl Fn([Real; 3]) -> Real,
        v: impl Fn([Real; 3]) -> Real,
        w: impl Fn([Real; 3]) -> Real,
    ) -> IncompressibleFields {
        let mut ux = Vec::with_capacity(mesh.num_cells());
        let mut uy = Vec::with_capacity(mesh.num_cells());
        let mut uz = Vec::with_capacity(mesh.num_cells());
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    let x = i as Real + 0.5;
                    let y = j as Real + 0.5;
                    let z = k as Real + 0.5;
                    let xyz = [x, y, z];
                    ux.push(u(xyz));
                    uy.push(v(xyz));
                    uz.push(w(xyz));
                }
            }
        }
        IncompressibleFields {
            pressure: ScalarField::uniform(mesh.num_cells(), pressure).expect("pressure"),
            velocity_x: ScalarField::from_values(ux).expect("u"),
            velocity_y: ScalarField::from_values(uy).expect("v"),
            velocity_z: ScalarField::from_values(uz).expect("w"),
        }
    }

    #[test]
    fn uniform_velocity_has_zero_divergence_and_laplacian() {
        let mesh = mesh_3x3x3();
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [1.0, -2.0, 0.5]).expect("fields");

        let div = compute_incompressible_divergence_3d(&mesh, &fields).expect("div");
        assert!(div.values().iter().all(|&v| approx_eq(v, 0.0, 1.0e-12)));

        let lap = compute_incompressible_velocity_laplacian_3d(&mesh, &fields).expect("lap");
        for field in [&lap.velocity_x, &lap.velocity_y, &lap.velocity_z] {
            assert!(field.values().iter().all(|&v| approx_eq(v, 0.0, 1.0e-12)));
        }
    }

    #[test]
    fn linear_velocity_divergence_matches_interior_cell() {
        let mesh = mesh_3x3x3();
        let fields =
            fields_from_components(&mesh, 0.0, |xyz| xyz[0], |xyz| 2.0 * xyz[1], |xyz| -xyz[2]);

        let div = compute_incompressible_divergence_3d(&mesh, &fields).expect("div");
        let center = mesh.cell_index(1, 1, 1);
        assert!(approx_eq(div.values()[center], 2.0, 1.0e-12));
    }

    #[test]
    fn quadratic_velocity_laplacian_matches_interior_cell() {
        let mesh = mesh_3x3x3();
        let fields = fields_from_components(
            &mesh,
            0.0,
            |xyz| xyz[0] * xyz[0],
            |xyz| xyz[1] * xyz[1],
            |xyz| xyz[2] * xyz[2],
        );

        let lap = compute_incompressible_velocity_laplacian_3d(&mesh, &fields).expect("lap");
        let center = mesh.cell_index(1, 1, 1);
        assert!(approx_eq(lap.velocity_x.values()[center], 2.0, 1.0e-12));
        assert!(approx_eq(lap.velocity_y.values()[center], 2.0, 1.0e-12));
        assert!(approx_eq(lap.velocity_z.values()[center], 2.0, 1.0e-12));
    }
}
