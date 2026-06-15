//! 不可压缩结构化 3D FVM 离散（SIMPLEC / PISO 算子层）。
//!
//! 理论映射：[`docs/theory/incompressible_simplec_piso.md`](../../../docs/theory/incompressible_simplec_piso.md)
//! §1–§3。I1 阶段覆盖 cell-centered 结构化网格上的连续性、动量预测、Rhie-Chow 与压力校正。

pub mod bc;
pub mod boundary_flux;
pub mod face_boundary;
pub mod face_flux;
pub mod momentum;
mod momentum_convection;
mod momentum_geometry;
#[cfg(test)]
mod momentum_tests;
pub mod phi;
pub mod pressure_correction;
pub mod projection;
pub mod rhie_chow;
pub mod velocity_correction;

pub use bc::{IncompressibleBoundaryApplyStats, apply_incompressible_boundary_conditions_3d};
pub use face_boundary::{
    IncompressibleBoundaryFaceState, IncompressibleMassFluxBoundaryKind,
    incompressible_boundary_face_state, incompressible_boundary_face_velocity,
    incompressible_pressure_correction_dirichlet,
};
pub use face_flux::compute_incompressible_face_flux_divergence_3d;
pub use momentum::{
    IncompressibleConvectionScheme, IncompressibleMomentumPredictorConfig,
    IncompressibleMomentumPredictorSystem, assemble_incompressible_momentum_predictor_3d,
    assemble_incompressible_momentum_predictor_with_boundary_3d,
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d,
};
pub use phi::IncompressibleFaceFluxField;
pub use pressure_correction::assemble_incompressible_pressure_correction_3d;
pub use projection::{
    apply_pressure_correction_to_fields, apply_rhie_chow_pressure_projection_to_fields,
    subtract_d_pressure_gradient_from_velocity_3d,
};
pub use rhie_chow::{
    PressureCorrectedRhieChowDivergenceConfig, compute_incompressible_rhie_chow_divergence_3d,
    compute_pressure_corrected_rhie_chow_divergence_3d,
};
pub use velocity_correction::{
    RhieChowVelocityCorrectionConfig, corrected_incompressible_fields_rhie_chow_3d,
};

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::CsrMatrix;
use crate::mesh::StructuredMesh3d;

/// 速度三分量的 cell-centered Laplacian。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleVelocityLaplacian {
    pub velocity_x: ScalarField,
    pub velocity_y: ScalarField,
    pub velocity_z: ScalarField,
}

/// 不可压缩压力校正 Poisson 装配配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressiblePressureCorrectionConfig {
    pub density: Real,
    pub pressure_reference_cell: usize,
    pub pressure_reference_value: Real,
}

impl IncompressiblePressureCorrectionConfig {
    /// 构造压力校正配置；`density` 必须为正。
    pub fn new(
        density: Real,
        pressure_reference_cell: usize,
        pressure_reference_value: Real,
    ) -> Result<Self> {
        if density <= 0.0 {
            return Err(AsimuError::Config(
                "不可压缩压力校正 density 必须大于 0".to_string(),
            ));
        }
        Ok(Self {
            density,
            pressure_reference_cell,
            pressure_reference_value,
        })
    }
}

impl Default for IncompressiblePressureCorrectionConfig {
    fn default() -> Self {
        Self {
            density: 1.0,
            pressure_reference_cell: 0,
            pressure_reference_value: 0.0,
        }
    }
}

/// 压力校正线性系统 \(A p' = b\)。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressiblePressureCorrectionSystem {
    pub matrix: CsrMatrix,
    pub rhs: Vec<Real>,
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

/// 装配不可压缩压力校正 Poisson 骨架。
///
/// I1 使用 Cartesian 7 点 stencil 装配 \(-\rho\nabla^2 p'=\rho R_c\)，边界为零
/// Neumann；`pressure_reference_cell` 行强制为 `p'=pressure_reference_value`。
pub fn assemble_incompressible_pressure_poisson_3d(
    mesh: &StructuredMesh3d,
    divergence: &ScalarField,
    config: IncompressiblePressureCorrectionConfig,
) -> Result<IncompressiblePressureCorrectionSystem> {
    let n = mesh.num_cells();
    if divergence.len() != n {
        return Err(AsimuError::Field(format!(
            "压力校正 RHS 长度 {} 与网格单元数 {n} 不一致",
            divergence.len()
        )));
    }
    if config.density <= 0.0 {
        return Err(AsimuError::Config(
            "不可压缩压力校正 density 必须大于 0".to_string(),
        ));
    }
    if config.pressure_reference_cell >= n {
        return Err(AsimuError::Config(format!(
            "压力参考单元 {} 越界，单元数 {n}",
            config.pressure_reference_cell
        )));
    }

    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let mut rows = (0..n).map(|_| Vec::with_capacity(7)).collect::<Vec<_>>();
    let mut rhs = divergence
        .values()
        .iter()
        .map(|value| config.density * value)
        .collect::<Vec<_>>();
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let row = mesh.cell_index(i, j, k);
                if row == config.pressure_reference_cell {
                    rows[row].push((row, 1.0));
                    rhs[row] = config.pressure_reference_value;
                    continue;
                }
                add_pressure_poisson_neighbors(mesh, &mut rows[row], i, j, k, spacing, config);
            }
        }
    }
    Ok(IncompressiblePressureCorrectionSystem {
        matrix: CsrMatrix::from_rows(n, n, rows)?,
        rhs,
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

fn add_pressure_poisson_neighbors(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    i: usize,
    j: usize,
    k: usize,
    spacing: CartesianSpacing,
    config: IncompressiblePressureCorrectionConfig,
) {
    let center = mesh.cell_index(i, j, k);
    let cx = config.density / (spacing.dx * spacing.dx);
    let cy = config.density / (spacing.dy * spacing.dy);
    let cz = config.density / (spacing.dz * spacing.dz);
    let mut diag = 0.0;
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(i > 0, || (i - 1, j, k)),
        cx,
    );
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(i + 1 < mesh.nx, || (i + 1, j, k)),
        cx,
    );
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(j > 0, || (i, j - 1, k)),
        cy,
    );
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(j + 1 < mesh.ny, || (i, j + 1, k)),
        cy,
    );
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(k > 0, || (i, j, k - 1)),
        cz,
    );
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(k + 1 < mesh.nz, || (i, j, k + 1)),
        cz,
    );
    row.push((center, diag));
}

fn neighbor_if(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
) -> Option<(usize, usize, usize)> {
    present.then(index)
}

fn add_neighbor_if_present(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    neighbor: Option<(usize, usize, usize)>,
    coeff: Real,
) {
    if let Some((i, j, k)) = neighbor {
        *diag += coeff;
        row.push((mesh.cell_index(i, j, k), -coeff));
    }
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
    use super::{
        IncompressibleMomentumPredictorConfig, assemble_incompressible_momentum_predictor_3d,
        assemble_incompressible_pressure_correction_3d,
    };
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::approx_eq;
    use crate::mesh::BoundaryMesh;

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

    fn fields_from_pressure_and_components(
        mesh: &StructuredMesh3d,
        pressure: impl Fn([Real; 3]) -> Real,
        u: impl Fn([Real; 3]) -> Real,
        v: impl Fn([Real; 3]) -> Real,
        w: impl Fn([Real; 3]) -> Real,
    ) -> IncompressibleFields {
        let mut p = Vec::with_capacity(mesh.num_cells());
        let mut ux = Vec::with_capacity(mesh.num_cells());
        let mut uy = Vec::with_capacity(mesh.num_cells());
        let mut uz = Vec::with_capacity(mesh.num_cells());
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    let xyz = [i as Real + 0.5, j as Real + 0.5, k as Real + 0.5];
                    p.push(pressure(xyz));
                    ux.push(u(xyz));
                    uy.push(v(xyz));
                    uz.push(w(xyz));
                }
            }
        }
        IncompressibleFields {
            pressure: ScalarField::from_values(p).expect("pressure"),
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

    #[test]
    fn pressure_poisson_uses_reference_cell_row() {
        let mesh = mesh_3x3x3();
        let divergence = ScalarField::uniform(mesh.num_cells(), 0.0).expect("div");
        let config = IncompressiblePressureCorrectionConfig::new(1.0, 5, 3.25).expect("config");

        let system =
            assemble_incompressible_pressure_poisson_3d(&mesh, &divergence, config).expect("sys");

        assert_eq!(system.matrix.nrows(), mesh.num_cells());
        assert_eq!(system.matrix.ncols(), mesh.num_cells());
        assert_eq!(system.rhs[5], 3.25);
        let row = system.matrix.row_entries(5).collect::<Vec<_>>();
        assert_eq!(row, vec![(5, 1.0)]);
    }

    #[test]
    fn pressure_poisson_zero_divergence_has_zero_rhs_except_reference() {
        let mesh = mesh_3x3x3();
        let divergence = ScalarField::uniform(mesh.num_cells(), 0.0).expect("div");
        let config = IncompressiblePressureCorrectionConfig::default();

        let system =
            assemble_incompressible_pressure_poisson_3d(&mesh, &divergence, config).expect("sys");

        assert!(
            system
                .rhs
                .iter()
                .all(|&value| approx_eq(value, 0.0, 1.0e-12))
        );
    }

    #[test]
    fn pressure_poisson_interior_cell_has_seven_point_stencil() {
        let mesh = mesh_3x3x3();
        let divergence = ScalarField::uniform(mesh.num_cells(), 1.5).expect("div");
        let config = IncompressiblePressureCorrectionConfig::new(2.0, 0, 0.0).expect("config");

        let system =
            assemble_incompressible_pressure_poisson_3d(&mesh, &divergence, config).expect("sys");

        let center = mesh.cell_index(1, 1, 1);
        let row = system.matrix.row_entries(center).collect::<Vec<_>>();
        assert_eq!(row.len(), 7);
        assert!(row_contains(&row, center, 12.0));
        assert!(row_contains(&row, mesh.cell_index(0, 1, 1), -2.0));
        assert!(row_contains(&row, mesh.cell_index(2, 1, 1), -2.0));
        assert!(row_contains(&row, mesh.cell_index(1, 0, 1), -2.0));
        assert!(row_contains(&row, mesh.cell_index(1, 2, 1), -2.0));
        assert!(row_contains(&row, mesh.cell_index(1, 1, 0), -2.0));
        assert!(row_contains(&row, mesh.cell_index(1, 1, 2), -2.0));
        assert!(approx_eq(system.rhs[center], 3.0, 1.0e-12));
    }

    #[test]
    fn pressure_correction_scales_stencil_by_d_coefficient() {
        let mesh = mesh_3x3x3();
        let divergence = ScalarField::uniform(mesh.num_cells(), 1.5).expect("div");
        let d = ScalarField::uniform(mesh.num_cells(), 2.0).expect("d");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_max",
            mesh.resolve_logical_boundary("i_max").expect("faces"),
            BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 },
        )]);
        let config = IncompressiblePressureCorrectionConfig::new(1.0, 0, 0.0).expect("config");

        let system = assemble_incompressible_pressure_correction_3d(
            &mesh,
            &divergence,
            &d,
            &boundary,
            config,
        )
        .expect("sys");

        let center = mesh.cell_index(1, 1, 1);
        let row = system.matrix.row_entries(center).collect::<Vec<_>>();
        assert!(row_contains(&row, center, 12.0));
        assert!(row_contains(&row, mesh.cell_index(0, 1, 1), -2.0));
        assert!(approx_eq(system.rhs[center], 1.5, 1.0e-12));
    }

    #[test]
    fn pressure_correction_pressure_outlet_sets_zero_correction_row() {
        let mesh = mesh_3x3x3();
        let divergence = ScalarField::uniform(mesh.num_cells(), 1.5).expect("div");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_max",
            mesh.resolve_logical_boundary("i_max").expect("faces"),
            BoundaryKind::IncompressiblePressureOutlet { pressure: 0.0 },
        )]);
        let config = IncompressiblePressureCorrectionConfig::new(1.0, 0, 7.0).expect("config");

        let system = assemble_incompressible_pressure_correction_3d(
            &mesh,
            &divergence,
            &d,
            &boundary,
            config,
        )
        .expect("sys");

        let outlet = mesh.cell_index(2, 1, 1);
        let outlet_row = system.matrix.row_entries(outlet).collect::<Vec<_>>();
        assert_eq!(outlet_row, vec![(outlet, 1.0)]);
        assert!(approx_eq(system.rhs[outlet], 0.0, 1.0e-12));
        let reference_row = system.matrix.row_entries(0).collect::<Vec<_>>();
        assert_ne!(reference_row, vec![(0, 1.0)]);
    }

    #[test]
    fn momentum_predictor_zero_velocity_preserves_time_rhs_and_d() {
        let mesh = mesh_3x3x3();
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let config = IncompressibleMomentumPredictorConfig::new(0.25, 0.5).expect("config");

        let system =
            assemble_incompressible_momentum_predictor_3d(&mesh, &fields, config).expect("system");

        assert_eq!(system.matrix.nrows(), mesh.num_cells());
        assert_eq!(system.matrix.ncols(), mesh.num_cells());
        assert!(
            system
                .rhs_x
                .iter()
                .all(|&value| approx_eq(value, 0.0, 1.0e-12))
        );
        assert!(
            system
                .rhs_y
                .iter()
                .all(|&value| approx_eq(value, 0.0, 1.0e-12))
        );
        assert!(
            system
                .rhs_z
                .iter()
                .all(|&value| approx_eq(value, 0.0, 1.0e-12))
        );
        assert!(
            system
                .d_coefficient
                .values()
                .iter()
                .all(|&value| value.is_finite() && value > 0.0 && value < 0.5)
        );
    }

    #[test]
    fn momentum_predictor_interior_cell_has_transient_diffusion_stencil() {
        let mesh = mesh_3x3x3();
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let config = IncompressibleMomentumPredictorConfig::new(0.25, 0.5).expect("config");

        let system =
            assemble_incompressible_momentum_predictor_3d(&mesh, &fields, config).expect("system");

        let center = mesh.cell_index(1, 1, 1);
        let row = system.matrix.row_entries(center).collect::<Vec<_>>();
        assert_eq!(row.len(), 7);
        assert!(row_contains(&row, center, 3.5));
        assert!(row_contains(&row, mesh.cell_index(0, 1, 1), -0.25));
        assert!(row_contains(&row, mesh.cell_index(2, 1, 1), -0.25));
        assert!(row_contains(&row, mesh.cell_index(1, 0, 1), -0.25));
        assert!(row_contains(&row, mesh.cell_index(1, 2, 1), -0.25));
        assert!(row_contains(&row, mesh.cell_index(1, 1, 0), -0.25));
        assert!(row_contains(&row, mesh.cell_index(1, 1, 2), -0.25));
    }

    #[test]
    fn momentum_predictor_pressure_gradient_enters_rhs() {
        let mesh = mesh_3x3x3();
        let fields = fields_from_pressure_and_components(
            &mesh,
            |xyz| 3.0 * xyz[0] - 2.0 * xyz[1] + xyz[2],
            |_| 0.0,
            |_| 0.0,
            |_| 0.0,
        );
        let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0).expect("config");

        let system =
            assemble_incompressible_momentum_predictor_3d(&mesh, &fields, config).expect("system");

        let center = mesh.cell_index(1, 1, 1);
        assert!(approx_eq(system.rhs_x[center], -3.0, 1.0e-12));
        assert!(approx_eq(system.rhs_y[center], 2.0, 1.0e-12));
        assert!(approx_eq(system.rhs_z[center], -1.0, 1.0e-12));
    }

    #[test]
    fn momentum_predictor_upwind_convection_enters_matrix() {
        let mesh = mesh_3x3x3();
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [2.0, 0.0, 0.0]).expect("fields");
        let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0).expect("config");

        let system =
            assemble_incompressible_momentum_predictor_3d(&mesh, &fields, config).expect("system");

        let center = mesh.cell_index(1, 1, 1);
        let row = system.matrix.row_entries(center).collect::<Vec<_>>();
        assert!(row_contains(&row, center, 3.0));
        assert!(row_contains(&row, mesh.cell_index(0, 1, 1), -2.0));
        assert!(approx_eq(system.rhs_x[center], 2.0, 1.0e-12));
        assert!(approx_eq(
            system.d_coefficient.values()[center],
            1.0 / 3.0,
            1.0e-12
        ));
    }

    #[test]
    fn momentum_predictor_under_relaxation_increases_diagonal_and_rhs() {
        let mesh = mesh_3x3x3();
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [2.0, 0.0, 0.0]).expect("fields");
        let config = IncompressibleMomentumPredictorConfig::new(0.25, 0.5)
            .expect("config")
            .with_velocity_under_relaxation(0.5)
            .expect("relax");

        let system =
            assemble_incompressible_momentum_predictor_3d(&mesh, &fields, config).expect("system");

        let center = mesh.cell_index(1, 1, 1);
        let row = system.matrix.row_entries(center).collect::<Vec<_>>();
        assert!(row_contains(&row, center, 11.0));
        assert!(approx_eq(system.rhs_x[center], 15.0, 1.0e-12));
    }

    fn row_contains(row: &[(usize, Real)], col: usize, value: Real) -> bool {
        row.iter()
            .any(|&(c, v)| c == col && approx_eq(v, value, 1.0e-12))
    }
}
