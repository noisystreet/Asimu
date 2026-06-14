//! 不可压缩结构化 3D 压力校正装配。

use super::face_boundary::incompressible_pressure_correction_dirichlet;
use super::{IncompressiblePressureCorrectionConfig, IncompressiblePressureCorrectionSystem};
use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::ScalarField;
use crate::linalg::CsrMatrix;
use crate::mesh::{BoundaryMesh, StructuredMesh3d};

/// 装配使用动量一致系数 \(d_P\) 的不可压缩压力校正方程。
///
/// 内部面系数使用相邻 cell-centered \(d_P\) 算术平均；`pressure_outlet`
/// 与不可压缩 `pressure_outlet` patch 将 owner 行替换为 `p'=0`。若没有压力
/// Dirichlet 边界，则使用 `pressure_reference_cell` 固定参考压力。
pub fn assemble_incompressible_pressure_correction_3d(
    mesh: &StructuredMesh3d,
    divergence: &ScalarField,
    d_coefficient: &ScalarField,
    boundary: &BoundarySet,
    config: IncompressiblePressureCorrectionConfig,
) -> Result<IncompressiblePressureCorrectionSystem> {
    let n = mesh.num_cells();
    validate_pressure_inputs(n, divergence, config)?;
    validate_d_coefficient(n, d_coefficient)?;
    let correction_dirichlet = pressure_correction_dirichlet_cells(mesh, boundary)?;
    let has_correction_dirichlet = correction_dirichlet.iter().any(|value| *value);
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    let mut rows = (0..n).map(|_| Vec::with_capacity(7)).collect::<Vec<_>>();
    let mut rhs = pressure_correction_integrated_rhs(mesh, divergence, config.density);
    let fixed_pressure = correction_dirichlet.clone();
    let fixed_values = vec![0.0; n];
    if is_closed_pressure_correction_cavity(boundary) {
        remove_active_pressure_rhs_mean(&mut rhs, &correction_dirichlet);
    } else if !has_correction_dirichlet {
        remove_closed_domain_rhs_mean(&mut rhs);
    }
    let ctx = PressureCorrectionCtx {
        mesh,
        density: config.density,
        d: d_coefficient.values(),
        fixed_pressure: &fixed_pressure,
        fixed_values: &fixed_values,
        periodic_x,
    };
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let row = mesh.cell_index(i, j, k);
                if fixed_pressure[row] {
                    rows[row].push((row, 1.0));
                    rhs[row] = fixed_values[row];
                    continue;
                }
                add_pressure_correction_neighbors(ctx, &mut rows[row], &mut rhs[row], (i, j, k));
            }
        }
    }
    Ok(IncompressiblePressureCorrectionSystem {
        matrix: CsrMatrix::from_rows(n, n, rows)?,
        rhs,
    })
}

#[derive(Debug, Clone, Copy)]
struct PressureCorrectionCtx<'a> {
    mesh: &'a StructuredMesh3d,
    density: Real,
    d: &'a [Real],
    fixed_pressure: &'a [bool],
    fixed_values: &'a [Real],
    periodic_x: bool,
}

fn add_pressure_correction_neighbors(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    rhs: &mut Real,
    cell: (usize, usize, usize),
) {
    let center = ctx.mesh.cell_index(cell.0, cell.1, cell.2);
    let mut diag = 0.0;
    add_x_neighbors(ctx, row, rhs, &mut diag, center, cell);
    add_y_neighbors(ctx, row, rhs, &mut diag, center, cell);
    add_z_neighbors(ctx, row, rhs, &mut diag, center, cell);
    row.push((center, diag));
}

fn add_x_neighbors(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    rhs: &mut Real,
    diag: &mut Real,
    center: usize,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    add_d_neighbor(
        ctx,
        row,
        rhs,
        diag,
        center,
        neighbor_with_coeff(i > 0, || (i - 1, j, k), || face_coeff_x(ctx, i - 1, j, k)).or_else(
            || {
                neighbor_with_coeff(
                    ctx.periodic_x && i == 0,
                    || (ctx.mesh.nx - 1, j, k),
                    || face_coeff_x(ctx, 0, j, k),
                )
            },
        ),
    );
    add_d_neighbor(
        ctx,
        row,
        rhs,
        diag,
        center,
        neighbor_with_coeff(
            i + 1 < ctx.mesh.nx,
            || (i + 1, j, k),
            || face_coeff_x(ctx, i, j, k),
        )
        .or_else(|| {
            neighbor_with_coeff(
                ctx.periodic_x && i + 1 == ctx.mesh.nx,
                || (0, j, k),
                || face_coeff_x(ctx, ctx.mesh.nx - 2, j, k),
            )
        }),
    );
}

fn add_y_neighbors(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    rhs: &mut Real,
    diag: &mut Real,
    center: usize,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    add_d_neighbor(
        ctx,
        row,
        rhs,
        diag,
        center,
        neighbor_with_coeff(j > 0, || (i, j - 1, k), || face_coeff_y(ctx, i, j - 1, k)),
    );
    add_d_neighbor(
        ctx,
        row,
        rhs,
        diag,
        center,
        neighbor_with_coeff(
            j + 1 < ctx.mesh.ny,
            || (i, j + 1, k),
            || face_coeff_y(ctx, i, j, k),
        ),
    );
}

fn add_z_neighbors(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    rhs: &mut Real,
    diag: &mut Real,
    center: usize,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    add_d_neighbor(
        ctx,
        row,
        rhs,
        diag,
        center,
        neighbor_with_coeff(k > 0, || (i, j, k - 1), || face_coeff_z(ctx, i, j, k - 1)),
    );
    add_d_neighbor(
        ctx,
        row,
        rhs,
        diag,
        center,
        neighbor_with_coeff(
            k + 1 < ctx.mesh.nz,
            || (i, j, k + 1),
            || face_coeff_z(ctx, i, j, k),
        ),
    );
}

fn add_d_neighbor(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    rhs: &mut Real,
    diag: &mut Real,
    _center: usize,
    neighbor: Option<((usize, usize, usize), Real)>,
) {
    if let Some(((i, j, k), coeff)) = neighbor {
        let col = ctx.mesh.cell_index(i, j, k);
        *diag += coeff;
        if ctx.fixed_pressure[col] {
            *rhs += coeff * ctx.fixed_values[col];
        } else {
            row.push((col, -coeff));
        }
    }
}

fn pressure_correction_integrated_rhs(
    mesh: &StructuredMesh3d,
    divergence: &ScalarField,
    density: Real,
) -> Vec<Real> {
    let mut rhs = vec![0.0; mesh.num_cells()];
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                rhs[cell] = density * divergence.values()[cell] * mesh.cell_metric(i, j, k).volume;
            }
        }
    }
    rhs
}

fn face_coeff_x(ctx: PressureCorrectionCtx<'_>, i: usize, j: usize, k: usize) -> Real {
    let owner = CellCoord { i, j, k };
    let neighbor = CellCoord { i: i + 1, j, k };
    let face = ctx.mesh.i_face_metric(i, j, k);
    pressure_face_coeff(ctx, owner, neighbor, face)
}

fn face_coeff_y(ctx: PressureCorrectionCtx<'_>, i: usize, j: usize, k: usize) -> Real {
    let owner = CellCoord { i, j, k };
    let neighbor = CellCoord { i, j: j + 1, k };
    let face = ctx.mesh.j_face_metric(i, j, k);
    pressure_face_coeff(ctx, owner, neighbor, face)
}

fn face_coeff_z(ctx: PressureCorrectionCtx<'_>, i: usize, j: usize, k: usize) -> Real {
    let owner = CellCoord { i, j, k };
    let neighbor = CellCoord { i, j, k: k + 1 };
    let face = ctx.mesh.k_face_metric(i, j, k);
    pressure_face_coeff(ctx, owner, neighbor, face)
}

fn pressure_face_coeff(
    ctx: PressureCorrectionCtx<'_>,
    owner: CellCoord,
    neighbor: CellCoord,
    face: crate::mesh::FaceMetric,
) -> Real {
    let left = ctx.mesh.cell_index(owner.i, owner.j, owner.k);
    let right = ctx.mesh.cell_index(neighbor.i, neighbor.j, neighbor.k);
    let d_face = 0.5 * (ctx.d[left] + ctx.d[right]);
    ctx.density * d_face * face.area / owner_neighbor_distance(ctx.mesh, owner, neighbor, &face)
}

#[derive(Debug, Clone, Copy)]
struct CellCoord {
    i: usize,
    j: usize,
    k: usize,
}

fn owner_neighbor_distance(
    mesh: &StructuredMesh3d,
    owner: CellCoord,
    neighbor: CellCoord,
    face: &crate::mesh::FaceMetric,
) -> Real {
    let owner_center = mesh.cell_metric(owner.i, owner.j, owner.k).center;
    let neighbor_center = mesh.cell_metric(neighbor.i, neighbor.j, neighbor.k).center;
    let dx = neighbor_center.x - owner_center.x;
    let dy = neighbor_center.y - owner_center.y;
    let dz = neighbor_center.z - owner_center.z;
    let projected = (dx * face.normal.x + dy * face.normal.y + dz * face.normal.z).abs();
    projected.max(Real::EPSILON)
}

fn validate_pressure_inputs(
    n: usize,
    divergence: &ScalarField,
    config: IncompressiblePressureCorrectionConfig,
) -> Result<()> {
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
    Ok(())
}

fn validate_d_coefficient(n: usize, d_coefficient: &ScalarField) -> Result<()> {
    if d_coefficient.len() != n {
        return Err(AsimuError::Field(format!(
            "压力校正 d_P 长度 {} 与网格单元数 {n} 不一致",
            d_coefficient.len()
        )));
    }
    if d_coefficient
        .values()
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(AsimuError::Field("压力校正 d_P 必须为有限正值".to_string()));
    }
    Ok(())
}

fn pressure_correction_dirichlet_cells(
    mesh: &StructuredMesh3d,
    boundary: &BoundarySet,
) -> Result<Vec<bool>> {
    let mut cells = vec![false; mesh.num_cells()];
    for patch in boundary.patches() {
        if !is_pressure_correction_dirichlet_kind(&patch.kind) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            cells[owner] = true;
        }
    }
    Ok(cells)
}

fn is_pressure_correction_dirichlet_kind(kind: &BoundaryKind) -> bool {
    incompressible_pressure_correction_dirichlet(kind)
}

fn is_closed_pressure_correction_cavity(boundary: &BoundarySet) -> bool {
    let has_dirichlet = boundary
        .patches()
        .iter()
        .any(|patch| is_pressure_correction_dirichlet_kind(&patch.kind));
    if !has_dirichlet {
        return false;
    }
    !boundary.patches().iter().any(|patch| {
        matches!(
            patch.kind,
            BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. }
        )
    })
}

fn remove_closed_domain_rhs_mean(rhs: &mut [Real]) {
    if rhs.is_empty() {
        return;
    }
    let mean = rhs.iter().sum::<Real>() / rhs.len() as Real;
    for value in rhs.iter_mut() {
        *value -= mean;
    }
}

/// 有 \(p'=0\) 边界 owner 时，对非约束行 RHS 去均值以满足闭域兼容性。
fn remove_active_pressure_rhs_mean(rhs: &mut [Real], pressure_correction_dirichlet: &[bool]) {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (value, &dirichlet) in rhs.iter().zip(pressure_correction_dirichlet.iter()) {
        if dirichlet {
            continue;
        }
        sum += *value;
        count += 1;
    }
    if count <= 1 {
        return;
    }
    let mean = sum / count as Real;
    for (value, &dirichlet) in rhs.iter_mut().zip(pressure_correction_dirichlet.iter()) {
        if !dirichlet {
            *value -= mean;
        }
    }
}

fn neighbor_with_coeff(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
    coeff: impl FnOnce() -> Real,
) -> Option<((usize, usize, usize), Real)> {
    present.then(|| (index(), coeff()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::WallHeat;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};

    #[test]
    fn closed_domain_pressure_rhs_removes_active_mean() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 2, 1, 1.0, 1.0, 0.1).expect("mesh");
        let divergence = ScalarField::from_values(vec![1.0, 2.0, 3.0, 4.0]).expect("divergence");
        let d = ScalarField::from_values(vec![1.0; mesh.num_cells()]).expect("d");
        let boundary = BoundarySet::new(Vec::new());

        let system = assemble_incompressible_pressure_correction_3d(
            &mesh,
            &divergence,
            &d,
            &boundary,
            IncompressiblePressureCorrectionConfig::new(1.0, 0, 0.0).expect("config"),
        )
        .expect("system");

        assert!((system.rhs[0] + 0.0375).abs() <= Real::EPSILON);
        assert!((system.rhs[1] + 0.0125).abs() <= Real::EPSILON);
        assert!((system.rhs[2] - 0.0125).abs() <= Real::EPSILON);
        assert!((system.rhs[3] - 0.0375).abs() <= Real::EPSILON);
    }

    #[test]
    fn closed_pressure_reference_keeps_continuity_rows_for_pcg() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 3, 1, 1.0, 1.0, 0.1).expect("mesh");
        let divergence = ScalarField::from_values(vec![1.0; mesh.num_cells()]).expect("divergence");
        let d = ScalarField::from_values(vec![1.0; mesh.num_cells()]).expect("d");
        let boundary = BoundarySet::new(Vec::new());

        let system = assemble_incompressible_pressure_correction_3d(
            &mesh,
            &divergence,
            &d,
            &boundary,
            IncompressiblePressureCorrectionConfig::new(1.0, 0, 0.0).expect("config"),
        )
        .expect("system");

        assert_ne!(
            system.matrix.row_entries(0).collect::<Vec<_>>(),
            vec![(0, 1.0)]
        );
        let neighbor = mesh.cell_index(1, 0, 0);
        let row = system.matrix.row_entries(neighbor).collect::<Vec<_>>();
        assert!(
            row.iter().any(|(col, _)| *col == 0),
            "closed-domain pressure reference should not drop continuity columns: {row:?}"
        );
        assert_matrix_symmetric(&system.matrix, 1.0e-12);
    }

    #[test]
    fn walled_cavity_pressure_rhs_removes_active_mean() {
        let mesh = StructuredMesh3d::uniform_box("box", 5, 5, 1, 1.0, 1.0, 0.1).expect("mesh");
        let divergence = ScalarField::uniform(mesh.num_cells(), 1.0).expect("divergence");
        let d = ScalarField::uniform(mesh.num_cells(), 1.0).expect("d");
        let boundary = BoundarySet::new(vec![
            BoundaryPatch::new(
                "i_min",
                mesh.resolve_logical_boundary("i_min").expect("faces"),
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "i_max",
                mesh.resolve_logical_boundary("i_max").expect("faces"),
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "j_min",
                mesh.resolve_logical_boundary("j_min").expect("faces"),
                BoundaryKind::Wall {
                    no_slip: true,
                    heat: WallHeat::Adiabatic,
                },
            ),
            BoundaryPatch::new(
                "j_max",
                mesh.resolve_logical_boundary("j_max").expect("faces"),
                BoundaryKind::MovingWall {
                    velocity: [1.0, 0.0, 0.0],
                },
            ),
        ]);
        let system = assemble_incompressible_pressure_correction_3d(
            &mesh,
            &divergence,
            &d,
            &boundary,
            IncompressiblePressureCorrectionConfig::new(1.0, 0, 0.0).expect("config"),
        )
        .expect("system");
        let active_sum = active_pressure_rhs_sum(&system.matrix, &system.rhs);
        assert!(active_sum.abs() <= 1.0e-12);
    }

    fn active_pressure_rhs_sum(matrix: &CsrMatrix, rhs: &[Real]) -> Real {
        let mut sum = 0.0;
        for (row, value) in rhs.iter().enumerate() {
            let mut entries = matrix.row_entries(row);
            let Some((col, diag)) = entries.next() else {
                continue;
            };
            if entries.next().is_none() && col == row && (diag - 1.0).abs() <= Real::EPSILON {
                continue;
            }
            sum += *value;
        }
        sum
    }

    fn assert_matrix_symmetric(matrix: &CsrMatrix, tolerance: Real) {
        for row in 0..matrix.nrows() {
            for (col, value) in matrix.row_entries(row) {
                let mirror = matrix_entry(matrix, col, row);
                assert!(
                    (value - mirror).abs() <= tolerance,
                    "matrix asymmetric at ({row},{col}): {value} vs {mirror}"
                );
            }
        }
    }

    fn matrix_entry(matrix: &CsrMatrix, row: usize, col: usize) -> Real {
        matrix
            .row_entries(row)
            .find_map(|(entry_col, value)| (entry_col == col).then_some(value))
            .unwrap_or(0.0)
    }
}
