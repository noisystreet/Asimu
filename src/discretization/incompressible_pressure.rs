//! 不可压缩结构化 3D 压力校正装配。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::{
    IncompressiblePressureCorrectionConfig, IncompressiblePressureCorrectionSystem,
};
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
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let pressure_dirichlet = pressure_dirichlet_cells(mesh, boundary)?;
    let has_pressure_dirichlet = pressure_dirichlet.iter().any(|value| *value);
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    let mut rows = (0..n).map(|_| Vec::with_capacity(7)).collect::<Vec<_>>();
    let mut rhs = divergence
        .values()
        .iter()
        .map(|value| config.density * value)
        .collect::<Vec<_>>();
    let ctx = PressureCorrectionCtx {
        mesh,
        spacing,
        density: config.density,
        d: d_coefficient.values(),
        periodic_x,
    };
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let row = mesh.cell_index(i, j, k);
                if pressure_dirichlet[row]
                    || (!has_pressure_dirichlet && row == config.pressure_reference_cell)
                {
                    rows[row].push((row, 1.0));
                    rhs[row] = if pressure_dirichlet[row] {
                        0.0
                    } else {
                        config.pressure_reference_value
                    };
                    continue;
                }
                add_pressure_correction_neighbors(ctx, &mut rows[row], (i, j, k));
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
                "不可压缩压力校正要求正的 Cartesian 网格间距".to_string(),
            ));
        }
        Ok(Self {
            dx: dx.abs(),
            dy: dy.abs(),
            dz: dz.abs(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct PressureCorrectionCtx<'a> {
    mesh: &'a StructuredMesh3d,
    spacing: CartesianSpacing,
    density: Real,
    d: &'a [Real],
    periodic_x: bool,
}

fn add_pressure_correction_neighbors(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let center = ctx.mesh.cell_index(i, j, k);
    let mut diag = 0.0;
    add_d_neighbor(
        ctx,
        row,
        &mut diag,
        center,
        neighbor_if(i > 0, || (i - 1, j, k))
            .or_else(|| neighbor_if(ctx.periodic_x && i == 0, || (ctx.mesh.nx - 1, j, k))),
        ctx.spacing.dx,
    );
    add_d_neighbor(
        ctx,
        row,
        &mut diag,
        center,
        neighbor_if(i + 1 < ctx.mesh.nx, || (i + 1, j, k))
            .or_else(|| neighbor_if(ctx.periodic_x && i + 1 == ctx.mesh.nx, || (0, j, k))),
        ctx.spacing.dx,
    );
    add_d_neighbor(
        ctx,
        row,
        &mut diag,
        center,
        neighbor_if(j > 0, || (i, j - 1, k)),
        ctx.spacing.dy,
    );
    add_d_neighbor(
        ctx,
        row,
        &mut diag,
        center,
        neighbor_if(j + 1 < ctx.mesh.ny, || (i, j + 1, k)),
        ctx.spacing.dy,
    );
    add_d_neighbor(
        ctx,
        row,
        &mut diag,
        center,
        neighbor_if(k > 0, || (i, j, k - 1)),
        ctx.spacing.dz,
    );
    add_d_neighbor(
        ctx,
        row,
        &mut diag,
        center,
        neighbor_if(k + 1 < ctx.mesh.nz, || (i, j, k + 1)),
        ctx.spacing.dz,
    );
    row.push((center, diag));
}

fn add_d_neighbor(
    ctx: PressureCorrectionCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    center: usize,
    neighbor: Option<(usize, usize, usize)>,
    spacing: Real,
) {
    if let Some((i, j, k)) = neighbor {
        let col = ctx.mesh.cell_index(i, j, k);
        let d_face = 0.5 * (ctx.d[center] + ctx.d[col]);
        let coeff = ctx.density * d_face / (spacing * spacing);
        *diag += coeff;
        row.push((col, -coeff));
    }
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

fn pressure_dirichlet_cells(mesh: &StructuredMesh3d, boundary: &BoundarySet) -> Result<Vec<bool>> {
    let mut cells = vec![false; mesh.num_cells()];
    for patch in boundary.patches() {
        if !is_pressure_dirichlet_kind(&patch.kind) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            cells[owner] = true;
        }
    }
    Ok(cells)
}

fn is_pressure_dirichlet_kind(kind: &BoundaryKind) -> bool {
    matches!(
        kind,
        BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. }
    )
}

fn neighbor_if(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
) -> Option<(usize, usize, usize)> {
    present.then(index)
}
