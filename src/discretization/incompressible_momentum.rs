//! 不可压缩结构化 3D 动量预测装配。
//!
//! 理论映射：`docs/theory/incompressible_simplec_piso.md` 式 (8a)–(10)。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::CsrMatrix;
use crate::mesh::StructuredMesh3d;

/// 不可压缩动量预测方程装配配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleMomentumPredictorConfig {
    pub kinematic_viscosity: Real,
    pub pseudo_time_step: Real,
    pub velocity_under_relaxation: Real,
}

impl IncompressibleMomentumPredictorConfig {
    pub fn new(kinematic_viscosity: Real, pseudo_time_step: Real) -> Result<Self> {
        if kinematic_viscosity < 0.0 {
            return Err(AsimuError::Config(
                "不可压缩动量预测 kinematic_viscosity 不能为负".to_string(),
            ));
        }
        if pseudo_time_step <= 0.0 {
            return Err(AsimuError::Config(
                "不可压缩动量预测 pseudo_time_step 必须大于 0".to_string(),
            ));
        }
        Ok(Self {
            kinematic_viscosity,
            pseudo_time_step,
            velocity_under_relaxation: 1.0,
        })
    }

    pub fn with_velocity_under_relaxation(mut self, value: Real) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) || value == 0.0 {
            return Err(AsimuError::Config(
                "不可压缩动量预测 velocity_under_relaxation 必须位于 (0, 1]".to_string(),
            ));
        }
        self.velocity_under_relaxation = value;
        Ok(self)
    }
}

/// 三个速度分量共用矩阵的动量预测系统。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleMomentumPredictorSystem {
    pub matrix: CsrMatrix,
    pub rhs_x: Vec<Real>,
    pub rhs_y: Vec<Real>,
    pub rhs_z: Vec<Real>,
    pub d_coefficient: ScalarField,
}

/// 装配不可压缩伪瞬态动量预测方程。
///
/// I1 包含伪时间项、扩散项、一阶迎风对流项、压力梯度源项与速度欠松弛；
/// 边界面真实通量会在边界条件阶段替换当前缺失邻居处理。
pub fn assemble_incompressible_momentum_predictor_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    config: IncompressibleMomentumPredictorConfig,
) -> Result<IncompressibleMomentumPredictorSystem> {
    fields.validate_len(mesh.num_cells())?;
    validate_config(config)?;
    let spacing = CartesianSpacing::from_mesh(mesh)?;
    let volume = spacing.volume();
    let time_coeff = volume / config.pseudo_time_step;
    let n = mesh.num_cells();
    let mut rows = (0..n).map(|_| Vec::with_capacity(7)).collect::<Vec<_>>();
    let mut rhs_x = Vec::with_capacity(n);
    let mut rhs_y = Vec::with_capacity(n);
    let mut rhs_z = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let ctx = MomentumAssemblyCtx {
        mesh,
        spacing,
        fields,
        config,
        time_coeff,
    };
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let row = mesh.cell_index(i, j, k);
                let consistent_coeff =
                    add_momentum_predictor_neighbors(ctx, &mut rows[row], (i, j, k));
                let grad_p = pressure_gradient(mesh, fields.pressure.values(), i, j, k, spacing);
                let relax_source = momentum_relaxation_source(rows[row].last_mut(), config)?;
                rhs_x.push(
                    time_coeff * fields.velocity_x.values()[row] - volume * grad_p[0]
                        + relax_source * fields.velocity_x.values()[row],
                );
                rhs_y.push(
                    time_coeff * fields.velocity_y.values()[row] - volume * grad_p[1]
                        + relax_source * fields.velocity_y.values()[row],
                );
                rhs_z.push(
                    time_coeff * fields.velocity_z.values()[row] - volume * grad_p[2]
                        + relax_source * fields.velocity_z.values()[row],
                );
                d.push(volume / consistent_coeff);
            }
        }
    }
    Ok(IncompressibleMomentumPredictorSystem {
        matrix: CsrMatrix::from_rows(n, n, rows)?,
        rhs_x,
        rhs_y,
        rhs_z,
        d_coefficient: ScalarField::from_values(d)?,
    })
}

fn validate_config(config: IncompressibleMomentumPredictorConfig) -> Result<()> {
    if config.kinematic_viscosity < 0.0 {
        return Err(AsimuError::Config(
            "不可压缩动量预测 kinematic_viscosity 不能为负".to_string(),
        ));
    }
    if config.pseudo_time_step <= 0.0 {
        return Err(AsimuError::Config(
            "不可压缩动量预测 pseudo_time_step 必须大于 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.velocity_under_relaxation)
        || config.velocity_under_relaxation == 0.0
    {
        return Err(AsimuError::Config(
            "不可压缩动量预测 velocity_under_relaxation 必须位于 (0, 1]".to_string(),
        ));
    }
    Ok(())
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
                "不可压缩 Cartesian 动量装配要求正的网格间距".to_string(),
            ));
        }
        Ok(Self {
            dx: dx.abs(),
            dy: dy.abs(),
            dz: dz.abs(),
        })
    }

    fn volume(self) -> Real {
        self.dx * self.dy * self.dz
    }
}

#[derive(Debug, Clone, Copy)]
struct MomentumAssemblyCtx<'a> {
    mesh: &'a StructuredMesh3d,
    spacing: CartesianSpacing,
    fields: &'a IncompressibleFields,
    config: IncompressibleMomentumPredictorConfig,
    time_coeff: Real,
}

fn add_momentum_predictor_neighbors(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    cell: (usize, usize, usize),
) -> Real {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    let center = mesh.cell_index(i, j, k);
    let cx = ctx.config.kinematic_viscosity * ctx.spacing.dy * ctx.spacing.dz / ctx.spacing.dx;
    let cy = ctx.config.kinematic_viscosity * ctx.spacing.dx * ctx.spacing.dz / ctx.spacing.dy;
    let cz = ctx.config.kinematic_viscosity * ctx.spacing.dx * ctx.spacing.dy / ctx.spacing.dz;
    let mut diag = ctx.time_coeff;
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
    add_momentum_convection(mesh, row, &mut diag, cell, ctx.spacing, ctx.fields);
    let consistent_coeff = diag + row.iter().map(|(_, value)| *value).sum::<Real>();
    row.push((center, diag));
    consistent_coeff.max(ctx.time_coeff)
}

fn momentum_relaxation_source(
    diagonal_entry: Option<&mut (usize, Real)>,
    config: IncompressibleMomentumPredictorConfig,
) -> Result<Real> {
    let (_, diagonal) = diagonal_entry
        .ok_or_else(|| AsimuError::Field("不可压缩动量预测缺少对角系数".to_string()))?;
    let original = *diagonal;
    *diagonal = original / config.velocity_under_relaxation;
    Ok((1.0 - config.velocity_under_relaxation) * original / config.velocity_under_relaxation)
}

fn add_momentum_convection(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
    spacing: CartesianSpacing,
    fields: &IncompressibleFields,
) {
    let (i, j, k) = cell;
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(i + 1 < mesh.nx, || (i + 1, j, k)),
        face_velocity_x(mesh, fields, i, j, k, true) * spacing.dy * spacing.dz,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(i > 0, || (i - 1, j, k)),
        -face_velocity_x(mesh, fields, i, j, k, false) * spacing.dy * spacing.dz,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(j + 1 < mesh.ny, || (i, j + 1, k)),
        face_velocity_y(mesh, fields, i, j, k, true) * spacing.dx * spacing.dz,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(j > 0, || (i, j - 1, k)),
        -face_velocity_y(mesh, fields, i, j, k, false) * spacing.dx * spacing.dz,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(k + 1 < mesh.nz, || (i, j, k + 1)),
        face_velocity_z(mesh, fields, i, j, k, true) * spacing.dx * spacing.dy,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(k > 0, || (i, j, k - 1)),
        -face_velocity_z(mesh, fields, i, j, k, false) * spacing.dx * spacing.dy,
    );
}

fn add_convective_face(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    neighbor: Option<(usize, usize, usize)>,
    flux: Real,
) {
    let Some((i, j, k)) = neighbor else {
        return;
    };
    if flux >= 0.0 {
        *diag += flux;
    } else {
        row.push((mesh.cell_index(i, j, k), flux));
    }
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

fn pressure_gradient(
    mesh: &StructuredMesh3d,
    pressure: &[Real],
    i: usize,
    j: usize,
    k: usize,
    spacing: CartesianSpacing,
) -> [Real; 3] {
    [
        central_diff_x(mesh, pressure, i, j, k, spacing.dx),
        central_diff_y(mesh, pressure, i, j, k, spacing.dy),
        central_diff_z(mesh, pressure, i, j, k, spacing.dz),
    ]
}

fn face_velocity_x(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
) -> Real {
    let neighbor_i = if upper { east(i, mesh.nx) } else { west(i) };
    0.5 * (cell_value(mesh, fields.velocity_x.values(), i, j, k)
        + cell_value(mesh, fields.velocity_x.values(), neighbor_i, j, k))
}

fn face_velocity_y(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
) -> Real {
    let neighbor_j = if upper { north(j, mesh.ny) } else { south(j) };
    0.5 * (cell_value(mesh, fields.velocity_y.values(), i, j, k)
        + cell_value(mesh, fields.velocity_y.values(), i, neighbor_j, k))
}

fn face_velocity_z(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
) -> Real {
    let neighbor_k = if upper { top(k, mesh.nz) } else { bottom(k) };
    0.5 * (cell_value(mesh, fields.velocity_z.values(), i, j, k)
        + cell_value(mesh, fields.velocity_z.values(), i, j, neighbor_k))
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

fn neighbor_if(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
) -> Option<(usize, usize, usize)> {
    present.then(index)
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
