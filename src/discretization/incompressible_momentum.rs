//! 不可压缩结构化 3D 动量预测装配。
//!
//! 理论映射：`docs/theory/incompressible_simplec_piso.md` 式 (8a)–(10)。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible_bc::{
    incompressible_boundary_owner_velocity_target, interior_neighbor_index,
};
use crate::discretization::incompressible_boundary_flux::{
    IncompressibleBoundaryOwnerMap, StructuredMomentumFaceQuery, structured_momentum_face_velocity,
};
use crate::discretization::incompressible_phi::IncompressibleFaceFluxField;
use crate::error::{AsimuError, Result};
use crate::field::{IncompressibleFields, ScalarField};
use crate::linalg::CsrMatrix;
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, FaceGeometry3d, StructuredMesh3d};

/// 不可压缩动量预测方程装配配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleMomentumPredictorConfig {
    pub kinematic_viscosity: Real,
    pub pseudo_time_step: Real,
    pub body_force: [Real; 3],
    pub velocity_under_relaxation: Real,
    pub convection_scheme: IncompressibleConvectionScheme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressibleConvectionScheme {
    Upwind,
    Central,
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
            body_force: [0.0, 0.0, 0.0],
            velocity_under_relaxation: 1.0,
            convection_scheme: IncompressibleConvectionScheme::Upwind,
        })
    }

    pub fn with_body_force(mut self, value: [Real; 3]) -> Result<Self> {
        if value.iter().any(|component| !component.is_finite()) {
            return Err(AsimuError::Config(
                "不可压缩动量预测 body_force 分量必须为有限值".to_string(),
            ));
        }
        self.body_force = value;
        Ok(self)
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

    pub fn with_convection_scheme(mut self, value: IncompressibleConvectionScheme) -> Self {
        self.convection_scheme = value;
        self
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
    assemble_incompressible_momentum_predictor_with_boundary_3d(
        mesh,
        fields,
        &BoundarySet::default(),
        config,
    )
}

/// 装配含边界面通量的不可压缩伪瞬态动量预测方程。
pub fn assemble_incompressible_momentum_predictor_with_boundary_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
    config: IncompressibleMomentumPredictorConfig,
) -> Result<IncompressibleMomentumPredictorSystem> {
    assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d(
        mesh, fields, boundary, config, None,
    )
}

pub fn assemble_incompressible_momentum_predictor_with_boundary_and_flux_3d(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
    config: IncompressibleMomentumPredictorConfig,
    face_flux: Option<&IncompressibleFaceFluxField>,
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
    let boundary_terms = boundary_momentum_contributions(mesh, fields, boundary, config)?;
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    let boundary_map = IncompressibleBoundaryOwnerMap::build(mesh, boundary);
    let ctx = MomentumAssemblyCtx {
        mesh,
        spacing,
        fields,
        config,
        time_coeff,
        periodic_x,
        boundary_map: &boundary_map,
        face_flux,
    };
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let row = mesh.cell_index(i, j, k);
                let consistent_coeff = add_momentum_predictor_neighbors(
                    ctx,
                    &mut rows[row],
                    (i, j, k),
                    boundary_terms.diagonal[row],
                );
                let grad_p =
                    pressure_gradient(mesh, fields.pressure.values(), i, j, k, spacing, periodic_x);
                let relax_source = momentum_relaxation_source(rows[row].last_mut(), config)?;
                let rhs_cell_x = time_coeff * fields.velocity_x.values()[row] - volume * grad_p[0]
                    + volume * config.body_force[0]
                    + relax_source * fields.velocity_x.values()[row]
                    + boundary_terms.rhs_x[row];
                let rhs_cell_y = time_coeff * fields.velocity_y.values()[row] - volume * grad_p[1]
                    + volume * config.body_force[1]
                    + relax_source * fields.velocity_y.values()[row]
                    + boundary_terms.rhs_y[row];
                let rhs_cell_z = time_coeff * fields.velocity_z.values()[row] - volume * grad_p[2]
                    + volume * config.body_force[2]
                    + relax_source * fields.velocity_z.values()[row]
                    + boundary_terms.rhs_z[row];
                rhs_x.push(rhs_cell_x);
                rhs_y.push(rhs_cell_y);
                rhs_z.push(rhs_cell_z);
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
    if config
        .body_force
        .iter()
        .any(|component| !component.is_finite())
    {
        return Err(AsimuError::Config(
            "不可压缩动量预测 body_force 分量必须为有限值".to_string(),
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
    periodic_x: bool,
    boundary_map: &'a IncompressibleBoundaryOwnerMap,
    face_flux: Option<&'a IncompressibleFaceFluxField>,
}

#[derive(Debug, Clone, PartialEq)]
struct BoundaryMomentumContributions {
    diagonal: Vec<Real>,
    rhs_x: Vec<Real>,
    rhs_y: Vec<Real>,
    rhs_z: Vec<Real>,
}

impl BoundaryMomentumContributions {
    fn zeros(n: usize) -> Self {
        Self {
            diagonal: vec![0.0; n],
            rhs_x: vec![0.0; n],
            rhs_y: vec![0.0; n],
            rhs_z: vec![0.0; n],
        }
    }
}

fn boundary_momentum_contributions(
    mesh: &StructuredMesh3d,
    fields: &IncompressibleFields,
    boundary: &BoundarySet,
    config: IncompressibleMomentumPredictorConfig,
) -> Result<BoundaryMomentumContributions> {
    let mut out = BoundaryMomentumContributions::zeros(mesh.num_cells());
    for patch in boundary.patches() {
        for &face in &patch.face_ids {
            let owner = mesh.face_owner(face)?.index() as usize;
            let geom = mesh.face_geometry_3d(face)?;
            let interior = interior_neighbor_index(mesh, face)?;
            add_boundary_face_momentum(
                owner,
                geom,
                &patch.kind,
                fields,
                interior,
                config,
                &mut out,
            );
        }
    }
    Ok(out)
}

fn add_boundary_face_momentum(
    owner: usize,
    geom: FaceGeometry3d,
    kind: &BoundaryKind,
    fields: &IncompressibleFields,
    interior: Option<usize>,
    config: IncompressibleMomentumPredictorConfig,
    out: &mut BoundaryMomentumContributions,
) {
    if matches!(kind, BoundaryKind::Periodic { .. }) {
        return;
    }
    if let Some(velocity) = incompressible_boundary_owner_velocity_target(
        kind,
        [geom.normal.x, geom.normal.y, geom.normal.z],
        fields,
        interior,
    ) {
        let diffusion = config.kinematic_viscosity * geom.area / geom.spacing;
        out.diagonal[owner] += diffusion;
        out.rhs_x[owner] += diffusion * velocity[0];
        out.rhs_y[owner] += diffusion * velocity[1];
        out.rhs_z[owner] += diffusion * velocity[2];
        add_boundary_convection(owner, geom, velocity, Some(velocity), fields, out);
        return;
    }
    if matches!(
        kind,
        BoundaryKind::Symmetry | BoundaryKind::Wall { no_slip: false, .. }
    ) {
        return;
    }
    if is_pressure_outlet(kind) {
        let owner_velocity = cell_velocity(fields, owner);
        add_boundary_convection(owner, geom, owner_velocity, None, fields, out);
    }
}

fn add_boundary_convection(
    owner: usize,
    geom: FaceGeometry3d,
    face_velocity: [Real; 3],
    boundary_value: Option<[Real; 3]>,
    fields: &IncompressibleFields,
    out: &mut BoundaryMomentumContributions,
) {
    let flux = (face_velocity[0] * geom.normal.x
        + face_velocity[1] * geom.normal.y
        + face_velocity[2] * geom.normal.z)
        * geom.area;
    if flux >= 0.0 {
        out.diagonal[owner] += flux;
        return;
    }
    if let Some(value) = boundary_value {
        out.rhs_x[owner] -= flux * value[0];
        out.rhs_y[owner] -= flux * value[1];
        out.rhs_z[owner] -= flux * value[2];
    } else {
        let owner_value = cell_velocity(fields, owner);
        out.diagonal[owner] += flux;
        out.rhs_x[owner] -= flux * owner_value[0];
        out.rhs_y[owner] -= flux * owner_value[1];
        out.rhs_z[owner] -= flux * owner_value[2];
    }
}

fn is_pressure_outlet(kind: &BoundaryKind) -> bool {
    matches!(
        kind,
        BoundaryKind::IncompressiblePressureOutlet { .. } | BoundaryKind::Outlet { .. }
    )
}

fn cell_velocity(fields: &IncompressibleFields, cell: usize) -> [Real; 3] {
    [
        fields.velocity_x.values()[cell],
        fields.velocity_y.values()[cell],
        fields.velocity_z.values()[cell],
    ]
}

fn add_momentum_predictor_neighbors(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    cell: (usize, usize, usize),
    boundary_diagonal: Real,
) -> Real {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    let center = mesh.cell_index(i, j, k);
    let cx = ctx.config.kinematic_viscosity * ctx.spacing.dy * ctx.spacing.dz / ctx.spacing.dx;
    let cy = ctx.config.kinematic_viscosity * ctx.spacing.dx * ctx.spacing.dz / ctx.spacing.dy;
    let cz = ctx.config.kinematic_viscosity * ctx.spacing.dx * ctx.spacing.dy / ctx.spacing.dz;
    let mut diag = ctx.time_coeff + boundary_diagonal;
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(i > 0, || (i - 1, j, k))
            .or_else(|| neighbor_if(ctx.periodic_x && i == 0, || (mesh.nx - 1, j, k))),
        cx,
    );
    add_neighbor_if_present(
        mesh,
        row,
        &mut diag,
        neighbor_if(i + 1 < mesh.nx, || (i + 1, j, k))
            .or_else(|| neighbor_if(ctx.periodic_x && i + 1 == mesh.nx, || (0, j, k))),
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
    add_momentum_convection(ctx, row, &mut diag, cell);
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
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    let face_velocity = |axis: usize, upper: bool| {
        structured_momentum_face_velocity(StructuredMomentumFaceQuery {
            mesh,
            fields: ctx.fields,
            axis,
            cell: (i, j, k),
            upper,
            periodic_x: ctx.periodic_x,
            boundary: ctx.boundary_map,
        })
    };
    let convective_flux = |axis: usize, upper: bool, area: Real| {
        ctx.face_flux
            .and_then(|flux| flux.cell_face_flux(mesh, axis, cell, upper))
            .unwrap_or_else(|| {
                let flux = face_velocity(axis, upper) * area;
                if upper { flux } else { -flux }
            })
    };
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(i + 1 < mesh.nx, || (i + 1, j, k))
            .or_else(|| neighbor_if(ctx.periodic_x && i + 1 == mesh.nx, || (0, j, k))),
        convective_flux(0, true, ctx.spacing.dy * ctx.spacing.dz),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(i > 0, || (i - 1, j, k))
            .or_else(|| neighbor_if(ctx.periodic_x && i == 0, || (mesh.nx - 1, j, k))),
        convective_flux(0, false, ctx.spacing.dy * ctx.spacing.dz),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(j + 1 < mesh.ny, || (i, j + 1, k)),
        convective_flux(1, true, ctx.spacing.dx * ctx.spacing.dz),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(j > 0, || (i, j - 1, k)),
        convective_flux(1, false, ctx.spacing.dx * ctx.spacing.dz),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(k + 1 < mesh.nz, || (i, j, k + 1)),
        convective_flux(2, true, ctx.spacing.dx * ctx.spacing.dy),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_if(k > 0, || (i, j, k - 1)),
        convective_flux(2, false, ctx.spacing.dx * ctx.spacing.dy),
        ctx.config.convection_scheme,
    );
}

fn add_convective_face(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    neighbor: Option<(usize, usize, usize)>,
    flux: Real,
    scheme: IncompressibleConvectionScheme,
) {
    let Some((i, j, k)) = neighbor else {
        return;
    };
    match scheme {
        IncompressibleConvectionScheme::Upwind => {
            if flux >= 0.0 {
                *diag += flux;
            } else {
                row.push((mesh.cell_index(i, j, k), flux));
            }
        }
        IncompressibleConvectionScheme::Central => {
            *diag += 0.5 * flux;
            row.push((mesh.cell_index(i, j, k), 0.5 * flux));
        }
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
    periodic_x: bool,
) -> [Real; 3] {
    [
        central_diff_x(mesh, pressure, i, j, k, spacing.dx, periodic_x),
        central_diff_y(mesh, pressure, i, j, k, spacing.dy),
        central_diff_z(mesh, pressure, i, j, k, spacing.dz),
    ]
}

fn central_diff_x(
    mesh: &StructuredMesh3d,
    values: &[Real],
    i: usize,
    j: usize,
    k: usize,
    dx: Real,
    periodic_x: bool,
) -> Real {
    (cell_value(
        mesh,
        values,
        east_with_periodic(i, mesh.nx, periodic_x),
        j,
        k,
    ) - cell_value(
        mesh,
        values,
        west_with_periodic(i, mesh.nx, periodic_x),
        j,
        k,
    )) / (2.0 * dx)
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

fn west_with_periodic(i: usize, nx: usize, periodic_x: bool) -> usize {
    if periodic_x && i == 0 {
        nx - 1
    } else {
        west(i)
    }
}

fn east_with_periodic(i: usize, nx: usize, periodic_x: bool) -> usize {
    if periodic_x && i + 1 == nx {
        0
    } else {
        east(i, nx)
    }
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
    use crate::boundary::BoundaryPatch;
    use crate::core::approx_eq;
    use crate::mesh::BoundaryMesh;

    #[test]
    fn moving_wall_adds_boundary_diffusion_source() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "j_max",
            mesh.resolve_logical_boundary("j_max").expect("faces"),
            BoundaryKind::MovingWall {
                velocity: [2.0, 0.0, 0.0],
            },
        )]);
        let config = IncompressibleMomentumPredictorConfig::new(0.25, 1.0).expect("config");

        let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
            &mesh, &fields, &boundary, config,
        )
        .expect("system");

        let row = system.matrix.row_entries(0).collect::<Vec<_>>();
        assert_eq!(row, vec![(0, 1.5)]);
        assert!(approx_eq(system.rhs_x[0], 1.0, 1.0e-12));
        assert!(approx_eq(system.rhs_y[0], 0.0, 1.0e-12));
        assert!(approx_eq(
            system.d_coefficient.values()[0],
            2.0 / 3.0,
            1.0e-12
        ));
    }

    #[test]
    fn velocity_inlet_adds_upwind_boundary_convection_source() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("faces"),
            BoundaryKind::IncompressibleVelocityInlet {
                velocity: [1.0, 0.0, 0.0],
            },
        )]);
        let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0).expect("config");

        let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
            &mesh, &fields, &boundary, config,
        )
        .expect("system");

        let row = system.matrix.row_entries(0).collect::<Vec<_>>();
        assert_eq!(row, vec![(0, 1.0)]);
        assert!(approx_eq(system.rhs_x[0], 1.0, 1.0e-12));
        assert!(approx_eq(system.rhs_y[0], 0.0, 1.0e-12));
        assert!(approx_eq(system.rhs_z[0], 0.0, 1.0e-12));
    }

    #[test]
    fn body_force_adds_component_rhs_source() {
        let mesh = StructuredMesh3d::uniform_box("box", 1, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let fields =
            IncompressibleFields::uniform(mesh.num_cells(), 0.0, [0.0, 0.0, 0.0]).expect("fields");
        let config = IncompressibleMomentumPredictorConfig::new(0.0, 1.0)
            .expect("config")
            .with_body_force([2.0, -3.0, 4.0])
            .expect("body force");

        let system = assemble_incompressible_momentum_predictor_with_boundary_3d(
            &mesh,
            &fields,
            &BoundarySet::default(),
            config,
        )
        .expect("system");

        assert!(approx_eq(system.rhs_x[0], 2.0, 1.0e-12));
        assert!(approx_eq(system.rhs_y[0], -3.0, 1.0e-12));
        assert!(approx_eq(system.rhs_z[0], 4.0, 1.0e-12));
    }
}
