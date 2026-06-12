//! 不可压缩结构化 3D 动量预测装配。
//!
//! 理论映射：`docs/theory/incompressible_simplec_piso.md` 式 (8a)–(10)。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::incompressible_bc::{
    incompressible_boundary_owner_velocity_target, interior_neighbor_index,
};
use crate::discretization::incompressible_boundary_flux::{
    IncompressibleBoundaryOwnerMap, interior_face_velocity,
};
use crate::discretization::incompressible_momentum_geometry::{
    owner_neighbor_distance, pressure_gradient, scalar_cross_diffusion_source,
    structured_scalar_gradients,
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
    let n = mesh.num_cells();
    let mut rows = (0..n).map(|_| Vec::with_capacity(7)).collect::<Vec<_>>();
    let mut rhs_x = Vec::with_capacity(n);
    let mut rhs_y = Vec::with_capacity(n);
    let mut rhs_z = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let boundary_terms = boundary_momentum_contributions(mesh, fields, boundary, config)?;
    let periodic_x = boundary.has_periodic_pair("i_min", "i_max");
    let boundary_map = IncompressibleBoundaryOwnerMap::build(mesh, boundary);
    let pressure_gradients =
        structured_scalar_gradients(mesh, fields.pressure.values(), periodic_x);
    let velocity_x_gradients =
        structured_scalar_gradients(mesh, fields.velocity_x.values(), periodic_x);
    let velocity_y_gradients =
        structured_scalar_gradients(mesh, fields.velocity_y.values(), periodic_x);
    let velocity_z_gradients =
        structured_scalar_gradients(mesh, fields.velocity_z.values(), periodic_x);
    let ctx = MomentumAssemblyCtx {
        mesh,
        fields,
        config,
        periodic_x,
        boundary_map: &boundary_map,
        face_flux,
    };
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let row = mesh.cell_index(i, j, k);
                let volume = mesh.cell_metric(i, j, k).volume;
                let time_coeff = volume / config.pseudo_time_step;
                let consistent_coeff = add_momentum_predictor_neighbors(
                    ctx,
                    &mut rows[row],
                    (i, j, k),
                    boundary_terms.diagonal[row],
                    time_coeff,
                );
                let grad_p = pressure_gradient(
                    mesh,
                    fields.pressure.values(),
                    &pressure_gradients,
                    i,
                    j,
                    k,
                    periodic_x,
                );
                let diffusion_source_x = scalar_cross_diffusion_source(
                    mesh,
                    &velocity_x_gradients,
                    (i, j, k),
                    config.kinematic_viscosity,
                    periodic_x,
                );
                let diffusion_source_y = scalar_cross_diffusion_source(
                    mesh,
                    &velocity_y_gradients,
                    (i, j, k),
                    config.kinematic_viscosity,
                    periodic_x,
                );
                let diffusion_source_z = scalar_cross_diffusion_source(
                    mesh,
                    &velocity_z_gradients,
                    (i, j, k),
                    config.kinematic_viscosity,
                    periodic_x,
                );
                let relax_source = momentum_relaxation_source(rows[row].last_mut(), config)?;
                let rhs_cell_x = time_coeff * fields.velocity_x.values()[row] - volume * grad_p[0]
                    + diffusion_source_x
                    + volume * config.body_force[0]
                    + relax_source * fields.velocity_x.values()[row]
                    + boundary_terms.rhs_x[row];
                let rhs_cell_y = time_coeff * fields.velocity_y.values()[row] - volume * grad_p[1]
                    + diffusion_source_y
                    + volume * config.body_force[1]
                    + relax_source * fields.velocity_y.values()[row]
                    + boundary_terms.rhs_y[row];
                let rhs_cell_z = time_coeff * fields.velocity_z.values()[row] - volume * grad_p[2]
                    + diffusion_source_z
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
struct MomentumAssemblyCtx<'a> {
    mesh: &'a StructuredMesh3d,
    fields: &'a IncompressibleFields,
    config: IncompressibleMomentumPredictorConfig,
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
    time_coeff: Real,
) -> Real {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    let center = mesh.cell_index(i, j, k);
    let mut diag = time_coeff + boundary_diagonal;
    add_diffusion_x_neighbors(ctx, row, &mut diag, cell);
    add_diffusion_y_neighbors(ctx, row, &mut diag, cell);
    add_diffusion_z_neighbors(ctx, row, &mut diag, cell);
    add_momentum_convection(ctx, row, &mut diag, cell);
    let consistent_coeff = diag + row.iter().map(|(_, value)| *value).sum::<Real>();
    row.push((center, diag));
    consistent_coeff.max(time_coeff)
}

fn add_diffusion_x_neighbors(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    add_neighbor_if_present(
        mesh,
        row,
        diag,
        neighbor_with_coeff(
            i > 0,
            || (i - 1, j, k),
            || diffusion_coeff_x(ctx, i - 1, j, k),
        )
        .or_else(|| {
            neighbor_with_coeff(
                ctx.periodic_x && i == 0,
                || (mesh.nx - 1, j, k),
                || diffusion_coeff_x(ctx, 0, j, k),
            )
        }),
    );
    add_neighbor_if_present(
        mesh,
        row,
        diag,
        neighbor_with_coeff(
            i + 1 < mesh.nx,
            || (i + 1, j, k),
            || diffusion_coeff_x(ctx, i, j, k),
        )
        .or_else(|| {
            neighbor_with_coeff(
                ctx.periodic_x && i + 1 == mesh.nx,
                || (0, j, k),
                || diffusion_coeff_x(ctx, mesh.nx.saturating_sub(2), j, k),
            )
        }),
    );
}

fn add_diffusion_y_neighbors(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    add_neighbor_if_present(
        mesh,
        row,
        diag,
        neighbor_with_coeff(
            j > 0,
            || (i, j - 1, k),
            || diffusion_coeff_y(ctx, i, j - 1, k),
        ),
    );
    add_neighbor_if_present(
        mesh,
        row,
        diag,
        neighbor_with_coeff(
            j + 1 < mesh.ny,
            || (i, j + 1, k),
            || diffusion_coeff_y(ctx, i, j, k),
        ),
    );
}

fn add_diffusion_z_neighbors(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    add_neighbor_if_present(
        mesh,
        row,
        diag,
        neighbor_with_coeff(
            k > 0,
            || (i, j, k - 1),
            || diffusion_coeff_z(ctx, i, j, k - 1),
        ),
    );
    add_neighbor_if_present(
        mesh,
        row,
        diag,
        neighbor_with_coeff(
            k + 1 < mesh.nz,
            || (i, j, k + 1),
            || diffusion_coeff_z(ctx, i, j, k),
        ),
    );
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

fn diffusion_coeff_x(ctx: MomentumAssemblyCtx<'_>, i: usize, j: usize, k: usize) -> Real {
    let face = ctx.mesh.i_face_metric(i, j, k);
    ctx.config.kinematic_viscosity * face.area
        / owner_neighbor_distance(ctx.mesh, (i, j, k), (i + 1, j, k), &face)
}

fn diffusion_coeff_y(ctx: MomentumAssemblyCtx<'_>, i: usize, j: usize, k: usize) -> Real {
    let face = ctx.mesh.j_face_metric(i, j, k);
    ctx.config.kinematic_viscosity * face.area
        / owner_neighbor_distance(ctx.mesh, (i, j, k), (i, j + 1, k), &face)
}

fn diffusion_coeff_z(ctx: MomentumAssemblyCtx<'_>, i: usize, j: usize, k: usize) -> Real {
    let face = ctx.mesh.k_face_metric(i, j, k);
    ctx.config.kinematic_viscosity * face.area
        / owner_neighbor_distance(ctx.mesh, (i, j, k), (i, j, k + 1), &face)
}

fn add_momentum_convection(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    add_convection_x(ctx, row, diag, cell);
    add_convection_y(ctx, row, diag, cell);
    add_convection_z(ctx, row, diag, cell);
}

fn add_convection_x(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_with_flux(
            i + 1 < mesh.nx,
            || (i + 1, j, k),
            || convective_flux(ctx, cell, 0, true),
        )
        .or_else(|| {
            neighbor_with_flux(
                ctx.periodic_x && i + 1 == mesh.nx,
                || (0, j, k),
                || convective_flux(ctx, cell, 0, true),
            )
        }),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_with_flux(
            i > 0,
            || (i - 1, j, k),
            || convective_flux(ctx, cell, 0, false),
        )
        .or_else(|| {
            neighbor_with_flux(
                ctx.periodic_x && i == 0,
                || (mesh.nx - 1, j, k),
                || convective_flux(ctx, cell, 0, false),
            )
        }),
        ctx.config.convection_scheme,
    );
}

fn add_convection_y(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_with_flux(
            j + 1 < mesh.ny,
            || (i, j + 1, k),
            || convective_flux(ctx, cell, 1, true),
        ),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_with_flux(
            j > 0,
            || (i, j - 1, k),
            || convective_flux(ctx, cell, 1, false),
        ),
        ctx.config.convection_scheme,
    );
}

fn add_convection_z(
    ctx: MomentumAssemblyCtx<'_>,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    cell: (usize, usize, usize),
) {
    let (i, j, k) = cell;
    let mesh = ctx.mesh;
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_with_flux(
            k + 1 < mesh.nz,
            || (i, j, k + 1),
            || convective_flux(ctx, cell, 2, true),
        ),
        ctx.config.convection_scheme,
    );
    add_convective_face(
        mesh,
        row,
        diag,
        neighbor_with_flux(
            k > 0,
            || (i, j, k - 1),
            || convective_flux(ctx, cell, 2, false),
        ),
        ctx.config.convection_scheme,
    );
}

fn convective_flux(
    ctx: MomentumAssemblyCtx<'_>,
    cell: (usize, usize, usize),
    axis: usize,
    upper: bool,
) -> Real {
    ctx.face_flux
        .and_then(|flux| flux.cell_face_flux(ctx.mesh, axis, cell, upper))
        .unwrap_or_else(|| fallback_convective_flux(ctx, cell, axis, upper))
}

fn fallback_convective_flux(
    ctx: MomentumAssemblyCtx<'_>,
    cell: (usize, usize, usize),
    axis: usize,
    upper: bool,
) -> Real {
    let mesh = ctx.mesh;
    let (i, j, k) = cell;
    let (left, right, metric) = match (axis, upper) {
        (0, true) if i + 1 < mesh.nx => (
            mesh.cell_index(i, j, k),
            mesh.cell_index(i + 1, j, k),
            mesh.i_face_metric(i, j, k),
        ),
        (0, true) if ctx.periodic_x && i + 1 == mesh.nx && mesh.nx > 1 => (
            mesh.cell_index(mesh.nx - 1, j, k),
            mesh.cell_index(0, j, k),
            mesh.i_face_metric(mesh.nx - 2, j, k),
        ),
        (0, false) if i > 0 => (
            mesh.cell_index(i - 1, j, k),
            mesh.cell_index(i, j, k),
            mesh.i_face_metric(i - 1, j, k),
        ),
        (0, false) if ctx.periodic_x && i == 0 && mesh.nx > 1 => (
            mesh.cell_index(mesh.nx - 1, j, k),
            mesh.cell_index(0, j, k),
            mesh.i_face_metric(mesh.nx - 2, j, k),
        ),
        (1, true) => (
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j + 1, k),
            mesh.j_face_metric(i, j, k),
        ),
        (1, false) => (
            mesh.cell_index(i, j - 1, k),
            mesh.cell_index(i, j, k),
            mesh.j_face_metric(i, j - 1, k),
        ),
        (2, true) => (
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j, k + 1),
            mesh.k_face_metric(i, j, k),
        ),
        (2, false) => (
            mesh.cell_index(i, j, k - 1),
            mesh.cell_index(i, j, k),
            mesh.k_face_metric(i, j, k - 1),
        ),
        _ => return 0.0,
    };
    let velocity = [
        interior_face_velocity(ctx.fields, left, right, 0, ctx.boundary_map),
        interior_face_velocity(ctx.fields, left, right, 1, ctx.boundary_map),
        interior_face_velocity(ctx.fields, left, right, 2, ctx.boundary_map),
    ];
    let flux = (velocity[0] * metric.normal.x
        + velocity[1] * metric.normal.y
        + velocity[2] * metric.normal.z)
        * metric.area;
    if upper { flux } else { -flux }
}

fn add_convective_face(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    neighbor: Option<((usize, usize, usize), Real)>,
    scheme: IncompressibleConvectionScheme,
) {
    let Some(((i, j, k), flux)) = neighbor else {
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

fn neighbor_with_flux(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
    flux: impl FnOnce() -> Real,
) -> Option<((usize, usize, usize), Real)> {
    present.then(|| (index(), flux()))
}

fn add_neighbor_if_present(
    mesh: &StructuredMesh3d,
    row: &mut Vec<(usize, Real)>,
    diag: &mut Real,
    neighbor: Option<((usize, usize, usize), Real)>,
) {
    if let Some(((i, j, k), coeff)) = neighbor {
        *diag += coeff;
        row.push((mesh.cell_index(i, j, k), -coeff));
    }
}

fn neighbor_with_coeff(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
    coeff: impl FnOnce() -> Real,
) -> Option<((usize, usize, usize), Real)> {
    present.then(|| (index(), coeff()))
}
