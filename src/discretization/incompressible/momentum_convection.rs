//! 不可压缩动量对流装配 helper。

use super::boundary_flux::interior_face_velocity;
use super::momentum::{IncompressibleConvectionScheme, MomentumAssemblyCtx};
use crate::core::Real;
use crate::mesh::StructuredMesh3d;

pub(super) fn add_momentum_convection(
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
                ctx.periodic.x && i + 1 == mesh.nx,
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
                ctx.periodic.x && i == 0,
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
        )
        .or_else(|| {
            neighbor_with_flux(
                ctx.periodic.y && j + 1 == mesh.ny,
                || (i, 0, k),
                || convective_flux(ctx, cell, 1, true),
            )
        }),
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
        )
        .or_else(|| {
            neighbor_with_flux(
                ctx.periodic.y && j == 0,
                || (i, mesh.ny - 1, k),
                || convective_flux(ctx, cell, 1, false),
            )
        }),
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
    let Some((left, right, metric)) = convective_face_geometry(ctx, cell, axis, upper) else {
        return 0.0;
    };
    let velocity = [
        interior_face_velocity(ctx.fields, left, right, 0),
        interior_face_velocity(ctx.fields, left, right, 1),
        interior_face_velocity(ctx.fields, left, right, 2),
    ];
    let flux = (velocity[0] * metric.normal.x
        + velocity[1] * metric.normal.y
        + velocity[2] * metric.normal.z)
        * metric.area;
    if upper { flux } else { -flux }
}

fn convective_face_geometry(
    ctx: MomentumAssemblyCtx<'_>,
    cell: (usize, usize, usize),
    axis: usize,
    upper: bool,
) -> Option<(usize, usize, crate::mesh::FaceMetric)> {
    let mesh = ctx.mesh;
    let (i, j, k) = cell;
    match axis {
        0 => convective_x_face_geometry(mesh, ctx.periodic.x, i, j, k, upper),
        1 => convective_y_face_geometry(mesh, ctx.periodic.y, i, j, k, upper),
        2 => convective_z_face_geometry(mesh, i, j, k, upper),
        _ => None,
    }
}

fn convective_x_face_geometry(
    mesh: &StructuredMesh3d,
    periodic_x: bool,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
) -> Option<(usize, usize, crate::mesh::FaceMetric)> {
    if upper {
        if i + 1 < mesh.nx {
            return Some((
                mesh.cell_index(i, j, k),
                mesh.cell_index(i + 1, j, k),
                mesh.i_face_metric(i, j, k),
            ));
        }
        if periodic_x && mesh.nx > 1 {
            return Some((
                mesh.cell_index(mesh.nx - 1, j, k),
                mesh.cell_index(0, j, k),
                mesh.i_face_metric(mesh.nx - 2, j, k),
            ));
        }
        return None;
    }
    if i > 0 {
        return Some((
            mesh.cell_index(i - 1, j, k),
            mesh.cell_index(i, j, k),
            mesh.i_face_metric(i - 1, j, k),
        ));
    }
    if periodic_x && mesh.nx > 1 {
        return Some((
            mesh.cell_index(mesh.nx - 1, j, k),
            mesh.cell_index(0, j, k),
            mesh.i_face_metric(mesh.nx - 2, j, k),
        ));
    }
    None
}

fn convective_y_face_geometry(
    mesh: &StructuredMesh3d,
    periodic_y: bool,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
) -> Option<(usize, usize, crate::mesh::FaceMetric)> {
    if upper {
        if j + 1 < mesh.ny {
            return Some((
                mesh.cell_index(i, j, k),
                mesh.cell_index(i, j + 1, k),
                mesh.j_face_metric(i, j, k),
            ));
        }
        if periodic_y && mesh.ny > 1 {
            return Some((
                mesh.cell_index(i, mesh.ny - 1, k),
                mesh.cell_index(i, 0, k),
                mesh.j_face_metric(i, mesh.ny - 2, k),
            ));
        }
        return None;
    }
    if j > 0 {
        return Some((
            mesh.cell_index(i, j - 1, k),
            mesh.cell_index(i, j, k),
            mesh.j_face_metric(i, j - 1, k),
        ));
    }
    if periodic_y && mesh.ny > 1 {
        return Some((
            mesh.cell_index(i, mesh.ny - 1, k),
            mesh.cell_index(i, 0, k),
            mesh.j_face_metric(i, mesh.ny - 2, k),
        ));
    }
    None
}

fn convective_z_face_geometry(
    mesh: &StructuredMesh3d,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
) -> Option<(usize, usize, crate::mesh::FaceMetric)> {
    Some(match (upper, k + 1 < mesh.nz, k > 0) {
        (true, true, _) => (
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j, k + 1),
            mesh.k_face_metric(i, j, k),
        ),
        (false, _, true) => (
            mesh.cell_index(i, j, k - 1),
            mesh.cell_index(i, j, k),
            mesh.k_face_metric(i, j, k - 1),
        ),
        _ => return None,
    })
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
