//! 不可压缩动量预测的结构化局部几何 helper。

use crate::core::{Real, Vector3};
use crate::discretization::gradient::compute_structured_scalar_gradients_3d;
use crate::mesh::{LogicalFace3d, StructuredMesh3d};

pub(crate) fn owner_neighbor_distance(
    mesh: &StructuredMesh3d,
    owner: (usize, usize, usize),
    neighbor: (usize, usize, usize),
    face: &crate::mesh::FaceMetric,
) -> Real {
    let owner_center = mesh.cell_metric(owner.0, owner.1, owner.2).center;
    let neighbor_center = mesh.cell_metric(neighbor.0, neighbor.1, neighbor.2).center;
    let dx = neighbor_center.x - owner_center.x;
    let dy = neighbor_center.y - owner_center.y;
    let dz = neighbor_center.z - owner_center.z;
    (dx * face.normal.x + dy * face.normal.y + dz * face.normal.z)
        .abs()
        .max(Real::EPSILON)
}

pub(crate) fn structured_scalar_gradients(
    mesh: &StructuredMesh3d,
    values: &[Real],
    periodic_x: bool,
) -> Vec<Vector3> {
    compute_structured_scalar_gradients_3d(mesh, values, periodic_x)
}

pub(crate) fn pressure_gradient(
    mesh: &StructuredMesh3d,
    pressure: &[Real],
    gradients: &[Vector3],
    i: usize,
    j: usize,
    k: usize,
    periodic_x: bool,
) -> [Real; 3] {
    let center = mesh.cell_index(i, j, k);
    let p_center = pressure[center];
    let mut sum = Vector3::new(0.0, 0.0, 0.0);
    let ctx = PressureFaceCtx {
        mesh,
        pressure,
        gradients,
        periodic_x,
    };
    let cell = CellCoord { i, j, k };
    add_i_faces(ctx, &mut sum, cell, p_center);
    add_j_faces(ctx, &mut sum, cell, p_center);
    add_k_faces(ctx, &mut sum, cell, p_center);
    let volume = mesh.cell_metric(i, j, k).volume;
    [sum.x / volume, sum.y / volume, sum.z / volume]
}

pub(crate) fn scalar_cross_diffusion_source(
    mesh: &StructuredMesh3d,
    gradients: &[Vector3],
    cell: (usize, usize, usize),
    diffusivity: Real,
    periodic_x: bool,
) -> Real {
    let mut source = 0.0;
    let ctx = CrossDiffusionCtx {
        mesh,
        gradients,
        diffusivity,
        periodic_x,
    };
    let cell = CellCoord::from_tuple(cell);
    add_cross_x_faces(ctx, cell, &mut source);
    add_cross_y_faces(ctx, cell, &mut source);
    add_cross_z_faces(ctx, cell, &mut source);
    source
}

#[derive(Debug, Clone, Copy)]
struct PressureFaceCtx<'a> {
    mesh: &'a StructuredMesh3d,
    pressure: &'a [Real],
    gradients: &'a [Vector3],
    periodic_x: bool,
}

#[derive(Debug, Clone, Copy)]
struct CrossDiffusionCtx<'a> {
    mesh: &'a StructuredMesh3d,
    gradients: &'a [Vector3],
    diffusivity: Real,
    periodic_x: bool,
}

#[derive(Debug, Clone, Copy)]
struct CellCoord {
    i: usize,
    j: usize,
    k: usize,
}

impl CellCoord {
    fn from_tuple(value: (usize, usize, usize)) -> Self {
        Self {
            i: value.0,
            j: value.1,
            k: value.2,
        }
    }
}

fn add_i_faces(ctx: PressureFaceCtx<'_>, sum: &mut Vector3, cell: CellCoord, p_center: Real) {
    let mesh = ctx.mesh;
    let CellCoord { i, j, k } = cell;
    if i + 1 < mesh.nx {
        let face = mesh.i_face_metric(i, j, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(i + 1, j, k),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, p_face);
    } else if ctx.periodic_x && mesh.nx > 1 {
        let face = mesh.i_face_metric(mesh.nx - 2, j, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(0, j, k),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, p_face);
    } else {
        add_boundary_pressure(mesh, sum, LogicalFace3d::IMax, i, j, k, p_center);
    }
    if i > 0 {
        let face = mesh.i_face_metric(i - 1, j, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(i - 1, j, k),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, -p_face);
    } else if ctx.periodic_x && mesh.nx > 1 {
        let face = mesh.i_face_metric(mesh.nx - 2, j, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(mesh.nx - 1, j, k),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, -p_face);
    } else {
        add_boundary_pressure(mesh, sum, LogicalFace3d::IMin, i, j, k, p_center);
    }
}

fn add_j_faces(ctx: PressureFaceCtx<'_>, sum: &mut Vector3, cell: CellCoord, p_center: Real) {
    let mesh = ctx.mesh;
    let CellCoord { i, j, k } = cell;
    if j + 1 < mesh.ny {
        let face = mesh.j_face_metric(i, j, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j + 1, k),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, p_face);
    } else {
        add_boundary_pressure(mesh, sum, LogicalFace3d::JMax, i, j, k, p_center);
    }
    if j > 0 {
        let face = mesh.j_face_metric(i, j - 1, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j - 1, k),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, -p_face);
    } else {
        add_boundary_pressure(mesh, sum, LogicalFace3d::JMin, i, j, k, p_center);
    }
}

fn add_k_faces(ctx: PressureFaceCtx<'_>, sum: &mut Vector3, cell: CellCoord, p_center: Real) {
    let mesh = ctx.mesh;
    let CellCoord { i, j, k } = cell;
    if k + 1 < mesh.nz {
        let face = mesh.k_face_metric(i, j, k);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j, k + 1),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, p_face);
    } else {
        add_boundary_pressure(mesh, sum, LogicalFace3d::KMax, i, j, k, p_center);
    }
    if k > 0 {
        let face = mesh.k_face_metric(i, j, k - 1);
        let p_face = reconstructed_face_value(
            mesh,
            ctx.pressure,
            ctx.gradients,
            mesh.cell_index(i, j, k),
            mesh.cell_index(i, j, k - 1),
            face.center,
        );
        add_area_pressure(sum, face.area_vector, -p_face);
    } else {
        add_boundary_pressure(mesh, sum, LogicalFace3d::KMin, i, j, k, p_center);
    }
}

fn add_boundary_pressure(
    mesh: &StructuredMesh3d,
    sum: &mut Vector3,
    face: LogicalFace3d,
    i: usize,
    j: usize,
    k: usize,
    pressure: Real,
) {
    add_area_pressure(
        sum,
        mesh.boundary_face_metric(face, i, j, k).area_vector,
        pressure,
    );
}

fn add_area_pressure(sum: &mut Vector3, area_vector: Vector3, pressure: Real) {
    sum.x += pressure * area_vector.x;
    sum.y += pressure * area_vector.y;
    sum.z += pressure * area_vector.z;
}

fn reconstructed_face_value(
    mesh: &StructuredMesh3d,
    values: &[Real],
    gradients: &[Vector3],
    owner: usize,
    neighbor: usize,
    face_center: Vector3,
) -> Real {
    0.5 * (reconstruct_cell_value(mesh, values, gradients, owner, face_center)
        + reconstruct_cell_value(mesh, values, gradients, neighbor, face_center))
}

fn reconstruct_cell_value(
    mesh: &StructuredMesh3d,
    values: &[Real],
    gradients: &[Vector3],
    cell: usize,
    face_center: Vector3,
) -> Real {
    let (i, j, k) = cell_ijk(mesh, cell);
    let center = mesh.cell_metric(i, j, k).center;
    let delta = vec_sub(face_center, center);
    values[cell] + dot(gradients[cell], delta)
}

fn add_cross_x_faces(ctx: CrossDiffusionCtx<'_>, cell: CellCoord, source: &mut Real) {
    let mesh = ctx.mesh;
    let CellCoord { i, j, k } = cell;
    if i + 1 < mesh.nx {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i: i + 1, j, k },
            mesh.i_face_metric(i, j, k),
            source,
        );
    } else if ctx.periodic_x && mesh.nx > 1 {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i: 0, j, k },
            mesh.i_face_metric(mesh.nx - 2, j, k),
            source,
        );
    }
    if i > 0 {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i: i - 1, j, k },
            reverse_face(mesh.i_face_metric(i - 1, j, k)),
            source,
        );
    } else if ctx.periodic_x && mesh.nx > 1 {
        add_cross_face(
            ctx,
            cell,
            CellCoord {
                i: mesh.nx - 1,
                j,
                k,
            },
            reverse_face(mesh.i_face_metric(mesh.nx - 2, j, k)),
            source,
        );
    }
}

fn add_cross_y_faces(ctx: CrossDiffusionCtx<'_>, cell: CellCoord, source: &mut Real) {
    let mesh = ctx.mesh;
    let CellCoord { i, j, k } = cell;
    if j + 1 < mesh.ny {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i, j: j + 1, k },
            mesh.j_face_metric(i, j, k),
            source,
        );
    }
    if j > 0 {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i, j: j - 1, k },
            reverse_face(mesh.j_face_metric(i, j - 1, k)),
            source,
        );
    }
}

fn add_cross_z_faces(ctx: CrossDiffusionCtx<'_>, cell: CellCoord, source: &mut Real) {
    let mesh = ctx.mesh;
    let CellCoord { i, j, k } = cell;
    if k + 1 < mesh.nz {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i, j, k: k + 1 },
            mesh.k_face_metric(i, j, k),
            source,
        );
    }
    if k > 0 {
        add_cross_face(
            ctx,
            cell,
            CellCoord { i, j, k: k - 1 },
            reverse_face(mesh.k_face_metric(i, j, k - 1)),
            source,
        );
    }
}

fn add_cross_face(
    ctx: CrossDiffusionCtx<'_>,
    owner: CellCoord,
    neighbor: CellCoord,
    face: crate::mesh::FaceMetric,
    source: &mut Real,
) {
    let mesh = ctx.mesh;
    let owner_idx = mesh.cell_index(owner.i, owner.j, owner.k);
    let neighbor_idx = mesh.cell_index(neighbor.i, neighbor.j, neighbor.k);
    let owner_center = mesh.cell_metric(owner.i, owner.j, owner.k).center;
    let neighbor_center = mesh.cell_metric(neighbor.i, neighbor.j, neighbor.k).center;
    let d = vec_sub(neighbor_center, owner_center);
    let denom = dot(face.area_vector, d);
    if denom.abs() <= Real::EPSILON {
        return;
    }
    let orthogonal = scale(d, dot(face.area_vector, face.area_vector) / denom);
    let correction = vec_sub(face.area_vector, orthogonal);
    let grad = average(ctx.gradients[owner_idx], ctx.gradients[neighbor_idx]);
    *source += ctx.diffusivity * dot(grad, correction);
}

fn reverse_face(mut face: crate::mesh::FaceMetric) -> crate::mesh::FaceMetric {
    face.area_vector = scale(face.area_vector, -1.0);
    face.normal = scale(face.normal, -1.0);
    face
}

fn average(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z))
}

fn scale(v: Vector3, value: Real) -> Vector3 {
    Vector3::new(v.x * value, v.y * value, v.z * value)
}

fn dot(a: Vector3, b: Vector3) -> Real {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn cell_ijk(mesh: &StructuredMesh3d, cell: usize) -> (usize, usize, usize) {
    let cells_per_layer = mesh.nx * mesh.ny;
    let k = cell / cells_per_layer;
    let rem = cell % cells_per_layer;
    let j = rem / mesh.nx;
    let i = rem % mesh.nx;
    (i, j, k)
}
