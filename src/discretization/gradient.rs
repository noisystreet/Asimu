//! 结构化网格单元中心梯度（有限差分）。
//!
//! 理论：[`docs/theory/structured_gradients.md`](../../docs/theory/structured_gradients.md)

#![allow(clippy::too_many_arguments)]

pub use crate::discretization::gradient_typed::{
    GradientFields, GradientFieldsT, InviscidPrimitiveGradients, InviscidPrimitiveGradientsT,
    VelocityGradient, VelocityGradientSlices, VelocityGradientSlicesT, VelocityGradientT,
};

use crate::boundary::BoundarySet;
use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::{LogicalFace3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

/// 结构化网格有限差分梯度。
///
/// 在每个单元沿逻辑 i/j/k 方向构造物理空间差分：
/// \(\Delta \phi_m = \Delta \mathbf{x}_m\cdot\nabla\phi\)，再解 3x3 系统得到笛卡尔梯度。
pub fn compute_structured_gradients_3d(
    mesh: &StructuredMesh3d,
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    boundaries: &BoundarySet,
    ghosts: &BoundaryGhostBuffer,
    min_pressure: Real,
    viscous: Option<&crate::physics::ViscousPhysicsConfig>,
    out: &mut GradientFields,
) -> Result<()> {
    let n = mesh.num_cells();
    if primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    out.clear();
    let temperatures = cell_temperatures(primitives, eos, viscous)?;
    let ctx = DifferenceGradientContext {
        mesh,
        primitives,
        eos,
        boundaries,
        ghosts,
        min_pressure,
        viscous,
        temperatures: &temperatures,
    };
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let cell = mesh.cell_index(i, j, k);
                let di = difference_along_axis(&ctx, i, j, k, Axis3d::I)?;
                let dj = difference_along_axis(&ctx, i, j, k, Axis3d::J)?;
                let dk = difference_along_axis(&ctx, i, j, k, Axis3d::K)?;
                write_cell_gradient(out, cell, di, dj, dk)?;
            }
        }
    }
    Ok(())
}

/// 结构化网格任意 cell-centered 标量的物理空间梯度。
///
/// 与可压缩结构梯度使用同一类局部物理差分假设：用相邻单元中心构造
/// \(\Delta\phi=\Delta\mathbf{x}\cdot\nabla\phi\)，再由局部最小二乘正规方程
/// 得到 Cartesian 梯度。边界缺失方向自然退化为单侧邻接；准二维网格的缺失
/// 方向返回零梯度分量。
pub(crate) fn compute_structured_scalar_gradients_3d(
    mesh: &StructuredMesh3d,
    values: &[Real],
    periodic_x: bool,
) -> Vec<Vector3> {
    let mut gradients = Vec::with_capacity(mesh.num_cells());
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                gradients.push(scalar_cell_lsq_gradient(mesh, values, i, j, k, periodic_x));
            }
        }
    }
    gradients
}

fn scalar_cell_lsq_gradient(
    mesh: &StructuredMesh3d,
    values: &[Real],
    i: usize,
    j: usize,
    k: usize,
    periodic_x: bool,
) -> Vector3 {
    let center = mesh.cell_index(i, j, k);
    let center_point = mesh.cell_metric(i, j, k).center;
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    accumulate_scalar_lsq_neighbor(
        mesh,
        values,
        center,
        center_point,
        scalar_neighbor_i(mesh, i, j, k, false, periodic_x),
        &mut normal,
        &mut rhs,
    );
    accumulate_scalar_lsq_neighbor(
        mesh,
        values,
        center,
        center_point,
        scalar_neighbor_i(mesh, i, j, k, true, periodic_x),
        &mut normal,
        &mut rhs,
    );
    accumulate_scalar_lsq_neighbor(
        mesh,
        values,
        center,
        center_point,
        scalar_neighbor(j > 0, || (i, j - 1, k)),
        &mut normal,
        &mut rhs,
    );
    accumulate_scalar_lsq_neighbor(
        mesh,
        values,
        center,
        center_point,
        scalar_neighbor(j + 1 < mesh.ny, || (i, j + 1, k)),
        &mut normal,
        &mut rhs,
    );
    accumulate_scalar_lsq_neighbor(
        mesh,
        values,
        center,
        center_point,
        scalar_neighbor(k > 0, || (i, j, k - 1)),
        &mut normal,
        &mut rhs,
    );
    accumulate_scalar_lsq_neighbor(
        mesh,
        values,
        center,
        center_point,
        scalar_neighbor(k + 1 < mesh.nz, || (i, j, k + 1)),
        &mut normal,
        &mut rhs,
    );
    solve_regularized_3x3(normal, rhs)
}

fn scalar_neighbor(
    present: bool,
    index: impl FnOnce() -> (usize, usize, usize),
) -> Option<(usize, usize, usize)> {
    present.then(index)
}

fn scalar_neighbor_i(
    mesh: &StructuredMesh3d,
    i: usize,
    j: usize,
    k: usize,
    upper: bool,
    periodic_x: bool,
) -> Option<(usize, usize, usize)> {
    match (upper, i) {
        (false, 0) if periodic_x && mesh.nx > 1 => Some((mesh.nx - 1, j, k)),
        (false, 0) => None,
        (false, _) => Some((i - 1, j, k)),
        (true, _) if i + 1 < mesh.nx => Some((i + 1, j, k)),
        (true, _) if periodic_x && mesh.nx > 1 => Some((0, j, k)),
        _ => None,
    }
}

fn accumulate_scalar_lsq_neighbor(
    mesh: &StructuredMesh3d,
    values: &[Real],
    center: usize,
    center_point: Vector3,
    neighbor: Option<(usize, usize, usize)>,
    normal: &mut [[Real; 3]; 3],
    rhs: &mut [Real; 3],
) {
    let Some((i, j, k)) = neighbor else {
        return;
    };
    let neighbor_idx = mesh.cell_index(i, j, k);
    let delta = vec_sub(mesh.cell_metric(i, j, k).center, center_point);
    let dphi = values[neighbor_idx] - values[center];
    let r = [delta.x, delta.y, delta.z];
    for a in 0..3 {
        rhs[a] += r[a] * dphi;
        for b in 0..3 {
            normal[a][b] += r[a] * r[b];
        }
    }
}

fn solve_regularized_3x3(mut matrix: [[Real; 3]; 3], mut rhs: [Real; 3]) -> Vector3 {
    let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    if trace.abs() <= Real::EPSILON {
        return Vector3::new(0.0, 0.0, 0.0);
    }
    let lambda = trace.abs() * 1.0e-14;
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] += lambda;
    }
    for pivot in 0..3 {
        let mut best = pivot;
        for row in (pivot + 1)..3 {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if matrix[best][pivot].abs() <= Real::EPSILON {
            continue;
        }
        if best != pivot {
            matrix.swap(pivot, best);
            rhs.swap(pivot, best);
        }
        let inv = 1.0 / matrix[pivot][pivot];
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value *= inv;
        }
        rhs[pivot] *= inv;
        let pivot_row = matrix[pivot];
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for (value, pivot_value) in matrix[row].iter_mut().zip(pivot_row.iter()).skip(pivot) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Vector3::new(rhs[0], rhs[1], rhs[2])
}

struct DifferenceGradientContext<'a> {
    mesh: &'a StructuredMesh3d,
    primitives: &'a PrimitiveFields,
    eos: &'a IdealGasEoS,
    boundaries: &'a BoundarySet,
    ghosts: &'a BoundaryGhostBuffer,
    min_pressure: Real,
    viscous: Option<&'a crate::physics::ViscousPhysicsConfig>,
    temperatures: &'a [Real],
}

#[derive(Clone, Copy)]
enum Axis3d {
    I,
    J,
    K,
}

#[derive(Clone, Copy)]
struct ScalarSample {
    point: Vector3,
    u: Real,
    v: Real,
    w: Real,
    t: Real,
}

pub(crate) fn cell_temperatures(
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    viscous: Option<&crate::physics::ViscousPhysicsConfig>,
) -> Result<Vec<Real>> {
    let n = primitives.num_cells();
    let mut t = vec![0.0; n];
    cell_temperatures_into(primitives, eos, viscous, &mut t)?;
    Ok(t)
}

pub(crate) fn cell_temperatures_into(
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    viscous: Option<&crate::physics::ViscousPhysicsConfig>,
    out: &mut Vec<Real>,
) -> Result<()> {
    let n = primitives.num_cells();
    out.resize(n, 0.0);
    for (i, ti) in out.iter_mut().enumerate().take(n) {
        let rho = primitives.density.values()[i];
        let p = primitives.pressure.values()[i];
        if rho <= 0.0 || p <= 0.0 {
            return Err(AsimuError::Field(format!(
                "单元 {i} 密度或压力非正，无法计算温度: rho={rho:.6e}, p={p:.6e}"
            )));
        }
        *ti = viscous
            .map(|v| v.static_temperature(p, rho, eos))
            .unwrap_or(p / (rho * eos.gas_constant));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AxisDifference {
    dr: Vector3,
    du: Real,
    dv: Real,
    dw: Real,
    dt: Real,
}

fn difference_along_axis(
    ctx: &DifferenceGradientContext<'_>,
    i: usize,
    j: usize,
    k: usize,
    axis: Axis3d,
) -> Result<AxisDifference> {
    let owner = cell_sample(ctx, i, j, k);
    let lower = axis_sample(ctx, i, j, k, axis, false)?;
    let upper = axis_sample(ctx, i, j, k, axis, true)?;
    match (lower, upper) {
        (Some(lo), Some(hi)) => Ok(sample_difference(lo, hi)),
        (None, Some(hi)) => Ok(sample_difference(owner, hi)),
        (Some(lo), None) => Ok(sample_difference(lo, owner)),
        (None, None) => Err(AsimuError::Mesh(format!(
            "单元 ({i},{j},{k}) 缺少 {} 方向差分样本",
            axis_label(axis)
        ))),
    }
}

fn axis_sample(
    ctx: &DifferenceGradientContext<'_>,
    i: usize,
    j: usize,
    k: usize,
    axis: Axis3d,
    positive: bool,
) -> Result<Option<ScalarSample>> {
    let mesh = ctx.mesh;
    match (axis, positive) {
        (Axis3d::I, false) if i > 0 => Ok(Some(cell_sample(ctx, i - 1, j, k))),
        (Axis3d::I, true) if i + 1 < mesh.nx => Ok(Some(cell_sample(ctx, i + 1, j, k))),
        (Axis3d::J, false) if j > 0 => Ok(Some(cell_sample(ctx, i, j - 1, k))),
        (Axis3d::J, true) if j + 1 < mesh.ny => Ok(Some(cell_sample(ctx, i, j + 1, k))),
        (Axis3d::K, false) if k > 0 => Ok(Some(cell_sample(ctx, i, j, k - 1))),
        (Axis3d::K, true) if k + 1 < mesh.nz => Ok(Some(cell_sample(ctx, i, j, k + 1))),
        _ => boundary_sample(ctx, i, j, k, boundary_face(axis, positive)),
    }
}

fn cell_sample(ctx: &DifferenceGradientContext<'_>, i: usize, j: usize, k: usize) -> ScalarSample {
    let cell = ctx.mesh.cell_index(i, j, k);
    let prim = ctx.primitives;
    ScalarSample {
        point: ctx.mesh.cell_metric(i, j, k).center,
        u: prim.velocity_x.values()[cell],
        v: prim.velocity_y.values()[cell],
        w: prim.velocity_z.values()[cell],
        t: ctx.temperatures[cell],
    }
}

fn boundary_sample(
    ctx: &DifferenceGradientContext<'_>,
    i: usize,
    j: usize,
    k: usize,
    face: LogicalFace3d,
) -> Result<Option<ScalarSample>> {
    let face_id = face.encode(boundary_local_index(ctx.mesh, face, i, j, k) as u32);
    if !ctx
        .boundaries
        .patches()
        .iter()
        .any(|patch| patch.face_ids.contains(&face_id))
    {
        return Ok(None);
    }
    let Some(ghost) = ctx.ghosts.get_face(face_id) else {
        return Ok(None);
    };
    let owner_center = ctx.mesh.cell_metric(i, j, k).center;
    let face_metric = ctx.mesh.boundary_face_metric(face, i, j, k);
    let ghost_prim = primitive_from_conserved_relaxed(ctx.eos, &ghost.conserved, ctx.min_pressure)?;
    let t_ghost = ctx
        .viscous
        .map(|v| v.static_temperature(ghost_prim.pressure, ghost_prim.density, ctx.eos))
        .unwrap_or(ghost_prim.pressure / (ghost_prim.density * ctx.eos.gas_constant));
    Ok(Some(ScalarSample {
        point: mirror_point(face_metric.center, owner_center),
        u: ghost_prim.velocity[0],
        v: ghost_prim.velocity[1],
        w: ghost_prim.velocity[2],
        t: t_ghost,
    }))
}

fn boundary_local_index(
    mesh: &StructuredMesh3d,
    face: LogicalFace3d,
    i: usize,
    j: usize,
    k: usize,
) -> usize {
    match face {
        LogicalFace3d::IMin | LogicalFace3d::IMax => j + k * mesh.ny,
        LogicalFace3d::JMin | LogicalFace3d::JMax => i + k * mesh.nx,
        LogicalFace3d::KMin | LogicalFace3d::KMax => i + j * mesh.nx,
    }
}

fn boundary_face(axis: Axis3d, positive: bool) -> LogicalFace3d {
    match (axis, positive) {
        (Axis3d::I, false) => LogicalFace3d::IMin,
        (Axis3d::I, true) => LogicalFace3d::IMax,
        (Axis3d::J, false) => LogicalFace3d::JMin,
        (Axis3d::J, true) => LogicalFace3d::JMax,
        (Axis3d::K, false) => LogicalFace3d::KMin,
        (Axis3d::K, true) => LogicalFace3d::KMax,
    }
}

fn axis_label(axis: Axis3d) -> &'static str {
    match axis {
        Axis3d::I => "i",
        Axis3d::J => "j",
        Axis3d::K => "k",
    }
}

fn sample_difference(lower: ScalarSample, upper: ScalarSample) -> AxisDifference {
    AxisDifference {
        dr: vec_sub(upper.point, lower.point),
        du: upper.u - lower.u,
        dv: upper.v - lower.v,
        dw: upper.w - lower.w,
        dt: upper.t - lower.t,
    }
}

fn mirror_point(face_center: Vector3, owner_center: Vector3) -> Vector3 {
    Vector3::new(
        2.0 * face_center.x - owner_center.x,
        2.0 * face_center.y - owner_center.y,
        2.0 * face_center.z - owner_center.z,
    )
}

fn write_cell_gradient(
    out: &mut GradientFields,
    cell: usize,
    di: AxisDifference,
    dj: AxisDifference,
    dk: AxisDifference,
) -> Result<()> {
    let du = solve_physical_gradient(di.dr, dj.dr, dk.dr, [di.du, dj.du, dk.du])?;
    let dv = solve_physical_gradient(di.dr, dj.dr, dk.dr, [di.dv, dj.dv, dk.dv])?;
    let dw = solve_physical_gradient(di.dr, dj.dr, dk.dr, [di.dw, dj.dw, dk.dw])?;
    let dt = solve_physical_gradient(di.dr, dj.dr, dk.dr, [di.dt, dj.dt, dk.dt])?;
    out.du_dx.values_mut()[cell] = du.x;
    out.du_dy.values_mut()[cell] = du.y;
    out.du_dz.values_mut()[cell] = du.z;
    out.dv_dx.values_mut()[cell] = dv.x;
    out.dv_dy.values_mut()[cell] = dv.y;
    out.dv_dz.values_mut()[cell] = dv.z;
    out.dw_dx.values_mut()[cell] = dw.x;
    out.dw_dy.values_mut()[cell] = dw.y;
    out.dw_dz.values_mut()[cell] = dw.z;
    out.dt_dx.values_mut()[cell] = dt.x;
    out.dt_dy.values_mut()[cell] = dt.y;
    out.dt_dz.values_mut()[cell] = dt.z;
    Ok(())
}

fn solve_physical_gradient(
    r1: Vector3,
    r2: Vector3,
    r3: Vector3,
    rhs: [Real; 3],
) -> Result<Vector3> {
    let c23 = vec_cross(r2, r3);
    let c31 = vec_cross(r3, r1);
    let c12 = vec_cross(r1, r2);
    let det = vec_dot(r1, c23);
    if det.abs() <= Real::EPSILON {
        return Err(AsimuError::Mesh(
            "结构网格差分方向退化，无法求梯度".to_string(),
        ));
    }
    let inv_det = 1.0 / det;
    Ok(Vector3::new(
        (rhs[0] * c23.x + rhs[1] * c31.x + rhs[2] * c12.x) * inv_det,
        (rhs[0] * c23.y + rhs[1] * c31.y + rhs[2] * c12.y) * inv_det,
        (rhs[0] * c23.z + rhs[1] * c31.z + rhs[2] * c12.z) * inv_det,
    ))
}

fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn vec_cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn vec_dot(a: Vector3, b: Vector3) -> Real {
    a.x * b.x + a.y * b.y + a.z * b.z
}
