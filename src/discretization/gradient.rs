//! 结构化网格单元中心梯度（有限差分）。
//!
//! 理论：[`docs/theory/structured_gradients.md`](../../docs/theory/structured_gradients.md)

#![allow(clippy::too_many_arguments)]

use crate::boundary::BoundarySet;
use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, ScalarField, primitive_from_conserved_relaxed};
use crate::mesh::{LogicalFace3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

/// 速度分量与温度的单元中心梯度（SoA）。
#[derive(Debug, Clone, PartialEq)]
pub struct GradientFields {
    pub du_dx: ScalarField,
    pub du_dy: ScalarField,
    pub du_dz: ScalarField,
    pub dv_dx: ScalarField,
    pub dv_dy: ScalarField,
    pub dv_dz: ScalarField,
    pub dw_dx: ScalarField,
    pub dw_dy: ScalarField,
    pub dw_dz: ScalarField,
    pub dt_dx: ScalarField,
    pub dt_dy: ScalarField,
    pub dt_dz: ScalarField,
}

impl GradientFields {
    pub fn zeros(num_cells: usize) -> Result<Self> {
        Ok(Self {
            du_dx: ScalarField::uniform(num_cells, 0.0)?,
            du_dy: ScalarField::uniform(num_cells, 0.0)?,
            du_dz: ScalarField::uniform(num_cells, 0.0)?,
            dv_dx: ScalarField::uniform(num_cells, 0.0)?,
            dv_dy: ScalarField::uniform(num_cells, 0.0)?,
            dv_dz: ScalarField::uniform(num_cells, 0.0)?,
            dw_dx: ScalarField::uniform(num_cells, 0.0)?,
            dw_dy: ScalarField::uniform(num_cells, 0.0)?,
            dw_dz: ScalarField::uniform(num_cells, 0.0)?,
            dt_dx: ScalarField::uniform(num_cells, 0.0)?,
            dt_dy: ScalarField::uniform(num_cells, 0.0)?,
            dt_dz: ScalarField::uniform(num_cells, 0.0)?,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.du_dx.len()
    }

    #[must_use]
    pub fn velocity_grad_at(&self, cell: usize) -> VelocityGradient {
        VelocityGradient {
            du: [
                self.du_dx.values()[cell],
                self.du_dy.values()[cell],
                self.du_dz.values()[cell],
            ],
            dv: [
                self.dv_dx.values()[cell],
                self.dv_dy.values()[cell],
                self.dv_dz.values()[cell],
            ],
            dw: [
                self.dw_dx.values()[cell],
                self.dw_dy.values()[cell],
                self.dw_dz.values()[cell],
            ],
            dt: [
                self.dt_dx.values()[cell],
                self.dt_dy.values()[cell],
                self.dt_dz.values()[cell],
            ],
        }
    }
}

/// 单元 \((u,v,w,T)\) 梯度张量分量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityGradient {
    pub du: [Real; 3],
    pub dv: [Real; 3],
    pub dw: [Real; 3],
    pub dt: [Real; 3],
}

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
    for (i, ti) in t.iter_mut().enumerate().take(n) {
        let rho = primitives.density.values()[i];
        let p = primitives.pressure.values()[i];
        if rho <= 0.0 || p <= 0.0 {
            return Err(AsimuError::Field(
                "密度或压力非正，无法计算温度".to_string(),
            ));
        }
        *ti = viscous
            .map(|v| v.static_temperature(p, rho, eos))
            .unwrap_or(p / (rho * eos.gas_constant));
    }
    Ok(t)
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

impl GradientFields {
    pub(crate) fn clear(&mut self) {
        for f in [
            &mut self.du_dx,
            &mut self.du_dy,
            &mut self.du_dz,
            &mut self.dv_dx,
            &mut self.dv_dy,
            &mut self.dv_dz,
            &mut self.dw_dx,
            &mut self.dw_dy,
            &mut self.dw_dz,
            &mut self.dt_dx,
            &mut self.dt_dy,
            &mut self.dt_dz,
        ] {
            for v in f.values_mut() {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};

    #[test]
    fn uniform_flow_has_zero_velocity_gradient() {
        let pair = FreestreamPairFixture::air_sutherland(0.1);
        pair.for_each_inviscid_side(|side| {
            let (mesh, boundary, _fields, ghosts) =
                uniform_farfield_box(4, 4, 4, 1.0, 1.0, 1.0, side);
            let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
            prim.fill_from_conserved(&_fields, side.eos, side.min_pressure)
                .expect("fill");
            let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
            compute_structured_gradients_3d(
                &mesh,
                &prim,
                side.eos,
                &boundary,
                &ghosts,
                side.min_pressure,
                side.viscous,
                &mut grad,
            )
            .expect("grad");
            for i in 0..mesh.num_cells() {
                let g = grad.velocity_grad_at(i);
                for comp in [g.du, g.dv, g.dw] {
                    assert!(
                        comp.iter().all(|&x| x.abs() < 1.0e-10),
                        "{} velocity gradient cell {i}",
                        side.label
                    );
                }
            }
        });
    }

    #[test]
    fn linear_field_recovers_constant_structured_gradient() {
        let mesh = StructuredMesh3d::uniform_box("box", 4, 4, 4, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let boundary = BoundarySet::new(Vec::new());
        let ghosts = BoundaryGhostBuffer::new();
        let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    let cell = mesh.cell_index(i, j, k);
                    let c = mesh.cell_metric(i, j, k).center;
                    prim.density.values_mut()[cell] = 1.0;
                    prim.pressure.values_mut()[cell] = 101_325.0;
                    prim.velocity_x.values_mut()[cell] = 2.0 * c.x + 3.0 * c.y - 4.0 * c.z;
                    prim.velocity_y.values_mut()[cell] = -c.x + 0.5 * c.y + c.z;
                    prim.velocity_z.values_mut()[cell] = 7.0 * c.x - 2.0 * c.y + 0.25 * c.z;
                }
            }
        }
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        compute_structured_gradients_3d(
            &mesh, &prim, &eos, &boundary, &ghosts, 1.0e-6, None, &mut grad,
        )
        .expect("grad");
        for cell in 0..mesh.num_cells() {
            let g = grad.velocity_grad_at(cell);
            assert!((g.du[0] - 2.0).abs() < 1.0e-12);
            assert!((g.du[1] - 3.0).abs() < 1.0e-12);
            assert!((g.du[2] + 4.0).abs() < 1.0e-12);
            assert!((g.dv[0] + 1.0).abs() < 1.0e-12);
            assert!((g.dv[1] - 0.5).abs() < 1.0e-12);
            assert!((g.dv[2] - 1.0).abs() < 1.0e-12);
            assert!((g.dw[0] - 7.0).abs() < 1.0e-12);
            assert!((g.dw[1] + 2.0).abs() < 1.0e-12);
            assert!((g.dw[2] - 0.25).abs() < 1.0e-12);
        }
    }
}
