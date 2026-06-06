//! 非结构网格单元中心梯度（逆距离加权最小二乘）。
//!
//! 理论：[`docs/theory/unstructured_fvm.md`](../../docs/theory/unstructured_fvm.md)

use crate::boundary::BoundarySet;
use crate::core::{CellId, FaceId, Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::{GradientFields, cell_temperatures};
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{ConservedState, IdealGasEoS, ViscousPhysicsConfig};

/// 非结构 IDWLS 梯度计算输入。
pub struct UnstructuredGradientLsqInput<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub primitives: &'a PrimitiveFields,
    pub eos: &'a IdealGasEoS,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// 非结构网格逆距离加权最小二乘梯度。
///
/// 内部面用相邻单元中心样本，边界面用 ghost 状态在面心关于 owner 单元中心的镜像点作为样本。
pub fn compute_unstructured_gradients_idw_lsq(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
) -> Result<()> {
    let mesh = input.mesh;
    let primitives = input.primitives;
    let n = mesh.num_cells();
    if primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "非结构梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    out.clear();
    let temperatures = cell_temperatures(primitives, input.eos, input.viscous)?;
    let mut accumulators = vec![LsqAccumulator::default(); n];
    let ctx = UnstructuredGradientContext {
        mesh,
        primitives,
        eos: input.eos,
        boundaries: input.boundaries,
        ghosts: input.ghosts,
        min_pressure: input.min_pressure,
        viscous: input.viscous,
        temperatures: &temperatures,
    };
    accumulate_interior_samples(&ctx, &mut accumulators)?;
    accumulate_boundary_samples(&ctx, &mut accumulators)?;
    write_lsq_gradients(out, &accumulators)
}

struct UnstructuredGradientContext<'a> {
    mesh: &'a UnstructuredMesh3d,
    primitives: &'a PrimitiveFields,
    eos: &'a IdealGasEoS,
    boundaries: &'a BoundarySet,
    ghosts: &'a BoundaryGhostBuffer,
    min_pressure: Real,
    viscous: Option<&'a ViscousPhysicsConfig>,
    temperatures: &'a [Real],
}

#[derive(Clone, Copy)]
struct ScalarSample {
    point: Vector3,
    u: Real,
    v: Real,
    w: Real,
    t: Real,
}

#[derive(Clone, Copy)]
struct LsqAccumulator {
    a_xx: Real,
    a_xy: Real,
    a_xz: Real,
    a_yy: Real,
    a_yz: Real,
    a_zz: Real,
    bu: Vector3,
    bv: Vector3,
    bw: Vector3,
    bt: Vector3,
}

impl Default for LsqAccumulator {
    fn default() -> Self {
        Self {
            a_xx: 0.0,
            a_xy: 0.0,
            a_xz: 0.0,
            a_yy: 0.0,
            a_yz: 0.0,
            a_zz: 0.0,
            bu: zero_vector(),
            bv: zero_vector(),
            bw: zero_vector(),
            bt: zero_vector(),
        }
    }
}

impl LsqAccumulator {
    fn add(&mut self, owner: ScalarSample, sample: ScalarSample) {
        let dr = vec_sub(sample.point, owner.point);
        let dist = dr.magnitude();
        if dist <= Real::EPSILON {
            return;
        }
        let w = 1.0 / dist;
        self.a_xx += w * dr.x * dr.x;
        self.a_xy += w * dr.x * dr.y;
        self.a_xz += w * dr.x * dr.z;
        self.a_yy += w * dr.y * dr.y;
        self.a_yz += w * dr.y * dr.z;
        self.a_zz += w * dr.z * dr.z;
        self.bu = vec_add_scaled(self.bu, dr, w * (sample.u - owner.u));
        self.bv = vec_add_scaled(self.bv, dr, w * (sample.v - owner.v));
        self.bw = vec_add_scaled(self.bw, dr, w * (sample.w - owner.w));
        self.bt = vec_add_scaled(self.bt, dr, w * (sample.t - owner.t));
    }

    fn solve(&self, cell: usize, label: &str, rhs: Vector3) -> Result<Vector3> {
        solve_symmetric_3x3(self, rhs).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 {label} 最小二乘梯度样本退化"))
        })
    }
}

fn accumulate_interior_samples(
    ctx: &UnstructuredGradientContext<'_>,
    accumulators: &mut [LsqAccumulator],
) -> Result<()> {
    for face in 0..ctx.mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let Some(neighbor_id) = ctx.mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner_id = ctx.mesh.face_owner(face_id)?;
        let owner = owner_id.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        let owner_sample = cell_sample(ctx, owner_id);
        let neighbor_sample = cell_sample(ctx, neighbor_id);
        accumulators[owner].add(owner_sample, neighbor_sample);
        accumulators[neighbor].add(neighbor_sample, owner_sample);
    }
    Ok(())
}

fn accumulate_boundary_samples(
    ctx: &UnstructuredGradientContext<'_>,
    accumulators: &mut [LsqAccumulator],
) -> Result<()> {
    for patch in ctx.boundaries.patches() {
        for &face in &patch.face_ids {
            let owner_id = ctx.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let Some(ghost) = ctx.ghosts.get_face(face) else {
                continue;
            };
            let owner_sample = cell_sample(ctx, owner_id);
            let ghost_sample = ghost_sample(ctx, face, owner_sample.point, ghost.conserved)?;
            accumulators[owner].add(owner_sample, ghost_sample);
        }
    }
    Ok(())
}

fn cell_sample(ctx: &UnstructuredGradientContext<'_>, cell: CellId) -> ScalarSample {
    let index = cell.index() as usize;
    ScalarSample {
        point: ctx.mesh.cell_metric(cell).center,
        u: ctx.primitives.velocity_x.values()[index],
        v: ctx.primitives.velocity_y.values()[index],
        w: ctx.primitives.velocity_z.values()[index],
        t: ctx.temperatures[index],
    }
}

fn ghost_sample(
    ctx: &UnstructuredGradientContext<'_>,
    face: FaceId,
    owner_center: Vector3,
    conserved: ConservedState,
) -> Result<ScalarSample> {
    let prim = primitive_from_conserved_relaxed(ctx.eos, &conserved, ctx.min_pressure)?;
    let t = ctx
        .viscous
        .map(|v| v.static_temperature(prim.pressure, prim.density, ctx.eos))
        .unwrap_or(prim.pressure / (prim.density.max(1.0e-30) * ctx.eos.gas_constant));
    Ok(ScalarSample {
        point: mirrored_face_sample_point(owner_center, ctx.mesh.face_metric(face).center),
        u: prim.velocity[0],
        v: prim.velocity[1],
        w: prim.velocity[2],
        t,
    })
}

fn write_lsq_gradients(out: &mut GradientFields, accumulators: &[LsqAccumulator]) -> Result<()> {
    for (cell, acc) in accumulators.iter().enumerate() {
        let du = acc.solve(cell, "u", acc.bu)?;
        let dv = acc.solve(cell, "v", acc.bv)?;
        let dw = acc.solve(cell, "w", acc.bw)?;
        let dt = acc.solve(cell, "T", acc.bt)?;
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
    }
    Ok(())
}

fn solve_symmetric_3x3(a: &LsqAccumulator, rhs: Vector3) -> Option<Vector3> {
    let c_xx = a.a_yy * a.a_zz - a.a_yz * a.a_yz;
    let c_xy = a.a_xz * a.a_yz - a.a_xy * a.a_zz;
    let c_xz = a.a_xy * a.a_yz - a.a_xz * a.a_yy;
    let c_yy = a.a_xx * a.a_zz - a.a_xz * a.a_xz;
    let c_yz = a.a_xy * a.a_xz - a.a_xx * a.a_yz;
    let c_zz = a.a_xx * a.a_yy - a.a_xy * a.a_xy;
    let det = a.a_xx * c_xx + a.a_xy * c_xy + a.a_xz * c_xz;
    if det.abs() <= Real::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(Vector3::new(
        (c_xx * rhs.x + c_xy * rhs.y + c_xz * rhs.z) * inv_det,
        (c_xy * rhs.x + c_yy * rhs.y + c_yz * rhs.z) * inv_det,
        (c_xz * rhs.x + c_yz * rhs.y + c_zz * rhs.z) * inv_det,
    ))
}

fn mirrored_face_sample_point(owner_center: Vector3, face_center: Vector3) -> Vector3 {
    Vector3::new(
        2.0 * face_center.x - owner_center.x,
        2.0 * face_center.y - owner_center.y,
        2.0 * face_center.z - owner_center.z,
    )
}

fn vec_sub(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn vec_add_scaled(a: Vector3, b: Vector3, scale: Real) -> Vector3 {
    Vector3::new(a.x + scale * b.x, a.y + scale * b.y, a.z + scale * b.z)
}

fn zero_vector() -> Vector3 {
    Vector3::new(0.0, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::GhostCellState;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::PrimitiveState;

    #[test]
    fn linear_field_recovers_constant_unstructured_idw_lsq_gradient() {
        let mesh = UnstructuredMesh3d::new(
            "hex",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Hex, vec![0, 1, 2, 3, 4, 5, 6, 7]).expect("cell")],
        )
        .expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let cell_center = mesh.cell_metric(CellId(0)).center;
        let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        let cell_prim = linear_primitive_at(cell_center, &eos);
        prim.density.values_mut()[0] = cell_prim.density;
        prim.pressure.values_mut()[0] = cell_prim.pressure;
        prim.velocity_x.values_mut()[0] = cell_prim.velocity[0];
        prim.velocity_y.values_mut()[0] = cell_prim.velocity[1];
        prim.velocity_z.values_mut()[0] = cell_prim.velocity[2];

        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let mut ghosts = BoundaryGhostBuffer::new();
        for &face in &faces {
            let sample_point =
                mirrored_face_sample_point(cell_center, mesh.face_metric(face).center);
            let ghost_prim = linear_primitive_at(sample_point, &eos);
            ghosts.insert_face(
                face,
                GhostCellState {
                    conserved: ConservedState::from_primitive(&eos, &ghost_prim).expect("cons"),
                },
            );
        }
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "all",
            faces,
            BoundaryKind::Farfield {
                mach: 0.0,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);

        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        compute_unstructured_gradients_idw_lsq(
            UnstructuredGradientLsqInput {
                mesh: &mesh,
                primitives: &prim,
                eos: &eos,
                boundaries: &boundary,
                ghosts: &ghosts,
                min_pressure: 1.0e-8,
                viscous: None,
            },
            &mut grad,
        )
        .expect("grad");

        let g = grad.velocity_grad_at(0);
        assert!((g.du[0] - 2.0).abs() < 1.0e-12);
        assert!((g.du[1] + 3.0).abs() < 1.0e-12);
        assert!((g.du[2] - 0.5).abs() < 1.0e-12);
        assert!((g.dv[0] + 1.5).abs() < 1.0e-12);
        assert!((g.dv[1] - 0.25).abs() < 1.0e-12);
        assert!((g.dv[2] - 4.0).abs() < 1.0e-12);
        assert!((g.dw[0] - 0.75).abs() < 1.0e-12);
        assert!((g.dw[1] - 1.25).abs() < 1.0e-12);
        assert!((g.dw[2] + 2.5).abs() < 1.0e-12);
        assert!((grad.dt_dx.values()[0] - 10.0).abs() < 1.0e-12);
        assert!((grad.dt_dy.values()[0] + 5.0).abs() < 1.0e-12);
        assert!((grad.dt_dz.values()[0] - 2.0).abs() < 1.0e-12);
    }

    fn linear_primitive_at(point: Vector3, eos: &IdealGasEoS) -> PrimitiveState {
        let density = 1.0;
        let temperature = 300.0 + 10.0 * point.x - 5.0 * point.y + 2.0 * point.z;
        PrimitiveState {
            density,
            velocity: [
                2.0 * point.x - 3.0 * point.y + 0.5 * point.z,
                -1.5 * point.x + 0.25 * point.y + 4.0 * point.z,
                0.75 * point.x + 1.25 * point.y - 2.5 * point.z,
            ],
            pressure: density * eos.gas_constant * temperature,
            temperature,
        }
    }
}
