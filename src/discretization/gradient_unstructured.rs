//! 非结构网格单元中心梯度（逆距离平方加权最小二乘，对标 SU2 WLS）。
//!
//! 理论：[`docs/theory/unstructured_fvm.md`](../../docs/theory/unstructured_fvm.md)

#[path = "gradient_unstructured_inviscid.rs"]
mod inviscid_linear;
pub use inviscid_linear::compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq;

use tracing::info_span;

use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::{GradientFields, cell_temperatures_into};
use crate::discretization::unstructured_face_cache::{
    LsqRhsCellIncidence, UnstructuredBoundaryFace, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache, accumulate_lsq_rhs_component, solve_lsq_gradient,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// 非结构 IDWLS 梯度计算输入。
pub struct UnstructuredGradientLsqInput<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub primitives: &'a PrimitiveFields,
    pub eos: &'a IdealGasEoS,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// 非结构 IDWLS 温度等 discretization 侧 scratch（RHS \(b\) 在 [`ExecutionContext`]）。
pub struct UnstructuredGradientScratch {
    pub temperatures: Vec<Real>,
}

impl UnstructuredGradientScratch {
    #[must_use]
    pub fn new(_num_cells: usize) -> Self {
        Self {
            temperatures: Vec::new(),
        }
    }

    pub(super) fn prepare_temperatures(&mut self, num_cells: usize) {
        self.temperatures.resize(num_cells, 0.0);
    }
}

/// 非结构网格逆距离加权最小二乘梯度。
pub fn compute_unstructured_gradients_idw_lsq(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let mut scratch = UnstructuredGradientScratch::new(input.mesh.num_cells());
    compute_unstructured_gradients_idw_lsq_with_scratch(input, out, &mut scratch, exec)
}

/// 使用调用方提供的 scratch 计算非结构 IDWLS 梯度。
pub fn compute_unstructured_gradients_idw_lsq_with_scratch(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
    scratch: &mut UnstructuredGradientScratch,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let mesh = input.mesh;
    let primitives = input.primitives;
    let n = mesh.num_cells();
    if primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "非结构梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    if input.mesh_cache.lsq_geometry.len() != n {
        return Err(AsimuError::Field(
            "非结构 IDWLS 几何缓存与网格单元数不一致".to_string(),
        ));
    }
    out.clear();
    scratch.prepare_temperatures(n);
    exec.idwls_prepare_viscous(n);
    {
        let _span = info_span!("unstructured_idw_lsq_cell_temperatures", cells = n).entered();
        cell_temperatures_into(
            primitives,
            input.eos,
            input.viscous,
            &mut scratch.temperatures,
        )?;
    }
    {
        let topology = &input.mesh_cache.face_topology;
        let _span = info_span!(
            "unstructured_idw_lsq_accumulate_rhs",
            interior_faces = topology.interior.len(),
            boundary_faces = topology.boundary.len(),
        )
        .entered();
        accumulate_lsq_rhs(&input, scratch, exec)?;
    }
    {
        let _span = info_span!("unstructured_idw_lsq_solve_gradients", cells = n).entered();
        write_lsq_gradients(input.mesh_cache, exec, out)
    }
}

fn accumulate_lsq_rhs(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &UnstructuredGradientScratch,
    exec: &mut ExecutionContext,
) -> Result<()> {
    #[cfg(feature = "parallel-fvm")]
    if exec.uses_parallel_cell_loops() {
        return accumulate_lsq_rhs_cell_parallel(input, scratch, exec);
    }
    accumulate_lsq_rhs_face_serial(input, scratch, exec)
}

pub(super) fn accumulate_lsq_rhs_face_serial(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &UnstructuredGradientScratch,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    let temperatures = &scratch.temperatures;
    let idwls = exec.scratch_mut().idwls_mut();
    let (bu, bv, bw, bt) = idwls.viscous_arrays_mut();
    for face in &topology.interior {
        accumulate_lsq_interior_as_owner(
            input,
            face,
            temperatures,
            &mut bu[face.owner],
            &mut bv[face.owner],
            &mut bw[face.owner],
            &mut bt[face.owner],
        )?;
        accumulate_lsq_interior_as_neighbor(
            input,
            face,
            temperatures,
            &mut bu[face.neighbor],
            &mut bv[face.neighbor],
            &mut bw[face.neighbor],
            &mut bt[face.neighbor],
        )?;
    }
    for face in &topology.boundary {
        accumulate_lsq_boundary_face(
            input,
            face,
            temperatures,
            &mut bu[face.owner],
            &mut bv[face.owner],
            &mut bw[face.owner],
            &mut bt[face.owner],
        )?;
    }
    Ok(())
}

#[cfg(feature = "parallel-fvm")]
pub(super) fn accumulate_lsq_rhs_cell_parallel(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &UnstructuredGradientScratch,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    let incidence = &input.mesh_cache.lsq_rhs_incidence;
    let temperatures = &scratch.temperatures;
    exec.idwls_accumulate_viscous_cells(|cell, bu, bv, bw, bt| {
        let mut rhs = LsqViscousCellRhsMut { bu, bv, bw, bt };
        accumulate_lsq_rhs_one_cell(input, topology, incidence, temperatures, cell, &mut rhs)
    })
}

struct LsqViscousCellRhsMut<'a> {
    bu: &'a mut Vector3,
    bv: &'a mut Vector3,
    bw: &'a mut Vector3,
    bt: &'a mut Vector3,
}

#[cfg(feature = "parallel-fvm")]
fn accumulate_lsq_rhs_one_cell(
    input: &UnstructuredGradientLsqInput<'_>,
    topology: &crate::discretization::unstructured_face_cache::UnstructuredFaceTopology,
    incidence: &LsqRhsCellIncidence,
    temperatures: &[Real],
    cell: usize,
    rhs: &mut LsqViscousCellRhsMut<'_>,
) -> Result<()> {
    for &face_idx in &incidence.interior_as_owner[cell] {
        accumulate_lsq_interior_as_owner(
            input,
            &topology.interior[face_idx],
            temperatures,
            rhs.bu,
            rhs.bv,
            rhs.bw,
            rhs.bt,
        )?;
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
        accumulate_lsq_interior_as_neighbor(
            input,
            &topology.interior[face_idx],
            temperatures,
            rhs.bu,
            rhs.bv,
            rhs.bw,
            rhs.bt,
        )?;
    }
    for &boundary_idx in &incidence.boundary_faces[cell] {
        accumulate_lsq_boundary_face(
            input,
            &topology.boundary[boundary_idx],
            temperatures,
            rhs.bu,
            rhs.bv,
            rhs.bw,
            rhs.bt,
        )?;
    }
    Ok(())
}

fn accumulate_lsq_interior_as_owner(
    input: &UnstructuredGradientLsqInput<'_>,
    face: &UnstructuredInteriorFace,
    temperatures: &[Real],
    bu: &mut Vector3,
    bv: &mut Vector3,
    bw: &mut Vector3,
    bt: &mut Vector3,
) -> Result<()> {
    let u_o = input.primitives.velocity_x.values()[face.owner];
    let v_o = input.primitives.velocity_y.values()[face.owner];
    let w_o = input.primitives.velocity_z.values()[face.owner];
    let t_o = temperatures[face.owner];
    let u_n = input.primitives.velocity_x.values()[face.neighbor];
    let v_n = input.primitives.velocity_y.values()[face.neighbor];
    let w_n = input.primitives.velocity_z.values()[face.neighbor];
    let t_n = temperatures[face.neighbor];
    accumulate_lsq_rhs_component(bu, face.lsq_dr, face.lsq_w, u_n - u_o);
    accumulate_lsq_rhs_component(bv, face.lsq_dr, face.lsq_w, v_n - v_o);
    accumulate_lsq_rhs_component(bw, face.lsq_dr, face.lsq_w, w_n - w_o);
    accumulate_lsq_rhs_component(bt, face.lsq_dr, face.lsq_w, t_n - t_o);
    Ok(())
}

fn accumulate_lsq_interior_as_neighbor(
    input: &UnstructuredGradientLsqInput<'_>,
    face: &UnstructuredInteriorFace,
    temperatures: &[Real],
    bu: &mut Vector3,
    bv: &mut Vector3,
    bw: &mut Vector3,
    bt: &mut Vector3,
) -> Result<()> {
    let u_o = input.primitives.velocity_x.values()[face.owner];
    let v_o = input.primitives.velocity_y.values()[face.owner];
    let w_o = input.primitives.velocity_z.values()[face.owner];
    let t_o = temperatures[face.owner];
    let u_n = input.primitives.velocity_x.values()[face.neighbor];
    let v_n = input.primitives.velocity_y.values()[face.neighbor];
    let w_n = input.primitives.velocity_z.values()[face.neighbor];
    let t_n = temperatures[face.neighbor];
    let dr_n = neg_vector(face.lsq_dr);
    accumulate_lsq_rhs_component(bu, dr_n, face.lsq_w, u_o - u_n);
    accumulate_lsq_rhs_component(bv, dr_n, face.lsq_w, v_o - v_n);
    accumulate_lsq_rhs_component(bw, dr_n, face.lsq_w, w_o - w_n);
    accumulate_lsq_rhs_component(bt, dr_n, face.lsq_w, t_o - t_n);
    Ok(())
}

fn accumulate_lsq_boundary_face(
    input: &UnstructuredGradientLsqInput<'_>,
    face: &UnstructuredBoundaryFace,
    temperatures: &[Real],
    bu: &mut Vector3,
    bv: &mut Vector3,
    bw: &mut Vector3,
    bt: &mut Vector3,
) -> Result<()> {
    let owner = face.owner;
    let u_o = input.primitives.velocity_x.values()[owner];
    let v_o = input.primitives.velocity_y.values()[owner];
    let w_o = input.primitives.velocity_z.values()[owner];
    let t_o = temperatures[owner];
    let ghost = input.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "非结构梯度边界面 FaceId({}) 缺少 ghost",
            face.face.index()
        ))
    })?;
    let ghost_sample = ghost_scalar_sample(input, ghost.conserved)?;
    accumulate_lsq_rhs_component(bu, face.lsq_dr, face.lsq_w, ghost_sample.u - u_o);
    accumulate_lsq_rhs_component(bv, face.lsq_dr, face.lsq_w, ghost_sample.v - v_o);
    accumulate_lsq_rhs_component(bw, face.lsq_dr, face.lsq_w, ghost_sample.w - w_o);
    accumulate_lsq_rhs_component(bt, face.lsq_dr, face.lsq_w, ghost_sample.t - t_o);
    Ok(())
}

fn neg_vector(v: Vector3) -> Vector3 {
    Vector3::new(-v.x, -v.y, -v.z)
}

#[derive(Clone, Copy)]
struct ScalarSample {
    u: Real,
    v: Real,
    w: Real,
    t: Real,
}

fn ghost_scalar_sample(
    input: &UnstructuredGradientLsqInput<'_>,
    conserved: crate::physics::ConservedState,
) -> Result<ScalarSample> {
    let prim = primitive_from_conserved_relaxed(input.eos, &conserved, input.min_pressure)?;
    let t = input
        .viscous
        .map(|v| v.static_temperature(prim.pressure, prim.density, input.eos))
        .unwrap_or(prim.pressure / (prim.density.max(1.0e-30) * input.eos.gas_constant));
    Ok(ScalarSample {
        u: prim.velocity[0],
        v: prim.velocity[1],
        w: prim.velocity[2],
        t,
    })
}

fn write_lsq_gradients(
    mesh_cache: &UnstructuredSolverMeshCache,
    exec: &ExecutionContext,
    out: &mut GradientFields,
) -> Result<()> {
    let idwls = exec.idwls_rhs();
    let n = mesh_cache.lsq_geometry.len();
    let mut cell = 0;
    while cell < n {
        #[cfg(feature = "simd-fvm")]
        if cell + 4 <= n && write_lsq_gradients_batch4(mesh_cache, idwls, out, cell)? {
            cell += 4;
            continue;
        }
        write_lsq_gradients_one_cell(mesh_cache, idwls, out, cell)?;
        cell += 1;
    }
    Ok(())
}

#[cfg(feature = "simd-fvm")]
fn write_lsq_gradients_batch4(
    mesh_cache: &UnstructuredSolverMeshCache,
    idwls: &crate::exec::IdwlsRhsBuffer,
    out: &mut GradientFields,
    start: usize,
) -> Result<bool> {
    use crate::discretization::unstructured_face_cache::sym3_from_lsq_for_exec;
    use crate::exec::cpu::{Symmetric3x3, solve_symmetric_3x3_batch4};

    let g0 = sym3_from_lsq_for_exec(&mesh_cache.lsq_geometry[start]);
    let g1 = sym3_from_lsq_for_exec(&mesh_cache.lsq_geometry[start + 1]);
    let g2 = sym3_from_lsq_for_exec(&mesh_cache.lsq_geometry[start + 2]);
    let g3 = sym3_from_lsq_for_exec(&mesh_cache.lsq_geometry[start + 3]);
    let mats: [&Symmetric3x3; 4] = [&g0, &g1, &g2, &g3];
    let bu = [
        idwls.bu()[start],
        idwls.bu()[start + 1],
        idwls.bu()[start + 2],
        idwls.bu()[start + 3],
    ];
    let bv = [
        idwls.bv()[start],
        idwls.bv()[start + 1],
        idwls.bv()[start + 2],
        idwls.bv()[start + 3],
    ];
    let bw = [
        idwls.bw()[start],
        idwls.bw()[start + 1],
        idwls.bw()[start + 2],
        idwls.bw()[start + 3],
    ];
    let bt = [
        idwls.bt()[start],
        idwls.bt()[start + 1],
        idwls.bt()[start + 2],
        idwls.bt()[start + 3],
    ];
    let du = solve_symmetric_3x3_batch4(mats, bu);
    let dv = solve_symmetric_3x3_batch4(mats, bv);
    let dw = solve_symmetric_3x3_batch4(mats, bw);
    let dt = solve_symmetric_3x3_batch4(mats, bt);
    for lane in 0..4 {
        let cell = start + lane;
        let du_v = du[lane].ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 u 最小二乘梯度样本退化"))
        })?;
        let dv_v = dv[lane].ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 v 最小二乘梯度样本退化"))
        })?;
        let dw_v = dw[lane].ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 w 最小二乘梯度样本退化"))
        })?;
        let dt_v = dt[lane].ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 T 最小二乘梯度样本退化"))
        })?;
        out.du_dx.values_mut()[cell] = du_v.x;
        out.du_dy.values_mut()[cell] = du_v.y;
        out.du_dz.values_mut()[cell] = du_v.z;
        out.dv_dx.values_mut()[cell] = dv_v.x;
        out.dv_dy.values_mut()[cell] = dv_v.y;
        out.dv_dz.values_mut()[cell] = dv_v.z;
        out.dw_dx.values_mut()[cell] = dw_v.x;
        out.dw_dy.values_mut()[cell] = dw_v.y;
        out.dw_dz.values_mut()[cell] = dw_v.z;
        out.dt_dx.values_mut()[cell] = dt_v.x;
        out.dt_dy.values_mut()[cell] = dt_v.y;
        out.dt_dz.values_mut()[cell] = dt_v.z;
    }
    Ok(true)
}

fn write_lsq_gradients_one_cell(
    mesh_cache: &UnstructuredSolverMeshCache,
    idwls: &crate::exec::IdwlsRhsBuffer,
    out: &mut GradientFields,
    cell: usize,
) -> Result<()> {
    let geometry = &mesh_cache.lsq_geometry[cell];
    let du = solve_lsq_gradient(geometry, idwls.bu()[cell])
        .ok_or_else(|| AsimuError::Mesh(format!("非结构单元 {cell} 的 u 最小二乘梯度样本退化")))?;
    let dv = solve_lsq_gradient(geometry, idwls.bv()[cell])
        .ok_or_else(|| AsimuError::Mesh(format!("非结构单元 {cell} 的 v 最小二乘梯度样本退化")))?;
    let dw = solve_lsq_gradient(geometry, idwls.bw()[cell])
        .ok_or_else(|| AsimuError::Mesh(format!("非结构单元 {cell} 的 w 最小二乘梯度样本退化")))?;
    let dt = solve_lsq_gradient(geometry, idwls.bt()[cell])
        .ok_or_else(|| AsimuError::Mesh(format!("非结构单元 {cell} 的 T 最小二乘梯度样本退化")))?;
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

#[cfg(test)]
#[path = "gradient_unstructured_tests.rs"]
mod tests;
