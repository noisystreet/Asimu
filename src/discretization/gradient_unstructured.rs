//! 非结构网格单元中心梯度（逆距离加权最小二乘）。
//!
//! 理论：[`docs/theory/unstructured_fvm.md`](../../docs/theory/unstructured_fvm.md)

use tracing::info_span;

use crate::core::{Real, Vector3};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::{GradientFields, cell_temperatures_into};
use crate::discretization::unstructured_face_cache::{
    LsqRhsCellIncidence, UnstructuredBoundaryFace, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache, accumulate_lsq_rhs_component, solve_lsq_gradient,
};
use crate::error::{AsimuError, Result};
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

/// 非结构 IDWLS 梯度计算复用缓冲。
pub struct UnstructuredGradientScratch {
    pub temperatures: Vec<Real>,
    bu: Vec<Vector3>,
    bv: Vec<Vector3>,
    bw: Vec<Vector3>,
    bt: Vec<Vector3>,
    br: Vec<Vector3>,
    bp: Vec<Vector3>,
}

impl UnstructuredGradientScratch {
    #[must_use]
    pub fn new(num_cells: usize) -> Self {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        Self {
            temperatures: vec![0.0; num_cells],
            bu: vec![zero; num_cells],
            bv: vec![zero; num_cells],
            bw: vec![zero; num_cells],
            bt: vec![zero; num_cells],
            br: vec![zero; num_cells],
            bp: vec![zero; num_cells],
        }
    }

    fn prepare(&mut self, num_cells: usize) {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        self.temperatures.resize(num_cells, 0.0);
        self.bu.resize(num_cells, zero);
        self.bv.resize(num_cells, zero);
        self.bw.resize(num_cells, zero);
        self.bt.resize(num_cells, zero);
        self.br.resize(num_cells, zero);
        self.bp.resize(num_cells, zero);
        for i in 0..num_cells {
            self.bu[i] = zero;
            self.bv[i] = zero;
            self.bw[i] = zero;
            self.bt[i] = zero;
            self.br[i] = zero;
            self.bp[i] = zero;
        }
    }

    fn prepare_inviscid_linear_reconstruction(&mut self, num_cells: usize) {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        self.bu.resize(num_cells, zero);
        self.bv.resize(num_cells, zero);
        self.bw.resize(num_cells, zero);
        self.br.resize(num_cells, zero);
        self.bp.resize(num_cells, zero);
        for i in 0..num_cells {
            self.bu[i] = zero;
            self.bv[i] = zero;
            self.bw[i] = zero;
            self.br[i] = zero;
            self.bp[i] = zero;
        }
    }
}

/// 非结构网格逆距离加权最小二乘梯度。
///
/// 内部面用相邻单元中心样本，边界面用 ghost 状态在面心关于 owner 单元中心的镜像点作为样本。
pub fn compute_unstructured_gradients_idw_lsq(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
) -> Result<()> {
    let mut scratch = UnstructuredGradientScratch::new(input.mesh.num_cells());
    compute_unstructured_gradients_idw_lsq_with_scratch(input, out, &mut scratch)
}

/// 使用调用方提供的 scratch 计算非结构 IDWLS 梯度。
pub fn compute_unstructured_gradients_idw_lsq_with_scratch(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
    scratch: &mut UnstructuredGradientScratch,
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
    scratch.prepare(n);
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
        accumulate_lsq_rhs(&input, scratch)?;
    }
    {
        let _span = info_span!("unstructured_idw_lsq_solve_gradients", cells = n).entered();
        write_lsq_gradients(input.mesh_cache, scratch, out)
    }
}

fn accumulate_lsq_rhs(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    #[cfg(feature = "parallel-fvm")]
    {
        accumulate_lsq_rhs_cell_parallel(input, scratch)
    }
    #[cfg(not(feature = "parallel-fvm"))]
    {
        accumulate_lsq_rhs_face_serial(input, scratch)
    }
}

#[cfg(any(not(feature = "parallel-fvm"), test))]
fn accumulate_lsq_rhs_face_serial(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    let temperatures = &scratch.temperatures;
    for face in &topology.interior {
        accumulate_lsq_interior_as_owner(
            input,
            face,
            temperatures,
            &mut scratch.bu[face.owner],
            &mut scratch.bv[face.owner],
            &mut scratch.bw[face.owner],
            &mut scratch.bt[face.owner],
        )?;
        accumulate_lsq_interior_as_neighbor(
            input,
            face,
            temperatures,
            &mut scratch.bu[face.neighbor],
            &mut scratch.bv[face.neighbor],
            &mut scratch.bw[face.neighbor],
            &mut scratch.bt[face.neighbor],
        )?;
    }
    for face in &topology.boundary {
        accumulate_lsq_boundary_face(
            input,
            face,
            temperatures,
            &mut scratch.bu[face.owner],
            &mut scratch.bv[face.owner],
            &mut scratch.bw[face.owner],
            &mut scratch.bt[face.owner],
        )?;
    }
    Ok(())
}

#[cfg(feature = "parallel-fvm")]
fn accumulate_lsq_rhs_cell_parallel(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    use rayon::prelude::*;

    let topology = &input.mesh_cache.face_topology;
    let incidence = &input.mesh_cache.lsq_rhs_incidence;
    let temperatures = &scratch.temperatures;
    (
        scratch.bu.par_iter_mut(),
        scratch.bv.par_iter_mut(),
        scratch.bw.par_iter_mut(),
        scratch.bt.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each(|(cell, (bu, bv, bw, bt))| {
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
    scratch: &UnstructuredGradientScratch,
    out: &mut GradientFields,
) -> Result<()> {
    for (cell, geometry) in mesh_cache.lsq_geometry.iter().enumerate() {
        let du = solve_lsq_gradient(geometry, scratch.bu[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 u 最小二乘梯度样本退化"))
        })?;
        let dv = solve_lsq_gradient(geometry, scratch.bv[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 v 最小二乘梯度样本退化"))
        })?;
        let dw = solve_lsq_gradient(geometry, scratch.bw[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 w 最小二乘梯度样本退化"))
        })?;
        let dt = solve_lsq_gradient(geometry, scratch.bt[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 T 最小二乘梯度样本退化"))
        })?;
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

/// 非结构二阶线性重构用 IDWLS 梯度（\(\rho,u,v,w,p\)）。
pub fn compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    let n = input.mesh.num_cells();
    if input.primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "非结构无粘梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    if input.mesh_cache.lsq_geometry.len() != n {
        return Err(AsimuError::Field(
            "非结构 IDWLS 几何缓存与网格单元数不一致".to_string(),
        ));
    }
    scratch.prepare_inviscid_linear_reconstruction(n);
    {
        let _span = info_span!(
            "unstructured_inviscid_linear_reconstruction_lsq_accumulate_rhs",
            cells = n
        )
        .entered();
        accumulate_lsq_rhs_inviscid_linear_reconstruction(&input, scratch)?;
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_linear_reconstruction_lsq_solve",
            cells = n
        )
        .entered();
        write_lsq_inviscid_linear_reconstruction_gradients(input.mesh_cache, scratch, out)
    }
}

fn accumulate_lsq_rhs_inviscid_linear_reconstruction(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    #[cfg(feature = "parallel-fvm")]
    {
        accumulate_lsq_rhs_inviscid_cell_parallel(input, scratch)
    }
    #[cfg(not(feature = "parallel-fvm"))]
    {
        accumulate_lsq_rhs_inviscid_face_serial(input, scratch)
    }
}

#[cfg(any(not(feature = "parallel-fvm"), test))]
fn accumulate_lsq_rhs_inviscid_face_serial(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    for face in &topology.interior {
        accumulate_inviscid_interior_as_owner(
            input,
            face,
            &mut scratch.br[face.owner],
            &mut scratch.bp[face.owner],
            &mut scratch.bu[face.owner],
            &mut scratch.bv[face.owner],
            &mut scratch.bw[face.owner],
        )?;
        accumulate_inviscid_interior_as_neighbor(
            input,
            face,
            &mut scratch.br[face.neighbor],
            &mut scratch.bp[face.neighbor],
            &mut scratch.bu[face.neighbor],
            &mut scratch.bv[face.neighbor],
            &mut scratch.bw[face.neighbor],
        )?;
    }
    for face in &topology.boundary {
        accumulate_inviscid_boundary_face(
            input,
            face,
            &mut scratch.br[face.owner],
            &mut scratch.bp[face.owner],
            &mut scratch.bu[face.owner],
            &mut scratch.bv[face.owner],
            &mut scratch.bw[face.owner],
        )?;
    }
    Ok(())
}

#[cfg(feature = "parallel-fvm")]
fn accumulate_lsq_rhs_inviscid_cell_parallel(
    input: &UnstructuredGradientLsqInput<'_>,
    scratch: &mut UnstructuredGradientScratch,
) -> Result<()> {
    use rayon::prelude::*;

    let topology = &input.mesh_cache.face_topology;
    let incidence = &input.mesh_cache.lsq_rhs_incidence;
    (
        scratch.br.par_iter_mut(),
        scratch.bp.par_iter_mut(),
        scratch.bu.par_iter_mut(),
        scratch.bv.par_iter_mut(),
        scratch.bw.par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .try_for_each(|(cell, (br, bp, bu, bv, bw))| {
            let mut rhs = LsqInviscidCellRhsMut { br, bp, bu, bv, bw };
            accumulate_lsq_rhs_inviscid_one_cell(input, topology, incidence, cell, &mut rhs)
        })
}

struct LsqInviscidCellRhsMut<'a> {
    br: &'a mut Vector3,
    bp: &'a mut Vector3,
    bu: &'a mut Vector3,
    bv: &'a mut Vector3,
    bw: &'a mut Vector3,
}

#[cfg(feature = "parallel-fvm")]
fn accumulate_lsq_rhs_inviscid_one_cell(
    input: &UnstructuredGradientLsqInput<'_>,
    topology: &crate::discretization::unstructured_face_cache::UnstructuredFaceTopology,
    incidence: &LsqRhsCellIncidence,
    cell: usize,
    rhs: &mut LsqInviscidCellRhsMut<'_>,
) -> Result<()> {
    for &face_idx in &incidence.interior_as_owner[cell] {
        accumulate_inviscid_interior_as_owner(
            input,
            &topology.interior[face_idx],
            rhs.br,
            rhs.bp,
            rhs.bu,
            rhs.bv,
            rhs.bw,
        )?;
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
        accumulate_inviscid_interior_as_neighbor(
            input,
            &topology.interior[face_idx],
            rhs.br,
            rhs.bp,
            rhs.bu,
            rhs.bv,
            rhs.bw,
        )?;
    }
    for &boundary_idx in &incidence.boundary_faces[cell] {
        accumulate_inviscid_boundary_face(
            input,
            &topology.boundary[boundary_idx],
            rhs.br,
            rhs.bp,
            rhs.bu,
            rhs.bv,
            rhs.bw,
        )?;
    }
    Ok(())
}

fn accumulate_inviscid_interior_as_owner(
    input: &UnstructuredGradientLsqInput<'_>,
    face: &UnstructuredInteriorFace,
    br: &mut Vector3,
    bp: &mut Vector3,
    bu: &mut Vector3,
    bv: &mut Vector3,
    bw: &mut Vector3,
) -> Result<()> {
    let prim = input.primitives;
    let rho_o = prim.density.values()[face.owner];
    let p_o = prim.pressure.values()[face.owner];
    let u_o = prim.velocity_x.values()[face.owner];
    let v_o = prim.velocity_y.values()[face.owner];
    let w_o = prim.velocity_z.values()[face.owner];
    let rho_n = prim.density.values()[face.neighbor];
    let p_n = prim.pressure.values()[face.neighbor];
    let u_n = prim.velocity_x.values()[face.neighbor];
    let v_n = prim.velocity_y.values()[face.neighbor];
    let w_n = prim.velocity_z.values()[face.neighbor];
    accumulate_inviscid_component(br, face.lsq_dr, face.lsq_w, rho_n - rho_o);
    accumulate_inviscid_component(bp, face.lsq_dr, face.lsq_w, p_n - p_o);
    accumulate_inviscid_component(bu, face.lsq_dr, face.lsq_w, u_n - u_o);
    accumulate_inviscid_component(bv, face.lsq_dr, face.lsq_w, v_n - v_o);
    accumulate_inviscid_component(bw, face.lsq_dr, face.lsq_w, w_n - w_o);
    Ok(())
}

fn accumulate_inviscid_interior_as_neighbor(
    input: &UnstructuredGradientLsqInput<'_>,
    face: &UnstructuredInteriorFace,
    br: &mut Vector3,
    bp: &mut Vector3,
    bu: &mut Vector3,
    bv: &mut Vector3,
    bw: &mut Vector3,
) -> Result<()> {
    let prim = input.primitives;
    let rho_o = prim.density.values()[face.owner];
    let p_o = prim.pressure.values()[face.owner];
    let u_o = prim.velocity_x.values()[face.owner];
    let v_o = prim.velocity_y.values()[face.owner];
    let w_o = prim.velocity_z.values()[face.owner];
    let rho_n = prim.density.values()[face.neighbor];
    let p_n = prim.pressure.values()[face.neighbor];
    let u_n = prim.velocity_x.values()[face.neighbor];
    let v_n = prim.velocity_y.values()[face.neighbor];
    let w_n = prim.velocity_z.values()[face.neighbor];
    let dr_n = neg_vector(face.lsq_dr);
    accumulate_inviscid_component(br, dr_n, face.lsq_w, rho_o - rho_n);
    accumulate_inviscid_component(bp, dr_n, face.lsq_w, p_o - p_n);
    accumulate_inviscid_component(bu, dr_n, face.lsq_w, u_o - u_n);
    accumulate_inviscid_component(bv, dr_n, face.lsq_w, v_o - v_n);
    accumulate_inviscid_component(bw, dr_n, face.lsq_w, w_o - w_n);
    Ok(())
}

fn accumulate_inviscid_boundary_face(
    input: &UnstructuredGradientLsqInput<'_>,
    face: &UnstructuredBoundaryFace,
    br: &mut Vector3,
    bp: &mut Vector3,
    bu: &mut Vector3,
    bv: &mut Vector3,
    bw: &mut Vector3,
) -> Result<()> {
    let owner = face.owner;
    let prim = input.primitives;
    let ghost = input.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "非结构无粘梯度边界面 FaceId({}) 缺少 ghost",
            face.face.index()
        ))
    })?;
    let ghost_prim =
        primitive_from_conserved_relaxed(input.eos, &ghost.conserved, input.min_pressure)?;
    let rho_o = prim.density.values()[owner];
    let p_o = prim.pressure.values()[owner];
    let u_o = prim.velocity_x.values()[owner];
    let v_o = prim.velocity_y.values()[owner];
    let w_o = prim.velocity_z.values()[owner];
    accumulate_inviscid_component(br, face.lsq_dr, face.lsq_w, ghost_prim.density - rho_o);
    accumulate_inviscid_component(bp, face.lsq_dr, face.lsq_w, ghost_prim.pressure - p_o);
    accumulate_inviscid_component(bu, face.lsq_dr, face.lsq_w, ghost_prim.velocity[0] - u_o);
    accumulate_inviscid_component(bv, face.lsq_dr, face.lsq_w, ghost_prim.velocity[1] - v_o);
    accumulate_inviscid_component(bw, face.lsq_dr, face.lsq_w, ghost_prim.velocity[2] - w_o);
    Ok(())
}

fn accumulate_inviscid_component(rhs: &mut Vector3, dr: Vector3, w: Real, delta: Real) {
    accumulate_lsq_rhs_component(rhs, dr, w, delta);
}

fn write_lsq_inviscid_linear_reconstruction_gradients(
    mesh_cache: &UnstructuredSolverMeshCache,
    scratch: &UnstructuredGradientScratch,
    out: &mut GradientFields,
) -> Result<()> {
    for (cell, geometry) in mesh_cache.lsq_geometry.iter().enumerate() {
        let drho = solve_lsq_gradient(geometry, scratch.br[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 rho 最小二乘梯度样本退化"))
        })?;
        let dp = solve_lsq_gradient(geometry, scratch.bp[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 p 最小二乘梯度样本退化"))
        })?;
        let du = solve_lsq_gradient(geometry, scratch.bu[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 u 最小二乘梯度样本退化"))
        })?;
        let dv = solve_lsq_gradient(geometry, scratch.bv[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 v 最小二乘梯度样本退化"))
        })?;
        let dw = solve_lsq_gradient(geometry, scratch.bw[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 w 最小二乘梯度样本退化"))
        })?;
        out.drho_dx.values_mut()[cell] = drho.x;
        out.drho_dy.values_mut()[cell] = drho.y;
        out.drho_dz.values_mut()[cell] = drho.z;
        out.dp_dx.values_mut()[cell] = dp.x;
        out.dp_dy.values_mut()[cell] = dp.y;
        out.dp_dz.values_mut()[cell] = dp.z;
        out.du_dx.values_mut()[cell] = du.x;
        out.du_dy.values_mut()[cell] = du.y;
        out.du_dz.values_mut()[cell] = du.z;
        out.dv_dx.values_mut()[cell] = dv.x;
        out.dv_dy.values_mut()[cell] = dv.y;
        out.dv_dz.values_mut()[cell] = dv.z;
        out.dw_dx.values_mut()[cell] = dw.x;
        out.dw_dy.values_mut()[cell] = dw.y;
        out.dw_dz.values_mut()[cell] = dw.z;
    }
    Ok(())
}

#[cfg(test)]
#[path = "gradient_unstructured_tests.rs"]
mod tests;
