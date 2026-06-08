//! 非结构二阶无粘线性重构 IDWLS 梯度（\(\rho,u,v,w,p\)）。

use tracing::info_span;

use crate::core::Vector3;
use crate::discretization::gradient::GradientFields;
use crate::discretization::unstructured_face_cache::{
    LsqRhsCellIncidence, UnstructuredBoundaryFace, UnstructuredInteriorFace,
    UnstructuredSolverMeshCache, accumulate_lsq_rhs_component, solve_lsq_gradient,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::field::primitive_from_conserved_relaxed;

use super::{UnstructuredGradientLsqInput, UnstructuredGradientScratch, neg_vector};

/// 非结构二阶线性重构用 IDWLS 梯度（\(\rho,u,v,w,p\)）。
pub fn compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
    input: UnstructuredGradientLsqInput<'_>,
    out: &mut GradientFields,
    scratch: &mut UnstructuredGradientScratch,
    exec: &mut ExecutionContext,
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
    let _ = scratch;
    exec.idwls_prepare_inviscid(n);
    {
        let _span = info_span!(
            "unstructured_inviscid_linear_reconstruction_lsq_accumulate_rhs",
            cells = n
        )
        .entered();
        accumulate_lsq_rhs_inviscid_linear_reconstruction(&input, exec)?;
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_linear_reconstruction_lsq_solve",
            cells = n
        )
        .entered();
        write_lsq_inviscid_linear_reconstruction_gradients(input.mesh_cache, exec, out)
    }
}

fn accumulate_lsq_rhs_inviscid_linear_reconstruction(
    input: &UnstructuredGradientLsqInput<'_>,
    exec: &mut ExecutionContext,
) -> Result<()> {
    #[cfg(feature = "parallel-fvm")]
    if exec.uses_parallel_cell_loops() {
        return accumulate_lsq_rhs_inviscid_cell_parallel(input, exec);
    }
    accumulate_lsq_rhs_inviscid_face_serial(input, exec)
}

pub(super) fn accumulate_lsq_rhs_inviscid_face_serial(
    input: &UnstructuredGradientLsqInput<'_>,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    let idwls = exec.scratch_mut().idwls_mut();
    let (br, bp, bu, bv, bw) = idwls.inviscid_arrays_mut();
    for face in &topology.interior {
        accumulate_inviscid_interior_as_owner(
            input,
            face,
            &mut br[face.owner],
            &mut bp[face.owner],
            &mut bu[face.owner],
            &mut bv[face.owner],
            &mut bw[face.owner],
        )?;
        accumulate_inviscid_interior_as_neighbor(
            input,
            face,
            &mut br[face.neighbor],
            &mut bp[face.neighbor],
            &mut bu[face.neighbor],
            &mut bv[face.neighbor],
            &mut bw[face.neighbor],
        )?;
    }
    for face in &topology.boundary {
        accumulate_inviscid_boundary_face(
            input,
            face,
            &mut br[face.owner],
            &mut bp[face.owner],
            &mut bu[face.owner],
            &mut bv[face.owner],
            &mut bw[face.owner],
        )?;
    }
    Ok(())
}

#[cfg(feature = "parallel-fvm")]
pub(super) fn accumulate_lsq_rhs_inviscid_cell_parallel(
    input: &UnstructuredGradientLsqInput<'_>,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    let incidence = &input.mesh_cache.lsq_rhs_incidence;
    exec.idwls_accumulate_inviscid_cells(|cell, br, bp, bu, bv, bw| {
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

fn accumulate_inviscid_component(
    rhs: &mut Vector3,
    dr: Vector3,
    w: crate::core::Real,
    delta: crate::core::Real,
) {
    accumulate_lsq_rhs_component(rhs, dr, w, delta);
}

fn write_lsq_inviscid_linear_reconstruction_gradients(
    mesh_cache: &UnstructuredSolverMeshCache,
    exec: &ExecutionContext,
    out: &mut GradientFields,
) -> Result<()> {
    let idwls = exec.idwls_rhs();
    for (cell, geometry) in mesh_cache.lsq_geometry.iter().enumerate() {
        let drho = solve_lsq_gradient(geometry, idwls.br()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 rho 最小二乘梯度样本退化"))
        })?;
        let dp = solve_lsq_gradient(geometry, idwls.bp()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 p 最小二乘梯度样本退化"))
        })?;
        let du = solve_lsq_gradient(geometry, idwls.bu()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 u 最小二乘梯度样本退化"))
        })?;
        let dv = solve_lsq_gradient(geometry, idwls.bv()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 v 最小二乘梯度样本退化"))
        })?;
        let dw = solve_lsq_gradient(geometry, idwls.bw()[cell]).ok_or_else(|| {
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
