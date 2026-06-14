//! 非结构二阶无粘线性重构 IDWLS 梯度（f32 串行路径）。

use tracing::info_span;

use crate::core::Vector3;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::UnstructuredGradientLsqInputF32;
use crate::discretization::unstructured_face_cache::{
    UnstructuredBoundaryFace, UnstructuredInteriorFace, UnstructuredSolverMeshCache,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::exec::cpu::{Symmetric3x3, accumulate_lsq_rhs_component_f32, solve_symmetric_3x3_f32};
use crate::field::primitive_from_conserved_relaxed;

/// 非结构二阶线性重构用 IDWLS 梯度（f32）。
pub fn compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32(
    input: UnstructuredGradientLsqInputF32<'_>,
    out: &mut GradientFieldsT<f32>,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let n = input.mesh.num_cells();
    if input.primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "非结构 f32 无粘梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    if input.mesh_cache.lsq_geometry.len() != n {
        return Err(AsimuError::Field(
            "非结构 IDWLS 几何缓存与网格单元数不一致".to_string(),
        ));
    }
    out.clear();
    exec.idwls_prepare_inviscid_f32(n);
    {
        let _span = info_span!(
            "unstructured_inviscid_linear_reconstruction_lsq_accumulate_rhs_f32",
            cells = n
        )
        .entered();
        accumulate_rhs_f32(&input, exec)?;
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_linear_reconstruction_lsq_solve_f32",
            cells = n
        )
        .entered();
        write_gradients_f32(input.mesh_cache, exec, out)
    }
}

fn accumulate_rhs_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology;
    let idwls = exec.scratch_mut().idwls_mut();
    let (br, bp, bu, bv, bw) = idwls.inviscid_arrays_mut_f32();
    for face in &topology.interior {
        accumulate_interior_owner_f32(
            input,
            face,
            &mut br[face.owner],
            &mut bp[face.owner],
            &mut bu[face.owner],
            &mut bv[face.owner],
            &mut bw[face.owner],
        )?;
        accumulate_interior_neighbor_f32(
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
        accumulate_boundary_f32(
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

fn accumulate_interior_owner_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredInteriorFace,
    br: &mut [f32; 3],
    bp: &mut [f32; 3],
    bu: &mut [f32; 3],
    bv: &mut [f32; 3],
    bw: &mut [f32; 3],
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
    accumulate_lsq_rhs_component_f32(br, face.lsq_dr, face.lsq_w, rho_n - rho_o);
    accumulate_lsq_rhs_component_f32(bp, face.lsq_dr, face.lsq_w, p_n - p_o);
    accumulate_lsq_rhs_component_f32(bu, face.lsq_dr, face.lsq_w, u_n - u_o);
    accumulate_lsq_rhs_component_f32(bv, face.lsq_dr, face.lsq_w, v_n - v_o);
    accumulate_lsq_rhs_component_f32(bw, face.lsq_dr, face.lsq_w, w_n - w_o);
    Ok(())
}

fn accumulate_interior_neighbor_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredInteriorFace,
    br: &mut [f32; 3],
    bp: &mut [f32; 3],
    bu: &mut [f32; 3],
    bv: &mut [f32; 3],
    bw: &mut [f32; 3],
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
    accumulate_lsq_rhs_component_f32(br, dr_n, face.lsq_w, rho_o - rho_n);
    accumulate_lsq_rhs_component_f32(bp, dr_n, face.lsq_w, p_o - p_n);
    accumulate_lsq_rhs_component_f32(bu, dr_n, face.lsq_w, u_o - u_n);
    accumulate_lsq_rhs_component_f32(bv, dr_n, face.lsq_w, v_o - v_n);
    accumulate_lsq_rhs_component_f32(bw, dr_n, face.lsq_w, w_o - w_n);
    Ok(())
}

fn accumulate_boundary_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredBoundaryFace,
    br: &mut [f32; 3],
    bp: &mut [f32; 3],
    bu: &mut [f32; 3],
    bv: &mut [f32; 3],
    bw: &mut [f32; 3],
) -> Result<()> {
    let owner = face.owner;
    let prim = input.primitives;
    let rho_o = prim.density.values()[owner];
    let p_o = prim.pressure.values()[owner];
    let u_o = prim.velocity_x.values()[owner];
    let v_o = prim.velocity_y.values()[owner];
    let w_o = prim.velocity_z.values()[owner];
    let ghost = input.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "非结构 f32 无粘梯度边界面 FaceId({}) 缺少 ghost",
            face.face.index()
        ))
    })?;
    let ghost_prim =
        primitive_from_conserved_relaxed(input.eos, &ghost.conserved, input.min_pressure)?;
    let rho_g = ghost_prim.density as f32;
    let p_g = ghost_prim.pressure as f32;
    let u_g = ghost_prim.velocity[0] as f32;
    let v_g = ghost_prim.velocity[1] as f32;
    let w_g = ghost_prim.velocity[2] as f32;
    accumulate_lsq_rhs_component_f32(br, face.lsq_dr, face.lsq_w, rho_g - rho_o);
    accumulate_lsq_rhs_component_f32(bp, face.lsq_dr, face.lsq_w, p_g - p_o);
    accumulate_lsq_rhs_component_f32(bu, face.lsq_dr, face.lsq_w, u_g - u_o);
    accumulate_lsq_rhs_component_f32(bv, face.lsq_dr, face.lsq_w, v_g - v_o);
    accumulate_lsq_rhs_component_f32(bw, face.lsq_dr, face.lsq_w, w_g - w_o);
    Ok(())
}

fn neg_vector(v: Vector3) -> Vector3 {
    Vector3::new(-v.x, -v.y, -v.z)
}

fn sym3_from_lsq(
    a: &crate::discretization::unstructured_face_cache::LsqPrecomputedCell,
) -> Symmetric3x3 {
    Symmetric3x3 {
        a_xx: a.a_xx,
        a_xy: a.a_xy,
        a_xz: a.a_xz,
        a_yy: a.a_yy,
        a_yz: a.a_yz,
        a_zz: a.a_zz,
    }
}

fn write_gradients_f32(
    mesh_cache: &UnstructuredSolverMeshCache,
    exec: &ExecutionContext,
    out: &mut GradientFieldsT<f32>,
) -> Result<()> {
    let idwls = exec.idwls_rhs_f32();
    for (cell, geometry) in mesh_cache.lsq_geometry.iter().enumerate() {
        let mat = sym3_from_lsq(geometry);
        let drho = solve_symmetric_3x3_f32(&mat, idwls.br_f32()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 rho 最小二乘梯度样本退化"))
        })?;
        let dp = solve_symmetric_3x3_f32(&mat, idwls.bp_f32()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 p 最小二乘梯度样本退化"))
        })?;
        let du = solve_symmetric_3x3_f32(&mat, idwls.bu_f32()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 u 最小二乘梯度样本退化"))
        })?;
        let dv = solve_symmetric_3x3_f32(&mat, idwls.bv_f32()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 v 最小二乘梯度样本退化"))
        })?;
        let dw = solve_symmetric_3x3_f32(&mat, idwls.bw_f32()[cell]).ok_or_else(|| {
            AsimuError::Mesh(format!("非结构单元 {cell} 的 w 最小二乘梯度样本退化"))
        })?;
        out.drho_dx.values_mut()[cell] = drho[0];
        out.drho_dy.values_mut()[cell] = drho[1];
        out.drho_dz.values_mut()[cell] = drho[2];
        out.dp_dx.values_mut()[cell] = dp[0];
        out.dp_dy.values_mut()[cell] = dp[1];
        out.dp_dz.values_mut()[cell] = dp[2];
        out.du_dx.values_mut()[cell] = du[0];
        out.du_dy.values_mut()[cell] = du[1];
        out.du_dz.values_mut()[cell] = du[2];
        out.dv_dx.values_mut()[cell] = dv[0];
        out.dv_dy.values_mut()[cell] = dv[1];
        out.dv_dz.values_mut()[cell] = dv[2];
        out.dw_dx.values_mut()[cell] = dw[0];
        out.dw_dy.values_mut()[cell] = dw[1];
        out.dw_dz.values_mut()[cell] = dw[2];
    }
    Ok(())
}
