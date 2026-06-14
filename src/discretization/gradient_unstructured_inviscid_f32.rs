//! 非结构二阶无粘线性重构 IDWLS 梯度（f32 串行路径；读 `face_topology_f32` / `lsq_geometry_f32`）。

use tracing::info_span;

use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::UnstructuredGradientLsqInputF32;
use crate::discretization::neg_dr;
use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
use crate::discretization::unstructured_face_cache_f32::{
    LsqPrecomputedCellF32, UnstructuredBoundaryFaceF32, UnstructuredInteriorFaceF32,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::exec::cpu::{accumulate_lsq_rhs_component_f32, solve_lsq_precomputed_cell_f32};
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
    if input.mesh_cache.lsq_geometry_f32.len() != n {
        return Err(AsimuError::Field(
            "非结构 f32 IDWLS 几何缓存与网格单元数不一致".to_string(),
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
    let topology = &input.mesh_cache.face_topology_f32;
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
    face: &UnstructuredInteriorFaceF32,
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
    face: &UnstructuredInteriorFaceF32,
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
    let dr_n = neg_dr(face.lsq_dr);
    accumulate_lsq_rhs_component_f32(br, dr_n, face.lsq_w, rho_o - rho_n);
    accumulate_lsq_rhs_component_f32(bp, dr_n, face.lsq_w, p_o - p_n);
    accumulate_lsq_rhs_component_f32(bu, dr_n, face.lsq_w, u_o - u_n);
    accumulate_lsq_rhs_component_f32(bv, dr_n, face.lsq_w, v_o - v_n);
    accumulate_lsq_rhs_component_f32(bw, dr_n, face.lsq_w, w_o - w_n);
    Ok(())
}

fn accumulate_boundary_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredBoundaryFaceF32,
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

fn write_gradients_f32(
    mesh_cache: &UnstructuredSolverMeshCache,
    exec: &ExecutionContext,
    out: &mut GradientFieldsT<f32>,
) -> Result<()> {
    let idwls = exec.idwls_rhs_f32();
    for (cell, geometry) in mesh_cache.lsq_geometry_f32.iter().enumerate() {
        let drho = solve_lsq_cell_f32(geometry, idwls.br_f32()[cell], "rho", cell)?;
        let dp = solve_lsq_cell_f32(geometry, idwls.bp_f32()[cell], "p", cell)?;
        let du = solve_lsq_cell_f32(geometry, idwls.bu_f32()[cell], "u", cell)?;
        let dv = solve_lsq_cell_f32(geometry, idwls.bv_f32()[cell], "v", cell)?;
        let dw = solve_lsq_cell_f32(geometry, idwls.bw_f32()[cell], "w", cell)?;
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

fn solve_lsq_cell_f32(
    geometry: &LsqPrecomputedCellF32,
    rhs: [f32; 3],
    component: &str,
    cell: usize,
) -> Result<[f32; 3]> {
    solve_lsq_precomputed_cell_f32(geometry, rhs).ok_or_else(|| {
        AsimuError::Mesh(format!(
            "非结构单元 {cell} 的 {component} 最小二乘梯度样本退化"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::{ComputeFloat, approx_eq};
    use crate::discretization::{
        BoundaryGhostBuffer, GhostCellState, UnstructuredGradientScratch,
        UnstructuredSolverMeshCache,
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq,
    };
    use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
    use crate::field::{ConservedFields, ConservedFieldsT, PrimitiveFields, PrimitiveFieldsT};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS};

    fn tet_mesh_and_boundary() -> (UnstructuredMesh3d, BoundarySet) {
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: 0.2,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        (mesh, boundary)
    }

    #[test]
    fn f32_inviscid_idwls_gradients_match_f64_on_uniform_freestream_tet() {
        let (mesh, boundary) = tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields_f64 =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let fields_f32 =
            ConservedFieldsT::<f32>::from_real_fields(&fields_f64).expect("fields f32");
        let mut ghosts_f64 = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let mut ghosts_f32 = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let state_f64 = fields_f64.cell_state(0).expect("state");
        let state_f32 = fields_f32.cell_state(0).expect("state");
        for face in 0..mesh.num_faces() {
            let face = crate::core::FaceId(face as u32);
            ghosts_f64.insert_face(
                face,
                GhostCellState {
                    conserved: state_f64,
                },
            );
            ghosts_f32.insert_face(
                face,
                GhostCellState {
                    conserved: state_f32,
                },
            );
        }
        let mut prim_f64 = PrimitiveFields::zeros(mesh.num_cells()).expect("prim f64");
        let mut prim_f32 = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim f32");
        prim_f64
            .fill_from_conserved(&fields_f64, &eos, 1.0e-8)
            .expect("fill f64");
        prim_f32
            .fill_from_conserved(&fields_f32, &eos, 1.0e-8)
            .expect("fill f32");
        let mut grad_f64 =
            crate::discretization::GradientFields::zeros(mesh.num_cells()).expect("g");
        let mut grad_f32 =
            crate::discretization::GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("g f32");
        let mut scratch_f64 = UnstructuredGradientScratch::new(mesh.num_cells());
        let mut exec_f64 = ExecutionContext::new(ExecConfig::default(), MeshExecMetrics::empty())
            .expect("exec f64");
        let mut exec_f32 = ExecutionContext::new(ExecConfig::default(), MeshExecMetrics::empty())
            .expect("exec f32");
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq(
            crate::discretization::UnstructuredGradientLsqInput {
                mesh: &mesh,
                mesh_cache: &mesh_cache,
                primitives: &prim_f64,
                eos: &eos,
                ghosts: &ghosts_f64,
                min_pressure: 1.0e-8,
                viscous: None,
            },
            &mut grad_f64,
            &mut scratch_f64,
            &mut exec_f64,
        )
        .expect("grad f64");
        compute_unstructured_inviscid_linear_reconstruction_gradients_idw_lsq_f32(
            UnstructuredGradientLsqInputF32 {
                mesh: &mesh,
                mesh_cache: &mesh_cache,
                primitives: &prim_f32,
                eos: &eos,
                ghosts: &ghosts_f32,
                min_pressure: 1.0e-8,
                viscous: None,
            },
            &mut grad_f32,
            &mut exec_f32,
        )
        .expect("grad f32");
        assert!(approx_eq(
            grad_f32.dp_dx.values()[0].to_real(),
            grad_f64.dp_dx.values()[0],
            1.0e-3
        ));
        assert!(approx_eq(
            grad_f32.du_dx.values()[0].to_real(),
            grad_f64.du_dx.values()[0],
            1.0e-3
        ));
    }
}
