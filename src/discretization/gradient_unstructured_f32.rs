//! 非结构 IDWLS 梯度（f32 串行路径；面样本与矩阵读 `face_topology_f32` / `lsq_geometry_f32`）。

use tracing::info_span;

use crate::core::{ComputeFloat, Real};
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::neg_dr;
use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
use crate::discretization::unstructured_face_cache_f32::{
    LsqPrecomputedCellF32, UnstructuredBoundaryFaceF32, UnstructuredInteriorFaceF32,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::exec::cpu::{accumulate_lsq_rhs_component_f32, solve_lsq_precomputed_cell_f32};
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed_f32_from_state};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

/// f32 非结构 IDWLS 梯度输入。
pub struct UnstructuredGradientLsqInputF32<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub eos: &'a IdealGasEoS,
    pub ghosts: &'a crate::discretization::BoundaryGhostBuffer,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

pub struct UnstructuredGradientScratchF32 {
    pub temperatures: Vec<f32>,
}

impl UnstructuredGradientScratchF32 {
    #[must_use]
    pub fn new(_num_cells: usize) -> Self {
        Self {
            temperatures: Vec::new(),
        }
    }

    fn prepare_temperatures(&mut self, num_cells: usize) {
        self.temperatures.resize(num_cells, 0.0);
    }
}

/// 非结构粘性 IDWLS 梯度（f32）。
pub fn compute_unstructured_gradients_idw_lsq_f32(
    input: UnstructuredGradientLsqInputF32<'_>,
    out: &mut GradientFieldsT<f32>,
    scratch: &mut UnstructuredGradientScratchF32,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let n = input.mesh.num_cells();
    if input.primitives.num_cells() != n || out.num_cells() != n {
        return Err(AsimuError::Field(
            "非结构 f32 梯度场与原始变量场尺寸不一致".to_string(),
        ));
    }
    if input.mesh_cache.lsq_geometry_f32.len() != n {
        return Err(AsimuError::Field(
            "非结构 f32 IDWLS 几何缓存与网格单元数不一致".to_string(),
        ));
    }
    out.clear();
    scratch.prepare_temperatures(n);
    cell_temperatures_f32_into(
        input.primitives,
        input.eos,
        input.viscous,
        &mut scratch.temperatures,
    )?;
    exec.idwls_prepare_viscous_f32(n);
    {
        let topology = &input.mesh_cache.face_topology_f32;
        let _span = info_span!(
            "unstructured_idw_lsq_accumulate_rhs_f32",
            interior_faces = topology.interior.len(),
            boundary_faces = topology.boundary.len(),
        )
        .entered();
        accumulate_lsq_rhs_f32(&input, scratch, exec)?;
    }
    {
        let _span = info_span!("unstructured_idw_lsq_solve_gradients_f32", cells = n).entered();
        write_lsq_gradients_f32(input.mesh_cache, exec, out)
    }
}

fn cell_temperatures_f32_into(
    primitives: &PrimitiveFieldsT<f32>,
    eos: &IdealGasEoS,
    viscous: Option<&ViscousPhysicsConfig>,
    out: &mut Vec<f32>,
) -> Result<()> {
    let n = primitives.num_cells();
    out.resize(n, 0.0);
    for (i, ti) in out.iter_mut().enumerate().take(n) {
        let rho = primitives.density.values()[i].to_real();
        let p = primitives.pressure.values()[i].to_real();
        if rho <= 0.0 || p <= 0.0 {
            return Err(AsimuError::Field(
                "密度或压力非正，无法计算温度".to_string(),
            ));
        }
        let t = viscous
            .map(|v| v.static_temperature(p, rho, eos))
            .unwrap_or(p / (rho * eos.gas_constant));
        *ti = t as f32;
    }
    Ok(())
}

fn accumulate_lsq_rhs_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    scratch: &UnstructuredGradientScratchF32,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let topology = &input.mesh_cache.face_topology_f32;
    let temperatures = &scratch.temperatures;
    let idwls = exec.scratch_mut().idwls_mut();
    let (bu, bv, bw, bt) = idwls.viscous_arrays_mut_f32();
    for face in &topology.interior {
        accumulate_interior_as_owner_f32(
            input,
            face,
            temperatures,
            &mut bu[face.owner],
            &mut bv[face.owner],
            &mut bw[face.owner],
            &mut bt[face.owner],
        )?;
        accumulate_interior_as_neighbor_f32(
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
        accumulate_boundary_f32(
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

fn accumulate_interior_as_owner_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredInteriorFaceF32,
    temperatures: &[f32],
    bu: &mut [f32; 3],
    bv: &mut [f32; 3],
    bw: &mut [f32; 3],
    bt: &mut [f32; 3],
) -> Result<()> {
    let prim = input.primitives;
    let u_o = prim.velocity_x.values()[face.owner];
    let v_o = prim.velocity_y.values()[face.owner];
    let w_o = prim.velocity_z.values()[face.owner];
    let t_o = temperatures[face.owner];
    let u_n = prim.velocity_x.values()[face.neighbor];
    let v_n = prim.velocity_y.values()[face.neighbor];
    let w_n = prim.velocity_z.values()[face.neighbor];
    let t_n = temperatures[face.neighbor];
    accumulate_lsq_rhs_component_f32(bu, face.lsq_dr, face.lsq_w, u_n - u_o);
    accumulate_lsq_rhs_component_f32(bv, face.lsq_dr, face.lsq_w, v_n - v_o);
    accumulate_lsq_rhs_component_f32(bw, face.lsq_dr, face.lsq_w, w_n - w_o);
    accumulate_lsq_rhs_component_f32(bt, face.lsq_dr, face.lsq_w, t_n - t_o);
    Ok(())
}

fn accumulate_interior_as_neighbor_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredInteriorFaceF32,
    temperatures: &[f32],
    bu: &mut [f32; 3],
    bv: &mut [f32; 3],
    bw: &mut [f32; 3],
    bt: &mut [f32; 3],
) -> Result<()> {
    let prim = input.primitives;
    let u_o = prim.velocity_x.values()[face.owner];
    let v_o = prim.velocity_y.values()[face.owner];
    let w_o = prim.velocity_z.values()[face.owner];
    let t_o = temperatures[face.owner];
    let u_n = prim.velocity_x.values()[face.neighbor];
    let v_n = prim.velocity_y.values()[face.neighbor];
    let w_n = prim.velocity_z.values()[face.neighbor];
    let t_n = temperatures[face.neighbor];
    let dr_n = neg_dr(face.lsq_dr);
    accumulate_lsq_rhs_component_f32(bu, dr_n, face.lsq_w, u_o - u_n);
    accumulate_lsq_rhs_component_f32(bv, dr_n, face.lsq_w, v_o - v_n);
    accumulate_lsq_rhs_component_f32(bw, dr_n, face.lsq_w, w_o - w_n);
    accumulate_lsq_rhs_component_f32(bt, dr_n, face.lsq_w, t_o - t_n);
    Ok(())
}

fn accumulate_boundary_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    face: &UnstructuredBoundaryFaceF32,
    temperatures: &[f32],
    bu: &mut [f32; 3],
    bv: &mut [f32; 3],
    bw: &mut [f32; 3],
    bt: &mut [f32; 3],
) -> Result<()> {
    let owner = face.owner;
    let prim = input.primitives;
    let u_o = prim.velocity_x.values()[owner];
    let v_o = prim.velocity_y.values()[owner];
    let w_o = prim.velocity_z.values()[owner];
    let t_o = temperatures[owner];
    let ghost = input.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "非结构 f32 梯度边界面 FaceId({}) 缺少 ghost",
            face.face.index()
        ))
    })?;
    let ghost_sample = ghost_scalar_sample_f32(input, ghost.conserved)?;
    accumulate_lsq_rhs_component_f32(bu, face.lsq_dr, face.lsq_w, ghost_sample.u - u_o);
    accumulate_lsq_rhs_component_f32(bv, face.lsq_dr, face.lsq_w, ghost_sample.v - v_o);
    accumulate_lsq_rhs_component_f32(bw, face.lsq_dr, face.lsq_w, ghost_sample.w - w_o);
    accumulate_lsq_rhs_component_f32(bt, face.lsq_dr, face.lsq_w, ghost_sample.t - t_o);
    Ok(())
}

struct ScalarSampleF32 {
    u: f32,
    v: f32,
    w: f32,
    t: f32,
}

fn ghost_scalar_sample_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
    conserved: crate::physics::ConservedState,
) -> Result<ScalarSampleF32> {
    let prim =
        primitive_from_conserved_relaxed_f32_from_state(input.eos, &conserved, input.min_pressure)?;
    let t = input
        .viscous
        .map(|v| {
            v.static_temperature(
                prim.pressure as crate::core::Real,
                prim.density as crate::core::Real,
                input.eos,
            ) as f32
        })
        .unwrap_or(prim.temperature);
    Ok(ScalarSampleF32 {
        u: prim.velocity[0],
        v: prim.velocity[1],
        w: prim.velocity[2],
        t,
    })
}

fn write_lsq_gradients_f32(
    mesh_cache: &UnstructuredSolverMeshCache,
    exec: &ExecutionContext,
    out: &mut GradientFieldsT<f32>,
) -> Result<()> {
    let idwls = exec.idwls_rhs_f32();
    for (cell, geometry) in mesh_cache.lsq_geometry_f32.iter().enumerate() {
        let du = solve_lsq_cell_f32(geometry, idwls.bu_f32()[cell], "u", cell)?;
        let dv = solve_lsq_cell_f32(geometry, idwls.bv_f32()[cell], "v", cell)?;
        let dw = solve_lsq_cell_f32(geometry, idwls.bw_f32()[cell], "w", cell)?;
        let dt = solve_lsq_cell_f32(geometry, idwls.bt_f32()[cell], "T", cell)?;
        out.du_dx.values_mut()[cell] = du[0];
        out.du_dy.values_mut()[cell] = du[1];
        out.du_dz.values_mut()[cell] = du[2];
        out.dv_dx.values_mut()[cell] = dv[0];
        out.dv_dy.values_mut()[cell] = dv[1];
        out.dv_dz.values_mut()[cell] = dv[2];
        out.dw_dx.values_mut()[cell] = dw[0];
        out.dw_dy.values_mut()[cell] = dw[1];
        out.dw_dz.values_mut()[cell] = dw[2];
        out.dt_dx.values_mut()[cell] = dt[0];
        out.dt_dy.values_mut()[cell] = dt[1];
        out.dt_dz.values_mut()[cell] = dt[2];
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
