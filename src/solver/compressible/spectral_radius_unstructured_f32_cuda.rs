//! 非结构 f32 谱半径 CUDA 分支。

use crate::core::Real;
use crate::discretization::unstructured_spectral_exec_topo::SpectralGhostPrimHost;
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::field::primitive_from_conserved_relaxed_f32_from_state;
use crate::solver::compressible::spectral_radius_f32::cell_viscous_diffusivity_max_f32;
use crate::solver::compressible::spectral_radius_unstructured::SpectralRadiusUnstructuredTypedParams;
use crate::solver::compressible::spectral_radius_unstructured_f32::{
    SpectralRadiusUnstructuredF32Params, cell_spectral_radius_unstructured_f32,
};
use crate::solver::finalize_cell_dts_from_sigma_f32;

/// CUDA 优先；不可用时回退 CPU 串行。
pub(crate) fn compute_spectral_radius_f32_with_exec(
    params: &SpectralRadiusUnstructuredTypedParams<'_, f32>,
    exec: &mut ExecutionContext,
    cfl: Real,
    fixed_dt: Option<Real>,
    local_time_step: bool,
    keep_timestep_on_device: bool,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let n = params.mesh.num_cells();
    #[cfg(feature = "cuda")]
    {
        let boundary_ghosts_storage = if exec.cuda_boundary_ghosts_on_device() {
            None
        } else {
            Some(prepare_boundary_ghost_prims_f32(params)?)
        };
        let boundary_ghosts = boundary_ghosts_storage.as_deref().unwrap_or(&[]);
        let diffusivity = if exec.cuda_spectral_diffusivity_on_device() {
            None
        } else if let Some(viscous) = params.viscous {
            Some(cell_viscous_diffusivity_max_f32(
                params.primitives,
                params.eos,
                viscous,
            )?)
        } else {
            None
        };
        let mut sigma = vec![0.0f32; n];
        let topo = &params.mesh_cache.spectral_radius_topo;
        let topo_key = std::ptr::from_ref(params.mesh_cache).addr();
        if crate::exec::spectral_radius_cuda::try_compute_spectral_radius_unstructured_f32(
            exec,
            &crate::exec::spectral_radius_cuda::SpectralRadiusCudaInput {
                primitives: params.primitives,
                topo,
                topo_key,
                gamma: params.eos.gamma as f32,
                boundary_ghosts,
                diffusivity: diffusivity.as_deref(),
                cfl: cfl as f32,
                fixed_dt: fixed_dt.map(|d| d as f32),
                defer_timestep_d2h: true,
            },
            &mut sigma,
        )? {
            if keep_timestep_on_device {
                return Ok((Vec::new(), None));
            }
            let mut cell_dts = vec![0.0f32; n];
            crate::exec::spectral_radius_cuda::download_timestep_f32(
                exec,
                &mut sigma,
                &mut cell_dts,
                local_time_step,
            )?;
            return Ok((sigma, Some(cell_dts)));
        }
    }
    let _ = (
        exec,
        cfl,
        fixed_dt,
        local_time_step,
        keep_timestep_on_device,
    );
    let sigma = cell_spectral_radius_unstructured_f32(&SpectralRadiusUnstructuredF32Params {
        mesh: params.mesh,
        mesh_cache: params.mesh_cache,
        boundaries: params.boundaries,
        ghosts: params.ghosts,
        primitives: params.primitives,
        eos: params.eos,
        min_pressure: params.min_pressure,
        viscous: params.viscous,
    })?;
    let volumes: Vec<f32> = params
        .mesh
        .cell_volumes()
        .iter()
        .map(|v| *v as f32)
        .collect();
    let cell_dts = finalize_cell_dts_from_sigma_f32(
        &volumes,
        &sigma,
        cfl as f32,
        fixed_dt.map(|d| d as f32),
        local_time_step,
    )?;
    Ok((sigma, Some(cell_dts)))
}

fn prepare_boundary_ghost_prims_f32(
    params: &SpectralRadiusUnstructuredTypedParams<'_, f32>,
) -> Result<Vec<SpectralGhostPrimHost>> {
    let mut out = Vec::with_capacity(params.mesh_cache.face_topology_f32.boundary.len());
    for face in &params.mesh_cache.face_topology_f32.boundary {
        let ghost = params.ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "谱半径 CUDA 边界面 FaceId({}) 缺少 ghost",
                face.face.index()
            ))
        })?;
        let prim = primitive_from_conserved_relaxed_f32_from_state(
            params.eos,
            &ghost.conserved,
            params.min_pressure,
        )?;
        out.push(SpectralGhostPrimHost {
            rho: prim.density,
            pressure: prim.pressure,
            u: prim.velocity[0],
            v: prim.velocity[1],
            w: prim.velocity[2],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ExecDevice;
    use crate::core::approx_eq;
    use crate::discretization::{BoundaryGhostBuffer, GhostCellState, UnstructuredSolverMeshCache};
    use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
    use crate::field::{ConservedFields, PrimitiveFieldsT};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS, ViscousPhysicsConfig};
    use crate::solver::compressible::spectral_radius_unstructured::SpectralRadiusUnstructuredTypedParams;
    #[cfg(feature = "cuda")]
    use crate::solver::compressible::spectral_radius_unstructured_f32::{
        SpectralRadiusUnstructuredF32Params, cell_spectral_radius_unstructured_f32,
    };

    fn uniform_closed_tet() -> (
        UnstructuredMesh3d,
        BoundarySet,
        UnstructuredSolverMeshCache,
        BoundaryGhostBuffer,
        IdealGasEoS,
        PrimitiveFieldsT<f32>,
    ) {
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
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let fields_f32 =
            crate::field::ConservedFieldsT::<f32>::from_real_fields(&fields).expect("f32");
        let mut primitives = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields_f32, &eos, 1.0e-8)
            .expect("fill");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        for &face in &faces {
            ghosts.insert_face(face, GhostCellState { conserved: state });
        }
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: fs.mach,
                pressure: fs.pressure,
                temperature: fs.temperature,
                alpha: fs.alpha,
                beta: fs.beta,
            },
        )]);
        let cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        (mesh, boundary, cache, ghosts, eos, primitives)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "gpu"]
    fn cuda_spectral_radius_matches_cpu_on_uniform_tet() {
        let (mesh, boundary, cache, ghosts, eos, primitives) = uniform_closed_tet();
        let viscous = ViscousPhysicsConfig::default();
        let params = SpectralRadiusUnstructuredTypedParams {
            mesh: &mesh,
            mesh_cache: &cache,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: 1.0e-8,
            viscous: Some(&viscous),
        };
        let cpu = cell_spectral_radius_unstructured_f32(&SpectralRadiusUnstructuredF32Params {
            mesh: params.mesh,
            mesh_cache: params.mesh_cache,
            boundaries: params.boundaries,
            ghosts: params.ghosts,
            primitives: params.primitives,
            eos: params.eos,
            min_pressure: params.min_pressure,
            viscous: params.viscous,
        })
        .expect("cpu");

        let mut exec = ExecutionContext::new(
            ExecConfig {
                device: ExecDevice::GpuCuda,
                ..ExecConfig::default()
            },
            MeshExecMetrics::new(mesh.num_cells(), cache.face_topology.interior.len(), 1),
        )
        .expect("cuda");
        let gpu = super::compute_spectral_radius_f32_with_exec(
            &params, &mut exec, 0.5, None, true, false,
        )
        .expect("cuda");
        assert!(approx_eq(gpu.0[0] as f64, cpu[0] as f64, 1.0e-3));
    }
}
