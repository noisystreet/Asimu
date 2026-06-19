//! 非结构 f32 粘性 IDWLS RHS CUDA 分支（ADR 0017 P4）。

use crate::discretization::gradient_unstructured_f32::UnstructuredGradientLsqInputF32;
use crate::discretization::unstructured_idwls_exec_topo::IdwlsGhostSampleHost;
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::field::primitive_from_conserved_relaxed_f32_from_state;

pub(super) fn try_accumulate_lsq_rhs_f32_cuda(
    input: &UnstructuredGradientLsqInputF32<'_>,
    temperatures: &[f32],
    exec: &mut ExecutionContext,
) -> Result<bool> {
    if try_accumulate_and_solve_idwls_f32_cuda(input, temperatures, exec)? {
        return Ok(true);
    }
    let boundary_ghosts_storage = if exec.cuda_boundary_ghosts_on_device() {
        None
    } else {
        Some(prepare_boundary_ghost_samples_f32(input)?)
    };
    let boundary_ghosts = boundary_ghosts_storage.as_deref().unwrap_or(&[]);
    let topo = &input.mesh_cache.idwls_viscous_topo;
    let topo_key = std::ptr::from_ref(input.mesh_cache).addr();
    crate::exec::idwls_cuda::try_accumulate_viscous_rhs_f32_cuda(
        exec,
        input.primitives,
        topo,
        topo_key,
        temperatures,
        boundary_ghosts,
    )
}

fn try_accumulate_and_solve_idwls_f32_cuda(
    input: &UnstructuredGradientLsqInputF32<'_>,
    temperatures: &[f32],
    exec: &mut ExecutionContext,
) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        if exec.device() == crate::core::ExecDevice::GpuCuda && exec.cuda_rhs_pipeline_active() {
            if temperatures.is_empty() {
                if let Some(viscous) = input.viscous {
                    exec.cuda_ensure_cell_temperatures_from_device_primitives(
                        input.mesh.num_cells(),
                        input.eos,
                        viscous,
                    )?;
                }
            }
            let boundary_ghosts_storage = if exec.cuda_boundary_ghosts_on_device() {
                None
            } else {
                Some(prepare_boundary_ghost_samples_f32(input)?)
            };
            let boundary_ghosts = boundary_ghosts_storage.as_deref().unwrap_or(&[]);
            let topo = &input.mesh_cache.idwls_viscous_topo;
            let topo_key = std::ptr::from_ref(input.mesh_cache).addr();
            exec.cuda_accumulate_and_solve_idwls_viscous_gradients(
                input.primitives,
                topo,
                topo_key,
                &input.mesh_cache.lsq_geometry_f32,
                temperatures,
                boundary_ghosts,
            )?;
            return Ok(true);
        }
    }
    let _ = (input, temperatures, exec);
    Ok(false)
}

fn prepare_boundary_ghost_samples_f32(
    input: &UnstructuredGradientLsqInputF32<'_>,
) -> Result<Vec<IdwlsGhostSampleHost>> {
    let mut out = Vec::with_capacity(input.mesh_cache.face_topology_f32.boundary.len());
    for face in &input.mesh_cache.face_topology_f32.boundary {
        let ghost = input.ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "非结构 f32 CUDA IDWLS 边界面 FaceId({}) 缺少 ghost",
                face.face.index()
            ))
        })?;
        let prim = primitive_from_conserved_relaxed_f32_from_state(
            input.eos,
            &ghost.conserved,
            input.min_pressure,
        )?;
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
        out.push(IdwlsGhostSampleHost {
            u: prim.velocity[0],
            v: prim.velocity[1],
            w: prim.velocity[2],
            t,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ExecDevice;
    use crate::core::approx_eq;
    use crate::discretization::gradient_unstructured_f32::{
        UnstructuredGradientLsqInputF32, UnstructuredGradientScratchF32,
    };
    use crate::discretization::{BoundaryGhostBuffer, GhostCellState, UnstructuredSolverMeshCache};
    use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
    use crate::field::{ConservedFields, ConservedFieldsT, PrimitiveFieldsT};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS, ViscousPhysicsConfig};

    fn uniform_closed_tet() -> (
        UnstructuredMesh3d,
        BoundarySet,
        UnstructuredSolverMeshCache,
        BoundaryGhostBuffer,
        IdealGasEoS,
        PrimitiveFieldsT<f32>,
        ViscousPhysicsConfig,
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
        let fields_f32 = ConservedFieldsT::<f32>::from_real_fields(&fields).expect("fields f32");
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
        let viscous = ViscousPhysicsConfig::default();
        (mesh, boundary, cache, ghosts, eos, primitives, viscous)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "gpu"]
    fn cuda_idwls_rhs_matches_cpu_serial_on_uniform_tet() {
        let (mesh, _boundary, cache, ghosts, eos, primitives, viscous) = uniform_closed_tet();
        let mut scratch = UnstructuredGradientScratchF32::new(mesh.num_cells());
        scratch.temperatures = vec![288.15f32; mesh.num_cells()];
        let input = UnstructuredGradientLsqInputF32 {
            mesh: &mesh,
            mesh_cache: &cache,
            primitives: &primitives,
            eos: &eos,
            ghosts: &ghosts,
            min_pressure: 1.0e-8,
            viscous: Some(&viscous),
        };
        let metrics = MeshExecMetrics::new(
            mesh.num_cells(),
            cache.face_topology.interior.len(),
            mesh.num_cells(),
        );

        let mut exec_cpu = ExecutionContext::new(ExecConfig::default(), metrics).expect("cpu ctx");
        exec_cpu.idwls_prepare_viscous_f32(mesh.num_cells());
        crate::discretization::gradient_unstructured_f32::accumulate_lsq_rhs_f32_cpu_serial(
            &input,
            &scratch,
            &mut exec_cpu,
        )
        .expect("cpu");
        let cpu = exec_cpu.idwls_rhs_f32();

        let mut exec_cuda = ExecutionContext::new(
            ExecConfig {
                device: ExecDevice::GpuCuda,
                ..ExecConfig::default()
            },
            metrics,
        )
        .expect("cuda ctx");
        exec_cuda.idwls_prepare_viscous_f32(mesh.num_cells());
        super::try_accumulate_lsq_rhs_f32_cuda(&input, &scratch.temperatures, &mut exec_cuda)
            .expect("cuda try")
            .then_some(())
            .expect("cuda path");
        let gpu = exec_cuda.idwls_rhs_f32();

        for comp in 0..3 {
            assert!(approx_eq(
                cpu.bu_f32()[0][comp] as f64,
                gpu.bu_f32()[0][comp] as f64,
                1.0e-4
            ));
            assert!(approx_eq(
                cpu.bv_f32()[0][comp] as f64,
                gpu.bv_f32()[0][comp] as f64,
                1.0e-4
            ));
            assert!(approx_eq(
                cpu.bw_f32()[0][comp] as f64,
                gpu.bw_f32()[0][comp] as f64,
                1.0e-4
            ));
            assert!(approx_eq(
                cpu.bt_f32()[0][comp] as f64,
                gpu.bt_f32()[0][comp] as f64,
                1.0e-4
            ));
        }
    }
}
