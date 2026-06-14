//! 非结构 f32 粘性内面 CUDA 分支（ADR 0017 G2）。

use tracing::info_span;

use crate::error::Result;

use super::super::assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredScratch, prepare_unstructured_viscous_transport_f32,
};
use super::ViscousAssemblyUnstructuredF32Input;
use super::ViscousInteriorAssemblyF32;

pub(super) fn cuda_viscous_f32_interior(
    residual: &mut crate::field::ConservedResidualT<f32>,
    params: &mut ViscousInteriorAssemblyF32<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<bool> {
    scratch.init_cuda_viscous_topo_from_mesh_cache(params.mesh_cache);
    let topo_key = std::ptr::from_ref(params.mesh_cache).addr();
    {
        let _span = info_span!(
            "unstructured_viscous_prepare_transport_f32",
            cells = params.primitives.num_cells(),
            interior_faces = params.face_topology.interior.len(),
        )
        .entered();
        let topo = scratch
            .cuda_viscous_topo_ref()
            .expect("cuda viscous topo after init");
        if !crate::exec::viscous::try_prepare_unstructured_viscous_transport_f32_cuda(
            params.exec,
            topo,
            topo_key,
            params.temperatures,
            params.viscous,
            params.eos,
        )? {
            let constant = prepare_unstructured_viscous_transport_f32(
                params.transport_topology,
                params.primitives.num_cells(),
                params.viscous,
                params.eos,
                params.temperatures,
                scratch,
            )?;
            scratch.apply_transport_to_cuda_viscous_topo(constant);
        }
    }
    let topo = scratch
        .cuda_viscous_topo_ref()
        .expect("cuda viscous topo after prepare");
    crate::exec::viscous::try_assemble_viscous_interior_f32(
        params.exec,
        residual,
        params.primitives,
        params.gradients,
        topo,
        topo_key,
    )
}

pub(super) fn cuda_viscous_f32_boundary(
    residual: &mut crate::field::ConservedResidualT<f32>,
    input: &mut ViscousAssemblyUnstructuredF32Input<'_>,
    grad_scratch: &crate::discretization::UnstructuredGradientScratchF32,
) -> Result<bool> {
    if input.mesh_cache.cuda_viscous_boundary_topo.num_faces() == 0 {
        return Ok(true);
    }
    let boundary_ghosts_storage = if input.exec.cuda_boundary_ghosts_on_device() {
        None
    } else {
        Some(
            crate::discretization::unstructured_boundary_exec_topo::prepare_viscous_boundary_ghost_prims_f32(
                &input.mesh_cache.face_topology_f32,
                input.ghosts,
                input.eos,
                input.viscous,
                input.min_pressure,
            )?,
        )
    };
    let boundary_ghosts = boundary_ghosts_storage.as_deref().unwrap_or(&[]);
    let topo = &input.mesh_cache.cuda_viscous_boundary_topo;
    let topo_key = std::ptr::from_ref(input.mesh_cache).addr();
    crate::exec::viscous::try_assemble_viscous_boundary_f32(
        input.exec,
        residual,
        input.primitives,
        input.gradient_scratch,
        crate::exec::gpu::cuda::CudaViscousBoundaryInput {
            topo,
            topo_key,
            boundary_ghosts,
            temperatures: &grad_scratch.temperatures,
            viscous: input.viscous,
            eos: input.eos,
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ComputeFloat;
    use crate::core::ExecDevice;
    use crate::core::approx_eq;
    use crate::discretization::UnstructuredGradientScratchF32;
    use crate::discretization::gradient_typed::GradientFieldsT;
    use crate::discretization::{BoundaryGhostBuffer, UnstructuredSolverMeshCache};
    use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
    use crate::field::{ConservedFields, ConservedResidualT, PrimitiveFieldsT};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::{FreestreamParams, IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

    use crate::discretization::residual::{
        ViscousAssemblyUnstructuredF32Input, ViscousAssemblyUnstructuredScratch,
        compute_gradients_and_assemble_viscous_unstructured_f32,
    };

    fn uniform_closed_tet() -> (
        UnstructuredMesh3d,
        IdealGasEoS,
        BoundarySet,
        UnstructuredSolverMeshCache,
        BoundaryGhostBuffer,
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
        let mut primitives_f32 = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        let fields_f32 =
            crate::field::ConservedFieldsT::<f32>::from_real_fields(&fields).expect("f32");
        primitives_f32
            .fill_from_conserved(&fields_f32, &eos, 1.0e-8)
            .expect("fill");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        for &face in &faces {
            ghosts.insert_face(
                face,
                crate::discretization::GhostCellState { conserved: state },
            );
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
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32)
    }

    #[test]
    #[ignore = "gpu"]
    fn cpu_f32_matches_cuda_f32_viscous_single_tet() {
        let (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32) = uniform_closed_tet();
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("viscous");

        let mut cpu_rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("cpu rhs");
        let mut cpu_exec = ExecutionContext::for_unit_test();
        let mut grad = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut grad_scratch = UnstructuredGradientScratchF32::new(mesh.num_cells());
        let mut cpu_input = ViscousAssemblyUnstructuredF32Input {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives_f32,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
            exec: &mut cpu_exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_f32(
            &mut cpu_rhs,
            &mut cpu_input,
            &mut scratch,
            &mut grad_scratch,
        )
        .expect("cpu visc");

        let mut cuda_rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("cuda rhs");
        let cuda_config = ExecConfig {
            device: ExecDevice::GpuCuda,
            ..Default::default()
        };
        let mut cuda_exec =
            ExecutionContext::new(cuda_config, MeshExecMetrics::empty()).expect("cuda exec");
        let mut grad2 = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut scratch2 = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut grad_scratch2 = UnstructuredGradientScratchF32::new(mesh.num_cells());
        let mut cuda_input = ViscousAssemblyUnstructuredF32Input {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives_f32,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad2,
            exec: &mut cuda_exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_f32(
            &mut cuda_rhs,
            &mut cuda_input,
            &mut scratch2,
            &mut grad_scratch2,
        )
        .expect("cuda visc");

        for i in 0..mesh.num_cells() {
            assert!(
                approx_eq(
                    cpu_rhs.momentum_x.values()[i].to_real(),
                    cuda_rhs.momentum_x.values()[i].to_real(),
                    1.0e-4
                ),
                "mx cell {i}"
            );
            assert!(
                approx_eq(
                    cpu_rhs.momentum_y.values()[i].to_real(),
                    cuda_rhs.momentum_y.values()[i].to_real(),
                    1.0e-4
                ),
                "my cell {i}"
            );
            assert!(
                approx_eq(
                    cpu_rhs.total_energy.values()[i].to_real(),
                    cuda_rhs.total_energy.values()[i].to_real(),
                    1.0e-3
                ),
                "energy cell {i}"
            );
        }
    }
}
