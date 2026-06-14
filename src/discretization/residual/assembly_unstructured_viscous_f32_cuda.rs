//! 非结构 f32 粘性内面 CUDA 分支（ADR 0017 G2）。

use tracing::info_span;

use crate::core::Real;
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;
use crate::error::Result;
use crate::exec::gpu::cuda::{
    DeviceViscousFaceGeom, ExecInteriorColorBucket, ExecViscousInteriorTopology,
};

use super::super::assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredScratch, prepare_unstructured_viscous_transport_f32,
};
use super::ViscousInteriorAssemblyF32;

pub(super) fn cuda_viscous_f32_interior(
    residual: &mut crate::field::ConservedResidualT<f32>,
    params: &mut ViscousInteriorAssemblyF32<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<bool> {
    let constant = {
        let _span = info_span!(
            "unstructured_viscous_prepare_transport_f32",
            cells = params.primitives.num_cells(),
            interior_faces = params.face_topology.interior.len(),
        )
        .entered();
        prepare_unstructured_viscous_transport_f32(
            params.transport_topology,
            params.primitives.num_cells(),
            params.viscous,
            params.eos,
            params.temperatures,
            scratch,
        )?
    };
    let exec_topo = {
        let _span = info_span!(
            "unstructured_viscous_build_exec_topo_f32",
            interior_faces = params.face_topology.interior.len(),
            colors = params.transport_topology.interior_coloring.buckets.len(),
        )
        .entered();
        build_exec_viscous_topology(
            params.face_topology,
            params.transport_topology,
            constant,
            scratch,
        )
    };
    let topo_key = std::ptr::from_ref(params.transport_topology).addr();
    crate::exec::viscous::try_assemble_viscous_interior_f32(
        params.exec,
        residual,
        params.primitives,
        params.gradients,
        &exec_topo,
        topo_key,
    )
}

fn build_exec_viscous_topology(
    topology_f32: &UnstructuredFaceTopologyF32,
    coloring: &UnstructuredFaceTopology,
    constant: Option<(Real, Real)>,
    scratch: &ViscousAssemblyUnstructuredScratch,
) -> ExecViscousInteriorTopology {
    let faces = topology_f32
        .interior
        .iter()
        .enumerate()
        .map(|(face_idx, face)| {
            let (mu, lambda) = if let Some((m, l)) = constant {
                (m as f32, l as f32)
            } else {
                let (m, l) = scratch.face_transport_at(face_idx);
                (m as f32, l as f32)
            };
            let mut nx = face.normal[0];
            let mut ny = face.normal[1];
            let mut nz = face.normal[2];
            let mag = (nx * nx + ny * ny + nz * nz).sqrt();
            if mag > 1.0e-30 {
                let inv = 1.0 / mag;
                nx *= inv;
                ny *= inv;
                nz *= inv;
            }
            DeviceViscousFaceGeom {
                owner: face.owner as u32,
                neighbor: face.neighbor as u32,
                nx,
                ny,
                nz,
                mu,
                lambda,
                owner_scale: face.owner_rhs_scale,
                neighbor_scale: face.neighbor_rhs_scale,
            }
        })
        .collect();
    let color_buckets = coloring
        .interior_coloring
        .buckets
        .iter()
        .map(|bucket| ExecInteriorColorBucket {
            face_indices: bucket.iter().map(|&i| i as u32).collect(),
        })
        .collect();
    ExecViscousInteriorTopology {
        faces,
        color_buckets,
    }
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
