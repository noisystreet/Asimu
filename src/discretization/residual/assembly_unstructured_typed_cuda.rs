//! 非结构 typed 无粘装配 CUDA 分支（ADR 0017 G1）。

use crate::discretization::flux_config::FluxScheme;
use crate::discretization::unstructured_face_cache::UnstructuredFaceTopology;
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;
use crate::error::Result;
use crate::exec::gpu::cuda::{
    CUDA_FLUX_SCHEME_HVL, CUDA_FLUX_SCHEME_ROE, CudaFirstOrderInviscidParams,
    ExecInteriorColorBucket, ExecInteriorFaceStatic, ExecInteriorFaceTopology,
};
use crate::field::ConservedResidualT;

use super::InviscidAssemblyUnstructuredTypedParams;

pub(super) fn cuda_first_order_f32_interior(
    residual: &mut ConservedResidualT<f32>,
    params: &mut InviscidAssemblyUnstructuredTypedParams<'_, f32>,
    topology: &UnstructuredFaceTopology,
) -> Result<bool> {
    let (flux_scheme, roe_entropy_fix) = match params.config.scheme {
        FluxScheme::Roe(roe_cfg) => (CUDA_FLUX_SCHEME_ROE, roe_cfg.entropy_fix),
        FluxScheme::HanelVanLeer => (CUDA_FLUX_SCHEME_HVL, false),
        _ => return Ok(false),
    };
    let exec_topo = build_exec_interior_topology(&params.mesh_cache.face_topology_f32, topology);
    let topo_key = std::ptr::from_ref(topology).addr();
    crate::exec::inviscid::try_assemble_first_order_interior_f32(
        params.exec,
        residual,
        params.primitives,
        &exec_topo,
        topo_key,
        CudaFirstOrderInviscidParams {
            gamma: params.eos.gamma as f32,
            flux_scheme,
            roe_entropy_fix,
        },
    )
}

fn build_exec_interior_topology(
    topology_f32: &UnstructuredFaceTopologyF32,
    coloring: &UnstructuredFaceTopology,
) -> ExecInteriorFaceTopology {
    let faces = topology_f32
        .interior
        .iter()
        .map(|face| {
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
            ExecInteriorFaceStatic {
                owner: face.owner as u32,
                neighbor: face.neighbor as u32,
                nx,
                ny,
                nz,
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
    ExecInteriorFaceTopology {
        faces,
        color_buckets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ComputeFloat;
    use crate::core::ExecDevice;
    use crate::core::approx_eq;
    use crate::discretization::freestream_pair::FreestreamPairFixture;
    use crate::discretization::{
        BoundaryGhostBuffer, InviscidFluxConfig, UnstructuredSolverMeshCache,
        apply_compressible_boundary_conditions_typed,
    };
    use crate::exec::{ExecConfig, ExecutionContext, MeshExecMetrics};
    use crate::field::{ConservedFieldsT, PrimitiveFieldsT};
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

    use super::super::{
        InviscidAssemblyUnstructuredTypedParams, assemble_inviscid_residual_unstructured_typed,
    };

    fn single_tet_fixture(
        side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
    ) -> (
        UnstructuredMesh3d,
        BoundarySet,
        ConservedFieldsT<f32>,
        BoundaryGhostBuffer,
        UnstructuredSolverMeshCache,
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
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: side.fs.mach,
                pressure: side.fs.pressure,
                temperature: side.fs.temperature,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        let fields = ConservedFieldsT::<f32>::from_real_fields(
            &crate::field::ConservedFields::from_freestream_context(
                mesh.num_cells(),
                &side.ctx,
                side.fs,
            )
            .expect("fields"),
        )
        .expect("typed");
        let mut ghosts = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        apply_compressible_boundary_conditions_typed(
            &mesh,
            &boundary,
            &fields,
            &mut ghosts,
            &side.ctx,
            side.fs,
            None,
        )
        .expect("bc");
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let mut primitives = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, side.eos, side.min_pressure)
            .expect("fill");
        (mesh, boundary, fields, ghosts, mesh_cache, primitives)
    }

    #[test]
    #[ignore = "gpu"]
    fn cpu_f32_matches_cuda_f32_inviscid_single_tet() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
        let config = InviscidFluxConfig::default();

        let mut cpu_rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("cpu rhs");
        let mut cpu_exec = ExecutionContext::for_unit_test();
        let mut cpu_params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            gradients: None,
            min_pressure: side.min_pressure,
            exec: &mut cpu_exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut cpu_rhs, &mut cpu_params)
            .expect("cpu assemble");

        let mut cuda_rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("cuda rhs");
        let cuda_config = ExecConfig {
            device: ExecDevice::GpuCuda,
            ..Default::default()
        };
        let mut cuda_exec =
            ExecutionContext::new(cuda_config, MeshExecMetrics::empty()).expect("cuda exec");
        let mut cuda_params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            gradients: None,
            min_pressure: side.min_pressure,
            exec: &mut cuda_exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut cuda_rhs, &mut cuda_params)
            .expect("cuda assemble");

        for i in 0..mesh.num_cells() {
            assert!(
                approx_eq(
                    cpu_rhs.density.values()[i].to_real(),
                    cuda_rhs.density.values()[i].to_real(),
                    1.0e-4
                ),
                "density cell {i}"
            );
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
                    cpu_rhs.total_energy.values()[i].to_real(),
                    cuda_rhs.total_energy.values()[i].to_real(),
                    1.0e-3
                ),
                "energy cell {i}"
            );
        }
    }

    #[test]
    #[ignore = "gpu"]
    fn cpu_f32_matches_cuda_f32_inviscid_hvl_single_tet() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mesh, boundary, fields, ghosts, mesh_cache, primitives) = single_tet_fixture(&side);
        let config = InviscidFluxConfig::hanel_van_leer_first_order();

        let mut cpu_rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("cpu rhs");
        let mut cpu_exec = ExecutionContext::for_unit_test();
        let mut cpu_params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            gradients: None,
            min_pressure: side.min_pressure,
            exec: &mut cpu_exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut cpu_rhs, &mut cpu_params)
            .expect("cpu assemble");

        let mut cuda_rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("cuda rhs");
        let cuda_config = ExecConfig {
            device: ExecDevice::GpuCuda,
            ..Default::default()
        };
        let mut cuda_exec =
            ExecutionContext::new(cuda_config, MeshExecMetrics::empty()).expect("cuda exec");
        let mut cuda_params = InviscidAssemblyUnstructuredTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            mesh_cache: &mesh_cache,
            gradients: None,
            min_pressure: side.min_pressure,
            exec: &mut cuda_exec,
        };
        assemble_inviscid_residual_unstructured_typed(&fields, &mut cuda_rhs, &mut cuda_params)
            .expect("cuda assemble");

        for i in 0..mesh.num_cells() {
            assert!(
                approx_eq(
                    cpu_rhs.density.values()[i].to_real(),
                    cuda_rhs.density.values()[i].to_real(),
                    1.0e-4
                ),
                "density cell {i}"
            );
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
                    cpu_rhs.total_energy.values()[i].to_real(),
                    cuda_rhs.total_energy.values()[i].to_real(),
                    1.0e-3
                ),
                "energy cell {i}"
            );
        }
    }
}
