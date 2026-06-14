//! 非结构 3D 粘性残差 f32 装配（梯度、内面与边界面通量均为 f32）。

#[cfg(feature = "cuda")]
#[path = "assembly_unstructured_viscous_f32_cuda.rs"]
mod assembly_unstructured_viscous_f32_cuda;

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::UnstructuredGradientScratchF32;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::{
    UnstructuredGradientLsqInputF32, compute_unstructured_gradients_idw_lsq_f32,
};
use crate::discretization::unstructured_face_cache::{
    UnstructuredFaceTopology, UnstructuredSolverMeshCache,
};
use crate::discretization::unstructured_face_cache_f32::{
    UnstructuredFaceTopologyF32, UnstructuredInteriorFaceF32,
};
#[cfg(not(feature = "parallel-fvm"))]
use crate::discretization::viscous_f32::scatter_fused_interior_viscous_face_f32;
use crate::discretization::viscous_f32::{
    ColoredViscousFaceFluxF32, InteriorViscousScatterGeomF32, average_face_lane_f32,
    fused_interior_viscous_face_flux_averaged_f32,
};
use crate::error::Result;
use crate::exec::ExecutionContext;
#[cfg(feature = "parallel-fvm")]
use crate::exec::scatter::{
    ViscousResidualMutF32, ViscousScatterOpF32, ViscousValidSlotScatterF32,
    scatter_viscous_valid_slots_f32,
};
use crate::field::{ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::assembly_unstructured_viscous::assemble_boundary_faces_f32;
use super::assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredScratch, prepare_unstructured_viscous_transport_f32,
};
use crate::discretization::viscous_boundary_f32::ViscousBoundaryFluxParamsF32;

pub(super) struct ViscousInteriorAssemblyF32<'a> {
    face_topology: &'a UnstructuredFaceTopologyF32,
    transport_topology: &'a UnstructuredFaceTopology,
    eos: &'a IdealGasEoS,
    viscous: &'a ViscousPhysicsConfig,
    primitives: &'a PrimitiveFieldsT<f32>,
    gradients: &'a GradientFieldsT<f32>,
    temperatures: &'a [f32],
    exec: &'a mut ExecutionContext,
}

/// f32 非结构粘性装配输入。
pub struct ViscousAssemblyUnstructuredF32Input<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a crate::discretization::BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFieldsT<f32>,
    pub exec: &'a mut ExecutionContext,
}

/// 计算 f32 IDWLS 梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured_f32(
    residual: &mut ConservedResidualT<f32>,
    input: &mut ViscousAssemblyUnstructuredF32Input<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    grad_scratch: &mut UnstructuredGradientScratchF32,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_viscous_idw_lsq_gradient_f32",
            cells = input.mesh.num_cells(),
        )
        .entered();
        compute_unstructured_gradients_idw_lsq_f32(
            UnstructuredGradientLsqInputF32 {
                mesh: input.mesh,
                mesh_cache: input.mesh_cache,
                primitives: input.primitives,
                eos: input.eos,
                ghosts: input.ghosts,
                min_pressure: input.min_pressure,
                viscous: Some(input.viscous),
            },
            input.gradient_scratch,
            grad_scratch,
            input.exec,
        )?;
    }
    assemble_viscous_residual_f32_interior(
        residual,
        &mut ViscousInteriorAssemblyF32 {
            face_topology: &input.mesh_cache.face_topology_f32,
            transport_topology: &input.mesh_cache.face_topology,
            eos: input.eos,
            viscous: input.viscous,
            primitives: input.primitives,
            gradients: input.gradient_scratch,
            temperatures: &grad_scratch.temperatures,
            exec: input.exec,
        },
        scratch,
    )?;
    #[cfg(feature = "cuda")]
    if input.exec.cuda_rhs_pipeline_active() {
        input
            .exec
            .cuda_flush_rhs_pipeline(residual, input.gradient_scratch)?;
    }
    let boundary_params = ViscousBoundaryFluxParamsF32 {
        eos: input.eos,
        viscous: input.viscous,
        primitives: input.primitives,
        gradients: input.gradient_scratch,
    };
    assemble_boundary_faces_f32(
        residual,
        &input.mesh_cache.face_topology_f32,
        input.ghosts,
        &boundary_params,
        input.min_pressure,
        &grad_scratch.temperatures,
    )
}

fn assemble_viscous_residual_f32_interior(
    residual: &mut ConservedResidualT<f32>,
    params: &mut ViscousInteriorAssemblyF32<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if assembly_unstructured_viscous_f32_cuda::cuda_viscous_f32_interior(
            residual, params, scratch,
        )? {
            return Ok(());
        }
    }

    let constant = prepare_unstructured_viscous_transport_f32(
        params.transport_topology,
        params.primitives.num_cells(),
        params.viscous,
        params.eos,
        params.temperatures,
        scratch,
    )?;
    let grad_slices = params.gradients.velocity_gradient_slices();
    let coloring = &params.transport_topology.interior_coloring;

    #[cfg(not(feature = "parallel-fvm"))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_f32",
            path = "colored_serial",
            faces = params.face_topology.interior.len(),
            colors = coloring.num_colors,
        )
        .entered();
        for bucket in &coloring.buckets {
            for &face_idx in bucket {
                if let Some((geom, flux)) = compute_interior_viscous_face_contribution_f32(
                    face_idx,
                    &params.face_topology.interior,
                    constant,
                    scratch,
                    params,
                    &grad_slices,
                )? {
                    scatter_fused_interior_viscous_face_f32(residual, &geom, &flux);
                }
            }
        }
        return Ok(());
    }

    #[cfg(feature = "parallel-fvm")]
    {
        use crate::exec::parallel::par_try_map_face_indices;

        let _span = info_span!(
            "unstructured_viscous_interior_flux_f32",
            path = "parallel_bucket",
            faces = params.face_topology.interior.len(),
            colors = coloring.num_colors,
        )
        .entered();
        for bucket in &coloring.buckets {
            let contributions = par_try_map_face_indices(bucket, 1024, |face_idx| {
                compute_interior_viscous_face_contribution_f32(
                    face_idx,
                    &params.face_topology.interior,
                    constant,
                    scratch,
                    params,
                    &grad_slices,
                )
            })?;
            let pairs: Vec<_> = contributions.into_iter().flatten().collect();
            if pairs.is_empty() {
                continue;
            }
            let geoms: Vec<InteriorViscousScatterGeomF32> = pairs.iter().map(|(g, _)| *g).collect();
            let fluxes: Vec<ColoredViscousFaceFluxF32> = pairs.iter().map(|(_, f)| *f).collect();
            let valid = vec![true; pairs.len()];
            scatter_viscous_valid_slots_f32(
                ViscousValidSlotScatterF32 {
                    ctx: params.exec,
                    bucket_len: bucket.len(),
                    geoms: &geoms,
                    fluxes: &fluxes,
                    valid: &valid,
                    residual: ViscousResidualMutF32 {
                        mx: residual.momentum_x.values_mut(),
                        my: residual.momentum_y.values_mut(),
                        mz: residual.momentum_z.values_mut(),
                        energy: residual.total_energy.values_mut(),
                    },
                },
                viscous_scatter_extract_f32,
            );
        }
        Ok(())
    }
}

#[cfg(feature = "parallel-fvm")]
fn viscous_scatter_extract_f32(
    g: &InteriorViscousScatterGeomF32,
    f: &ColoredViscousFaceFluxF32,
) -> ViscousScatterOpF32 {
    ViscousScatterOpF32 {
        owner: g.owner,
        neighbor: g.neighbor,
        owner_scale: g.owner_scale,
        neighbor_scale: g.neighbor_scale,
        flux_mx: f.mx,
        flux_my: f.my,
        flux_mz: f.mz,
        flux_energy: f.energy,
    }
}

fn compute_interior_viscous_face_contribution_f32(
    face_idx: usize,
    interior: &[UnstructuredInteriorFaceF32],
    constant: Option<(Real, Real)>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    params: &ViscousInteriorAssemblyF32<'_>,
    grad_slices: &crate::discretization::gradient_typed::VelocityGradientSlicesT<'_, f32>,
) -> Result<Option<(InteriorViscousScatterGeomF32, ColoredViscousFaceFluxF32)>> {
    let face = &interior[face_idx];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return Ok(None);
    }
    let (mu, lambda) = if let Some(c) = constant {
        (c.0 as f32, c.1 as f32)
    } else {
        let (m, l) = scratch.face_transport_at(face_idx);
        (m as f32, l as f32)
    };
    let normal = face.normal;
    let geom = InteriorViscousScatterGeomF32 {
        owner: face.owner,
        neighbor: face.neighbor,
        nx: normal[0],
        ny: normal[1],
        nz: normal[2],
        mu,
        lambda,
        owner_scale: face.owner_rhs_scale,
        neighbor_scale: face.neighbor_rhs_scale,
    };
    let lane = average_face_lane_f32(face.owner, face.neighbor, params.primitives, grad_slices);
    let flux = fused_interior_viscous_face_flux_averaged_f32(
        lane, normal[0], normal[1], normal[2], mu, lambda,
    );
    Ok(Some((geom, flux)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ComputeFloat;
    use crate::discretization::BoundaryGhostBuffer;
    use crate::discretization::GhostCellState;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::FreestreamParams;
    use crate::physics::ViscosityModel;

    fn uniform_closed_tet_viscous_rhs() -> (
        UnstructuredMesh3d,
        IdealGasEoS,
        BoundarySet,
        UnstructuredSolverMeshCache,
        BoundaryGhostBuffer,
        PrimitiveFieldsT<f32>,
        FreestreamParams,
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
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32, fs)
    }

    #[test]
    fn f32_sutherland_uniform_closed_tet_has_near_zero_viscous_rhs() {
        let (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32, _fs) =
            uniform_closed_tet_viscous_rhs();
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("viscous");
        let mut grad = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let mut exec = ExecutionContext::for_unit_test();
        let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut grad_scratch = UnstructuredGradientScratchF32::new(mesh.num_cells());
        let mut input = ViscousAssemblyUnstructuredF32Input {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives_f32,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
            exec: &mut exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_f32(
            &mut rhs,
            &mut input,
            &mut scratch,
            &mut grad_scratch,
        )
        .expect("visc");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-12)
        );
        assert!(
            rhs.momentum_x
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
        assert!(
            rhs.total_energy
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
    }

    #[test]
    fn f32_native_boundary_uniform_closed_tet_has_near_zero_viscous_rhs() {
        let (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32, _fs) =
            uniform_closed_tet_viscous_rhs();
        let viscous = ViscousPhysicsConfig::default();
        let mut grad = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let mut exec = ExecutionContext::for_unit_test();
        let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut grad_scratch = UnstructuredGradientScratchF32::new(mesh.num_cells());
        let mut input = ViscousAssemblyUnstructuredF32Input {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives_f32,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
            exec: &mut exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_f32(
            &mut rhs,
            &mut input,
            &mut scratch,
            &mut grad_scratch,
        )
        .expect("visc");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-12)
        );
        assert!(
            rhs.momentum_x
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
        assert!(
            rhs.total_energy
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
    }
}
