//! 非结构 3D 粘性残差 typed 装配（P3/P5：梯度/通量 `f64`，残差 `f32`/`f64` scatter）。

#[cfg(feature = "parallel-fvm")]
#[path = "assembly_unstructured_viscous_typed_parallel.rs"]
mod parallel;

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::{ComputeFloat, Real};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::GradientFields;
use crate::discretization::gradient_unstructured::{
    UnstructuredGradientLsqInput, compute_unstructured_gradients_idw_lsq_with_scratch,
};
use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
use crate::discretization::viscous::{
    InteriorViscousFaceFlux, InteriorViscousFaceGeom, scatter_fused_interior_viscous_face_typed,
};
use crate::error::{AsimuError, Result};
use crate::exec::ExecutionContext;
use crate::exec::scatter::{
    ViscousResidualMut, ViscousResidualMutF32, ViscousScatterOp, ViscousScatterOpF32,
    ViscousValidSlotScatter, ViscousValidSlotScatterF32, scatter_viscous_valid_slots,
    scatter_viscous_valid_slots_f32,
};
use crate::field::{ConservedResidualT, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

#[cfg(not(feature = "simd-fvm"))]
use super::assembly_unstructured_viscous::fill_face_averaged_viscous_soa_typed;
use super::assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch,
    assemble_boundary_faces_typed, prepare_unstructured_viscous_transport,
};

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
use super::assembly_unstructured_viscous::compute_viscous_batch4_into;
#[cfg(not(feature = "parallel-fvm"))]
use super::assembly_unstructured_viscous::interior_viscous_face_geom_and_flux;

/// typed 粘性 scatter dispatch（`ComputeFloat` 密封子集；ADR 0016 P5）。
pub trait ViscousTypedScatterBackend: ComputeFloat {
    fn scatter_viscous_valid_slots(
        residual: &mut ConservedResidualT<Self>,
        ctx: &ExecutionContext,
        bucket_len: usize,
        geoms: &[InteriorViscousFaceGeom],
        fluxes: &[InteriorViscousFaceFlux],
        valid: &[bool],
        extract: impl Fn(&InteriorViscousFaceGeom, &InteriorViscousFaceFlux) -> ViscousScatterOp + Sync,
    );

    fn scatter_fused_interior_face(
        residual: &mut ConservedResidualT<Self>,
        geom: &InteriorViscousFaceGeom,
        flux: &InteriorViscousFaceFlux,
    );
}

#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
impl ViscousTypedScatterBackend for f64 {
    fn scatter_viscous_valid_slots(
        residual: &mut ConservedResidualT<f64>,
        ctx: &ExecutionContext,
        bucket_len: usize,
        geoms: &[InteriorViscousFaceGeom],
        fluxes: &[InteriorViscousFaceFlux],
        valid: &[bool],
        extract: impl Fn(&InteriorViscousFaceGeom, &InteriorViscousFaceFlux) -> ViscousScatterOp + Sync,
    ) {
        scatter_viscous_valid_slots(
            ViscousValidSlotScatter {
                ctx,
                bucket_len,
                geoms,
                fluxes,
                valid,
                residual: ViscousResidualMut {
                    mx: residual.momentum_x.values_mut(),
                    my: residual.momentum_y.values_mut(),
                    mz: residual.momentum_z.values_mut(),
                    energy: residual.total_energy.values_mut(),
                },
            },
            extract,
        );
    }

    fn scatter_fused_interior_face(
        residual: &mut ConservedResidualT<f64>,
        geom: &InteriorViscousFaceGeom,
        flux: &InteriorViscousFaceFlux,
    ) {
        scatter_fused_interior_viscous_face_typed(residual, geom, flux);
    }
}

#[cfg_attr(feature = "parallel-fvm", allow(dead_code))]
impl ViscousTypedScatterBackend for f32 {
    fn scatter_viscous_valid_slots(
        residual: &mut ConservedResidualT<f32>,
        ctx: &ExecutionContext,
        bucket_len: usize,
        geoms: &[InteriorViscousFaceGeom],
        fluxes: &[InteriorViscousFaceFlux],
        valid: &[bool],
        extract: impl Fn(&InteriorViscousFaceGeom, &InteriorViscousFaceFlux) -> ViscousScatterOp + Sync,
    ) {
        scatter_viscous_valid_slots_f32(
            ViscousValidSlotScatterF32 {
                ctx,
                bucket_len,
                geoms,
                fluxes,
                valid,
                residual: ViscousResidualMutF32 {
                    mx: residual.momentum_x.values_mut(),
                    my: residual.momentum_y.values_mut(),
                    mz: residual.momentum_z.values_mut(),
                    energy: residual.total_energy.values_mut(),
                },
            },
            |g, f| viscous_scatter_op_f32_from_real(extract(g, f)),
        );
    }

    fn scatter_fused_interior_face(
        residual: &mut ConservedResidualT<f32>,
        geom: &InteriorViscousFaceGeom,
        flux: &InteriorViscousFaceFlux,
    ) {
        scatter_fused_interior_viscous_face_typed(residual, geom, flux);
    }
}

fn viscous_scatter_op_f32_from_real(op: ViscousScatterOp) -> ViscousScatterOpF32 {
    ViscousScatterOpF32 {
        owner: op.owner,
        neighbor: op.neighbor,
        owner_scale: op.owner_scale as f32,
        neighbor_scale: op.neighbor_scale as f32,
        flux_mx: op.flux_mx as f32,
        flux_my: op.flux_my as f32,
        flux_mz: op.flux_mz as f32,
        flux_energy: op.flux_energy as f32,
    }
}

/// typed 非结构粘性装配输入（原始变量/梯度仍用 `f64`，见 ADR 0016 §4）。
pub struct ViscousAssemblyUnstructuredTypedInput<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFields,
    pub exec: &'a mut ExecutionContext,
}

/// 计算 IDWLS 梯度并在 typed 残差上叠加粘性通量。
pub fn compute_gradients_and_assemble_viscous_unstructured_typed<T: ViscousTypedScatterBackend>(
    residual: &mut ConservedResidualT<T>,
    input: &mut ViscousAssemblyUnstructuredTypedInput<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_viscous_idw_lsq_gradient_typed",
            cells = input.mesh.num_cells(),
            precision = T::PRECISION.label(),
        )
        .entered();
        compute_unstructured_gradients_idw_lsq_with_scratch(
            UnstructuredGradientLsqInput {
                mesh: input.mesh,
                mesh_cache: input.mesh_cache,
                primitives: input.primitives,
                eos: input.eos,
                ghosts: input.ghosts,
                min_pressure: input.min_pressure,
                viscous: Some(input.viscous),
            },
            input.gradient_scratch,
            &mut scratch.gradient,
            input.exec,
        )?;
    }
    let params = ViscousAssemblyUnstructuredParams {
        mesh: input.mesh,
        face_topology: &input.mesh_cache.face_topology,
        eos: input.eos,
        viscous: input.viscous,
        ghosts: input.ghosts,
        primitives: input.primitives,
        gradients: input.gradient_scratch,
        min_pressure: input.min_pressure,
    };
    assemble_viscous_residual_unstructured_typed(residual, &params, scratch, input.exec)
}

fn assemble_viscous_residual_unstructured_typed<T: ViscousTypedScatterBackend>(
    residual: &mut ConservedResidualT<T>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let n = params.mesh.num_cells();
    if residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构 typed 粘性装配：场/残差长度须等于网格单元数 {n}"
        )));
    }
    crate::discretization::gradient::cell_temperatures_into(
        params.primitives,
        params.eos,
        Some(params.viscous),
        &mut scratch.gradient.temperatures,
    )?;
    assemble_interior_faces_typed(residual, params, scratch, exec)?;
    assemble_boundary_faces_typed(residual, params, scratch)
}

fn assemble_interior_faces_typed<T: ViscousTypedScatterBackend>(
    residual: &mut ConservedResidualT<T>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    exec: &mut ExecutionContext,
) -> Result<()> {
    let num_faces = params.face_topology.interior.len();
    let constant = prepare_unstructured_viscous_transport(params, scratch)?;
    #[cfg(not(feature = "simd-fvm"))]
    {
        let _span = info_span!("unstructured_viscous_face_avg_typed", faces = num_faces).entered();
        fill_face_averaged_viscous_soa_typed(params, scratch);
    }
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_typed",
            faces = num_faces,
            colors = params.face_topology.interior_coloring.num_colors,
            precision = T::PRECISION.label(),
        )
        .entered();
        accumulate_interior_faces_typed_fused(residual, params, scratch, constant, exec)?;
    }
    Ok(())
}

fn accumulate_interior_faces_typed_fused<T: ViscousTypedScatterBackend>(
    residual: &mut ConservedResidualT<T>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
    exec: &mut ExecutionContext,
) -> Result<()> {
    #[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_typed",
            path = "simd_batch4",
            faces = params.face_topology.interior.len(),
        )
        .entered();
        for layout in &params.face_topology.interior_coloring.bucket_batch_layouts {
            accumulate_viscous_bucket_batch4_typed_serial(
                residual, layout, params, scratch, constant,
            )?;
        }
        return Ok(());
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_typed",
            path = "colored_serial",
            faces = params.face_topology.interior.len(),
        )
        .entered();
        params
            .face_topology
            .interior_coloring
            .for_each_face_index(|i| {
                if let Some((geom, flux)) =
                    interior_viscous_face_geom_and_flux(i, params, scratch, constant)
                {
                    T::scatter_fused_interior_face(residual, &geom, &flux);
                }
            });
    }

    #[cfg(all(feature = "parallel-fvm", feature = "simd-fvm"))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_typed",
            path = "simd_batch4",
            faces = params.face_topology.interior.len(),
            colors = params.face_topology.interior_coloring.num_colors,
        )
        .entered();
        for layout in &params.face_topology.interior_coloring.bucket_batch_layouts {
            parallel::accumulate_viscous_bucket_batch4_typed_fused(
                residual, layout, params, scratch, constant, exec,
            );
        }
    }

    #[cfg(all(feature = "parallel-fvm", not(feature = "simd-fvm")))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_typed",
            path = "parallel_bucket",
            faces = params.face_topology.interior.len(),
            colors = params.face_topology.interior_coloring.num_colors,
        )
        .entered();
        for bucket in &params.face_topology.interior_coloring.buckets {
            parallel::accumulate_viscous_color_bucket_typed_fused(
                residual, bucket, params, scratch, constant, exec,
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
fn accumulate_viscous_bucket_batch4_typed_serial<T: ViscousTypedScatterBackend>(
    residual: &mut ConservedResidualT<T>,
    layout: &crate::discretization::InteriorFaceBucketBatchLayout,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> Result<()> {
    let mut geoms = [InteriorViscousFaceGeom::default(); 4];
    let mut fluxes = [InteriorViscousFaceFlux::default(); 4];
    for batch in &layout.full_batches {
        let count =
            compute_viscous_batch4_into(batch, params, scratch, constant, &mut geoms, &mut fluxes);
        if count == 0 {
            for &face_idx in &batch.face_indices {
                if let Some((geom, flux)) =
                    interior_viscous_face_geom_and_flux(face_idx, params, scratch, constant)
                {
                    T::scatter_fused_interior_face(residual, &geom, &flux);
                }
            }
            continue;
        }
        for lane in 0..count as usize {
            T::scatter_fused_interior_face(residual, &geoms[lane], &fluxes[lane]);
        }
    }
    for &face_idx in &layout.remainder {
        if let Some((geom, flux)) =
            interior_viscous_face_geom_and_flux(face_idx, params, scratch, constant)
        {
            T::scatter_fused_interior_face(residual, &geom, &flux);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::BoundaryGhostBuffer;
    use crate::discretization::GhostCellState;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::FreestreamParams;

    #[test]
    fn f32_uniform_closed_tet_has_near_zero_unstructured_viscous_rhs() {
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
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-8)
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
        let viscous = ViscousPhysicsConfig::default();
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let mut exec = ExecutionContext::for_unit_test();
        let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut input = ViscousAssemblyUnstructuredTypedInput {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
            exec: &mut exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_typed::<f32>(
            &mut rhs,
            &mut input,
            &mut scratch,
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
