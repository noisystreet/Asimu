//! 非结构 3D 网格粘性残差装配。

#[path = "assembly_unstructured_viscous_boundary.rs"]
mod boundary;
#[path = "assembly_unstructured_viscous_face_avg.rs"]
mod face_avg;

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::GradientFields;
use crate::discretization::gradient_unstructured::{
    UnstructuredGradientLsqInput, UnstructuredGradientScratch,
    compute_unstructured_gradients_idw_lsq_with_scratch,
};
use crate::discretization::unstructured_face_cache::{
    UnstructuredFaceTopology, UnstructuredSolverMeshCache,
};
use crate::discretization::viscous::{
    InteriorViscousFaceGeom, InteriorViscousResidualMut, ViscousFaceAveragedSoA,
    face_transport_coefficients, fused_interior_viscous_face_flux_averaged,
    scatter_fused_interior_viscous_face,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

use face_avg::fill_face_averaged_viscous_soa;
#[cfg(feature = "simd-fvm")]
use face_avg::gather_viscous_face_batch4_from_face_averaged;

/// 非结构粘性残差装配输入。
pub struct ViscousAssemblyUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub face_topology: &'a UnstructuredFaceTopology,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub gradients: &'a GradientFields,
    pub min_pressure: Real,
}

/// 在已有残差上叠加非结构粘性通量贡献（不清零 residual）。
pub fn assemble_viscous_residual_unstructured(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(params.mesh.num_cells());
    crate::discretization::gradient::cell_temperatures_into(
        params.primitives,
        params.eos,
        Some(params.viscous),
        &mut scratch.gradient.temperatures,
    )?;
    assemble_viscous_residual_unstructured_with_scratch(residual, params, &mut scratch)
}

fn assemble_viscous_residual_unstructured_with_scratch(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let n = params.mesh.num_cells();
    if residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构粘性装配：场/残差长度须等于网格单元数 {n}"
        )));
    }
    if scratch.gradient.temperatures.len() != n {
        return Err(AsimuError::Field(format!(
            "非结构粘性装配：温度缓冲长度 {} 与单元数 {n} 不一致",
            scratch.gradient.temperatures.len()
        )));
    }
    {
        let _span = info_span!(
            "unstructured_viscous_assemble_interior_faces",
            faces = params.face_topology.interior.len(),
        )
        .entered();
        assemble_interior_faces(residual, params, scratch)?;
    }
    {
        let _span = info_span!(
            "unstructured_viscous_assemble_boundary_faces",
            faces = params.face_topology.boundary.len(),
        )
        .entered();
        boundary::assemble_boundary_faces(residual, params, scratch)?;
    }
    Ok(())
}

/// 非结构粘性梯度 + 装配输入。
pub struct ViscousAssemblyUnstructuredInput<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFields,
}

/// 非结构粘性 RHS 复用缓冲。
pub struct ViscousAssemblyUnstructuredScratch {
    pub gradient: UnstructuredGradientScratch,
    /// 内面心预平均速度与梯度（P7，flux 顺序读）。
    pub(crate) face_averaged: ViscousFaceAveragedSoA,
    cell_mu: Vec<Real>,
    cell_lambda: Vec<Real>,
    face_mu: Vec<Real>,
    face_lambda: Vec<Real>,
    constant_transport: Option<(Real, Real)>,
}

impl ViscousAssemblyUnstructuredScratch {
    #[must_use]
    pub fn new(num_cells: usize) -> Self {
        Self {
            gradient: UnstructuredGradientScratch::new(num_cells),
            face_averaged: ViscousFaceAveragedSoA::default(),
            cell_mu: Vec::new(),
            cell_lambda: Vec::new(),
            face_mu: Vec::new(),
            face_lambda: Vec::new(),
            constant_transport: None,
        }
    }

    fn ensure_face_averaged(&mut self, num_faces: usize) {
        self.face_averaged.ensure(num_faces);
    }

    fn ensure_cell_transport(&mut self, num_cells: usize) {
        self.cell_mu.resize(num_cells, 0.0);
        self.cell_lambda.resize(num_cells, 0.0);
    }

    fn ensure_face_transport(&mut self, num_faces: usize) {
        self.face_mu.resize(num_faces, 0.0);
        self.face_lambda.resize(num_faces, 0.0);
    }
}

/// 计算非结构 IDWLS 梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured(
    residual: &mut ConservedResidual,
    input: &mut ViscousAssemblyUnstructuredInput<'_>,
) -> Result<()> {
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(input.mesh.num_cells());
    compute_gradients_and_assemble_viscous_unstructured_with_scratch(residual, input, &mut scratch)
}

/// 使用调用方提供的 scratch 计算非结构梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured_with_scratch(
    residual: &mut ConservedResidual,
    input: &mut ViscousAssemblyUnstructuredInput<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_viscous_idw_lsq_gradient",
            cells = input.mesh.num_cells(),
            interior_faces = input.mesh_cache.face_topology.interior.len(),
            boundary_faces = input.mesh_cache.face_topology.boundary.len(),
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
    assemble_viscous_residual_unstructured_with_scratch(residual, &params, scratch)
}

fn assemble_interior_faces(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let num_faces = params.face_topology.interior.len();
    scratch.ensure_face_transport(num_faces);
    if matches!(params.viscous.model, ViscosityModel::Constant { .. }) {
        scratch.constant_transport = Some(face_transport_coefficients(
            1.0,
            1.0,
            params.viscous,
            params.eos,
        )?);
    } else {
        scratch.constant_transport = None;
        let num_cells = params.mesh.num_cells();
        scratch.ensure_cell_transport(num_cells);
        {
            let _span =
                info_span!("unstructured_viscous_interior_transport", cells = num_cells).entered();
            fill_cell_transport_coefficients(params, scratch)?;
            fill_face_transport_coefficients(params, scratch)?;
        }
    }
    {
        let _span = info_span!("unstructured_viscous_face_avg", faces = num_faces,).entered();
        fill_face_averaged_viscous_soa(params, scratch);
    }
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux",
            faces = num_faces,
            colors = params.face_topology.interior_coloring.num_colors,
        )
        .entered();
        accumulate_interior_faces_fused(residual, params, scratch)?;
    }
    Ok(())
}

fn fill_face_transport_coefficients(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let cell_mu = &scratch.cell_mu;
    let cell_lambda = &scratch.cell_lambda;
    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;
        scratch
            .face_mu
            .par_iter_mut()
            .zip(scratch.face_lambda.par_iter_mut())
            .zip(params.face_topology.interior.par_iter())
            .for_each(|((mu, lambda), face)| {
                if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
                    return;
                }
                *mu = 0.5 * (cell_mu[face.owner] + cell_mu[face.neighbor]);
                *lambda = 0.5 * (cell_lambda[face.owner] + cell_lambda[face.neighbor]);
            });
    }
    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (i, face) in params.face_topology.interior.iter().enumerate() {
            if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
                continue;
            }
            scratch.face_mu[i] = 0.5 * (cell_mu[face.owner] + cell_mu[face.neighbor]);
            scratch.face_lambda[i] = 0.5 * (cell_lambda[face.owner] + cell_lambda[face.neighbor]);
        }
    }
    Ok(())
}

fn accumulate_interior_faces_fused(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let face_averaged = &scratch.face_averaged;
    let mut residual_mut = InteriorViscousResidualMut {
        mx: residual.momentum_x.values_mut(),
        my: residual.momentum_y.values_mut(),
        mz: residual.momentum_z.values_mut(),
        energy: residual.total_energy.values_mut(),
    };
    let constant = scratch.constant_transport;

    #[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_fused",
            path = "simd_batch4",
            faces = params.face_topology.interior.len(),
        )
        .entered();
        for layout in &params.face_topology.interior_coloring.bucket_batch_layouts {
            accumulate_viscous_bucket_batch4(
                layout,
                face_averaged,
                &mut residual_mut,
                params,
                scratch,
                constant,
            )?;
        }
        return Ok(());
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_fused",
            path = "colored_serial",
            faces = params.face_topology.interior.len(),
        )
        .entered();
        params
            .face_topology
            .interior_coloring
            .for_each_face_index(|i| {
                accumulate_one_interior_face(
                    i,
                    face_averaged,
                    &mut residual_mut,
                    params,
                    scratch,
                    constant,
                );
            });
    }

    #[cfg(all(feature = "parallel-fvm", feature = "simd-fvm"))]
    {
        let bucket_results = {
            let _span = info_span!(
                "unstructured_viscous_interior_flux_compute",
                path = "simd_batch4",
                faces = params.face_topology.interior.len(),
                colors = params.face_topology.interior_coloring.num_colors,
            )
            .entered();
            params
                .face_topology
                .interior_coloring
                .bucket_batch_layouts
                .iter()
                .map(|layout| {
                    accumulate_viscous_bucket_batch4_to_vec(
                        layout,
                        face_averaged,
                        params,
                        scratch,
                        constant,
                    )
                })
                .collect::<Vec<_>>()
        };
        {
            let _span = info_span!(
                "unstructured_viscous_interior_flux_scatter",
                path = "simd_batch4",
                buckets = params
                    .face_topology
                    .interior_coloring
                    .bucket_batch_layouts
                    .len(),
            )
            .entered();
            for bucket in bucket_results {
                for (geom, flux) in bucket {
                    scatter_fused_interior_viscous_face(&mut residual_mut, &geom, &flux);
                }
            }
        }
    }

    #[cfg(all(feature = "parallel-fvm", not(feature = "simd-fvm")))]
    {
        let bucket_results = {
            let _span = info_span!(
                "unstructured_viscous_interior_flux_compute",
                faces = params.face_topology.interior.len(),
                colors = params.face_topology.interior_coloring.num_colors,
            )
            .entered();
            params.face_topology.interior_coloring.par_map_buckets(|i| {
                interior_face_flux_contribution(i, face_averaged, params, scratch, constant)
            })
        };
        {
            let _span = info_span!(
                "unstructured_viscous_interior_flux_scatter",
                buckets = params.face_topology.interior_coloring.buckets.len(),
            )
            .entered();
            for bucket in bucket_results {
                for item in bucket.into_iter().flatten() {
                    let (geom, flux) = item;
                    scatter_fused_interior_viscous_face(&mut residual_mut, &geom, &flux);
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
fn accumulate_viscous_bucket_batch4(
    layout: &crate::discretization::InteriorFaceBucketBatchLayout,
    face_averaged: &ViscousFaceAveragedSoA,
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> Result<()> {
    for batch in &layout.full_batches {
        if let Some(items) = viscous_face_batch4_static(batch, face_averaged, scratch, constant) {
            for (geom, flux) in items {
                scatter_fused_interior_viscous_face(residual_mut, &geom, &flux);
            }
            continue;
        }
        for &face_idx in &batch.face_indices {
            accumulate_one_interior_face(
                face_idx,
                face_averaged,
                residual_mut,
                params,
                scratch,
                constant,
            );
        }
    }
    for &face_idx in &layout.remainder {
        accumulate_one_interior_face(
            face_idx,
            face_averaged,
            residual_mut,
            params,
            scratch,
            constant,
        );
    }
    Ok(())
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
fn accumulate_viscous_bucket_batch4_to_vec(
    layout: &crate::discretization::InteriorFaceBucketBatchLayout,
    face_averaged: &ViscousFaceAveragedSoA,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> Vec<(
    InteriorViscousFaceGeom,
    crate::discretization::viscous::InteriorViscousFaceFlux,
)> {
    use rayon::prelude::*;

    let mut out = Vec::with_capacity(layout.num_faces());
    for part in layout
        .full_batches
        .par_iter()
        .with_min_len(128)
        .map(|batch| viscous_full_batch_to_vec(batch, face_averaged, params, scratch, constant))
        .collect::<Vec<_>>()
    {
        out.extend(part);
    }
    out.extend(
        layout
            .remainder
            .par_iter()
            .with_min_len(1024)
            .filter_map(|&face_idx| {
                interior_face_flux_contribution(face_idx, face_averaged, params, scratch, constant)
            })
            .collect::<Vec<_>>(),
    );
    out
}

#[cfg(all(feature = "simd-fvm", feature = "parallel-fvm"))]
fn viscous_full_batch_to_vec(
    batch: &crate::discretization::InteriorFaceBatchStatic4,
    face_averaged: &ViscousFaceAveragedSoA,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> Vec<(
    InteriorViscousFaceGeom,
    crate::discretization::viscous::InteriorViscousFaceFlux,
)> {
    if let Some(items) = viscous_face_batch4_static(batch, face_averaged, scratch, constant) {
        return items;
    }
    batch
        .face_indices
        .iter()
        .filter_map(|&face_idx| {
            interior_face_flux_contribution(face_idx, face_averaged, params, scratch, constant)
        })
        .collect()
}

#[cfg(feature = "simd-fvm")]
fn viscous_face_batch4_static(
    batch: &crate::discretization::InteriorFaceBatchStatic4,
    face_averaged: &ViscousFaceAveragedSoA,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> Option<
    Vec<(
        InteriorViscousFaceGeom,
        crate::discretization::viscous::InteriorViscousFaceFlux,
    )>,
> {
    use crate::exec::cpu::{ViscousFaceBatchGeom, fused_interior_viscous_face_flux_batch4};

    if !batch.simd_eligible() {
        return None;
    }

    let mut mu = [0.0; 4];
    let mut lambda = [0.0; 4];
    let mut geoms: [InteriorViscousFaceGeom; 4] = [InteriorViscousFaceGeom {
        owner: 0,
        neighbor: 0,
        nx: 0.0,
        ny: 0.0,
        nz: 0.0,
        mu: 0.0,
        lambda: 0.0,
        owner_scale: 0.0,
        neighbor_scale: 0.0,
    }; 4];
    for (lane, &face_idx) in batch.face_indices.iter().enumerate() {
        let (m, l) = transport_at_face(face_idx, scratch, constant);
        mu[lane] = m;
        lambda[lane] = l;
        geoms[lane] = InteriorViscousFaceGeom {
            owner: batch.owners[lane],
            neighbor: batch.neighbors[lane],
            nx: batch.nx[lane],
            ny: batch.ny[lane],
            nz: batch.nz[lane],
            mu: m,
            lambda: l,
            owner_scale: batch.owner_rhs_scale[lane],
            neighbor_scale: batch.neighbor_rhs_scale[lane],
        };
    }
    let gathered = gather_viscous_face_batch4_from_face_averaged(
        ViscousFaceBatchGeom {
            owners: batch.owners,
            neighbors: batch.neighbors,
            nx: batch.nx,
            ny: batch.ny,
            nz: batch.nz,
            mu,
            lambda,
        },
        batch.face_indices,
        &face_averaged.lanes,
    );
    let flux4 = fused_interior_viscous_face_flux_batch4(&gathered);
    let mut out = Vec::with_capacity(4);
    for (lane, geom) in geoms.into_iter().enumerate() {
        out.push((
            geom,
            crate::discretization::viscous::InteriorViscousFaceFlux {
                mx: flux4.mx[lane],
                my: flux4.my[lane],
                mz: flux4.mz[lane],
                energy: flux4.energy[lane],
            },
        ));
    }
    Some(out)
}

#[cfg(feature = "parallel-fvm")]
fn interior_face_flux_contribution(
    i: usize,
    face_averaged: &ViscousFaceAveragedSoA,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> Option<(
    InteriorViscousFaceGeom,
    crate::discretization::viscous::InteriorViscousFaceFlux,
)> {
    let face = &params.face_topology.interior[i];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return None;
    }
    let (mu, lambda) = transport_at_face(i, scratch, constant);
    let normal = face.normal;
    let geom = InteriorViscousFaceGeom {
        owner: face.owner,
        neighbor: face.neighbor,
        nx: normal.x,
        ny: normal.y,
        nz: normal.z,
        mu,
        lambda,
        owner_scale: face.owner_rhs_scale,
        neighbor_scale: face.neighbor_rhs_scale,
    };
    let flux = fused_interior_viscous_face_flux_averaged(
        face_averaged.lane(i),
        geom.nx,
        geom.ny,
        geom.nz,
        geom.mu,
        geom.lambda,
    );
    Some((geom, flux))
}

fn transport_at_face(
    i: usize,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) -> (Real, Real) {
    if let Some(coeffs) = constant {
        coeffs
    } else {
        (scratch.face_mu[i], scratch.face_lambda[i])
    }
}

#[cfg(any(not(feature = "parallel-fvm"), test))]
fn accumulate_one_interior_face(
    i: usize,
    face_averaged: &ViscousFaceAveragedSoA,
    residual_mut: &mut InteriorViscousResidualMut<'_>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
) {
    let face = &params.face_topology.interior[i];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return;
    }
    let (mu, lambda) = transport_at_face(i, scratch, constant);
    let normal = face.normal;
    let geom = InteriorViscousFaceGeom {
        owner: face.owner,
        neighbor: face.neighbor,
        nx: normal.x,
        ny: normal.y,
        nz: normal.z,
        mu,
        lambda,
        owner_scale: face.owner_rhs_scale,
        neighbor_scale: face.neighbor_rhs_scale,
    };
    let flux = fused_interior_viscous_face_flux_averaged(
        face_averaged.lane(i),
        geom.nx,
        geom.ny,
        geom.nz,
        geom.mu,
        geom.lambda,
    );
    scatter_fused_interior_viscous_face(residual_mut, &geom, &flux);
}

fn fill_cell_transport_coefficients(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let temperatures = &scratch.gradient.temperatures;
    #[cfg(feature = "parallel-fvm")]
    {
        use rayon::prelude::*;
        scratch
            .cell_mu
            .par_iter_mut()
            .zip(scratch.cell_lambda.par_iter_mut())
            .zip(temperatures.par_iter())
            .try_for_each(|((mu, lambda), &t)| -> Result<()> {
                let (m, l) = face_transport_coefficients(t, t, params.viscous, params.eos)?;
                *mu = m;
                *lambda = l;
                Ok(())
            })?;
    }
    #[cfg(not(feature = "parallel-fvm"))]
    {
        for (cell, &t) in temperatures.iter().enumerate() {
            let (mu, lambda) = face_transport_coefficients(t, t, params.viscous, params.eos)?;
            scratch.cell_mu[cell] = mu;
            scratch.cell_lambda[cell] = lambda;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "assembly_unstructured_viscous_tests.rs"]
mod tests;
