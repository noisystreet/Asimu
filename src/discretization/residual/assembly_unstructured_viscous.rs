//! 非结构 3D 网格粘性残差装配。

#[path = "assembly_unstructured_viscous_boundary.rs"]
mod boundary;
#[path = "assembly_unstructured_viscous_face_avg.rs"]
mod face_avg;
#[cfg(feature = "parallel-fvm")]
#[path = "assembly_unstructured_viscous_parallel.rs"]
mod parallel;

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
#[cfg(feature = "simd-fvm")]
use crate::discretization::viscous::InteriorViscousFaceFlux;
#[cfg(not(feature = "simd-fvm"))]
use crate::discretization::viscous::ViscousFaceAveragedSoA;
#[cfg(any(not(feature = "parallel-fvm"), test))]
use crate::discretization::viscous::scatter_fused_interior_viscous_face;
use crate::discretization::viscous::{
    InteriorViscousFaceGeom, InteriorViscousResidualMut, face_transport_coefficients,
    fused_interior_viscous_face_flux_averaged,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

#[cfg(feature = "simd-fvm")]
use face_avg::face_averaged_lane_at;

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
    exec: &mut crate::exec::ExecutionContext,
) -> Result<()> {
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(params.mesh.num_cells());
    crate::discretization::gradient::cell_temperatures_into(
        params.primitives,
        params.eos,
        Some(params.viscous),
        &mut scratch.gradient.temperatures,
    )?;
    assemble_viscous_residual_unstructured_with_scratch(residual, params, &mut scratch, exec)
}

fn assemble_viscous_residual_unstructured_with_scratch(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    exec: &mut crate::exec::ExecutionContext,
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
        assemble_interior_faces(residual, params, scratch, exec)?;
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
    pub exec: &'a mut crate::exec::ExecutionContext,
}

/// 非结构粘性 RHS 复用缓冲。
pub struct ViscousAssemblyUnstructuredScratch {
    pub gradient: UnstructuredGradientScratch,
    /// 内面心预平均速度与梯度（P7 非 SIMD，flux 顺序读）。
    #[cfg(not(feature = "simd-fvm"))]
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
            #[cfg(not(feature = "simd-fvm"))]
            face_averaged: ViscousFaceAveragedSoA::default(),
            cell_mu: Vec::new(),
            cell_lambda: Vec::new(),
            face_mu: Vec::new(),
            face_lambda: Vec::new(),
            constant_transport: None,
        }
    }

    #[cfg(not(feature = "simd-fvm"))]
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
    assemble_viscous_residual_unstructured_with_scratch(residual, &params, scratch, input.exec)
}

fn assemble_interior_faces(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    exec: &mut crate::exec::ExecutionContext,
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
    #[cfg(not(feature = "simd-fvm"))]
    {
        let _span = info_span!("unstructured_viscous_face_avg", faces = num_faces,).entered();
        face_avg::fill_face_averaged_viscous_soa(params, scratch);
    }
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux",
            faces = num_faces,
            colors = params.face_topology.interior_coloring.num_colors,
        )
        .entered();
        accumulate_interior_faces_fused(residual, params, scratch, exec)?;
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
        crate::exec::parallel::par_for_each_zip3_mut(
            &mut scratch.face_mu,
            &mut scratch.face_lambda,
            &params.face_topology.interior,
            |mu, lambda, face| {
                if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
                    return;
                }
                *mu = 0.5 * (cell_mu[face.owner] + cell_mu[face.neighbor]);
                *lambda = 0.5 * (cell_lambda[face.owner] + cell_lambda[face.neighbor]);
            },
        );
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
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    exec: &mut crate::exec::ExecutionContext,
) -> Result<()> {
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
            accumulate_viscous_bucket_batch4(layout, &mut residual_mut, params, scratch, constant)?;
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
                accumulate_one_interior_face(i, &mut residual_mut, params, scratch, constant);
            });
    }

    #[cfg(all(feature = "parallel-fvm", feature = "simd-fvm"))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_fused",
            path = "simd_batch4",
            faces = params.face_topology.interior.len(),
            colors = params.face_topology.interior_coloring.num_colors,
        )
        .entered();
        for layout in &params.face_topology.interior_coloring.bucket_batch_layouts {
            parallel::accumulate_viscous_bucket_batch4_fused(
                &mut residual_mut,
                layout,
                params,
                scratch,
                constant,
                exec,
            );
        }
    }

    #[cfg(all(feature = "parallel-fvm", not(feature = "simd-fvm")))]
    {
        let _span = info_span!(
            "unstructured_viscous_interior_flux_fused",
            path = "parallel_bucket",
            faces = params.face_topology.interior.len(),
            colors = params.face_topology.interior_coloring.num_colors,
        )
        .entered();
        for bucket in &params.face_topology.interior_coloring.buckets {
            parallel::accumulate_viscous_color_bucket_fused(
                &mut residual_mut,
                bucket,
                params,
                scratch,
                constant,
                exec,
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "simd-fvm", not(feature = "parallel-fvm")))]
fn accumulate_viscous_bucket_batch4(
    layout: &crate::discretization::InteriorFaceBucketBatchLayout,
    residual_mut: &mut InteriorViscousResidualMut<'_>,
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
                accumulate_one_interior_face(face_idx, residual_mut, params, scratch, constant);
            }
            continue;
        }
        for lane in 0..count as usize {
            scatter_fused_interior_viscous_face(residual_mut, &geoms[lane], &fluxes[lane]);
        }
    }
    for &face_idx in &layout.remainder {
        accumulate_one_interior_face(face_idx, residual_mut, params, scratch, constant);
    }
    Ok(())
}

#[cfg(feature = "simd-fvm")]
fn velocity_gradient_soa<'a>(
    params: &'a ViscousAssemblyUnstructuredParams<'a>,
) -> crate::exec::cpu::VelocityGradientSoA<'a> {
    use crate::exec::cpu::VelocityGradientSoA;

    let prim = params.primitives;
    let grad = params.gradients.velocity_gradient_slices();
    VelocityGradientSoA {
        ux: prim.velocity_x.values(),
        uy: prim.velocity_y.values(),
        uz: prim.velocity_z.values(),
        du_dx: grad.du_dx,
        du_dy: grad.du_dy,
        du_dz: grad.du_dz,
        dv_dx: grad.dv_dx,
        dv_dy: grad.dv_dy,
        dv_dz: grad.dv_dz,
        dw_dx: grad.dw_dx,
        dw_dy: grad.dw_dy,
        dw_dz: grad.dw_dz,
        dt_dx: grad.dt_dx,
        dt_dy: grad.dt_dy,
        dt_dz: grad.dt_dz,
    }
}

#[cfg(feature = "simd-fvm")]
fn viscous_batch_geom_from_static(
    batch: &crate::exec::ExecFaceBatchStatic4,
    mu: [Real; 4],
    lambda: [Real; 4],
) -> crate::exec::cpu::ViscousFaceBatchGeom {
    crate::exec::cpu::ViscousFaceBatchGeom {
        owners: batch.owners,
        neighbors: batch.neighbors,
        nx: batch.nx,
        ny: batch.ny,
        nz: batch.nz,
        mu,
        lambda,
    }
}

#[cfg(feature = "simd-fvm")]
pub(super) fn compute_viscous_batch4_into(
    batch: &crate::exec::ExecFaceBatchStatic4,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
    constant: Option<(Real, Real)>,
    geoms: &mut [InteriorViscousFaceGeom],
    fluxes: &mut [InteriorViscousFaceFlux],
) -> u8 {
    use crate::exec::cpu::fused_interior_viscous_face_flux_batch4_from_soa;

    debug_assert!(geoms.len() >= 4 && fluxes.len() >= 4);

    if batch.simd_eligible() {
        let mut mu = [0.0; 4];
        let mut lambda = [0.0; 4];
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
        let geom = viscous_batch_geom_from_static(batch, mu, lambda);
        let vel = velocity_gradient_soa(params);
        let flux4 = fused_interior_viscous_face_flux_batch4_from_soa(&geom, &vel);
        for (i, flux) in fluxes.iter_mut().enumerate().take(4) {
            *flux = InteriorViscousFaceFlux {
                mx: flux4.mx[i],
                my: flux4.my[i],
                mz: flux4.mz[i],
                energy: flux4.energy[i],
            };
        }
        return 4;
    }

    // 紧凑写入 `[0, count)`：与 `fill_batch_slot_valid` / 串行 `0..count` scatter 一致。
    // 不可按 enumerate lane 稀疏写入——跳过退化面会在槽位间留空，导致 scatter 错位。
    let mut count = 0u8;
    for &face_idx in &batch.face_indices {
        let face = &params.face_topology.interior[face_idx];
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        let slot = count as usize;
        let (mu, lambda) = transport_at_face(face_idx, scratch, constant);
        geoms[slot] = InteriorViscousFaceGeom {
            owner: face.owner,
            neighbor: face.neighbor,
            nx: face.normal.x,
            ny: face.normal.y,
            nz: face.normal.z,
            mu,
            lambda,
            owner_scale: face.owner_rhs_scale,
            neighbor_scale: face.neighbor_rhs_scale,
        };
        let lane_avg = viscous_averaged_lane(face_idx, scratch, params);
        fluxes[slot] = fused_interior_viscous_face_flux_averaged(
            lane_avg,
            geoms[slot].nx,
            geoms[slot].ny,
            geoms[slot].nz,
            mu,
            lambda,
        );
        count += 1;
    }
    count
}

#[inline(always)]
fn viscous_averaged_lane(
    i: usize,
    scratch: &ViscousAssemblyUnstructuredScratch,
    params: &ViscousAssemblyUnstructuredParams<'_>,
) -> crate::discretization::viscous::ViscousFaceAveragedLane {
    #[cfg(feature = "simd-fvm")]
    {
        let _ = scratch;
        face_averaged_lane_at(i, params)
    }
    #[cfg(not(feature = "simd-fvm"))]
    {
        let _ = params;
        scratch.face_averaged.lane(i)
    }
}

#[cfg(feature = "parallel-fvm")]
pub(super) fn interior_face_flux_contribution(
    i: usize,
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
    let lane = viscous_averaged_lane(i, scratch, params);
    let flux = fused_interior_viscous_face_flux_averaged(
        lane,
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
    let lane = viscous_averaged_lane(i, scratch, params);
    let flux = fused_interior_viscous_face_flux_averaged(
        lane,
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
        crate::exec::parallel::par_try_for_each_zip3(
            &mut scratch.cell_mu,
            &mut scratch.cell_lambda,
            temperatures,
            |mu, lambda, &t| -> Result<()> {
                let (m, l) = face_transport_coefficients(t, t, params.viscous, params.eos)?;
                *mu = m;
                *lambda = l;
                Ok(())
            },
        )?;
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
