//! 非结构 3D 网格无粘残差装配（一阶面循环）。

#[cfg(feature = "simd-fvm")]
#[path = "assembly_unstructured_inviscid_simd.rs"]
mod assembly_unstructured_inviscid_simd;

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::{FaceId, Real};
pub(super) use crate::discretization::inviscid::InteriorInviscidScatterGeom;
#[cfg(any(not(feature = "parallel-fvm"), test))]
use crate::discretization::inviscid::scatter_fused_interior_inviscid_face;
use crate::discretization::inviscid::{
    interior_inviscid_residual_mut, scatter_fused_boundary_inviscid_face,
};
use crate::discretization::unstructured_face_cache::{
    UnstructuredFaceTopology, UnstructuredSolverMeshCache,
};
use crate::discretization::{
    BoundaryGhostBuffer, FaceFluxInput, GradientFields, InviscidFlux, InviscidFluxConfig,
    ReconstructionKind, UnstructuredGradientLimiter, UnstructuredLinearReconstructionCtx,
    face_inviscid_flux, face_inviscid_flux_first_order_boundary_soa,
    face_inviscid_flux_first_order_interior_soa, face_inviscid_flux_from_interface,
    reconstruct_unstructured_boundary_face, reconstruct_unstructured_interior_face,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;
use tracing::info_span;

use super::{accumulate_boundary_face, accumulate_interior_face, is_degenerate_volume};

pub struct InviscidAssemblyUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    /// 若提供，内面走缓存拓扑 + 着色桶顺序（与粘性共用 `InteriorFaceColoring`）。
    pub face_topology: Option<&'a UnstructuredFaceTopology>,
    /// 二阶重构：完整 mesh cache（含限制器样本与面心偏移）。
    pub mesh_cache: Option<&'a UnstructuredSolverMeshCache>,
    /// 二阶重构：IDWLS 原始变量梯度。
    pub gradients: Option<&'a GradientFields>,
    pub min_pressure: Real,
    pub exec: &'a crate::exec::ExecutionContext,
}

/// 非结构一阶 Euler 残差：遍历显式 face owner/neighbor 拓扑。
pub fn assemble_inviscid_residual_unstructured(
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构场/残差/primitive 长度须等于网格单元数 {n}"
        )));
    }
    if params.config.reconstruction == ReconstructionKind::Muscl {
        validate_unstructured_linear_reconstruction_params(params)?;
    }
    residual.clear();
    {
        let _span = if let Some(topology) = params.face_topology {
            info_span!(
                "unstructured_inviscid_interior_faces",
                faces = topology.interior.len(),
                colors = topology.interior_coloring.num_colors,
            )
            .entered()
        } else {
            info_span!("unstructured_inviscid_interior_faces", path = "mesh_loop").entered()
        };
        if let Some(topology) = params.face_topology {
            assemble_interior_faces_cached(residual, fields, params, topology)?;
        } else {
            assemble_interior_faces(mesh, residual, params)?;
        }
    }
    {
        let _span = info_span!(
            "unstructured_inviscid_boundary_faces",
            faces = inviscid_boundary_face_count(params),
        )
        .entered();
        assemble_boundary_faces(residual, params)?;
    }
    Ok(())
}

fn inviscid_boundary_face_count(params: &InviscidAssemblyUnstructuredParams<'_>) -> usize {
    if let Some(topology) = params.face_topology {
        return topology.boundary.len();
    }
    params
        .boundaries
        .patches()
        .iter()
        .filter(|patch| !matches!(patch.kind, BoundaryKind::Periodic { .. }))
        .map(|patch| patch.face_ids.len())
        .sum()
}

#[cfg(feature = "simd-fvm")]
pub(crate) fn try_assemble_first_order_interior_simd_f64(
    residual: &mut crate::field::ConservedResidual,
    fields: &crate::field::ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &crate::discretization::UnstructuredFaceTopology,
) -> Result<bool> {
    assembly_unstructured_inviscid_simd::try_assemble_interior_faces_cached(
        residual, fields, params, topology,
    )
}

fn validate_unstructured_linear_reconstruction_params(
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    if params.mesh_cache.is_none() || params.face_topology.is_none() || params.gradients.is_none() {
        return Err(AsimuError::Config(
            "非结构二阶线性重构须同时提供 mesh_cache、face_topology 与 gradients".to_string(),
        ));
    }
    if params.config.unstructured_gradient_limiter.is_none() {
        return Err(AsimuError::Config(
            "非结构二阶线性重构须设置 unstructured_limiter（barth_jespersen 或 venkatakrishnan）"
                .to_string(),
        ));
    }
    Ok(())
}

fn unstructured_limiter(
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> UnstructuredGradientLimiter {
    params
        .config
        .unstructured_gradient_limiter
        .unwrap_or(UnstructuredGradientLimiter::BarthJespersen)
}

pub(crate) fn compute_interior_inviscid_face_contribution(
    face_idx: usize,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<Option<(InteriorInviscidScatterGeom, InviscidFlux)>> {
    let face = &topology.interior[face_idx];
    if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
        return Ok(None);
    }
    let flux = if params.config.reconstruction == ReconstructionKind::Muscl {
        let mesh_cache = params.mesh_cache.expect("linear reconstruction cache");
        let gradients = params.gradients.expect("linear reconstruction gradients");
        let limiter = unstructured_limiter(params);
        let ctx = UnstructuredLinearReconstructionCtx {
            mesh_cache,
            primitives: params.primitives,
            ghosts: params.ghosts,
            eos: params.eos,
            min_pressure: params.min_pressure,
            limiter,
        };
        let iface = reconstruct_unstructured_interior_face(
            face,
            ctx,
            gradients.inviscid_primitive_grad_at(face.owner),
            gradients.inviscid_primitive_grad_at(face.neighbor),
        )?;
        face_inviscid_flux_from_interface(iface, face.normal, params.eos, params.config)?
    } else {
        face_inviscid_flux_first_order_interior_soa(
            face.owner,
            face.neighbor,
            params.primitives,
            face.normal,
            params.eos,
            params.config,
        )?
    };
    let geom = InteriorInviscidScatterGeom {
        owner: face.owner,
        neighbor: face.neighbor,
        owner_scale: face.owner_rhs_scale,
        neighbor_scale: face.neighbor_rhs_scale,
    };
    Ok(Some((geom, flux)))
}

#[cfg(any(not(feature = "parallel-fvm"), test))]
pub(super) fn accumulate_one_interior_inviscid_face_fused(
    face_idx: usize,
    residual_mut: &mut crate::discretization::inviscid::InteriorInviscidResidualMut<'_>,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    if let Some((geom, flux)) =
        compute_interior_inviscid_face_contribution(face_idx, params, topology)?
    {
        scatter_fused_interior_inviscid_face(residual_mut, &geom, &flux);
    }
    Ok(())
}

#[cfg(any(not(feature = "parallel-fvm"), test))]
pub(super) fn accumulate_one_interior_inviscid_face(
    face_idx: usize,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    accumulate_one_interior_inviscid_face_fused(
        face_idx,
        &mut interior_inviscid_residual_mut(residual),
        params,
        topology,
    )
}

fn assemble_interior_faces_cached(
    residual: &mut ConservedResidual,
    #[cfg_attr(not(feature = "simd-fvm"), allow(unused_variables))] fields: &ConservedFields,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    #[cfg(feature = "simd-fvm")]
    if assembly_unstructured_inviscid_simd::try_assemble_interior_faces_cached(
        residual, fields, params, topology,
    )? {
        return Ok(());
    }

    #[cfg(not(feature = "parallel-fvm"))]
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_flux_fused",
            path = "colored_serial"
        )
        .entered();
        let mut residual_mut = interior_inviscid_residual_mut(residual);
        for bucket in &topology.interior_coloring.buckets {
            for &face_idx in bucket {
                accumulate_one_interior_inviscid_face_fused(
                    face_idx,
                    &mut residual_mut,
                    params,
                    topology,
                )?;
            }
        }
        return Ok(());
    }

    #[cfg(feature = "parallel-fvm")]
    {
        let _span = info_span!(
            "unstructured_inviscid_interior_flux_fused",
            path = "parallel_bucket",
            faces = topology.interior.len(),
            colors = topology.interior_coloring.num_colors,
        )
        .entered();
        let residual_mut = interior_inviscid_residual_mut(residual);
        for bucket in &topology.interior_coloring.buckets {
            let contributions =
                crate::exec::parallel::par_try_map_face_indices(bucket, 1024, |face_idx| {
                    compute_interior_inviscid_face_contribution(face_idx, params, topology)
                })?;
            let pairs: Vec<_> = contributions.into_iter().flatten().collect();
            crate::exec::scatter::scatter_inviscid_pairs(
                crate::exec::scatter::InviscidPairScatter {
                    ctx: params.exec,
                    bucket_len: bucket.len(),
                    pairs: &pairs,
                    residual: crate::exec::scatter::InviscidResidualMut {
                        density: residual_mut.density,
                        mx: residual_mut.mx,
                        my: residual_mut.my,
                        mz: residual_mut.mz,
                        energy: residual_mut.energy,
                    },
                },
                |g, f| crate::exec::scatter::InviscidScatterOp {
                    owner: g.owner,
                    neighbor: g.neighbor,
                    owner_scale: g.owner_scale,
                    neighbor_scale: g.neighbor_scale,
                    mass: f.mass,
                    momentum: f.momentum,
                    energy: f.energy,
                },
            );
        }
    }
    Ok(())
}

fn assemble_interior_faces(
    mesh: &UnstructuredMesh3d,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    for face in 0..mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let Some(neighbor_id) = mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner_id = mesh.face_owner(face_id)?;
        let owner = owner_id.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        let metric = mesh.face_metric(face_id);
        let owner_volume = mesh.cell_metric(owner_id).volume;
        let neighbor_volume = mesh.cell_metric(neighbor_id).volume;
        if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
            continue;
        }
        let flux = face_inviscid_flux_first_order_interior_soa(
            owner,
            neighbor,
            params.primitives,
            metric.normal,
            params.eos,
            params.config,
        )?;
        accumulate_interior_face(
            residual,
            owner,
            neighbor,
            &flux,
            metric.area,
            owner_volume,
            neighbor_volume,
        )?;
    }
    Ok(())
}

fn assemble_boundary_faces(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    if params.config.reconstruction == ReconstructionKind::Muscl {
        assemble_boundary_faces_linear_reconstruction(residual, params)
    } else if let Some(topology) = params.face_topology {
        assemble_boundary_faces_first_order_cached(residual, params, topology)
    } else {
        assemble_boundary_faces_first_order_mesh(residual, params)
    }
}

fn assemble_boundary_faces_linear_reconstruction(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mesh_cache = params.mesh_cache.expect("linear reconstruction cache");
    let gradients = params.gradients.expect("linear reconstruction gradients");
    let ctx = UnstructuredLinearReconstructionCtx {
        mesh_cache,
        primitives: params.primitives,
        ghosts: params.ghosts,
        eos: params.eos,
        min_pressure: params.min_pressure,
        limiter: unstructured_limiter(params),
    };
    for bface in &mesh_cache.face_topology.boundary {
        if is_degenerate_volume(bface.owner_volume) {
            continue;
        }
        let iface = reconstruct_unstructured_boundary_face(
            bface,
            ctx,
            gradients.inviscid_primitive_grad_at(bface.owner),
        )?;
        let flux =
            face_inviscid_flux_from_interface(iface, bface.normal, params.eos, params.config)?;
        accumulate_boundary_face(residual, bface.owner, &flux, bface.area, bface.owner_volume)?;
    }
    Ok(())
}

fn assemble_boundary_faces_first_order_cached(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    let mut residual_mut = interior_inviscid_residual_mut(residual);
    for bface in &topology.boundary {
        if bface.owner_rhs_scale == 0.0 {
            continue;
        }
        let ghost = params.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let flux = face_inviscid_flux_first_order_boundary_soa(
            bface.owner,
            params.primitives,
            &ghost.conserved,
            bface.normal,
            params.eos,
            params.config,
            params.min_pressure,
        )?;
        scatter_fused_boundary_inviscid_face(
            &mut residual_mut,
            bface.owner,
            bface.owner_rhs_scale,
            &flux,
        );
    }
    Ok(())
}

fn assemble_boundary_faces_first_order_mesh(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    for patch in params.boundaries.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner_id = mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let metric = mesh.face_metric(face);
            let owner_volume = mesh.cell_metric(owner_id).volume;
            if is_degenerate_volume(owner_volume) {
                continue;
            }
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
            })?;
            let owner_prim = params.primitives.cell_primitive(owner);
            let ghost_prim = crate::field::primitive_from_conserved_relaxed(
                params.eos,
                &ghost.conserved,
                params.min_pressure,
            )?;
            let flux = face_inviscid_flux(
                FaceFluxInput::first_order(&owner_prim, &ghost_prim),
                metric.normal,
                params.eos,
                params.config,
            )?;
            accumulate_boundary_face(residual, owner, &flux, metric.area, owner_volume)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "assembly_unstructured_tests.rs"]
mod tests;
