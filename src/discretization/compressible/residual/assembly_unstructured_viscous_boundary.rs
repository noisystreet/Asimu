//! 非结构粘性边界面装配。

use crate::core::ComputeFloat;
use crate::discretization::viscous_assembly::{
    ViscousBoundaryFaceKind, ViscousBoundaryFluxParams, accumulate_viscous_boundary,
    accumulate_viscous_boundary_typed, viscous_flux_at_boundary,
};
use crate::discretization::viscous_boundary_f32::{
    ViscousBoundaryFluxParamsF32, scatter_viscous_boundary_f32, viscous_flux_at_boundary_f32,
};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedResidual, ConservedResidualT, primitive_from_conserved_relaxed,
    primitive_from_conserved_relaxed_f32_from_state,
};

use super::{ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch};

use super::super::is_degenerate_volume;

pub(super) fn assemble_boundary_faces(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let boundary_params = ViscousBoundaryFluxParams {
        eos: params.eos,
        viscous: params.viscous,
        primitives: params.primitives,
        gradients: params.gradients,
    };
    let contributions = collect_viscous_boundary_contributions(
        params,
        &boundary_params,
        &scratch.gradient.temperatures,
    )?;
    apply_viscous_boundary_contributions(residual, &contributions)
}

pub(crate) fn assemble_boundary_faces_typed<T: ComputeFloat>(
    residual: &mut ConservedResidualT<T>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let boundary_params = ViscousBoundaryFluxParams {
        eos: params.eos,
        viscous: params.viscous,
        primitives: params.primitives,
        gradients: params.gradients,
    };
    let contributions = collect_viscous_boundary_contributions(
        params,
        &boundary_params,
        &scratch.gradient.temperatures,
    )?;
    for contrib in &contributions {
        accumulate_viscous_boundary_typed(
            residual,
            contrib.owner,
            &contrib.flux,
            contrib.area,
            contrib.owner_volume,
        )?;
    }
    Ok(())
}

/// f32 非结构粘性边界面装配（通量 compute + scatter 均为 f32）。
pub(crate) fn assemble_boundary_faces_f32(
    residual: &mut ConservedResidualT<f32>,
    face_topology: &crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32,
    ghosts: &crate::discretization::BoundaryGhostBuffer,
    params: &ViscousBoundaryFluxParamsF32<'_>,
    min_pressure: crate::core::Real,
    temperatures: &[f32],
) -> Result<()> {
    for face in &face_topology.boundary {
        if is_degenerate_volume(face.owner_volume as crate::core::Real) {
            continue;
        }
        let ghost = ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.face.index()))
        })?;
        let ghost_prim = primitive_from_conserved_relaxed_f32_from_state(
            params.eos,
            &ghost.conserved,
            min_pressure,
        )?;
        let kind = face.viscous;
        let flux = viscous_flux_at_boundary_f32(
            params,
            face.owner,
            ghost_prim,
            face.normal,
            face.spacing as crate::core::Real,
            ViscousBoundaryFaceKind {
                is_wall: kind.is_wall,
                no_slip: kind.no_slip,
                wall_heat: kind.wall_heat,
            },
            temperatures,
        )?;
        scatter_viscous_boundary_f32(residual, face.owner, &flux, face.area, face.owner_volume);
    }
    Ok(())
}

fn collect_viscous_boundary_contributions(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    boundary_params: &ViscousBoundaryFluxParams<'_>,
    temperatures: &[crate::core::Real],
) -> Result<Vec<ViscousBoundaryContribution>> {
    let mut out = Vec::with_capacity(params.face_topology.boundary.len());
    for face in &params.face_topology.boundary {
        if let Some(contrib) =
            compute_viscous_boundary_contribution(face, params, boundary_params, temperatures)?
        {
            out.push(contrib);
        }
    }
    Ok(out)
}

fn apply_viscous_boundary_contributions(
    residual: &mut ConservedResidual,
    contributions: &[ViscousBoundaryContribution],
) -> Result<()> {
    for contrib in contributions {
        accumulate_viscous_boundary(
            residual,
            contrib.owner,
            &contrib.flux,
            contrib.area,
            contrib.owner_volume,
        )?;
    }
    Ok(())
}

struct ViscousBoundaryContribution {
    owner: usize,
    flux: crate::discretization::viscous::ViscousFlux,
    area: crate::core::Real,
    owner_volume: crate::core::Real,
}

fn compute_viscous_boundary_contribution(
    face: &crate::discretization::unstructured_face_cache::UnstructuredBoundaryFace,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    boundary_params: &ViscousBoundaryFluxParams<'_>,
    temperatures: &[crate::core::Real],
) -> Result<Option<ViscousBoundaryContribution>> {
    if is_degenerate_volume(face.owner_volume) {
        return Ok(None);
    }
    let ghost = params.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.face.index()))
    })?;
    let ghost_prim =
        primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
    let kind = face.viscous;
    let flux = viscous_flux_at_boundary(
        boundary_params,
        face.owner,
        ghost_prim,
        face.normal,
        face.spacing,
        ViscousBoundaryFaceKind {
            is_wall: kind.is_wall,
            no_slip: kind.no_slip,
            wall_heat: kind.wall_heat,
        },
        temperatures,
    )?;
    Ok(Some(ViscousBoundaryContribution {
        owner: face.owner,
        flux,
        area: face.area,
        owner_volume: face.owner_volume,
    }))
}
