//! 多块 1-to-1 接口共享无粘通量装配。

use tracing::info_span;

use crate::core::Real;
use crate::discretization::residual::inviscid_boundary_face_flux_with_normal;
use crate::discretization::{BoundaryInviscidFluxInput, InviscidFlux};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::StructuredBlock3d;
use crate::physics::{FreestreamParams, IdealGasEoS};
use crate::solver::compressible::multiblock::SharedInterfaceFace;

#[derive(Debug, Clone)]
pub(crate) struct InterfaceResidualContribution {
    pub(crate) cell: usize,
    flux: InviscidFlux,
    scale: Real,
}

pub(crate) struct SharedInterfaceResidualParams<'a> {
    pub(crate) blocks: &'a [StructuredBlock3d],
    pub(crate) shared_faces: &'a [SharedInterfaceFace],
    pub(crate) snapshots: &'a [ConservedFields],
    pub(crate) eos: &'a IdealGasEoS,
    pub(crate) freestream: &'a FreestreamParams,
    pub(crate) inviscid: &'a crate::discretization::InviscidFluxConfig,
}

pub(crate) fn compute_shared_interface_residuals(
    params: &SharedInterfaceResidualParams<'_>,
) -> Result<Vec<Vec<InterfaceResidualContribution>>> {
    let mut primitives = Vec::with_capacity(params.blocks.len());
    for (block, fields) in params.blocks.iter().zip(params.snapshots.iter()) {
        let _span = info_span!(
            "fill_interface_primitives",
            block = %block.name,
            cells = block.mesh.num_cells()
        )
        .entered();
        let mut prim = PrimitiveFields::zeros(block.mesh.num_cells())?;
        prim.fill_from_conserved(fields, params.eos, p_floor(params.freestream))?;
        primitives.push(prim);
    }
    {
        let _span = info_span!("assemble_shared_interface_residuals").entered();
        assemble_shared_interface_residuals(params, &primitives)
    }
}

fn assemble_shared_interface_residuals(
    params: &SharedInterfaceResidualParams<'_>,
    primitives: &[PrimitiveFields],
) -> Result<Vec<Vec<InterfaceResidualContribution>>> {
    let mut out = new_contribution_buffers(params.blocks.len(), params.shared_faces);
    for face in params.shared_faces {
        add_shared_interface_face(params, primitives, face, &mut out)?;
    }
    Ok(out)
}

fn new_contribution_buffers(
    blocks: usize,
    shared_faces: &[SharedInterfaceFace],
) -> Vec<Vec<InterfaceResidualContribution>> {
    let mut counts = vec![0usize; blocks];
    for face in shared_faces {
        counts[face.owner_block_index] += 1;
        counts[face.donor_block_index] += 1;
    }
    counts.into_iter().map(Vec::with_capacity).collect()
}

fn add_shared_interface_face(
    params: &SharedInterfaceResidualParams<'_>,
    primitives: &[PrimitiveFields],
    face: &SharedInterfaceFace,
    out: &mut [Vec<InterfaceResidualContribution>],
) -> Result<()> {
    let owner_mesh = &params.blocks[face.owner_block_index].mesh;
    let exterior = params.snapshots[face.donor_block_index].cell_state(face.donor_cell)?;
    let flux = inviscid_boundary_face_flux_with_normal(
        BoundaryInviscidFluxInput {
            mesh: owner_mesh,
            structured: owner_mesh,
            primitives: &primitives[face.owner_block_index],
            eos: params.eos,
            config: params.inviscid,
            min_pressure: p_floor(params.freestream),
            face: face.face,
            exterior,
        },
        face.normal,
    )?;
    out[face.owner_block_index].push(InterfaceResidualContribution {
        cell: face.owner_cell,
        flux,
        scale: face.owner_scale,
    });
    out[face.donor_block_index].push(InterfaceResidualContribution {
        cell: face.donor_cell,
        flux,
        scale: face.donor_scale,
    });
    Ok(())
}

pub(crate) fn apply_interface_residuals(
    residual: &mut ConservedResidual,
    contributions: &[InterfaceResidualContribution],
) -> Result<()> {
    for contribution in contributions {
        residual.add_flux_to_cell(
            contribution.cell,
            contribution.flux.mass,
            contribution.flux.momentum,
            contribution.flux.energy,
            contribution.scale,
        )?;
    }
    Ok(())
}

/// typed 多块共享接口残差修正（通量仍以 f64 装配，写入 `ConservedResidualT<T>`）。
pub(crate) fn apply_interface_residuals_typed<T: crate::core::ComputeFloat>(
    residual: &mut crate::field::ConservedResidualT<T>,
    contributions: &[InterfaceResidualContribution],
) -> Result<()> {
    for contribution in contributions {
        residual.add_flux_to_cell(
            contribution.cell,
            contribution.flux.mass,
            contribution.flux.momentum,
            contribution.flux.energy,
            contribution.scale,
        )?;
    }
    Ok(())
}

fn p_floor(freestream: &FreestreamParams) -> Real {
    crate::field::positivity_pressure_floor(freestream.pressure)
}
