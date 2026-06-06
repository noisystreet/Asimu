//! 多块 1-to-1 接口共享无粘通量装配。

use std::collections::BTreeSet;

use tracing::info_span;

use super::BlockInterfaceLink;
use crate::core::Real;
use crate::discretization::residual::inviscid_boundary_face_flux;
use crate::discretization::{BoundaryInviscidFluxInput, InviscidFlux};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh3d, StructuredBlock3d};
use crate::physics::{FreestreamParams, IdealGasEoS};

#[derive(Debug, Clone)]
pub(super) struct InterfaceResidualContribution {
    pub(super) cell: usize,
    flux: InviscidFlux,
    scale: Real,
}

pub(super) struct SharedInterfaceResidualParams<'a> {
    pub(super) blocks: &'a [StructuredBlock3d],
    pub(super) links: &'a [Vec<BlockInterfaceLink>],
    pub(super) snapshots: &'a [ConservedFields],
    pub(super) eos: &'a IdealGasEoS,
    pub(super) freestream: &'a FreestreamParams,
    pub(super) inviscid: &'a crate::discretization::InviscidFluxConfig,
}

pub(super) fn compute_shared_interface_residuals(
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
    let mut out = (0..params.blocks.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    for (owner_block, links) in params.links.iter().enumerate() {
        for link in links {
            let key = canonical_interface_key(owner_block, link);
            if !seen.insert(key) {
                continue;
            }
            add_shared_interface_face(params, primitives, owner_block, link, &mut out)?;
        }
    }
    Ok(out)
}

fn canonical_interface_key(
    owner_block: usize,
    link: &BlockInterfaceLink,
) -> (usize, usize, usize, usize) {
    let a = (owner_block, link.owner_cell);
    let b = (link.donor_block_index, link.donor_cell);
    if a <= b {
        (a.0, a.1, b.0, b.1)
    } else {
        (b.0, b.1, a.0, a.1)
    }
}

fn add_shared_interface_face(
    params: &SharedInterfaceResidualParams<'_>,
    primitives: &[PrimitiveFields],
    owner_block: usize,
    link: &BlockInterfaceLink,
    out: &mut [Vec<InterfaceResidualContribution>],
) -> Result<()> {
    let owner_mesh = &params.blocks[owner_block].mesh;
    let donor_mesh = &params.blocks[link.donor_block_index].mesh;
    let exterior = params.snapshots[link.donor_block_index].cell_state(link.donor_cell)?;
    let flux = inviscid_boundary_face_flux(BoundaryInviscidFluxInput {
        mesh: owner_mesh,
        structured: owner_mesh,
        primitives: &primitives[owner_block],
        eos: params.eos,
        config: params.inviscid,
        min_pressure: p_floor(params.freestream),
        face: link.face,
        exterior,
    })?;
    let geom = owner_mesh.face_geometry_3d(link.face)?;
    let donor_volumes = donor_mesh.cell_volumes();
    out[owner_block].push(InterfaceResidualContribution {
        cell: link.owner_cell,
        flux,
        scale: -geom.area / owner_mesh.cell_volumes()[link.owner_cell],
    });
    out[link.donor_block_index].push(InterfaceResidualContribution {
        cell: link.donor_cell,
        flux,
        scale: geom.area / donor_volumes[link.donor_cell],
    });
    Ok(())
}

pub(super) fn apply_interface_residuals(
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

fn p_floor(freestream: &FreestreamParams) -> Real {
    crate::field::positivity_pressure_floor(freestream.pressure)
}
