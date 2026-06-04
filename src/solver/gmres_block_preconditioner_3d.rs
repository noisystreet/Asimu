//! GMRES 单元块对角预条件器的局部 Jacobian 构造。

use crate::core::Real;
use crate::discretization::residual::{
    BoundaryInviscidFluxInput, InviscidAssembly3dParams, inviscid_boundary_face_flux,
    inviscid_i_face_flux, inviscid_j_face_flux, inviscid_k_face_flux, is_degenerate_volume,
};
use crate::discretization::{InviscidFlux, InviscidFluxConfig};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, PrimitiveFields, primitive_from_conserved_relaxed};
use crate::linalg::CellBlockDiagonalPreconditioner;
use crate::mesh::{LogicalFace3d, StructuredMesh3d};
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use super::CompressibleAdvanceContext3d;
use super::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, component_basis_increment, conserved_component_scales,
};
use crate::field::{max_physical_increment_scale, state_after_increment};

pub(super) fn build_cell_block_preconditioner(
    ctx: &mut CompressibleAdvanceContext3d<'_>,
    fields: &ConservedFields,
    inviscid: &InviscidFluxConfig,
    dt: &[Real],
    p_floor: Real,
    epsilon_rel: Real,
) -> Result<CellBlockDiagonalPreconditioner> {
    let n = fields.num_cells();
    let mut blocks = vec![0.0; n * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    for k in 0..ctx.structured.nz {
        for j in 0..ctx.structured.ny {
            for i in 0..ctx.structured.nx {
                let cell = ctx.structured.cell_index(i, j, k);
                fill_cell_block(
                    &mut blocks,
                    CellBlockBuild {
                        ctx,
                        fields,
                        inviscid,
                        dt_cell: dt[cell],
                        p_floor,
                        epsilon_rel,
                        cell,
                        i,
                        j,
                        k,
                    },
                )?;
            }
        }
    }
    CellBlockDiagonalPreconditioner::from_blocks(CONSERVED_COMPONENTS_3D, blocks)
}

struct CellBlockBuild<'a, 'ctx> {
    ctx: &'a mut CompressibleAdvanceContext3d<'ctx>,
    fields: &'a ConservedFields,
    inviscid: &'a InviscidFluxConfig,
    dt_cell: Real,
    p_floor: Real,
    epsilon_rel: Real,
    cell: usize,
    i: usize,
    j: usize,
    k: usize,
}

fn fill_cell_block(blocks: &mut [Real], params: CellBlockBuild<'_, '_>) -> Result<()> {
    let CellBlockBuild {
        ctx,
        fields,
        inviscid,
        dt_cell,
        p_floor,
        epsilon_rel,
        cell,
        i,
        j,
        k,
    } = params;
    let base_state = fields.cell_state(cell)?;
    let base_primitive = ctx.primitive_scratch.cell_primitive(cell);
    let base_local = local_inviscid_residual(ctx, inviscid, p_floor, i, j, k)?;
    let scales = conserved_component_scales(&base_state);
    for (col, scale) in scales
        .iter()
        .copied()
        .enumerate()
        .take(CONSERVED_COMPONENTS_3D)
    {
        let requested_eps = epsilon_rel.sqrt() * scale;
        let unit = component_basis_increment(col);
        let eps =
            max_physical_increment_scale(&base_state, unit, requested_eps, ctx.eos.gamma, p_floor);
        if eps <= 0.0 {
            return Err(AsimuError::Solver(format!(
                "GMRES 局部块预条件器：cell {cell} 分量 {col} 无法构造正性扰动"
            )));
        }
        let perturbed_state = state_after_increment(&base_state, unit, eps);
        write_cell_primitive(
            &mut ctx.primitive_scratch,
            cell,
            &perturbed_state,
            ctx.eos,
            p_floor,
        )?;
        let perturbed_local = local_inviscid_residual(ctx, inviscid, p_floor, i, j, k)?;
        let jv = local_residual_difference(perturbed_local, base_local, eps);
        write_block_column(blocks, cell, col, dt_cell, jv);
    }
    restore_cell_primitive(&mut ctx.primitive_scratch, cell, base_primitive);
    Ok(())
}

fn local_inviscid_residual(
    ctx: &CompressibleAdvanceContext3d<'_>,
    inviscid: &InviscidFluxConfig,
    p_floor: Real,
    i: usize,
    j: usize,
    k: usize,
) -> Result<[Real; CONSERVED_COMPONENTS_3D]> {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    let mesh = ctx.structured;
    let volume = mesh.cell_metric(i, j, k).volume;
    if is_degenerate_volume(volume) {
        return Ok(out);
    }
    let assembly = local_assembly(ctx, inviscid, p_floor);
    add_local_interior_faces(&mut out, &assembly, mesh, LocalIndex { i, j, k }, volume)?;
    add_local_boundary_faces(
        &mut out,
        ctx,
        inviscid,
        p_floor,
        LocalIndex { i, j, k },
        volume,
    )?;
    Ok(out)
}

fn local_assembly<'a>(
    ctx: &'a CompressibleAdvanceContext3d<'_>,
    inviscid: &'a InviscidFluxConfig,
    p_floor: Real,
) -> InviscidAssembly3dParams<'a> {
    InviscidAssembly3dParams {
        mesh: ctx.structured,
        eos: ctx.eos,
        config: inviscid,
        boundaries: ctx.patches,
        ghosts: &*ctx.ghosts,
        primitives: &ctx.primitive_scratch,
        min_pressure: p_floor,
    }
}

#[derive(Debug, Clone, Copy)]
struct LocalIndex {
    i: usize,
    j: usize,
    k: usize,
}

fn add_local_interior_faces(
    out: &mut [Real; CONSERVED_COMPONENTS_3D],
    assembly: &InviscidAssembly3dParams<'_>,
    mesh: &StructuredMesh3d,
    idx: LocalIndex,
    volume: Real,
) -> Result<()> {
    let LocalIndex { i, j, k } = idx;
    if i > 0 {
        let face = mesh.i_face_metric(i - 1, j, k);
        add_flux_vector(
            out,
            &inviscid_i_face_flux(assembly, i - 1, j, k)?,
            face.area / volume,
        );
    }
    if i + 1 < mesh.nx {
        let face = mesh.i_face_metric(i, j, k);
        add_flux_vector(
            out,
            &inviscid_i_face_flux(assembly, i, j, k)?,
            -face.area / volume,
        );
    }
    if j > 0 {
        let face = mesh.j_face_metric(i, j - 1, k);
        add_flux_vector(
            out,
            &inviscid_j_face_flux(assembly, i, j - 1, k)?,
            face.area / volume,
        );
    }
    if j + 1 < mesh.ny {
        let face = mesh.j_face_metric(i, j, k);
        add_flux_vector(
            out,
            &inviscid_j_face_flux(assembly, i, j, k)?,
            -face.area / volume,
        );
    }
    if k > 0 {
        let face = mesh.k_face_metric(i, j, k - 1);
        add_flux_vector(
            out,
            &inviscid_k_face_flux(assembly, i, j, k - 1)?,
            face.area / volume,
        );
    }
    if k + 1 < mesh.nz {
        let face = mesh.k_face_metric(i, j, k);
        add_flux_vector(
            out,
            &inviscid_k_face_flux(assembly, i, j, k)?,
            -face.area / volume,
        );
    }
    Ok(())
}

fn add_local_boundary_faces(
    out: &mut [Real; CONSERVED_COMPONENTS_3D],
    ctx: &CompressibleAdvanceContext3d<'_>,
    inviscid: &InviscidFluxConfig,
    p_floor: Real,
    idx: LocalIndex,
    volume: Real,
) -> Result<()> {
    let mesh = ctx.structured;
    let LocalIndex { i, j, k } = idx;
    if i == 0 {
        add_local_boundary_face(
            out,
            ctx,
            inviscid,
            p_floor,
            LogicalFace3d::IMin,
            j + k * mesh.ny,
            volume,
        )?;
    }
    if i + 1 == mesh.nx {
        add_local_boundary_face(
            out,
            ctx,
            inviscid,
            p_floor,
            LogicalFace3d::IMax,
            j + k * mesh.ny,
            volume,
        )?;
    }
    if j == 0 {
        add_local_boundary_face(
            out,
            ctx,
            inviscid,
            p_floor,
            LogicalFace3d::JMin,
            i + k * mesh.nx,
            volume,
        )?;
    }
    if j + 1 == mesh.ny {
        add_local_boundary_face(
            out,
            ctx,
            inviscid,
            p_floor,
            LogicalFace3d::JMax,
            i + k * mesh.nx,
            volume,
        )?;
    }
    if k == 0 {
        add_local_boundary_face(
            out,
            ctx,
            inviscid,
            p_floor,
            LogicalFace3d::KMin,
            i + j * mesh.nx,
            volume,
        )?;
    }
    if k + 1 == mesh.nz {
        add_local_boundary_face(
            out,
            ctx,
            inviscid,
            p_floor,
            LogicalFace3d::KMax,
            i + j * mesh.nx,
            volume,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_local_boundary_face(
    out: &mut [Real; CONSERVED_COMPONENTS_3D],
    ctx: &CompressibleAdvanceContext3d<'_>,
    inviscid: &InviscidFluxConfig,
    p_floor: Real,
    logical: LogicalFace3d,
    local: usize,
    volume: Real,
) -> Result<()> {
    let face = logical.encode(local as u32);
    let Some(ghost) = ctx.ghosts.get_face(face) else {
        return Ok(());
    };
    let geom = ctx.mesh.face_geometry_3d(face)?;
    let flux = inviscid_boundary_face_flux(BoundaryInviscidFluxInput {
        mesh: ctx.mesh,
        structured: ctx.structured,
        primitives: &ctx.primitive_scratch,
        eos: ctx.eos,
        config: inviscid,
        min_pressure: p_floor,
        face,
        exterior: ghost.conserved,
    })?;
    add_flux_vector(out, &flux, -geom.area / volume);
    Ok(())
}

fn add_flux_vector(out: &mut [Real; CONSERVED_COMPONENTS_3D], flux: &InviscidFlux, scale: Real) {
    out[0] += scale * flux.mass;
    out[1] += scale * flux.momentum[0];
    out[2] += scale * flux.momentum[1];
    out[3] += scale * flux.momentum[2];
    out[4] += scale * flux.energy;
}

fn local_residual_difference(
    perturbed: [Real; CONSERVED_COMPONENTS_3D],
    base: [Real; CONSERVED_COMPONENTS_3D],
    eps: Real,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    for component in 0..CONSERVED_COMPONENTS_3D {
        out[component] = (perturbed[component] - base[component]) / eps;
    }
    out
}

fn write_block_column(
    blocks: &mut [Real],
    cell: usize,
    col: usize,
    dt_cell: Real,
    jv: [Real; CONSERVED_COMPONENTS_3D],
) {
    let block_offset = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
    for row in 0..CONSERVED_COMPONENTS_3D {
        let diagonal = if row == col { 1.0 / dt_cell } else { 0.0 };
        blocks[block_offset + row * CONSERVED_COMPONENTS_3D + col] = diagonal - jv[row];
    }
}

fn write_cell_primitive(
    primitives: &mut PrimitiveFields,
    cell: usize,
    state: &ConservedState,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<()> {
    let prim = primitive_from_conserved_relaxed(eos, state, p_floor)?;
    primitives.density.values_mut()[cell] = prim.density;
    primitives.pressure.values_mut()[cell] = prim.pressure;
    primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
    primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
    primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
    Ok(())
}

fn restore_cell_primitive(primitives: &mut PrimitiveFields, cell: usize, prim: PrimitiveState) {
    primitives.density.values_mut()[cell] = prim.density;
    primitives.pressure.values_mut()[cell] = prim.pressure;
    primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
    primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
    primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
}
