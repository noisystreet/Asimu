//! 非结构 3D GMRES 单元块对角预条件器（局部无粘 Jacobian，一阶面通量）。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::compressible::residual::{
    InviscidAssemblyUnstructuredParams, is_degenerate_volume,
};
use crate::discretization::{
    BoundaryGhostBuffer, InviscidFlux, InviscidFluxConfig, ReconstructionKind,
    UnstructuredFaceTopology, face_inviscid_flux_first_order_boundary_soa,
    face_inviscid_flux_first_order_interior_soa,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, PrimitiveFields, primitive_from_conserved_relaxed};
use crate::linalg::CellBlockDiagonalPreconditioner;
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState};

use crate::field::{max_physical_increment_scale, state_after_increment};
use crate::solver::compressible::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, component_basis_increment, conserved_component_scales,
};

struct CellFaceIncidence {
    interior: Vec<Vec<usize>>,
    boundary: Vec<Vec<usize>>,
}

struct UnstructuredCellBlockContext<'a> {
    mesh: &'a UnstructuredMesh3d,
    eos: &'a IdealGasEoS,
    patches: &'a BoundarySet,
    topology: &'a UnstructuredFaceTopology,
    incidence: &'a CellFaceIncidence,
    volumes: &'a [Real],
    ghosts: &'a BoundaryGhostBuffer,
    exec: &'a crate::exec::ExecutionContext,
    inviscid: &'a InviscidFluxConfig,
    fields: &'a ConservedFields,
    p_floor: Real,
    epsilon_rel: Real,
}

pub(super) struct UnstructuredCellBlockPreconditionerBuild<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub patches: &'a BoundarySet,
    pub topology: &'a UnstructuredFaceTopology,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub exec: &'a crate::exec::ExecutionContext,
    pub fields: &'a ConservedFields,
    pub primitives: &'a mut PrimitiveFields,
    pub inviscid: &'a InviscidFluxConfig,
    pub dt: &'a [Real],
    pub p_floor: Real,
    pub epsilon_rel: Real,
}

pub(super) fn build_cell_block_preconditioner_unstructured(
    params: UnstructuredCellBlockPreconditionerBuild<'_>,
) -> Result<CellBlockDiagonalPreconditioner> {
    let UnstructuredCellBlockPreconditionerBuild {
        mesh,
        eos,
        patches,
        topology,
        ghosts,
        exec,
        fields,
        primitives,
        inviscid,
        dt,
        p_floor,
        epsilon_rel,
    } = params;
    if inviscid.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(
            "非结构 cell_block_diagonal 预条件暂要求 reconstruction = first_order".to_string(),
        ));
    }
    let n = fields.num_cells();
    let incidence = build_cell_face_incidence(topology, n);
    let volumes = mesh.cell_volumes();
    let mut blocks = vec![0.0; n * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    let ctx = UnstructuredCellBlockContext {
        mesh,
        eos,
        patches,
        topology,
        incidence: &incidence,
        volumes: &volumes,
        ghosts,
        exec,
        inviscid,
        fields,
        p_floor,
        epsilon_rel,
    };
    for (cell, &dt_cell) in dt.iter().enumerate().take(n) {
        fill_cell_block(&mut blocks, &ctx, primitives, cell, dt_cell)?;
    }
    CellBlockDiagonalPreconditioner::from_blocks(CONSERVED_COMPONENTS_3D, blocks)
}

fn build_cell_face_incidence(
    topology: &UnstructuredFaceTopology,
    n_cells: usize,
) -> CellFaceIncidence {
    let mut interior = vec![Vec::new(); n_cells];
    let mut boundary = vec![Vec::new(); n_cells];
    for (idx, face) in topology.interior.iter().enumerate() {
        interior[face.owner].push(idx);
        interior[face.neighbor].push(idx);
    }
    for (idx, bface) in topology.boundary.iter().enumerate() {
        boundary[bface.owner].push(idx);
    }
    CellFaceIncidence { interior, boundary }
}

fn fill_cell_block(
    blocks: &mut [Real],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    cell: usize,
    dt_cell: Real,
) -> Result<()> {
    let base_state = ctx.fields.cell_state(cell)?;
    let base_primitive = primitives.cell_primitive(cell);
    let base_local = local_inviscid_residual(
        cell,
        &inviscid_params(ctx, primitives),
        ctx.topology,
        ctx.incidence,
        ctx.volumes,
    )?;
    let scales = conserved_component_scales(&base_state);
    for (col, scale) in scales
        .iter()
        .copied()
        .enumerate()
        .take(CONSERVED_COMPONENTS_3D)
    {
        let requested_eps = ctx.epsilon_rel.sqrt() * scale;
        let unit = component_basis_increment(col);
        let eps = max_physical_increment_scale(
            &base_state,
            unit,
            requested_eps,
            ctx.eos.gamma,
            ctx.p_floor,
        );
        if eps <= 0.0 {
            return Err(AsimuError::Solver(format!(
                "GMRES 局部块预条件器：cell {cell} 分量 {col} 无法构造正性扰动"
            )));
        }
        let perturbed_state = state_after_increment(&base_state, unit, eps);
        write_cell_primitive(primitives, cell, &perturbed_state, ctx.eos, ctx.p_floor)?;
        let perturbed_local = local_inviscid_residual(
            cell,
            &inviscid_params(ctx, primitives),
            ctx.topology,
            ctx.incidence,
            ctx.volumes,
        )?;
        let jv = local_residual_difference(perturbed_local, base_local, eps);
        write_block_column(blocks, cell, col, dt_cell, jv);
    }
    restore_cell_primitive(primitives, cell, base_primitive);
    Ok(())
}

fn inviscid_params<'a>(
    ctx: &'a UnstructuredCellBlockContext<'_>,
    primitives: &'a PrimitiveFields,
) -> InviscidAssemblyUnstructuredParams<'a> {
    InviscidAssemblyUnstructuredParams {
        mesh: ctx.mesh,
        eos: ctx.eos,
        config: ctx.inviscid,
        boundaries: ctx.patches,
        ghosts: ctx.ghosts,
        primitives,
        face_topology: Some(ctx.topology),
        mesh_cache: None,
        gradients: None,
        min_pressure: ctx.p_floor,
        exec: ctx.exec,
    }
}

fn local_inviscid_residual(
    cell: usize,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    incidence: &CellFaceIncidence,
    volumes: &[Real],
) -> Result<[Real; CONSERVED_COMPONENTS_3D]> {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    if is_degenerate_volume(volumes[cell]) {
        return Ok(out);
    }
    for &face_idx in &incidence.interior[cell] {
        let face = &topology.interior[face_idx];
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        let flux = face_inviscid_flux_first_order_interior_soa(
            face.owner,
            face.neighbor,
            params.primitives,
            face.normal,
            params.eos,
            params.config,
        )?;
        if face.owner == cell {
            add_flux_vector(&mut out, &flux, face.owner_rhs_scale);
        } else if face.neighbor == cell {
            add_flux_vector(&mut out, &flux, face.neighbor_rhs_scale);
        }
    }
    for &bidx in &incidence.boundary[cell] {
        let bface = &topology.boundary[bidx];
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
        add_flux_vector(&mut out, &flux, bface.owner_rhs_scale);
    }
    Ok(out)
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
