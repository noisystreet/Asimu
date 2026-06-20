//! 非结构 3D GMRES 单元块对角预条件器（局部无粘 Jacobian，一阶面通量）。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::block_lusgs_preconditioner_topology::{
    BlockLusgsOffDiagonalSlot, BlockLusgsPreconditionerTopology,
};
use crate::discretization::compressible::residual::{
    InviscidAssemblyUnstructuredParams, is_degenerate_volume,
};
use crate::discretization::unstructured_face_cache::LsqRhsCellIncidence;
use crate::discretization::{
    BoundaryGhostBuffer, InviscidFlux, InviscidFluxConfig, ReconstructionKind,
    UnstructuredFaceTopology, face_inviscid_flux_first_order_boundary_soa,
    face_inviscid_flux_first_order_interior_soa, first_order_face_flux_jacobian_supported,
    first_order_interior_flux_jacobian,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, PrimitiveFields};
use crate::linalg::{CellBlockDiagonalPreconditioner, Preconditioner, ensure_vector_len};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use crate::field::{max_physical_increment_scale, state_after_increment};
use crate::solver::compressible::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, component_basis_increment, conserved_component_scales,
};

use super::gmres_block_preconditioner_unstructured_math::{
    block_slice, block_vector_product, cell_vector, subtract_block_product,
    write_cell_vector_from_block_product,
};
use super::gmres_block_preconditioner_unstructured_state::{
    restore_cell_primitive, write_cell_primitive,
};
use super::gmres_block_preconditioner_unstructured_viscous::{
    ViscousCellDiffusivity, add_component_sigma, add_viscous_off_diagonal,
    local_viscous_diffusivity, viscous_component_sigma, viscous_coupling_from_scale,
};

#[cfg(test)]
#[path = "gmres_block_preconditioner_unstructured_tests.rs"]
mod tests;

struct UnstructuredCellBlockContext<'a> {
    mesh: &'a UnstructuredMesh3d,
    eos: &'a IdealGasEoS,
    patches: &'a BoundarySet,
    topology: &'a UnstructuredFaceTopology,
    incidence: &'a LsqRhsCellIncidence,
    volumes: &'a [Real],
    ghosts: &'a BoundaryGhostBuffer,
    exec: &'a crate::exec::ExecutionContext,
    inviscid: &'a InviscidFluxConfig,
    viscous_diffusivity: Option<&'a [ViscousCellDiffusivity]>,
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
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub incidence: &'a LsqRhsCellIncidence,
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
        viscous,
        incidence,
        dt,
        p_floor,
        epsilon_rel,
        ..
    } = params;
    if inviscid.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(
            "非结构 cell_block_diagonal 预条件暂要求 reconstruction = first_order".to_string(),
        ));
    }
    let n = fields.num_cells();
    let volumes = mesh.cell_volumes();
    let viscous_diffusivity = local_viscous_diffusivity(primitives, eos, viscous)?;
    let mut blocks = vec![0.0; n * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    let ctx = UnstructuredCellBlockContext {
        mesh,
        eos,
        patches,
        topology,
        incidence,
        volumes: &volumes,
        ghosts,
        exec,
        inviscid,
        viscous_diffusivity: viscous_diffusivity.as_deref(),
        fields,
        p_floor,
        epsilon_rel,
    };
    for (cell, &dt_cell) in dt.iter().enumerate().take(n) {
        face_blocks::fill_cell_block(&mut blocks, &ctx, primitives, cell, dt_cell)?;
    }
    CellBlockDiagonalPreconditioner::from_blocks(CONSERVED_COMPONENTS_3D, blocks)
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct UnstructuredBlockLusgsPreconditioner {
    diagonal_blocks: Vec<Real>,
    inverse_diagonal_blocks: Vec<Real>,
    row_offsets: Vec<usize>,
    entries: Vec<BlockLusgsEntry>,
    off_diagonal_slots: Vec<BlockLusgsOffDiagonalSlot>,
    solver_order: Vec<usize>,
    solver_rank: Vec<usize>,
    forward: Vec<Real>,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct BlockLusgsEntry {
    col: usize,
    block: [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
}

impl UnstructuredBlockLusgsPreconditioner {
    pub(super) fn allocate(
        topology: &BlockLusgsPreconditionerTopology,
        solver_order: &[usize],
        solver_rank: &[usize],
    ) -> Result<Self> {
        let num_cells = topology.num_cells();
        crate::mesh_order::validate_cell_order(solver_order, num_cells)?;
        if solver_rank.len() != num_cells {
            return Err(AsimuError::Linalg(
                "block_lusgs solver_rank 长度与单元数不一致".to_string(),
            ));
        }
        let block_entries = CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
        let mut entries = vec![BlockLusgsEntry::default(); topology.num_off_diagonal_blocks()];
        for (entry, slot) in entries.iter_mut().zip(&topology.off_diagonal) {
            entry.col = slot.col;
        }
        Ok(Self {
            diagonal_blocks: vec![0.0; num_cells * block_entries],
            inverse_diagonal_blocks: vec![0.0; num_cells * block_entries],
            row_offsets: topology.row_offsets.clone(),
            entries,
            off_diagonal_slots: topology.off_diagonal.clone(),
            solver_order: solver_order.to_vec(),
            solver_rank: solver_rank.to_vec(),
            forward: vec![0.0; num_cells * CONSERVED_COMPONENTS_3D],
        })
    }

    pub(super) fn refresh(
        &mut self,
        params: UnstructuredCellBlockPreconditionerBuild<'_>,
    ) -> Result<()> {
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
            viscous,
            incidence,
            dt,
            p_floor,
            epsilon_rel,
            ..
        } = params;
        if inviscid.reconstruction != ReconstructionKind::FirstOrder {
            return Err(AsimuError::Config(
                "非结构 block_lusgs 预条件暂要求 reconstruction = first_order".to_string(),
            ));
        }
        let n = fields.num_cells();
        if self.num_cells() != n {
            return Err(AsimuError::Linalg(
                "block_lusgs 预条件器缓存尺寸与场不一致".to_string(),
            ));
        }
        let volumes = mesh.cell_volumes();
        let viscous_diffusivity = local_viscous_diffusivity(primitives, eos, viscous)?;
        let ctx = UnstructuredCellBlockContext {
            mesh,
            eos,
            patches,
            topology,
            incidence,
            volumes: &volumes,
            ghosts,
            exec,
            inviscid,
            viscous_diffusivity: viscous_diffusivity.as_deref(),
            fields,
            p_floor,
            epsilon_rel,
        };
        for (cell, &dt_cell) in dt.iter().enumerate().take(n) {
            face_blocks::fill_cell_block(
                &mut self.diagonal_blocks,
                &ctx,
                primitives,
                cell,
                dt_cell,
            )?;
        }
        for (entry, slot) in self.entries.iter_mut().zip(&self.off_diagonal_slots) {
            entry.block = face_blocks::off_diagonal_block(&ctx, primitives, slot)?;
        }
        self.refresh_inverse_diagonal_blocks()?;
        Ok(())
    }

    fn refresh_inverse_diagonal_blocks(&mut self) -> Result<()> {
        let block_entries = CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
        for cell in 0..self.num_cells() {
            let offset = cell * block_entries;
            let inverse = face_blocks::invert_fixed_block(
                &self.diagonal_blocks[offset..offset + block_entries],
            )?;
            self.inverse_diagonal_blocks[offset..offset + block_entries].copy_from_slice(&inverse);
        }
        Ok(())
    }

    fn num_cells(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }
}

impl Preconditioner for UnstructuredBlockLusgsPreconditioner {
    fn dimension(&self) -> usize {
        self.num_cells() * CONSERVED_COMPONENTS_3D
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "block_lusgs rhs")?;
        ensure_vector_len(out, self.dimension(), "block_lusgs out")?;
        for &cell in &self.solver_order {
            let mut local = cell_vector(rhs, cell);
            for entry in face_blocks::row_entries(&self.entries, &self.row_offsets, cell) {
                if self.solver_rank[entry.col] < self.solver_rank[cell] {
                    subtract_block_product(
                        &mut local,
                        &entry.block,
                        &cell_vector(&self.forward, entry.col),
                    );
                }
            }
            write_cell_vector_from_block_product(
                &mut self.forward,
                cell,
                block_slice(&self.inverse_diagonal_blocks, cell),
                &local,
            );
        }
        for &cell in self.solver_order.iter().rev() {
            let mut local = block_vector_product(
                block_slice(&self.diagonal_blocks, cell),
                &cell_vector(&self.forward, cell),
            );
            for entry in face_blocks::row_entries(&self.entries, &self.row_offsets, cell) {
                if self.solver_rank[entry.col] > self.solver_rank[cell] {
                    subtract_block_product(&mut local, &entry.block, &cell_vector(out, entry.col));
                }
            }
            write_cell_vector_from_block_product(
                out,
                cell,
                block_slice(&self.inverse_diagonal_blocks, cell),
                &local,
            );
        }
        Ok(())
    }
}

#[path = "gmres_block_preconditioner_unstructured_face_blocks.rs"]
mod face_blocks;

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

fn local_boundary_inviscid_residual(
    cell: usize,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    incidence: &LsqRhsCellIncidence,
) -> Result<[Real; CONSERVED_COMPONENTS_3D]> {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    for &bidx in &incidence.boundary_faces[cell] {
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

fn local_inviscid_residual(
    cell: usize,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    incidence: &LsqRhsCellIncidence,
    volumes: &[Real],
) -> Result<[Real; CONSERVED_COMPONENTS_3D]> {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    if is_degenerate_volume(volumes[cell]) {
        return Ok(out);
    }
    for &face_idx in &incidence.interior_as_owner[cell] {
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
        add_flux_vector(&mut out, &flux, face.owner_rhs_scale);
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
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
        add_flux_vector(&mut out, &flux, face.neighbor_rhs_scale);
    }
    for &bidx in &incidence.boundary_faces[cell] {
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
    viscous_diagonal_sigma: [Real; CONSERVED_COMPONENTS_3D],
) {
    let block_offset = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
    for row in 0..CONSERVED_COMPONENTS_3D {
        let diagonal = if row == col {
            1.0 / dt_cell + viscous_diagonal_sigma[row]
        } else {
            0.0
        };
        blocks[block_offset + row * CONSERVED_COMPONENTS_3D + col] = diagonal - jv[row];
    }
}
