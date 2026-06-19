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
use crate::linalg::{CellBlockDiagonalPreconditioner, Preconditioner, ensure_vector_len};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{ConservedState, IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

use crate::field::{max_physical_increment_scale, state_after_increment};
use crate::solver::compressible::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, component_basis_increment, conserved_component_scales,
};
use crate::solver::compressible::spectral_radius::cell_viscous_diffusivity_max;

const PARABOLIC_SPECTRAL_FACTOR_3D: Real = 6.0;

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
    viscous_diffusivity: Option<&'a [Real]>,
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
    let viscous_diffusivity = local_viscous_diffusivity(primitives, eos, viscous)?;
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
        viscous_diffusivity: viscous_diffusivity.as_deref(),
        fields,
        p_floor,
        epsilon_rel,
    };
    for (cell, &dt_cell) in dt.iter().enumerate().take(n) {
        fill_cell_block(&mut blocks, &ctx, primitives, cell, dt_cell)?;
    }
    CellBlockDiagonalPreconditioner::from_blocks(CONSERVED_COMPONENTS_3D, blocks)
}

pub(super) fn build_block_lusgs_preconditioner_unstructured(
    params: UnstructuredCellBlockPreconditionerBuild<'_>,
) -> Result<UnstructuredBlockLusgsPreconditioner> {
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
        dt,
        p_floor,
        epsilon_rel,
    } = params;
    if inviscid.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(
            "非结构 block_lusgs 预条件暂要求 reconstruction = first_order".to_string(),
        ));
    }
    let n = fields.num_cells();
    let incidence = build_cell_face_incidence(topology, n);
    let volumes = mesh.cell_volumes();
    let viscous_diffusivity = local_viscous_diffusivity(primitives, eos, viscous)?;
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
        viscous_diffusivity: viscous_diffusivity.as_deref(),
        fields,
        p_floor,
        epsilon_rel,
    };
    let mut diagonal_blocks = vec![0.0; n * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    for (cell, &dt_cell) in dt.iter().enumerate().take(n) {
        fill_cell_block(&mut diagonal_blocks, &ctx, primitives, cell, dt_cell)?;
    }
    let off_diagonal = fill_off_diagonal_blocks(&ctx, primitives)?;
    UnstructuredBlockLusgsPreconditioner::from_blocks(diagonal_blocks, off_diagonal)
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct UnstructuredBlockLusgsPreconditioner {
    diagonal_blocks: Vec<Real>,
    inverse_diagonal_blocks: Vec<Real>,
    row_blocks: Vec<Vec<BlockLusgsEntry>>,
    forward: Vec<Real>,
}

#[derive(Debug, Clone, PartialEq)]
struct BlockLusgsEntry {
    col: usize,
    block: [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
}

impl UnstructuredBlockLusgsPreconditioner {
    fn from_blocks(
        diagonal_blocks: Vec<Real>,
        row_blocks: Vec<Vec<BlockLusgsEntry>>,
    ) -> Result<Self> {
        let block_entries = CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
        if diagonal_blocks.is_empty() || diagonal_blocks.len() % block_entries != 0 {
            return Err(AsimuError::Linalg(
                "block_lusgs 对角块尺寸不一致".to_string(),
            ));
        }
        let num_cells = diagonal_blocks.len() / block_entries;
        if row_blocks.len() != num_cells {
            return Err(AsimuError::Linalg(
                "block_lusgs 行块数量与对角块数量不一致".to_string(),
            ));
        }
        let mut inverse_diagonal_blocks = vec![0.0; diagonal_blocks.len()];
        for cell in 0..num_cells {
            let offset = cell * block_entries;
            let inverse = invert_fixed_block(&diagonal_blocks[offset..offset + block_entries])?;
            inverse_diagonal_blocks[offset..offset + block_entries].copy_from_slice(&inverse);
        }
        Ok(Self {
            diagonal_blocks,
            inverse_diagonal_blocks,
            row_blocks,
            forward: vec![0.0; num_cells * CONSERVED_COMPONENTS_3D],
        })
    }

    fn num_cells(&self) -> usize {
        self.row_blocks.len()
    }

    fn diagonal_block(&self, cell: usize) -> &[Real] {
        let start = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
        &self.diagonal_blocks[start..start + CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]
    }

    fn inverse_diagonal_block(&self, cell: usize) -> &[Real] {
        let start = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
        &self.inverse_diagonal_blocks
            [start..start + CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]
    }
}

impl Preconditioner for UnstructuredBlockLusgsPreconditioner {
    fn dimension(&self) -> usize {
        self.num_cells() * CONSERVED_COMPONENTS_3D
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        ensure_vector_len(rhs, self.dimension(), "block_lusgs rhs")?;
        ensure_vector_len(out, self.dimension(), "block_lusgs out")?;
        for cell in 0..self.num_cells() {
            let mut local = cell_vector(rhs, cell);
            for entry in self.row_blocks[cell]
                .iter()
                .filter(|entry| entry.col < cell)
            {
                subtract_block_product(
                    &mut local,
                    &entry.block,
                    &cell_vector(&self.forward, entry.col),
                );
            }
            let inverse_diagonal = fixed_block(self.inverse_diagonal_block(cell));
            write_cell_vector_from_block_product(
                &mut self.forward,
                cell,
                &inverse_diagonal,
                &local,
            );
        }
        for cell in (0..self.num_cells()).rev() {
            let mut local =
                block_vector_product(self.diagonal_block(cell), &cell_vector(&self.forward, cell));
            for entry in self.row_blocks[cell]
                .iter()
                .filter(|entry| entry.col > cell)
            {
                subtract_block_product(&mut local, &entry.block, &cell_vector(out, entry.col));
            }
            write_cell_vector_from_block_product(
                out,
                cell,
                self.inverse_diagonal_block(cell),
                &local,
            );
        }
        Ok(())
    }
}

fn fill_off_diagonal_blocks(
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
) -> Result<Vec<Vec<BlockLusgsEntry>>> {
    let n = ctx.fields.num_cells();
    let mut rows = vec![Vec::new(); n];
    for face in &ctx.topology.interior {
        let owner_viscous =
            interior_viscous_coupling(ctx, face.owner, face.area, face.owner_volume);
        let owner_block =
            off_diagonal_block(ctx, primitives, face.owner, face.neighbor, owner_viscous)?;
        rows[face.owner].push(BlockLusgsEntry {
            col: face.neighbor,
            block: owner_block,
        });
        let neighbor_viscous =
            interior_viscous_coupling(ctx, face.neighbor, face.area, face.neighbor_volume);
        let neighbor_block =
            off_diagonal_block(ctx, primitives, face.neighbor, face.owner, neighbor_viscous)?;
        rows[face.neighbor].push(BlockLusgsEntry {
            col: face.owner,
            block: neighbor_block,
        });
    }
    Ok(rows)
}

fn off_diagonal_block(
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    row_cell: usize,
    source_cell: usize,
    viscous_coupling: Real,
) -> Result<[Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]> {
    let base_state = ctx.fields.cell_state(source_cell)?;
    let base_primitive = primitives.cell_primitive(source_cell);
    let base_local = local_inviscid_residual(
        row_cell,
        &inviscid_params(ctx, primitives),
        ctx.topology,
        ctx.incidence,
        ctx.volumes,
    )?;
    let scales = conserved_component_scales(&base_state);
    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
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
                "GMRES block_lusgs 预条件器：cell {source_cell} 分量 {col} 无法构造正性扰动"
            )));
        }
        let perturbed_state = state_after_increment(&base_state, unit, eps);
        write_cell_primitive(
            primitives,
            source_cell,
            &perturbed_state,
            ctx.eos,
            ctx.p_floor,
        )?;
        let perturbed_local = local_inviscid_residual(
            row_cell,
            &inviscid_params(ctx, primitives),
            ctx.topology,
            ctx.incidence,
            ctx.volumes,
        )?;
        let jv = local_residual_difference(perturbed_local, base_local, eps);
        for row in 0..CONSERVED_COMPONENTS_3D {
            block[row * CONSERVED_COMPONENTS_3D + col] = -jv[row];
        }
    }
    add_viscous_off_diagonal(&mut block, viscous_coupling);
    restore_cell_primitive(primitives, source_cell, base_primitive);
    Ok(block)
}

fn local_viscous_diffusivity(
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    viscous: Option<&ViscousPhysicsConfig>,
) -> Result<Option<Vec<Real>>> {
    viscous
        .map(|config| cell_viscous_diffusivity_max(primitives, eos, config))
        .transpose()
}

fn interior_viscous_coupling(
    ctx: &UnstructuredCellBlockContext<'_>,
    cell: usize,
    area: Real,
    volume: Real,
) -> Real {
    let Some(diffusivity) = ctx.viscous_diffusivity else {
        return 0.0;
    };
    viscous_face_sigma(diffusivity[cell], area, volume)
}

fn cell_viscous_diagonal_sigma(ctx: &UnstructuredCellBlockContext<'_>, cell: usize) -> Real {
    if ctx.viscous_diffusivity.is_none() {
        return 0.0;
    }
    let mut sigma = 0.0;
    for &face_idx in &ctx.incidence.interior[cell] {
        let face = &ctx.topology.interior[face_idx];
        let volume = if face.owner == cell {
            face.owner_volume
        } else {
            face.neighbor_volume
        };
        sigma += interior_viscous_coupling(ctx, cell, face.area, volume);
    }
    for &face_idx in &ctx.incidence.boundary[cell] {
        let face = &ctx.topology.boundary[face_idx];
        sigma += interior_viscous_coupling(ctx, cell, face.area, face.owner_volume);
    }
    sigma
}

fn viscous_face_sigma(diffusivity: Real, area: Real, volume: Real) -> Real {
    if diffusivity > 0.0 && area > Real::EPSILON && volume > 1.0e-30 {
        PARABOLIC_SPECTRAL_FACTOR_3D * diffusivity * area * area / (volume * volume)
    } else {
        0.0
    }
}

fn add_viscous_off_diagonal(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    coupling: Real,
) {
    if coupling <= 0.0 {
        return;
    }
    for component in 0..CONSERVED_COMPONENTS_3D {
        block[component * CONSERVED_COMPONENTS_3D + component] -= coupling;
    }
}

fn cell_vector(values: &[Real], cell: usize) -> [Real; CONSERVED_COMPONENTS_3D] {
    let start = cell * CONSERVED_COMPONENTS_3D;
    [
        values[start],
        values[start + 1],
        values[start + 2],
        values[start + 3],
        values[start + 4],
    ]
}

fn fixed_block(block: &[Real]) -> [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D] {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    out.copy_from_slice(block);
    out
}

fn subtract_block_product(
    out: &mut [Real; CONSERVED_COMPONENTS_3D],
    block: &[Real],
    vector: &[Real; CONSERVED_COMPONENTS_3D],
) {
    for row in 0..CONSERVED_COMPONENTS_3D {
        let mut value = 0.0;
        for col in 0..CONSERVED_COMPONENTS_3D {
            value += block[row * CONSERVED_COMPONENTS_3D + col] * vector[col];
        }
        out[row] -= value;
    }
}

fn block_vector_product(
    block: &[Real],
    vector: &[Real; CONSERVED_COMPONENTS_3D],
) -> [Real; CONSERVED_COMPONENTS_3D] {
    let mut out = [0.0; CONSERVED_COMPONENTS_3D];
    for row in 0..CONSERVED_COMPONENTS_3D {
        for col in 0..CONSERVED_COMPONENTS_3D {
            out[row] += block[row * CONSERVED_COMPONENTS_3D + col] * vector[col];
        }
    }
    out
}

fn write_cell_vector_from_block_product(
    out: &mut [Real],
    cell: usize,
    block: &[Real],
    vector: &[Real; CONSERVED_COMPONENTS_3D],
) {
    let start = cell * CONSERVED_COMPONENTS_3D;
    let product = block_vector_product(block, vector);
    out[start..start + CONSERVED_COMPONENTS_3D].copy_from_slice(&product);
}

fn invert_fixed_block(
    block: &[Real],
) -> Result<[Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]> {
    let n = CONSERVED_COMPONENTS_3D;
    let width = n * 2;
    let mut aug = initialized_inverse_augmented_block(block, n, width);
    for pivot in 0..n {
        let (pivot_row, pivot_abs) = find_fixed_pivot_row(&aug, n, width, pivot);
        if pivot_abs <= Real::EPSILON {
            return Err(AsimuError::Linalg(
                "block_lusgs 预条件器遇到奇异对角块".to_string(),
            ));
        }
        swap_fixed_augmented_rows(&mut aug, width, pivot, pivot_row);
        normalize_fixed_pivot_row(&mut aug, width, pivot);
        eliminate_fixed_pivot_column(&mut aug, n, width, pivot);
    }
    Ok(extract_fixed_inverse(&aug, n, width))
}

fn initialized_inverse_augmented_block(
    block: &[Real],
    n: usize,
    width: usize,
) -> [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D * 2] {
    let mut aug = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D * 2];
    for row in 0..n {
        for col in 0..n {
            aug[row * width + col] = block[row * n + col];
        }
        aug[row * width + n + row] = 1.0;
    }
    aug
}

fn find_fixed_pivot_row(aug: &[Real], n: usize, width: usize, pivot: usize) -> (usize, Real) {
    let mut pivot_row = pivot;
    let mut pivot_abs = aug[pivot * width + pivot].abs();
    for row in pivot + 1..n {
        let candidate = aug[row * width + pivot].abs();
        if candidate > pivot_abs {
            pivot_abs = candidate;
            pivot_row = row;
        }
    }
    (pivot_row, pivot_abs)
}

fn swap_fixed_augmented_rows(aug: &mut [Real], width: usize, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    for col in 0..width {
        aug.swap(lhs * width + col, rhs * width + col);
    }
}

fn normalize_fixed_pivot_row(aug: &mut [Real], width: usize, pivot: usize) {
    let pivot_value = aug[pivot * width + pivot];
    for col in 0..width {
        aug[pivot * width + col] /= pivot_value;
    }
}

fn eliminate_fixed_pivot_column(aug: &mut [Real], n: usize, width: usize, pivot: usize) {
    for row in 0..n {
        if row == pivot {
            continue;
        }
        let factor = aug[row * width + pivot];
        if factor == 0.0 {
            continue;
        }
        for col in 0..width {
            aug[row * width + col] -= factor * aug[pivot * width + col];
        }
    }
}

fn extract_fixed_inverse(
    aug: &[Real],
    n: usize,
    width: usize,
) -> [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D] {
    let mut inverse = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    for row in 0..n {
        for col in 0..n {
            inverse[row * n + col] = aug[row * width + n + col];
        }
    }
    inverse
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
        write_block_column(
            blocks,
            cell,
            col,
            dt_cell,
            jv,
            cell_viscous_diagonal_sigma(ctx, cell),
        );
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
    viscous_diagonal_sigma: Real,
) {
    let block_offset = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D;
    for row in 0..CONSERVED_COMPONENTS_3D {
        let diagonal = if row == col {
            1.0 / dt_cell + viscous_diagonal_sigma
        } else {
            0.0
        };
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
