//! block_lusgs 面块 Jacobian 装配（解析 + 有限差分）。

use tracing::warn;

use super::*;

struct FiniteDifferencePerturbation {
    increment: [Real; CONSERVED_COMPONENTS_3D],
    epsilon: Real,
}

/// 守恒量有限差分扰动：先试 \(+\mathbf{e}_j\)，再试 \(-\mathbf{e}_j\)；均不可行则返回 `None`。
fn finite_difference_perturbation(
    base_state: &crate::physics::ConservedState,
    col: usize,
    epsilon_rel: Real,
    gamma: Real,
    p_floor: Real,
) -> Option<FiniteDifferencePerturbation> {
    let scales = conserved_component_scales(base_state);
    let requested_eps = epsilon_rel.sqrt() * scales[col];
    if requested_eps <= 0.0 || !requested_eps.is_finite() {
        return None;
    }
    for sign in [1.0, -1.0] {
        let mut increment = component_basis_increment(col);
        if sign < 0.0 {
            for entry in &mut increment {
                *entry *= -1.0;
            }
        }
        let eps =
            max_physical_increment_scale(base_state, increment, requested_eps, gamma, p_floor);
        if eps > 0.0 {
            return Some(FiniteDifferencePerturbation {
                increment,
                epsilon: eps,
            });
        }
    }
    None
}

fn finite_difference_jacobian_column(
    base: [Real; CONSERVED_COMPONENTS_3D],
    perturbed: [Real; CONSERVED_COMPONENTS_3D],
    perturbation: &FiniteDifferencePerturbation,
    col: usize,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    if perturbation.increment[col] >= 0.0 {
        local_residual_difference(perturbed, base, perturbation.epsilon)
    } else {
        local_residual_difference(base, perturbed, perturbation.epsilon)
    }
}

fn warn_zero_finite_difference_column(cell: usize, col: usize, context: &str) {
    warn!(
        cell,
        component = col,
        "{context}：cell {cell} 分量 {col} 无法构造正性扰动，Jacobian 列置零"
    );
}

pub(super) fn off_diagonal_block(
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    slot: &BlockLusgsOffDiagonalSlot,
) -> Result<[Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]> {
    if first_order_face_flux_jacobian_supported(ctx.inviscid) {
        off_diagonal_block_analytic(ctx, primitives, slot)
    } else {
        off_diagonal_block_finite_difference(ctx, primitives, slot)
    }
}

fn off_diagonal_block_analytic(
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &PrimitiveFields,
    slot: &BlockLusgsOffDiagonalSlot,
) -> Result<[Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]> {
    let face = &ctx.topology.interior[slot.face_idx];
    let left = ctx.fields.cell_state(face.owner)?;
    let right = ctx.fields.cell_state(face.neighbor)?;
    let prim_l = primitives.cell_primitive(face.owner);
    let prim_r = primitives.cell_primitive(face.neighbor);
    let (d_fl, d_fr) = first_order_interior_flux_jacobian(
        &left,
        &right,
        &prim_l,
        &prim_r,
        face.normal,
        ctx.eos,
        ctx.inviscid,
    )?;
    let (scale, d_source) = if slot.row == face.owner {
        (face.owner_rhs_scale, d_fr)
    } else {
        (face.neighbor_rhs_scale, d_fl)
    };
    debug_assert_eq!(
        slot.col,
        if slot.row == face.owner {
            face.neighbor
        } else {
            face.owner
        }
    );
    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    for row in 0..CONSERVED_COMPONENTS_3D {
        for col in 0..CONSERVED_COMPONENTS_3D {
            block[row * CONSERVED_COMPONENTS_3D + col] = -scale * d_source.data[row][col];
        }
    }
    add_viscous_off_diagonal_from_slot(&mut block, ctx, slot, face.normal);
    Ok(block)
}

fn off_diagonal_block_finite_difference(
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    slot: &BlockLusgsOffDiagonalSlot,
) -> Result<[Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D]> {
    let row_cell = slot.row;
    let source_cell = slot.col;
    let base_state = ctx.fields.cell_state(source_cell)?;
    let base_primitive = primitives.cell_primitive(source_cell);
    let base_local = local_inviscid_residual(
        row_cell,
        &inviscid_params(ctx, primitives),
        ctx.topology,
        ctx.incidence,
        ctx.volumes,
    )?;
    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    for col in 0..CONSERVED_COMPONENTS_3D {
        let Some(perturbation) = finite_difference_perturbation(
            &base_state,
            col,
            ctx.epsilon_rel,
            ctx.eos.gamma,
            ctx.p_floor,
        ) else {
            warn_zero_finite_difference_column(
                source_cell,
                col,
                "block_lusgs off-diagonal 有限差分",
            );
            continue;
        };
        let perturbed_state =
            state_after_increment(&base_state, perturbation.increment, perturbation.epsilon);
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
        let jv = finite_difference_jacobian_column(base_local, perturbed_local, &perturbation, col);
        for row in 0..CONSERVED_COMPONENTS_3D {
            block[row * CONSERVED_COMPONENTS_3D + col] = -jv[row];
        }
    }
    let face = &ctx.topology.interior[slot.face_idx];
    add_viscous_off_diagonal_from_slot(&mut block, ctx, slot, face.normal);
    restore_cell_primitive(primitives, source_cell, base_primitive);
    Ok(block)
}

fn add_viscous_off_diagonal_from_slot(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    ctx: &UnstructuredCellBlockContext<'_>,
    slot: &BlockLusgsOffDiagonalSlot,
    normal: crate::core::Vector3,
) {
    let Some(diffusivity) = ctx.viscous_diffusivity else {
        return;
    };
    add_viscous_off_diagonal(
        block,
        diffusivity[slot.row],
        slot.viscous_parabolic_scale,
        normal,
    );
}

fn interior_viscous_coupling(
    ctx: &UnstructuredCellBlockContext<'_>,
    cell: usize,
    area: Real,
    volume: Real,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    let Some(diffusivity) = ctx.viscous_diffusivity else {
        return [0.0; CONSERVED_COMPONENTS_3D];
    };
    viscous_component_sigma(diffusivity[cell], area, volume)
}

fn cell_viscous_diagonal_sigma(
    ctx: &UnstructuredCellBlockContext<'_>,
    cell: usize,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    if ctx.viscous_diffusivity.is_none() {
        return [0.0; CONSERVED_COMPONENTS_3D];
    }
    let mut sigma = [0.0; CONSERVED_COMPONENTS_3D];
    for &face_idx in &ctx.incidence.interior_as_owner[cell] {
        let face = &ctx.topology.interior[face_idx];
        add_component_sigma(
            &mut sigma,
            interior_viscous_coupling(ctx, cell, face.area, face.owner_volume),
        );
    }
    for &face_idx in &ctx.incidence.interior_as_neighbor[cell] {
        let face = &ctx.topology.interior[face_idx];
        add_component_sigma(
            &mut sigma,
            interior_viscous_coupling(ctx, cell, face.area, face.neighbor_volume),
        );
    }
    for &face_idx in &ctx.incidence.boundary_faces[cell] {
        let face = &ctx.topology.boundary[face_idx];
        add_component_sigma(
            &mut sigma,
            interior_viscous_coupling(ctx, cell, face.area, face.owner_volume),
        );
    }
    sigma
}

pub(super) fn row_entries<'a>(
    entries: &'a [BlockLusgsEntry],
    row_offsets: &[usize],
    cell: usize,
) -> &'a [BlockLusgsEntry] {
    &entries[row_offsets[cell]..row_offsets[cell + 1]]
}

pub(super) fn invert_fixed_block(
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

pub(super) fn fill_cell_block(
    blocks: &mut [Real],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    cell: usize,
    dt_cell: Real,
) -> Result<()> {
    if first_order_face_flux_jacobian_supported(ctx.inviscid) {
        fill_cell_block_analytic(blocks, ctx, primitives, cell, dt_cell)
    } else {
        fill_cell_block_finite_difference(blocks, ctx, primitives, cell, dt_cell)
    }
}

fn fill_cell_block_analytic(
    blocks: &mut [Real],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    cell: usize,
    dt_cell: Real,
) -> Result<()> {
    let viscous_diagonal_sigma = cell_viscous_diagonal_sigma(ctx, cell);
    let mut block = [0.0; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D];
    for &face_idx in &ctx.incidence.interior_as_owner[cell] {
        let face = &ctx.topology.interior[face_idx];
        let left = ctx.fields.cell_state(face.owner)?;
        let right = ctx.fields.cell_state(face.neighbor)?;
        let prim_l = primitives.cell_primitive(face.owner);
        let prim_r = primitives.cell_primitive(face.neighbor);
        let (d_fl, _d_fr) = first_order_interior_flux_jacobian(
            &left,
            &right,
            &prim_l,
            &prim_r,
            face.normal,
            ctx.eos,
            ctx.inviscid,
        )?;
        subtract_scaled_flux_jacobian(&mut block, face.owner_rhs_scale, &d_fl);
    }
    for &face_idx in &ctx.incidence.interior_as_neighbor[cell] {
        let face = &ctx.topology.interior[face_idx];
        let left = ctx.fields.cell_state(face.owner)?;
        let right = ctx.fields.cell_state(face.neighbor)?;
        let prim_l = primitives.cell_primitive(face.owner);
        let prim_r = primitives.cell_primitive(face.neighbor);
        let (_d_fl, d_fr) = first_order_interior_flux_jacobian(
            &left,
            &right,
            &prim_l,
            &prim_r,
            face.normal,
            ctx.eos,
            ctx.inviscid,
        )?;
        subtract_scaled_flux_jacobian(&mut block, face.neighbor_rhs_scale, &d_fr);
    }
    let base_state = ctx.fields.cell_state(cell)?;
    accumulate_boundary_cell_block(&mut block, ctx, primitives, cell, &base_state)?;
    for col in 0..CONSERVED_COMPONENTS_3D {
        for row in 0..CONSERVED_COMPONENTS_3D {
            let diagonal = if row == col {
                1.0 / dt_cell + viscous_diagonal_sigma[row]
            } else {
                0.0
            };
            let offset = cell * CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D
                + row * CONSERVED_COMPONENTS_3D
                + col;
            blocks[offset] = diagonal + block[row * CONSERVED_COMPONENTS_3D + col];
        }
    }
    Ok(())
}

fn subtract_scaled_flux_jacobian(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    scale: Real,
    jacobian: &crate::discretization::ConservedFluxJacobian,
) {
    for row in 0..CONSERVED_COMPONENTS_3D {
        for col in 0..CONSERVED_COMPONENTS_3D {
            block[row * CONSERVED_COMPONENTS_3D + col] -= scale * jacobian.data[row][col];
        }
    }
}

fn accumulate_boundary_cell_block(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    cell: usize,
    base_state: &crate::physics::ConservedState,
) -> Result<()> {
    if first_order_face_flux_jacobian_supported(ctx.inviscid) {
        accumulate_boundary_cell_block_analytic(block, ctx, primitives, cell)
    } else {
        let base_boundary = local_boundary_inviscid_residual(
            cell,
            &inviscid_params(ctx, primitives),
            ctx.topology,
            ctx.incidence,
        )?;
        accumulate_boundary_cell_block_finite_difference(
            block,
            ctx,
            primitives,
            cell,
            base_state,
            &base_boundary,
        )
    }
}

fn accumulate_boundary_cell_block_analytic(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &PrimitiveFields,
    cell: usize,
) -> Result<()> {
    use crate::discretization::BoundaryFluxJacobianContext;
    use crate::discretization::first_order_boundary_flux_jacobian_wrt_owner;
    use crate::mesh::FaceGeometry3d;

    let owner_state = ctx.fields.cell_state(cell)?;
    let prim_owner = primitives.cell_primitive(cell);
    for &bidx in &ctx.incidence.boundary_faces[cell] {
        let bface = &ctx.topology.boundary[bidx];
        if bface.owner_rhs_scale == 0.0 {
            continue;
        }
        let ghost = ctx.ghosts.get_face(bface.face).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界面 FaceId({}) 缺少 ghost 状态",
                bface.face.index()
            ))
        })?;
        let geom = FaceGeometry3d {
            normal: bface.normal,
            spacing: bface.spacing,
            area: bface.area,
            center: crate::core::Vector3::new(0.0, 0.0, 0.0),
        };
        let jac_ctx = BoundaryFluxJacobianContext {
            normal: bface.normal,
            inviscid_kind: bface.inviscid,
            geom: &geom,
            eos: ctx.eos,
            config: ctx.inviscid,
            p_floor: ctx.p_floor,
            viscous: ctx.viscous,
            epsilon_rel: ctx.epsilon_rel,
        };
        let jac = first_order_boundary_flux_jacobian_wrt_owner(
            &owner_state,
            &ghost.conserved,
            &prim_owner,
            &jac_ctx,
        )?;
        subtract_scaled_flux_jacobian(block, bface.owner_rhs_scale, &jac);
    }
    Ok(())
}

fn accumulate_boundary_cell_block_finite_difference(
    block: &mut [Real; CONSERVED_COMPONENTS_3D * CONSERVED_COMPONENTS_3D],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    cell: usize,
    base_state: &crate::physics::ConservedState,
    base_boundary: &[Real; CONSERVED_COMPONENTS_3D],
) -> Result<()> {
    let base_primitive = primitives.cell_primitive(cell);
    for col in 0..CONSERVED_COMPONENTS_3D {
        let Some(perturbation) = finite_difference_perturbation(
            base_state,
            col,
            ctx.epsilon_rel,
            ctx.eos.gamma,
            ctx.p_floor,
        ) else {
            warn_zero_finite_difference_column(cell, col, "block_lusgs 边界面有限差分");
            continue;
        };
        let perturbed_state =
            state_after_increment(base_state, perturbation.increment, perturbation.epsilon);
        write_cell_primitive(primitives, cell, &perturbed_state, ctx.eos, ctx.p_floor)?;
        let perturbed_boundary = local_boundary_inviscid_residual(
            cell,
            &inviscid_params(ctx, primitives),
            ctx.topology,
            ctx.incidence,
        )?;
        let jv = finite_difference_jacobian_column(
            *base_boundary,
            perturbed_boundary,
            &perturbation,
            col,
        );
        for row in 0..CONSERVED_COMPONENTS_3D {
            block[row * CONSERVED_COMPONENTS_3D + col] -= jv[row];
        }
    }
    restore_cell_primitive(primitives, cell, base_primitive);
    Ok(())
}

fn fill_cell_block_finite_difference(
    blocks: &mut [Real],
    ctx: &UnstructuredCellBlockContext<'_>,
    primitives: &mut PrimitiveFields,
    cell: usize,
    dt_cell: Real,
) -> Result<()> {
    let base_state = ctx.fields.cell_state(cell)?;
    let base_primitive = primitives.cell_primitive(cell);
    let viscous_diagonal_sigma = cell_viscous_diagonal_sigma(ctx, cell);
    let base_local = local_inviscid_residual(
        cell,
        &inviscid_params(ctx, primitives),
        ctx.topology,
        ctx.incidence,
        ctx.volumes,
    )?;
    for col in 0..CONSERVED_COMPONENTS_3D {
        let Some(perturbation) = finite_difference_perturbation(
            &base_state,
            col,
            ctx.epsilon_rel,
            ctx.eos.gamma,
            ctx.p_floor,
        ) else {
            warn_zero_finite_difference_column(cell, col, "block_lusgs 对角块有限差分");
            write_block_column(
                blocks,
                cell,
                col,
                dt_cell,
                [0.0; CONSERVED_COMPONENTS_3D],
                viscous_diagonal_sigma,
            );
            continue;
        };
        let perturbed_state =
            state_after_increment(&base_state, perturbation.increment, perturbation.epsilon);
        write_cell_primitive(primitives, cell, &perturbed_state, ctx.eos, ctx.p_floor)?;
        let perturbed_local = local_inviscid_residual(
            cell,
            &inviscid_params(ctx, primitives),
            ctx.topology,
            ctx.incidence,
            ctx.volumes,
        )?;
        let jv = finite_difference_jacobian_column(base_local, perturbed_local, &perturbation, col);
        write_block_column(blocks, cell, col, dt_cell, jv, viscous_diagonal_sigma);
    }
    restore_cell_primitive(primitives, cell, base_primitive);
    Ok(())
}
