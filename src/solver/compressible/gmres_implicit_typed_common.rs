//! typed 可压缩 GMRES 共用辅助（结构化/非结构 matrix-free 路径）。

use crate::core::{ComputeFloat, Real};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFields, ConservedFieldsT, ConservedResidual, ConservedResidualT,
    is_physical_conserved, max_physical_increment_scale, state_after_increment,
};
use crate::physics::{ConservedState, IdealGasEoS};

use super::{
    CONSERVED_COMPONENTS_3D, GmresImplicitDelta, GmresUpdateDiagnostics, conserved_component_scales,
};

pub(crate) fn apply_delta_with_line_search_typed<T: ComputeFloat>(
    fields: &mut ConservedFieldsT<T>,
    stage: &mut ConservedFieldsT<T>,
    base: &ConservedFieldsT<T>,
    delta: &GmresImplicitDelta,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<GmresUpdateDiagnostics> {
    const MIN_ALPHA: Real = 1.0 / 1024.0;
    let mut alpha = 1.0;
    loop {
        let mut diagnostics = assign_delta_limited_scaled_typed(
            stage,
            base,
            &delta.delta,
            alpha,
            eos.gamma,
            p_floor,
        )?;
        if fields_are_physical_typed(stage, eos.gamma, p_floor)? {
            diagnostics.alpha = alpha;
            fields.copy_from(stage)?;
            return Ok(diagnostics);
        }
        alpha *= 0.5;
        if alpha < MIN_ALPHA {
            return Err(AsimuError::Solver(format!(
                "GMRES 隐式更新线搜索失败：alpha < {MIN_ALPHA:.3e}"
            )));
        }
    }
}

pub(crate) fn residual_to_vector_typed<T: ComputeFloat>(
    residual: &ConservedResidualT<T>,
) -> Vec<Real> {
    let n = residual.num_cells();
    let mut out = vec![0.0; n * CONSERVED_COMPONENTS_3D];
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        out[offset] = residual.density.values()[cell].to_real();
        out[offset + 1] = residual.momentum_x.values()[cell].to_real();
        out[offset + 2] = residual.momentum_y.values()[cell].to_real();
        out[offset + 3] = residual.momentum_z.values()[cell].to_real();
        out[offset + 4] = residual.total_energy.values()[cell].to_real();
    }
    out
}

pub(crate) fn assign_vector_to_residual(
    residual: &mut ConservedResidual,
    values: &[Real],
) -> Result<()> {
    let n = residual.num_cells();
    ensure_vector_len(
        values,
        n * CONSERVED_COMPONENTS_3D,
        "gmres preconditioner rhs",
    )?;
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        residual.density.values_mut()[cell] = values[offset];
        residual.momentum_x.values_mut()[cell] = values[offset + 1];
        residual.momentum_y.values_mut()[cell] = values[offset + 2];
        residual.momentum_z.values_mut()[cell] = values[offset + 3];
        residual.total_energy.values_mut()[cell] = values[offset + 4];
    }
    Ok(())
}

pub(crate) fn fields_delta_to_vector(
    base: &ConservedFields,
    updated: &ConservedFields,
    out: &mut [Real],
) -> Result<()> {
    let n = base.num_cells();
    ensure_vector_len(out, n * CONSERVED_COMPONENTS_3D, "gmres preconditioner out")?;
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        out[offset] = updated.density.values()[cell] - base.density.values()[cell];
        out[offset + 1] = updated.momentum_x.values()[cell] - base.momentum_x.values()[cell];
        out[offset + 2] = updated.momentum_y.values()[cell] - base.momentum_y.values()[cell];
        out[offset + 3] = updated.momentum_z.values()[cell] - base.momentum_z.values()[cell];
        out[offset + 4] = updated.total_energy.values()[cell] - base.total_energy.values()[cell];
    }
    Ok(())
}

pub(crate) fn finite_difference_epsilon_typed<T: ComputeFloat>(
    base: &ConservedFieldsT<T>,
    direction: &[Real],
    epsilon_rel: Real,
) -> Result<Real> {
    let n = base.num_cells();
    ensure_vector_len(direction, n * CONSERVED_COMPONENTS_3D, "gmres direction")?;
    let mut scaled_norm_sq = 0.0;
    for cell in 0..n {
        let state = base.cell_state(cell)?;
        let scales = conserved_component_scales(&state);
        let offset = cell * CONSERVED_COMPONENTS_3D;
        for comp in 0..CONSERVED_COMPONENTS_3D {
            let scaled = direction[offset + comp] / scales[comp];
            scaled_norm_sq += scaled * scaled;
        }
    }
    let norm = scaled_norm_sq.sqrt();
    if !norm.is_finite() {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：方向向量含非有限值".to_string(),
        ));
    }
    Ok(epsilon_rel.sqrt() / norm.max(1.0))
}

pub(crate) fn assign_physical_perturbed_fields_typed<T: ComputeFloat>(
    out: &mut ConservedFieldsT<T>,
    base: &ConservedFieldsT<T>,
    direction: &[Real],
    epsilon: Real,
    gamma: Real,
    min_pressure: Real,
) -> Result<Real> {
    let effective =
        max_physical_vector_increment_scale_typed(base, direction, epsilon, gamma, min_pressure)?;
    if effective <= 0.0 {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：有限差分扰动无法保持正性".to_string(),
        ));
    }
    assign_perturbed_fields_typed(out, base, direction, effective)?;
    Ok(effective)
}

pub(crate) fn residual_difference_at_typed<T: ComputeFloat>(
    residual: &ConservedResidualT<T>,
    base: &ConservedResidualT<T>,
    cell: usize,
    epsilon: Real,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    [
        (residual.density.values()[cell].to_real() - base.density.values()[cell].to_real())
            / epsilon,
        (residual.momentum_x.values()[cell].to_real() - base.momentum_x.values()[cell].to_real())
            / epsilon,
        (residual.momentum_y.values()[cell].to_real() - base.momentum_y.values()[cell].to_real())
            / epsilon,
        (residual.momentum_z.values()[cell].to_real() - base.momentum_z.values()[cell].to_real())
            / epsilon,
        (residual.total_energy.values()[cell].to_real()
            - base.total_energy.values()[cell].to_real())
            / epsilon,
    ]
}

pub(crate) fn ensure_vector_len(values: &[Real], expected: usize, label: &str) -> Result<()> {
    if values.len() != expected {
        return Err(AsimuError::Solver(format!(
            "{label} 长度 {} 与期望 {expected} 不一致",
            values.len()
        )));
    }
    Ok(())
}

fn assign_perturbed_fields_typed<T: ComputeFloat>(
    out: &mut ConservedFieldsT<T>,
    base: &ConservedFieldsT<T>,
    direction: &[Real],
    epsilon: Real,
) -> Result<()> {
    let n = base.num_cells();
    ensure_vector_len(direction, n * CONSERVED_COMPONENTS_3D, "gmres direction")?;
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        out.density.values_mut()[cell] =
            base.density.values()[cell].add_mul_real(T::from_real(direction[offset]), epsilon);
        out.momentum_x.values_mut()[cell] = base.momentum_x.values()[cell]
            .add_mul_real(T::from_real(direction[offset + 1]), epsilon);
        out.momentum_y.values_mut()[cell] = base.momentum_y.values()[cell]
            .add_mul_real(T::from_real(direction[offset + 2]), epsilon);
        out.momentum_z.values_mut()[cell] = base.momentum_z.values()[cell]
            .add_mul_real(T::from_real(direction[offset + 3]), epsilon);
        out.total_energy.values_mut()[cell] = base.total_energy.values()[cell]
            .add_mul_real(T::from_real(direction[offset + 4]), epsilon);
    }
    Ok(())
}

fn assign_delta_limited_scaled_typed<T: ComputeFloat>(
    out: &mut ConservedFieldsT<T>,
    base: &ConservedFieldsT<T>,
    delta: &[Real],
    alpha: Real,
    gamma: Real,
    min_pressure: Real,
) -> Result<GmresUpdateDiagnostics> {
    let n = base.num_cells();
    ensure_vector_len(delta, n * CONSERVED_COMPONENTS_3D, "gmres delta")?;
    let mut limited_cells = 0;
    let mut min_update_scale: Real = 1.0;
    for cell in 0..n {
        let base_state = base.cell_state(cell)?;
        let increment = vector_increment_at(delta, cell);
        let effective =
            max_physical_increment_scale(&base_state, increment, alpha, gamma, min_pressure);
        let scale_ratio = if alpha > 0.0 { effective / alpha } else { 0.0 };
        min_update_scale = min_update_scale.min(scale_ratio);
        if scale_ratio < 1.0 - 1.0e-12 {
            limited_cells += 1;
        }
        let updated = if effective > 0.0 {
            state_after_increment(&base_state, increment, effective)
        } else {
            base_state
        };
        write_cell_state_typed(out, cell, &updated);
    }
    Ok(GmresUpdateDiagnostics {
        alpha,
        limited_cells,
        min_update_scale,
    })
}

fn fields_are_physical_typed<T: ComputeFloat>(
    fields: &ConservedFieldsT<T>,
    gamma: Real,
    min_pressure: Real,
) -> Result<bool> {
    for cell in 0..fields.num_cells() {
        if !is_physical_conserved(&fields.cell_state(cell)?, gamma, min_pressure) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn max_physical_vector_increment_scale_typed<T: ComputeFloat>(
    base: &ConservedFieldsT<T>,
    delta: &[Real],
    scale: Real,
    gamma: Real,
    min_pressure: Real,
) -> Result<Real> {
    let n = base.num_cells();
    ensure_vector_len(delta, n * CONSERVED_COMPONENTS_3D, "gmres vector increment")?;
    let mut effective = scale;
    for cell in 0..n {
        let base_state = base.cell_state(cell)?;
        let increment = vector_increment_at(delta, cell);
        effective = effective.min(max_physical_increment_scale(
            &base_state,
            increment,
            scale,
            gamma,
            min_pressure,
        ));
    }
    Ok(effective)
}

fn write_cell_state_typed<T: ComputeFloat>(
    fields: &mut ConservedFieldsT<T>,
    cell: usize,
    state: &ConservedState,
) {
    fields.density.values_mut()[cell] = T::from_real(state.density);
    fields.momentum_x.values_mut()[cell] = T::from_real(state.momentum[0]);
    fields.momentum_y.values_mut()[cell] = T::from_real(state.momentum[1]);
    fields.momentum_z.values_mut()[cell] = T::from_real(state.momentum[2]);
    fields.total_energy.values_mut()[cell] = T::from_real(state.total_energy);
}

fn vector_increment_at(values: &[Real], cell: usize) -> [Real; CONSERVED_COMPONENTS_3D] {
    let offset = cell * CONSERVED_COMPONENTS_3D;
    [
        values[offset],
        values[offset + 1],
        values[offset + 2],
        values[offset + 3],
        values[offset + 4],
    ]
}
