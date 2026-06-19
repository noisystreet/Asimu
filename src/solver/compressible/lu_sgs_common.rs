//! LU-SGS 扫掠共用辅助：正性限制、线搜索、对角回退、原始变量刷新。

use crate::core::Real;
use crate::error::Result;
use crate::field::{
    ConservedFields, ConservedResidual, PrimitiveFields, is_physical_conserved,
    is_physical_conserved_f32, max_physical_increment_scale, max_physical_increment_scale_f32,
    state_after_increment, state_after_increment_f32,
};
use crate::physics::{ConservedState, IdealGasEoS};

/// LU-SGS 扫掠标量参数（f32 热路径）。
pub(crate) struct LuSgsSweepScalarsF32<'a> {
    pub dt: &'a [f32],
    pub sigma: &'a [f32],
    pub volumes: &'a [f32],
    pub omega: f32,
    pub gamma: f32,
    /// \(1/\Delta t_{\mathrm{phys}}\)；稳态伪时间为 0。
    pub inv_dt_phys: f32,
}

/// 与对角 LU-SGS 一致（f32）：\(\omega\Delta t/(1+\Delta t\sigma+\Delta t/\Delta t_{\mathrm{phys}})\)。
#[inline]
pub(crate) fn implicit_scale_f32(dt: f32, sigma: f32, omega: f32, inv_dt_phys: f32) -> f32 {
    let denom = 1.0 + dt * sigma + dt * inv_dt_phys;
    if !(dt > 0.0 && omega > 0.0 && denom > 0.0) {
        return 0.0;
    }
    omega * dt / denom
}

/// LU-SGS 扫掠标量参数（结构化/非结构共用）。
pub(crate) struct LuSgsSweepScalars<'a> {
    pub dt: &'a [Real],
    pub sigma: &'a [Real],
    pub volumes: &'a [Real],
    pub omega: Real,
    pub gamma: Real,
    /// \(1/\Delta t_{\mathrm{phys}}\)；稳态伪时间为 0。
    pub inv_dt_phys: Real,
}

/// 与对角 LU-SGS 一致：\(\omega\Delta t/(1+\Delta t\sigma+\Delta t/\Delta t_{\mathrm{phys}})\)。
#[inline]
pub(crate) fn implicit_scale(dt: Real, sigma: Real, omega: Real, inv_dt_phys: Real) -> Real {
    let denom = 1.0 + dt * sigma + dt * inv_dt_phys;
    if !(dt > 0.0 && omega > 0.0 && denom > 0.0) {
        return 0.0;
    }
    omega * dt / denom
}

#[inline]
pub(crate) fn residual_cell_vector(residual: &ConservedResidual, cell: usize) -> [Real; 5] {
    [
        residual.density.values()[cell],
        residual.momentum_x.values()[cell],
        residual.momentum_y.values()[cell],
        residual.momentum_z.values()[cell],
        residual.total_energy.values()[cell],
    ]
}

#[inline]
pub(crate) fn conserved_vector(fields: &ConservedFields, cell: usize) -> [Real; 5] {
    [
        fields.density.values()[cell],
        fields.momentum_x.values()[cell],
        fields.momentum_y.values()[cell],
        fields.momentum_z.values()[cell],
        fields.total_energy.values()[cell],
    ]
}

#[inline]
pub(crate) fn scale_source(source: [Real; 5], factor: Real) -> [Real; 5] {
    [
        source[0] * factor,
        source[1] * factor,
        source[2] * factor,
        source[3] * factor,
        source[4] * factor,
    ]
}

pub(crate) fn apply_cell_increment(
    fields: &mut ConservedFields,
    cell: usize,
    scale: Real,
    increment: [Real; 5],
) {
    let base = fields.cell_state(cell).expect("cell state");
    let updated = state_after_increment(&base, increment, scale);
    write_cell_state(fields, cell, &updated);
}

pub(crate) fn apply_limited_cell_increment(
    fields: &mut ConservedFields,
    cell: usize,
    scale: Real,
    increment: [Real; 5],
    gamma: Real,
    min_pressure: Real,
) -> Result<()> {
    let base = fields.cell_state(cell)?;
    let effective = max_physical_increment_scale(&base, increment, scale, gamma, min_pressure);
    if effective <= 0.0 {
        return Ok(());
    }
    let updated = state_after_increment(&base, increment, effective);
    write_cell_state(fields, cell, &updated);
    Ok(())
}

pub(crate) fn write_cell_state(fields: &mut ConservedFields, cell: usize, state: &ConservedState) {
    fields.density.values_mut()[cell] = state.density;
    fields.momentum_x.values_mut()[cell] = state.momentum[0];
    fields.momentum_y.values_mut()[cell] = state.momentum[1];
    fields.momentum_z.values_mut()[cell] = state.momentum[2];
    fields.total_energy.values_mut()[cell] = state.total_energy;
}

pub(crate) fn fields_are_physical(
    fields: &ConservedFields,
    gamma: Real,
    min_pressure: Real,
) -> Result<bool> {
    for cell in 0..fields.num_cells() {
        let state = fields.cell_state(cell)?;
        if !is_physical_conserved(&state, gamma, min_pressure) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn blend_fields(
    out: &mut ConservedFields,
    base: &ConservedFields,
    target: &ConservedFields,
    alpha: Real,
) -> Result<()> {
    for cell in 0..base.num_cells() {
        let b = base.cell_state(cell)?;
        let t = target.cell_state(cell)?;
        let delta = [
            t.density - b.density,
            t.momentum[0] - b.momentum[0],
            t.momentum[1] - b.momentum[1],
            t.momentum[2] - b.momentum[2],
            t.total_energy - b.total_energy,
        ];
        write_cell_state(out, cell, &state_after_increment(&b, delta, alpha));
    }
    Ok(())
}

pub(crate) fn stabilize_sweep_update(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    u_sweep: &ConservedFields,
    residual: &ConservedResidual,
    min_pressure: Real,
    gamma: Real,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    if fields_are_physical(u_sweep, gamma, min_pressure)? {
        return Ok(());
    }
    const MIN_ALPHA: Real = 1.0 / 1024.0;
    let mut alpha = 1.0;
    loop {
        blend_fields(fields, u0, u_sweep, alpha)?;
        if fields_are_physical(fields, gamma, min_pressure)? {
            return Ok(());
        }
        alpha *= 0.5;
        if alpha < MIN_ALPHA {
            apply_diagonal_fallback(fields, u0, residual, gamma, min_pressure, scalars)?;
            return Ok(());
        }
    }
}

pub(crate) fn apply_diagonal_fallback(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    residual: &ConservedResidual,
    gamma: Real,
    min_pressure: Real,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for cell in 0..fields.num_cells() {
        let scale = implicit_scale(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
        let increment = residual_cell_vector(residual, cell);
        let base = u0.cell_state(cell)?;
        let effective = max_physical_increment_scale(&base, increment, scale, gamma, min_pressure);
        if effective > 0.0 {
            write_cell_state(
                fields,
                cell,
                &state_after_increment(&base, increment, effective),
            );
        } else {
            write_cell_state(fields, cell, &base);
        }
    }
    Ok(())
}

pub(crate) fn refresh_primitive_at_cell(
    fields: &ConservedFields,
    cell: usize,
    eos: &IdealGasEoS,
    min_pressure: Real,
    primitives: &mut PrimitiveFields,
) -> Result<()> {
    let cons = fields.cell_state(cell)?;
    let prim = crate::field::primitive_from_conserved_relaxed(eos, &cons, min_pressure)?;
    primitives.density.values_mut()[cell] = prim.density;
    primitives.pressure.values_mut()[cell] = prim.pressure;
    primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
    primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
    primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
    Ok(())
}

// --- typed 扫掠辅助（f32/f64 共用；正性限制仍经 `ConservedState`）---

pub(crate) fn residual_cell_vector_typed<T: crate::core::ComputeFloat>(
    residual: &crate::field::ConservedResidualT<T>,
    cell: usize,
) -> [Real; 5] {
    [
        residual.density.values()[cell].to_real(),
        residual.momentum_x.values()[cell].to_real(),
        residual.momentum_y.values()[cell].to_real(),
        residual.momentum_z.values()[cell].to_real(),
        residual.total_energy.values()[cell].to_real(),
    ]
}

pub(crate) fn conserved_vector_typed<T: crate::core::ComputeFloat>(
    fields: &crate::field::ConservedFieldsT<T>,
    cell: usize,
) -> [Real; 5] {
    [
        fields.density.values()[cell].to_real(),
        fields.momentum_x.values()[cell].to_real(),
        fields.momentum_y.values()[cell].to_real(),
        fields.momentum_z.values()[cell].to_real(),
        fields.total_energy.values()[cell].to_real(),
    ]
}

/// f32 残差单元向量（LU-SGS 扫掠热路径，无 Real 桥接）。
#[inline]
pub(crate) fn residual_cell_vector_f32(
    residual: &crate::field::ConservedResidualT<f32>,
    cell: usize,
) -> [f32; 5] {
    [
        residual.density.values()[cell],
        residual.momentum_x.values()[cell],
        residual.momentum_y.values()[cell],
        residual.momentum_z.values()[cell],
        residual.total_energy.values()[cell],
    ]
}

/// f32 守恒场单元向量（LU-SGS 耦合差分热路径）。
#[inline]
pub(crate) fn conserved_vector_f32(
    fields: &crate::field::ConservedFieldsT<f32>,
    cell: usize,
) -> [f32; 5] {
    [
        fields.density.values()[cell],
        fields.momentum_x.values()[cell],
        fields.momentum_y.values()[cell],
        fields.momentum_z.values()[cell],
        fields.total_energy.values()[cell],
    ]
}

/// f32 source 阻尼（backward sweep）。
#[inline]
pub(crate) fn scale_source_f32(source: [f32; 5], factor: f32) -> [f32; 5] {
    [
        source[0] * factor,
        source[1] * factor,
        source[2] * factor,
        source[3] * factor,
        source[4] * factor,
    ]
}

#[inline]
pub(crate) fn write_conserved_lane_f32(
    fields: &mut crate::field::ConservedFieldsT<f32>,
    cell: usize,
    lane: [f32; 5],
) {
    fields.density.values_mut()[cell] = lane[0];
    fields.momentum_x.values_mut()[cell] = lane[1];
    fields.momentum_y.values_mut()[cell] = lane[2];
    fields.momentum_z.values_mut()[cell] = lane[3];
    fields.total_energy.values_mut()[cell] = lane[4];
}

pub(crate) fn apply_limited_cell_increment_f32(
    fields: &mut crate::field::ConservedFieldsT<f32>,
    cell: usize,
    scale: f32,
    increment: [f32; 5],
    gamma: f32,
    min_pressure: f32,
) -> Result<()> {
    let base = conserved_vector_f32(fields, cell);
    let effective = max_physical_increment_scale_f32(base, increment, scale, gamma, min_pressure);
    if effective <= 0.0 {
        return Ok(());
    }
    write_conserved_lane_f32(
        fields,
        cell,
        state_after_increment_f32(base, increment, effective),
    );
    Ok(())
}

pub(crate) fn fields_are_physical_f32(
    fields: &crate::field::ConservedFieldsT<f32>,
    gamma: f32,
    min_pressure: f32,
) -> Result<bool> {
    for cell in 0..fields.num_cells() {
        let lane = conserved_vector_f32(fields, cell);
        if !is_physical_conserved_f32(
            lane[0],
            lane[1],
            lane[2],
            lane[3],
            lane[4],
            gamma,
            min_pressure,
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn blend_fields_f32(
    out: &mut crate::field::ConservedFieldsT<f32>,
    base: &crate::field::ConservedFieldsT<f32>,
    target: &crate::field::ConservedFieldsT<f32>,
    alpha: f32,
) -> Result<()> {
    for cell in 0..base.num_cells() {
        let b = conserved_vector_f32(base, cell);
        let t = conserved_vector_f32(target, cell);
        let delta = [
            t[0] - b[0],
            t[1] - b[1],
            t[2] - b[2],
            t[3] - b[3],
            t[4] - b[4],
        ];
        write_conserved_lane_f32(out, cell, state_after_increment_f32(b, delta, alpha));
    }
    Ok(())
}

pub(crate) fn stabilize_sweep_update_f32(
    fields: &mut crate::field::ConservedFieldsT<f32>,
    u0: &crate::field::ConservedFieldsT<f32>,
    u_sweep: &crate::field::ConservedFieldsT<f32>,
    residual: &crate::field::ConservedResidualT<f32>,
    min_pressure: f32,
    gamma: f32,
    scalars: &LuSgsSweepScalarsF32<'_>,
) -> Result<()> {
    if fields_are_physical_f32(u_sweep, gamma, min_pressure)? {
        return Ok(());
    }
    const MIN_ALPHA: f32 = 1.0 / 1024.0;
    let mut alpha = 1.0_f32;
    loop {
        blend_fields_f32(fields, u0, u_sweep, alpha)?;
        if fields_are_physical_f32(fields, gamma, min_pressure)? {
            return Ok(());
        }
        alpha *= 0.5;
        if alpha < MIN_ALPHA {
            apply_diagonal_fallback_f32(fields, u0, residual, gamma, min_pressure, scalars)?;
            return Ok(());
        }
    }
}

pub(crate) fn apply_diagonal_fallback_f32(
    fields: &mut crate::field::ConservedFieldsT<f32>,
    u0: &crate::field::ConservedFieldsT<f32>,
    residual: &crate::field::ConservedResidualT<f32>,
    gamma: f32,
    min_pressure: f32,
    scalars: &LuSgsSweepScalarsF32<'_>,
) -> Result<()> {
    for cell in 0..fields.num_cells() {
        let scale = implicit_scale_f32(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
        let increment = residual_cell_vector_f32(residual, cell);
        let base = conserved_vector_f32(u0, cell);
        let effective =
            max_physical_increment_scale_f32(base, increment, scale, gamma, min_pressure);
        if effective > 0.0 {
            write_conserved_lane_f32(
                fields,
                cell,
                state_after_increment_f32(base, increment, effective),
            );
        } else {
            write_conserved_lane_f32(fields, cell, base);
        }
    }
    Ok(())
}

pub(crate) fn apply_limited_cell_increment_typed<T: crate::core::ComputeFloat>(
    fields: &mut crate::field::ConservedFieldsT<T>,
    cell: usize,
    scale: Real,
    increment: [Real; 5],
    gamma: Real,
    min_pressure: Real,
) -> Result<()> {
    let base = fields.cell_state(cell)?;
    let effective = max_physical_increment_scale(&base, increment, scale, gamma, min_pressure);
    if effective <= 0.0 {
        return Ok(());
    }
    let updated = state_after_increment(&base, increment, effective);
    write_cell_state_typed(fields, cell, &updated);
    Ok(())
}

pub(crate) fn write_cell_state_typed<T: crate::core::ComputeFloat>(
    fields: &mut crate::field::ConservedFieldsT<T>,
    cell: usize,
    state: &ConservedState,
) {
    fields.density.values_mut()[cell] = T::from_real(state.density);
    fields.momentum_x.values_mut()[cell] = T::from_real(state.momentum[0]);
    fields.momentum_y.values_mut()[cell] = T::from_real(state.momentum[1]);
    fields.momentum_z.values_mut()[cell] = T::from_real(state.momentum[2]);
    fields.total_energy.values_mut()[cell] = T::from_real(state.total_energy);
}

pub(crate) fn fields_are_physical_typed<T: crate::core::ComputeFloat>(
    fields: &crate::field::ConservedFieldsT<T>,
    gamma: Real,
    min_pressure: Real,
) -> Result<bool> {
    for cell in 0..fields.num_cells() {
        let state = fields.cell_state(cell)?;
        if !is_physical_conserved(&state, gamma, min_pressure) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn blend_fields_typed<T: crate::core::ComputeFloat>(
    out: &mut crate::field::ConservedFieldsT<T>,
    base: &crate::field::ConservedFieldsT<T>,
    target: &crate::field::ConservedFieldsT<T>,
    alpha: Real,
) -> Result<()> {
    for cell in 0..base.num_cells() {
        let b = base.cell_state(cell)?;
        let t = target.cell_state(cell)?;
        let delta = [
            t.density - b.density,
            t.momentum[0] - b.momentum[0],
            t.momentum[1] - b.momentum[1],
            t.momentum[2] - b.momentum[2],
            t.total_energy - b.total_energy,
        ];
        write_cell_state_typed(out, cell, &state_after_increment(&b, delta, alpha));
    }
    Ok(())
}

pub(crate) fn stabilize_sweep_update_typed<T: crate::core::ComputeFloat>(
    fields: &mut crate::field::ConservedFieldsT<T>,
    u0: &crate::field::ConservedFieldsT<T>,
    u_sweep: &crate::field::ConservedFieldsT<T>,
    residual: &crate::field::ConservedResidualT<T>,
    min_pressure: Real,
    gamma: Real,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    if fields_are_physical_typed(u_sweep, gamma, min_pressure)? {
        return Ok(());
    }
    const MIN_ALPHA: Real = 1.0 / 1024.0;
    let mut alpha = 1.0;
    loop {
        blend_fields_typed(fields, u0, u_sweep, alpha)?;
        if fields_are_physical_typed(fields, gamma, min_pressure)? {
            return Ok(());
        }
        alpha *= 0.5;
        if alpha < MIN_ALPHA {
            apply_diagonal_fallback_typed(fields, u0, residual, gamma, min_pressure, scalars)?;
            return Ok(());
        }
    }
}

pub(crate) fn apply_diagonal_fallback_typed<T: crate::core::ComputeFloat>(
    fields: &mut crate::field::ConservedFieldsT<T>,
    u0: &crate::field::ConservedFieldsT<T>,
    residual: &crate::field::ConservedResidualT<T>,
    gamma: Real,
    min_pressure: Real,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for cell in 0..fields.num_cells() {
        let scale = implicit_scale(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
        let increment = residual_cell_vector_typed(residual, cell);
        let base = u0.cell_state(cell)?;
        let effective = max_physical_increment_scale(&base, increment, scale, gamma, min_pressure);
        if effective > 0.0 {
            write_cell_state_typed(
                fields,
                cell,
                &state_after_increment(&base, increment, effective),
            );
        } else {
            write_cell_state_typed(fields, cell, &base);
        }
    }
    Ok(())
}

/// 按精度从守恒场刷新单单元原始变量（f32 无 `cell_state` 往返）。
pub trait PrimitiveRefreshLane: crate::core::ComputeFloat {
    fn refresh_primitive_at_cell(
        fields: &crate::field::ConservedFieldsT<Self>,
        cell: usize,
        eos: &IdealGasEoS,
        min_pressure: Real,
        primitives: &mut crate::field::PrimitiveFieldsT<Self>,
    ) -> Result<()>;
}

impl PrimitiveRefreshLane for f32 {
    fn refresh_primitive_at_cell(
        fields: &crate::field::ConservedFieldsT<f32>,
        cell: usize,
        eos: &IdealGasEoS,
        min_pressure: Real,
        primitives: &mut crate::field::PrimitiveFieldsT<f32>,
    ) -> Result<()> {
        let prim = crate::field::primitive_from_conserved_relaxed_f32(
            eos,
            fields.density.values()[cell],
            [
                fields.momentum_x.values()[cell],
                fields.momentum_y.values()[cell],
                fields.momentum_z.values()[cell],
            ],
            fields.total_energy.values()[cell],
            min_pressure,
        )?;
        primitives.density.values_mut()[cell] = prim.density;
        primitives.pressure.values_mut()[cell] = prim.pressure;
        primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
        primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
        primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
        Ok(())
    }
}

impl PrimitiveRefreshLane for f64 {
    fn refresh_primitive_at_cell(
        fields: &crate::field::ConservedFieldsT<f64>,
        cell: usize,
        eos: &IdealGasEoS,
        min_pressure: Real,
        primitives: &mut crate::field::PrimitiveFieldsT<f64>,
    ) -> Result<()> {
        let cons = fields.cell_state(cell)?;
        let prim = crate::field::primitive_from_conserved_relaxed(eos, &cons, min_pressure)?;
        primitives.density.values_mut()[cell] = prim.density;
        primitives.pressure.values_mut()[cell] = prim.pressure;
        primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
        primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
        primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
        Ok(())
    }
}

pub(crate) fn refresh_primitive_at_cell_typed<T: PrimitiveRefreshLane>(
    fields: &crate::field::ConservedFieldsT<T>,
    cell: usize,
    eos: &IdealGasEoS,
    min_pressure: Real,
    primitives: &mut crate::field::PrimitiveFieldsT<T>,
) -> Result<()> {
    T::refresh_primitive_at_cell(fields, cell, eos, min_pressure, primitives)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_scale_decreases_with_inv_dt_phys() {
        let steady = implicit_scale(0.1, 2.0, 1.0, 0.0);
        let dual = implicit_scale(0.1, 2.0, 1.0, 10.0);
        assert!(steady > 0.0);
        assert!(dual > 0.0);
        assert!(dual < steady);
    }

    #[test]
    fn implicit_scale_f32_decreases_with_inv_dt_phys() {
        let steady = implicit_scale_f32(0.1_f32, 2.0_f32, 1.0_f32, 0.0_f32);
        let dual = implicit_scale_f32(0.1_f32, 2.0_f32, 1.0_f32, 10.0_f32);
        assert!(steady > 0.0);
        assert!(dual > 0.0);
        assert!(dual < steady);
    }
}
