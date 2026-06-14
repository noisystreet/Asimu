//! 3D 可压缩 typed 场 matrix-free GMRES 隐式线性化（ADR 0016 P4）。
//!
//! 线性代数向量与 GMRES 内积保持 `f64`；场存储与 RHS 装配使用 `ConservedFieldsT<T>`。

use std::time::Instant;

use crate::core::{ComputeFloat, Real, elapsed_ms};
use crate::discretization::{InviscidFaceFluxTyped, InviscidFluxConfig};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFieldsT, ConservedResidualT, is_physical_conserved, max_physical_increment_scale,
    state_after_increment,
};
use crate::linalg::{GmresSolver, LinearOperator};
use crate::physics::{ConservedState, IdealGasEoS};

use super::super::CompressibleAdvanceContext3dTyped;
use super::super::CompressibleEulerSolver;
use super::super::gmres_block_preconditioner_3d::build_cell_block_preconditioner;
use super::super::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, GmresImplicitConfig, GmresImplicitDelta, GmresImplicitDiagnostics,
    GmresImplicitPreconditioner, GmresImplicitTiming, GmresPreconditionerBuild,
    GmresPreconditionerKind, GmresUpdateDiagnostics, build_gmres_preconditioner,
    conserved_component_scales, validate_gmres_inputs,
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

impl CompressibleEulerSolver {
    /// typed 场 matrix-free GMRES 隐式伪时间步。
    pub fn solve_gmres_implicit_delta_3d_typed<T: ComputeFloat + InviscidFaceFluxTyped>(
        &self,
        ctx: &mut CompressibleAdvanceContext3dTyped<'_, T>,
        fields: &ConservedFieldsT<T>,
        dt: &[Real],
        sigma: &[Real],
        p_floor: Real,
        config: GmresImplicitConfig,
    ) -> Result<GmresImplicitDelta> {
        let total_start = Instant::now();
        validate_gmres_inputs(fields.num_cells(), dt, sigma, config.epsilon)?;
        let inviscid = self.config.inviscid;
        let mut base_residual = ConservedResidualT::<T>::zeros(fields.num_cells())?;
        let base_residual_start = Instant::now();
        self.rhs_context_3d_typed(ctx, &inviscid, p_floor)
            .run(fields, &mut base_residual)?;
        let base_residual_ms = elapsed_ms(base_residual_start);
        let base_residual_rms = base_residual.density_rms_norm();
        let rhs = residual_to_vector_typed(&base_residual);
        let preconditioner_kind = config.preconditioner;
        let preconditioner_start = Instant::now();
        let base_real = fields.cast_real()?;
        let mut ctx_f64 = ctx.f64_preconditioner_context();
        let precond = match config.preconditioner {
            GmresPreconditionerKind::ScalarDiagonal => {
                build_gmres_preconditioner(GmresPreconditionerBuild {
                    solver: self,
                    ctx: &mut ctx_f64,
                    fields: &base_real,
                    inviscid: &inviscid,
                    dt,
                    sigma,
                    p_floor,
                    config,
                })?
            }
            GmresPreconditionerKind::CellBlockDiagonal => {
                GmresImplicitPreconditioner::CellBlock(build_cell_block_preconditioner(
                    &mut ctx_f64,
                    &base_real,
                    &inviscid,
                    dt,
                    p_floor,
                    config.epsilon,
                )?)
            }
        };
        let preconditioner_build_ms = elapsed_ms(preconditioner_start);
        let mut diagnostics = GmresImplicitDiagnostics::new(preconditioner_kind);
        let zero_state = ConservedState {
            density: 0.0,
            momentum: [0.0; 3],
            total_energy: 0.0,
        };
        let mut op = MatrixFreeResidualOperator3dTyped {
            solver: self,
            ctx,
            base: fields,
            base_residual: &base_residual,
            inviscid: &inviscid,
            dt,
            p_floor,
            epsilon_rel: config.epsilon,
            diagnostics: GmresImplicitDiagnostics::new(preconditioner_kind),
            perturbed: ConservedFieldsT::<T>::uniform(fields.num_cells(), zero_state)?,
            perturbed_residual: ConservedResidualT::<T>::zeros(fields.num_cells())?,
        };
        let mut delta = vec![0.0; rhs.len()];
        let linear_solve_start = Instant::now();
        let report = GmresSolver::new(config.gmres)?.solve(&mut op, &precond, &rhs, &mut delta)?;
        let linear_solve_ms = elapsed_ms(linear_solve_start);
        diagnostics.perturbation_evals = op.diagnostics.perturbation_evals;
        diagnostics.perturbation_limited_evals = op.diagnostics.perturbation_limited_evals;
        diagnostics.min_perturbation_scale = op.diagnostics.min_perturbation_scale;
        diagnostics.timing = GmresImplicitTiming {
            base_residual_ms,
            preconditioner_build_ms,
            linear_solve_ms,
            total_ms: elapsed_ms(total_start),
        };
        Ok(GmresImplicitDelta {
            delta,
            report,
            base_residual_rms,
            diagnostics,
        })
    }
}

struct MatrixFreeResidualOperator3dTyped<'a, 'ctx, T: ComputeFloat + InviscidFaceFluxTyped> {
    solver: &'a CompressibleEulerSolver,
    ctx: &'a mut CompressibleAdvanceContext3dTyped<'ctx, T>,
    base: &'a ConservedFieldsT<T>,
    base_residual: &'a ConservedResidualT<T>,
    inviscid: &'a InviscidFluxConfig,
    dt: &'a [Real],
    p_floor: Real,
    epsilon_rel: Real,
    diagnostics: GmresImplicitDiagnostics,
    perturbed: ConservedFieldsT<T>,
    perturbed_residual: ConservedResidualT<T>,
}

impl<T: ComputeFloat + InviscidFaceFluxTyped> LinearOperator
    for MatrixFreeResidualOperator3dTyped<'_, '_, T>
{
    fn dimension(&self) -> usize {
        self.base.num_cells() * CONSERVED_COMPONENTS_3D
    }

    fn apply(&mut self, x: &[Real], y: &mut [Real]) -> Result<()> {
        let n = self.base.num_cells();
        ensure_vector_len(x, self.dimension(), "gmres implicit input")?;
        ensure_vector_len(y, self.dimension(), "gmres implicit output")?;
        let requested_eps = finite_difference_epsilon_typed(self.base, x, self.epsilon_rel)?;
        let eps = assign_physical_perturbed_fields_typed(
            &mut self.perturbed,
            self.base,
            x,
            requested_eps,
            self.ctx.eos.gamma,
            self.p_floor,
        )?;
        self.record_perturbation_scale(eps / requested_eps);
        self.solver
            .rhs_context_3d_typed(self.ctx, self.inviscid, self.p_floor)
            .run(&self.perturbed, &mut self.perturbed_residual)?;
        for cell in 0..n {
            let offset = cell * CONSERVED_COMPONENTS_3D;
            let jv = residual_difference_at_typed(
                &self.perturbed_residual,
                self.base_residual,
                cell,
                eps,
            );
            for comp in 0..CONSERVED_COMPONENTS_3D {
                y[offset + comp] = x[offset + comp] / self.dt[cell] - jv[comp];
            }
        }
        Ok(())
    }
}

impl<T: ComputeFloat + InviscidFaceFluxTyped> MatrixFreeResidualOperator3dTyped<'_, '_, T> {
    fn record_perturbation_scale(&mut self, scale: Real) {
        self.diagnostics.perturbation_evals += 1;
        self.diagnostics.min_perturbation_scale =
            self.diagnostics.min_perturbation_scale.min(scale);
        if scale < 1.0 - 1.0e-12 {
            self.diagnostics.perturbation_limited_evals += 1;
        }
    }
}

fn residual_to_vector_typed<T: ComputeFloat>(residual: &ConservedResidualT<T>) -> Vec<Real> {
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

fn finite_difference_epsilon_typed<T: ComputeFloat>(
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

fn assign_physical_perturbed_fields_typed<T: ComputeFloat>(
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

fn residual_difference_at_typed<T: ComputeFloat>(
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

fn ensure_vector_len(values: &[Real], expected: usize, label: &str) -> Result<()> {
    if values.len() != expected {
        return Err(AsimuError::Solver(format!(
            "{label} 长度 {} 与期望 {expected} 不一致",
            values.len()
        )));
    }
    Ok(())
}
