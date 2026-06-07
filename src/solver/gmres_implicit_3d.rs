//! 3D 可压缩残差的 matrix-free GMRES 隐式线性化。

use std::time::Instant;

use tracing::info;

use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive};
use crate::discretization::InviscidFluxConfig;
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFields, ConservedResidual, is_physical_conserved, max_physical_increment_scale,
    state_after_increment,
};
use crate::linalg::{
    CellBlockDiagonalPreconditioner, GmresConfig, GmresReport, GmresSolver, LinearOperator,
    LusgsDiagonalPreconditioner, Preconditioner,
};
use crate::physics::IdealGasEoS;

use super::gmres_block_preconditioner_3d::build_cell_block_preconditioner;
use super::{CompressibleAdvanceContext3d, CompressibleEulerSolver};

pub(super) const CONSERVED_COMPONENTS_3D: usize = 5;

/// GMRES 隐式更新结果。
#[derive(Debug, Clone, PartialEq)]
pub struct GmresImplicitDelta {
    pub delta: Vec<Real>,
    pub report: GmresReport,
    /// 步初 \(R(U^0)\) 的 RMS（与 `log10_residual` 监控语义一致）。
    pub base_residual_rms: Real,
    pub diagnostics: GmresImplicitDiagnostics,
}

/// GMRES 隐式步的数值诊断。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresImplicitDiagnostics {
    pub preconditioner: GmresPreconditionerKind,
    pub perturbation_evals: usize,
    pub perturbation_limited_evals: usize,
    pub min_perturbation_scale: Real,
    pub timing: GmresImplicitTiming,
}

/// GMRES 隐式线性化内部阶段耗时（毫秒）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresImplicitTiming {
    pub base_residual_ms: Real,
    pub preconditioner_build_ms: Real,
    pub linear_solve_ms: Real,
    pub total_ms: Real,
}

impl GmresImplicitTiming {
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            base_residual_ms: 0.0,
            preconditioner_build_ms: 0.0,
            linear_solve_ms: 0.0,
            total_ms: 0.0,
        }
    }
}

impl GmresImplicitDiagnostics {
    fn new(preconditioner: GmresPreconditionerKind) -> Self {
        Self {
            preconditioner,
            perturbation_evals: 0,
            perturbation_limited_evals: 0,
            min_perturbation_scale: 1.0,
            timing: GmresImplicitTiming::zero(),
        }
    }
}

/// GMRES 更新写回诊断。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresUpdateDiagnostics {
    pub alpha: Real,
    pub limited_cells: usize,
    pub min_update_scale: Real,
}

/// GMRES 外层推进阶段耗时（毫秒）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresStepTiming {
    pub compute_dt_ms: Real,
    pub implicit_solve_ms: Real,
    pub line_search_ms: Real,
    pub post_residual_ms: Real,
    pub step_total_ms: Real,
}

/// GMRES 单步诊断日志参数。
#[derive(Debug, Clone, Copy)]
pub struct GmresStepLog<'a> {
    pub step: u64,
    pub dt: Real,
    pub cfl: Real,
    pub delta: &'a GmresImplicitDelta,
    pub update: GmresUpdateDiagnostics,
    pub residual_rms: Real,
    pub timing: GmresStepTiming,
}

pub(crate) fn log_gmres_step_diagnostics(params: GmresStepLog<'_>) {
    let inner = params.delta.diagnostics.timing;
    info!(
        step = params.step,
        dt = %format_log_sci4(params.dt),
        cfl = params.cfl,
        gmres_converged = params.delta.report.converged,
        gmres_iters = params.delta.report.iterations,
        gmres_residual = %format_log_sci4(params.delta.report.residual_norm),
        gmres_preconditioner = params.delta.diagnostics.preconditioner.as_str(),
        line_search_alpha = params.update.alpha,
        update_limited_cells = params.update.limited_cells,
        update_min_scale = %format_log_sci4(params.update.min_update_scale),
        perturb_limited_evals = params.delta.diagnostics.perturbation_limited_evals,
        perturb_min_scale = %format_log_sci4(params.delta.diagnostics.min_perturbation_scale),
        log10_residual = %format_log_fixed4(log10_positive(params.residual_rms)),
        profile_compute_dt_ms = %format_log_fixed4(params.timing.compute_dt_ms),
        profile_implicit_solve_ms = %format_log_fixed4(params.timing.implicit_solve_ms),
        profile_base_residual_ms = %format_log_fixed4(inner.base_residual_ms),
        profile_preconditioner_build_ms = %format_log_fixed4(inner.preconditioner_build_ms),
        profile_linear_solve_ms = %format_log_fixed4(inner.linear_solve_ms),
        profile_line_search_ms = %format_log_fixed4(params.timing.line_search_ms),
        profile_post_residual_ms = %format_log_fixed4(params.timing.post_residual_ms),
        profile_step_total_ms = %format_log_fixed4(params.timing.step_total_ms),
        "GMRES 隐式步诊断"
    );
}

impl GmresImplicitDelta {
    /// 将 GMRES 求得的 \(\Delta U\) 按给定线搜索系数写入输出场。
    pub fn assign_scaled_to(
        &self,
        out: &mut ConservedFields,
        base: &ConservedFields,
        alpha: Real,
    ) -> Result<()> {
        assign_delta_scaled(out, base, &self.delta, alpha)
    }

    /// 将 GMRES 增量按正性约束裁剪后写入输出场。
    pub fn assign_limited_scaled_to(
        &self,
        out: &mut ConservedFields,
        base: &ConservedFields,
        alpha: Real,
        gamma: Real,
        p_floor: Real,
    ) -> Result<GmresUpdateDiagnostics> {
        assign_delta_limited_scaled(out, base, &self.delta, alpha, gamma, p_floor)
    }
}

pub(crate) fn apply_delta_with_line_search(
    fields: &mut ConservedFields,
    stage: &mut ConservedFields,
    base: &ConservedFields,
    delta: &GmresImplicitDelta,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<GmresUpdateDiagnostics> {
    const MIN_ALPHA: Real = 1.0 / 1024.0;
    let mut alpha = 1.0;
    loop {
        let mut diagnostics =
            delta.assign_limited_scaled_to(stage, base, alpha, eos.gamma, p_floor)?;
        if fields_are_physical(stage, eos.gamma, p_floor)? {
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

/// GMRES 隐式线性化参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmresImplicitConfig {
    pub gmres: GmresConfig,
    /// 有限差分扰动系数，实际 \(\epsilon\) 会按方向范数缩放。
    pub epsilon: Real,
    pub preconditioner: GmresPreconditionerKind,
}

/// Matrix-free GMRES 使用的左预条件器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmresPreconditionerKind {
    ScalarDiagonal,
    CellBlockDiagonal,
}

impl GmresPreconditionerKind {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw {
            "scalar" | "scalar_diagonal" | "lusgs_diagonal" => Ok(Self::ScalarDiagonal),
            "block" | "cell_block" | "cell_block_diagonal" => Ok(Self::CellBlockDiagonal),
            other => Err(AsimuError::Config(format!(
                "不支持的 GMRES 预条件器 \"{other}\"（可用 scalar_diagonal / cell_block_diagonal）"
            ))),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScalarDiagonal => "scalar_diagonal",
            Self::CellBlockDiagonal => "cell_block_diagonal",
        }
    }
}

impl Default for GmresImplicitConfig {
    fn default() -> Self {
        Self {
            gmres: GmresConfig {
                restart: 20,
                max_iters: 60,
                tolerance: 1.0e-6,
            },
            epsilon: 1.0e-7,
            preconditioner: GmresPreconditionerKind::ScalarDiagonal,
        }
    }
}

impl CompressibleEulerSolver {
    /// 求解 matrix-free 隐式伪时间线性系统
    /// \((D_{\Delta t}-J_R)\Delta U = R(U)\)。
    ///
    /// 当前入口用于把 GMRES + LU-SGS 对角预条件器接入 3D 可压缩残差；默认时间推进
    /// 仍保持原 LU-SGS/RK 路径，调用方可用返回的 `delta` 自行做线搜索与正性检查。
    pub fn solve_gmres_implicit_delta_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &ConservedFields,
        dt: &[Real],
        sigma: &[Real],
        p_floor: Real,
        config: GmresImplicitConfig,
    ) -> Result<GmresImplicitDelta> {
        let total_start = Instant::now();
        validate_gmres_inputs(fields.num_cells(), dt, sigma, config.epsilon)?;
        let inviscid = self.config.inviscid;
        let mut base_residual = ConservedResidual::zeros(fields.num_cells())?;
        let base_residual_start = Instant::now();
        self.rhs_context_3d(ctx, &inviscid, p_floor)
            .run(fields, &mut base_residual)?;
        let base_residual_ms = elapsed_ms(base_residual_start);
        let base_residual_rms = base_residual.density_rms_norm();
        let rhs = residual_to_vector(&base_residual);
        let preconditioner_kind = config.preconditioner;
        let preconditioner_start = Instant::now();
        let precond = build_gmres_preconditioner(GmresPreconditionerBuild {
            solver: self,
            ctx,
            fields,
            inviscid: &inviscid,
            dt,
            sigma,
            p_floor,
            config,
        })?;
        let preconditioner_build_ms = elapsed_ms(preconditioner_start);
        let mut diagnostics = GmresImplicitDiagnostics::new(preconditioner_kind);
        let mut op = MatrixFreeResidualOperator3d {
            solver: self,
            ctx,
            base: fields,
            base_residual: &base_residual,
            inviscid: &inviscid,
            dt,
            p_floor,
            epsilon_rel: config.epsilon,
            diagnostics: GmresImplicitDiagnostics::new(preconditioner_kind),
            perturbed: zero_conserved_fields(fields.num_cells())?,
            perturbed_residual: ConservedResidual::zeros(fields.num_cells())?,
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

enum GmresImplicitPreconditioner {
    Scalar(LusgsDiagonalPreconditioner),
    CellBlock(CellBlockDiagonalPreconditioner),
}

impl Preconditioner for GmresImplicitPreconditioner {
    fn dimension(&self) -> usize {
        match self {
            Self::Scalar(p) => p.dimension(),
            Self::CellBlock(p) => p.dimension(),
        }
    }

    fn apply(&self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        match self {
            Self::Scalar(p) => p.apply(rhs, out),
            Self::CellBlock(p) => p.apply(rhs, out),
        }
    }
}

struct GmresPreconditionerBuild<'a, 'ctx> {
    solver: &'a CompressibleEulerSolver,
    ctx: &'a mut CompressibleAdvanceContext3d<'ctx>,
    fields: &'a ConservedFields,
    inviscid: &'a InviscidFluxConfig,
    dt: &'a [Real],
    sigma: &'a [Real],
    p_floor: Real,
    config: GmresImplicitConfig,
}

fn build_gmres_preconditioner(
    params: GmresPreconditionerBuild<'_, '_>,
) -> Result<GmresImplicitPreconditioner> {
    match params.config.preconditioner {
        GmresPreconditionerKind::ScalarDiagonal => Ok(GmresImplicitPreconditioner::Scalar(
            LusgsDiagonalPreconditioner::from_lusgs_diagonal(
                params.dt,
                params.sigma,
                params.solver.config.lu_sgs.omega,
                CONSERVED_COMPONENTS_3D,
            )?,
        )),
        GmresPreconditionerKind::CellBlockDiagonal => Ok(GmresImplicitPreconditioner::CellBlock(
            build_cell_block_preconditioner(
                params.ctx,
                params.fields,
                params.inviscid,
                params.dt,
                params.p_floor,
                params.config.epsilon,
            )?,
        )),
    }
}

struct MatrixFreeResidualOperator3d<'a, 'ctx> {
    solver: &'a CompressibleEulerSolver,
    ctx: &'a mut CompressibleAdvanceContext3d<'ctx>,
    base: &'a ConservedFields,
    base_residual: &'a ConservedResidual,
    inviscid: &'a InviscidFluxConfig,
    dt: &'a [Real],
    p_floor: Real,
    epsilon_rel: Real,
    diagnostics: GmresImplicitDiagnostics,
    perturbed: ConservedFields,
    perturbed_residual: ConservedResidual,
}

impl LinearOperator for MatrixFreeResidualOperator3d<'_, '_> {
    fn dimension(&self) -> usize {
        self.base.num_cells() * CONSERVED_COMPONENTS_3D
    }

    fn apply(&mut self, x: &[Real], y: &mut [Real]) -> Result<()> {
        let n = self.base.num_cells();
        ensure_vector_len(x, self.dimension(), "gmres implicit input")?;
        ensure_vector_len(y, self.dimension(), "gmres implicit output")?;
        let requested_eps = finite_difference_epsilon(self.base, x, self.epsilon_rel)?;
        let eps = assign_physical_perturbed_fields(
            &mut self.perturbed,
            self.base,
            x,
            requested_eps,
            self.ctx.eos.gamma,
            self.p_floor,
        )?;
        self.record_perturbation_scale(eps / requested_eps);
        self.solver
            .rhs_context_3d(self.ctx, self.inviscid, self.p_floor)
            .run(&self.perturbed, &mut self.perturbed_residual)?;
        for cell in 0..n {
            let offset = cell * CONSERVED_COMPONENTS_3D;
            let jv =
                residual_difference_at(&self.perturbed_residual, self.base_residual, cell, eps);
            for comp in 0..CONSERVED_COMPONENTS_3D {
                y[offset + comp] = x[offset + comp] / self.dt[cell] - jv[comp];
            }
        }
        Ok(())
    }
}

impl MatrixFreeResidualOperator3d<'_, '_> {
    fn record_perturbation_scale(&mut self, scale: Real) {
        self.diagnostics.perturbation_evals += 1;
        self.diagnostics.min_perturbation_scale =
            self.diagnostics.min_perturbation_scale.min(scale);
        if scale < 1.0 - 1.0e-12 {
            self.diagnostics.perturbation_limited_evals += 1;
        }
    }
}

fn validate_gmres_inputs(
    num_cells: usize,
    dt: &[Real],
    sigma: &[Real],
    epsilon: Real,
) -> Result<()> {
    if dt.len() != num_cells || sigma.len() != num_cells {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：dt/sigma 长度与场不一致".to_string(),
        ));
    }
    if dt.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || sigma.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：dt 须为正且 sigma 非负".to_string(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：epsilon 须为正".to_string(),
        ));
    }
    Ok(())
}

fn finite_difference_epsilon(
    base: &ConservedFields,
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

pub(super) fn conserved_component_scales(state: &crate::physics::ConservedState) -> [Real; 5] {
    [
        state.density.abs().max(1.0),
        state.momentum[0].abs().max(state.density.abs()).max(1.0),
        state.momentum[1].abs().max(state.density.abs()).max(1.0),
        state.momentum[2].abs().max(state.density.abs()).max(1.0),
        state.total_energy.abs().max(1.0),
    ]
}

fn assign_perturbed_fields(
    out: &mut ConservedFields,
    base: &ConservedFields,
    direction: &[Real],
    epsilon: Real,
) -> Result<()> {
    let n = base.num_cells();
    ensure_vector_len(direction, n * CONSERVED_COMPONENTS_3D, "gmres direction")?;
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        out.density.values_mut()[cell] = base.density.values()[cell] + epsilon * direction[offset];
        out.momentum_x.values_mut()[cell] =
            base.momentum_x.values()[cell] + epsilon * direction[offset + 1];
        out.momentum_y.values_mut()[cell] =
            base.momentum_y.values()[cell] + epsilon * direction[offset + 2];
        out.momentum_z.values_mut()[cell] =
            base.momentum_z.values()[cell] + epsilon * direction[offset + 3];
        out.total_energy.values_mut()[cell] =
            base.total_energy.values()[cell] + epsilon * direction[offset + 4];
    }
    Ok(())
}

fn assign_physical_perturbed_fields(
    out: &mut ConservedFields,
    base: &ConservedFields,
    direction: &[Real],
    epsilon: Real,
    gamma: Real,
    min_pressure: Real,
) -> Result<Real> {
    let effective =
        max_physical_vector_increment_scale(base, direction, epsilon, gamma, min_pressure)?;
    if effective <= 0.0 {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：有限差分扰动无法保持正性".to_string(),
        ));
    }
    assign_perturbed_fields(out, base, direction, effective)?;
    Ok(effective)
}

pub(crate) fn assign_delta_scaled(
    out: &mut ConservedFields,
    base: &ConservedFields,
    delta: &[Real],
    alpha: Real,
) -> Result<()> {
    let n = base.num_cells();
    ensure_vector_len(delta, n * CONSERVED_COMPONENTS_3D, "gmres delta")?;
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        out.density.values_mut()[cell] = base.density.values()[cell] + alpha * delta[offset];
        out.momentum_x.values_mut()[cell] =
            base.momentum_x.values()[cell] + alpha * delta[offset + 1];
        out.momentum_y.values_mut()[cell] =
            base.momentum_y.values()[cell] + alpha * delta[offset + 2];
        out.momentum_z.values_mut()[cell] =
            base.momentum_z.values()[cell] + alpha * delta[offset + 3];
        out.total_energy.values_mut()[cell] =
            base.total_energy.values()[cell] + alpha * delta[offset + 4];
    }
    Ok(())
}

pub(crate) fn assign_delta_limited_scaled(
    out: &mut ConservedFields,
    base: &ConservedFields,
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
        write_cell_state(out, cell, &updated);
    }
    Ok(GmresUpdateDiagnostics {
        alpha,
        limited_cells,
        min_update_scale,
    })
}

pub(crate) fn fields_are_physical(
    fields: &ConservedFields,
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

fn max_physical_vector_increment_scale(
    base: &ConservedFields,
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

pub(super) fn component_basis_increment(component: usize) -> [Real; CONSERVED_COMPONENTS_3D] {
    let mut increment = [0.0; CONSERVED_COMPONENTS_3D];
    increment[component] = 1.0;
    increment
}

fn write_cell_state(
    fields: &mut ConservedFields,
    cell: usize,
    state: &crate::physics::ConservedState,
) {
    fields.density.values_mut()[cell] = state.density;
    fields.momentum_x.values_mut()[cell] = state.momentum[0];
    fields.momentum_y.values_mut()[cell] = state.momentum[1];
    fields.momentum_z.values_mut()[cell] = state.momentum[2];
    fields.total_energy.values_mut()[cell] = state.total_energy;
}

fn residual_to_vector(residual: &ConservedResidual) -> Vec<Real> {
    let n = residual.num_cells();
    let mut out = vec![0.0; n * CONSERVED_COMPONENTS_3D];
    for cell in 0..n {
        let offset = cell * CONSERVED_COMPONENTS_3D;
        out[offset] = residual.density.values()[cell];
        out[offset + 1] = residual.momentum_x.values()[cell];
        out[offset + 2] = residual.momentum_y.values()[cell];
        out[offset + 3] = residual.momentum_z.values()[cell];
        out[offset + 4] = residual.total_energy.values()[cell];
    }
    out
}

fn residual_difference_at(
    residual: &ConservedResidual,
    base: &ConservedResidual,
    cell: usize,
    epsilon: Real,
) -> [Real; CONSERVED_COMPONENTS_3D] {
    [
        (residual.density.values()[cell] - base.density.values()[cell]) / epsilon,
        (residual.momentum_x.values()[cell] - base.momentum_x.values()[cell]) / epsilon,
        (residual.momentum_y.values()[cell] - base.momentum_y.values()[cell]) / epsilon,
        (residual.momentum_z.values()[cell] - base.momentum_z.values()[cell]) / epsilon,
        (residual.total_energy.values()[cell] - base.total_energy.values()[cell]) / epsilon,
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

fn zero_conserved_fields(num_cells: usize) -> Result<ConservedFields> {
    ConservedFields::uniform(
        num_cells,
        crate::physics::ConservedState {
            density: 0.0,
            momentum: [0.0; 3],
            total_energy: 0.0,
        },
    )
}

fn elapsed_ms(start: Instant) -> Real {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
#[path = "gmres_implicit_3d_tests.rs"]
mod tests;
