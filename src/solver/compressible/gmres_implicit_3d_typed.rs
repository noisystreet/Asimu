//! 3D 可压缩 typed 场 matrix-free GMRES 隐式线性化（ADR 0016 P4）。
//!
//! 线性代数向量与 GMRES 内积保持 `f64`；场存储与 RHS 装配使用 `ConservedFieldsT<T>`。

use std::time::Instant;

use crate::core::{Real, elapsed_ms};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::linalg::{GmresSolver, LinearOperator};
use crate::physics::ConservedState;

use super::super::CompressibleAdvanceContext3dTyped;
use super::super::CompressibleEulerSolver;
use super::super::gmres_block_preconditioner_3d::build_cell_block_preconditioner;
use super::super::gmres_implicit_3d::gmres_implicit_typed_common::{
    assign_physical_perturbed_fields_typed, ensure_vector_len, finite_difference_epsilon_typed,
    residual_difference_at_typed, residual_to_vector_typed,
};
use super::super::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, GmresImplicitConfig, GmresImplicitDelta, GmresImplicitDiagnostics,
    GmresImplicitPreconditioner, GmresImplicitTiming, GmresPreconditionerBuild,
    GmresPreconditionerKind, build_gmres_preconditioner, validate_gmres_inputs,
};
use crate::solver::compressible::structured_compute_backend::StructuredComputeBackend;

impl CompressibleEulerSolver {
    /// typed 场 matrix-free GMRES 隐式伪时间步。
    #[allow(private_bounds)]
    pub fn solve_gmres_implicit_delta_3d_typed<T: StructuredComputeBackend>(
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
        let mut precond = match config.preconditioner {
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
            GmresPreconditionerKind::LusgsSweep => {
                return Err(crate::error::AsimuError::Config(
                    "结构化 GMRES 暂不支持 gmres_preconditioner = \"lusgs_sweep\"（请用非结构路径）"
                        .to_string(),
                ));
            }
        };
        let preconditioner_build_ms = elapsed_ms(preconditioner_start);
        let zero_state = ConservedState {
            density: 0.0,
            momentum: [0.0; 3],
            total_energy: 0.0,
        };
        let mut diagnostics = GmresImplicitDiagnostics::new(preconditioner_kind);
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
        let report =
            GmresSolver::new(config.gmres)?.solve(&mut op, &mut precond, &rhs, &mut delta)?;
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

struct MatrixFreeResidualOperator3dTyped<'a, 'ctx, T: StructuredComputeBackend> {
    solver: &'a CompressibleEulerSolver,
    ctx: &'a mut CompressibleAdvanceContext3dTyped<'ctx, T>,
    base: &'a ConservedFieldsT<T>,
    base_residual: &'a ConservedResidualT<T>,
    inviscid: &'a crate::discretization::InviscidFluxConfig,
    dt: &'a [Real],
    p_floor: Real,
    epsilon_rel: Real,
    diagnostics: GmresImplicitDiagnostics,
    perturbed: ConservedFieldsT<T>,
    perturbed_residual: ConservedResidualT<T>,
}

impl<T: StructuredComputeBackend> LinearOperator for MatrixFreeResidualOperator3dTyped<'_, '_, T> {
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

impl<T: StructuredComputeBackend> MatrixFreeResidualOperator3dTyped<'_, '_, T> {
    fn record_perturbation_scale(&mut self, scale: Real) {
        self.diagnostics.perturbation_evals += 1;
        self.diagnostics.min_perturbation_scale =
            self.diagnostics.min_perturbation_scale.min(scale);
        if scale < 1.0 - 1.0e-12 {
            self.diagnostics.perturbation_limited_evals += 1;
        }
    }
}

pub(crate) use super::super::gmres_implicit_3d::gmres_implicit_typed_common::apply_delta_with_line_search_typed;
