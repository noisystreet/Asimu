//! 非结构 3D 可压缩 typed matrix-free GMRES 隐式线性化（CPU 预条件：scalar / cell_block / lusgs_sweep）。

use std::time::Instant;

use super::gmres_block_preconditioner_unstructured::{
    UnstructuredBlockLusgsPreconditioner, UnstructuredCellBlockPreconditionerBuild,
    build_block_lusgs_preconditioner_unstructured, build_cell_block_preconditioner_unstructured,
};
use super::gmres_lusgs_sweep_preconditioner_unstructured::{
    LusgsSweepUnstructuredGmresPreconditioner, LusgsSweepUnstructuredGmresPreconditionerBuild,
};
use super::{
    UnstructuredComputeBackend, UnstructuredRunEnvTyped, UnstructuredStepWorkTyped,
    UnstructuredTypedRhsWork, assemble_unstructured_typed_rhs,
};
use crate::core::{ComputeFloat, ComputePrecision, Real, elapsed_ms};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::linalg::{
    CellBlockDiagonalPreconditioner, GmresSolver, LinearOperator, LusgsDiagonalPreconditioner,
    Preconditioner,
};
use crate::physics::ConservedState;
use crate::solver::compressible::gmres_implicit_3d::gmres_implicit_typed_common::{
    assign_physical_perturbed_fields_typed, ensure_vector_len, finite_difference_epsilon_typed,
    residual_difference_at_typed, residual_to_vector_typed,
};
use crate::solver::compressible::gmres_implicit_3d::{
    CONSERVED_COMPONENTS_3D, GmresImplicitConfig, GmresImplicitDelta, GmresImplicitDiagnostics,
    GmresImplicitTiming, GmresPreconditionerKind, validate_gmres_inputs,
};

pub(crate) struct UnstructuredGmresSolveResult {
    pub delta: GmresImplicitDelta,
}

enum UnstructuredGmresPreconditioner {
    Scalar(LusgsDiagonalPreconditioner),
    CellBlock(CellBlockDiagonalPreconditioner),
    LusgsSweep(Box<LusgsSweepUnstructuredGmresPreconditioner>),
    BlockLusgs(Box<UnstructuredBlockLusgsPreconditioner>),
}

impl Preconditioner for UnstructuredGmresPreconditioner {
    fn dimension(&self) -> usize {
        match self {
            Self::Scalar(p) => p.dimension(),
            Self::CellBlock(p) => p.dimension(),
            Self::LusgsSweep(p) => p.dimension(),
            Self::BlockLusgs(p) => p.dimension(),
        }
    }

    fn apply(&mut self, rhs: &[Real], out: &mut [Real]) -> Result<()> {
        match self {
            Self::Scalar(p) => p.apply(rhs, out),
            Self::CellBlock(p) => p.apply(rhs, out),
            Self::LusgsSweep(p) => p.apply(rhs, out),
            Self::BlockLusgs(p) => p.apply(rhs, out),
        }
    }
}

pub(crate) fn solve_gmres_implicit_delta_unstructured_typed<
    T: ComputeFloat + UnstructuredComputeBackend,
>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    fields: &ConservedFieldsT<T>,
    dt: &[Real],
    sigma: &[Real],
    p_floor: Real,
    config: GmresImplicitConfig,
) -> Result<UnstructuredGmresSolveResult> {
    let total_start = Instant::now();
    validate_gmres_inputs(fields.num_cells(), dt, sigma, config.epsilon)?;
    let mut base_residual = ConservedResidualT::<T>::zeros(fields.num_cells())?;
    let base_residual_start = Instant::now();
    evaluate_unstructured_rhs(env, work, fields, &mut base_residual, p_floor)?;
    let base_residual_ms = elapsed_ms(base_residual_start);
    let base_residual_rms = base_residual.density_rms_norm();
    let rhs = residual_to_vector_typed(&base_residual);
    let preconditioner_kind = config.preconditioner;
    let preconditioner_start = Instant::now();
    let mut precond =
        build_unstructured_gmres_preconditioner(UnstructuredGmresPreconditionerBuild {
            env,
            work,
            fields,
            dt,
            sigma,
            p_floor,
            config,
            lu_sgs: env.config.solver.config.lu_sgs,
        })?;
    let preconditioner_build_ms = elapsed_ms(preconditioner_start);
    let zero_state = ConservedState {
        density: 0.0,
        momentum: [0.0; 3],
        total_energy: 0.0,
    };
    let mut diagnostics = GmresImplicitDiagnostics::new(preconditioner_kind);
    let mut op = MatrixFreeUnstructuredResidualOperatorTyped {
        env,
        work,
        base: fields,
        base_residual: &base_residual,
        dt,
        p_floor,
        epsilon_rel: config.epsilon,
        diagnostics: GmresImplicitDiagnostics::new(preconditioner_kind),
        perturbed: ConservedFieldsT::<T>::uniform(fields.num_cells(), zero_state)?,
        perturbed_residual: ConservedResidualT::<T>::zeros(fields.num_cells())?,
    };
    let mut delta = vec![0.0; rhs.len()];
    let linear_solve_start = Instant::now();
    let report = GmresSolver::new(config.gmres)?.solve(&mut op, &mut precond, &rhs, &mut delta)?;
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
    Ok(UnstructuredGmresSolveResult {
        delta: GmresImplicitDelta {
            delta,
            report,
            base_residual_rms,
            diagnostics,
        },
    })
}

struct UnstructuredGmresPreconditionerBuild<'a, 'b, T: ComputeFloat> {
    env: &'a UnstructuredRunEnvTyped<'a>,
    work: &'b mut UnstructuredStepWorkTyped<T>,
    fields: &'a ConservedFieldsT<T>,
    dt: &'a [Real],
    sigma: &'a [Real],
    p_floor: Real,
    config: GmresImplicitConfig,
    lu_sgs: crate::solver::time::LuSgsConfig,
}

fn build_unstructured_gmres_preconditioner<T: ComputeFloat>(
    params: UnstructuredGmresPreconditionerBuild<'_, '_, T>,
) -> Result<UnstructuredGmresPreconditioner> {
    let UnstructuredGmresPreconditionerBuild {
        env,
        work,
        fields,
        dt,
        sigma,
        p_floor,
        config,
        lu_sgs,
    } = params;
    Ok(match config.preconditioner {
        GmresPreconditionerKind::ScalarDiagonal => UnstructuredGmresPreconditioner::Scalar(
            LusgsDiagonalPreconditioner::from_lusgs_diagonal(
                dt,
                sigma,
                lu_sgs.omega,
                CONSERVED_COMPONENTS_3D,
            )?,
        ),
        GmresPreconditionerKind::CellBlockDiagonal => {
            let fields_f64 = fields.cast_real()?;
            let mut primitives_f64 = work.primitives.cast_real()?;
            UnstructuredGmresPreconditioner::CellBlock(
                build_cell_block_preconditioner_unstructured(
                    UnstructuredCellBlockPreconditionerBuild {
                        mesh: env.config.mesh,
                        eos: env.config.eos,
                        patches: env.config.patches,
                        topology: &work.mesh_cache.face_topology,
                        ghosts: &work.ghosts,
                        exec: &work.exec,
                        fields: &fields_f64,
                        primitives: &mut primitives_f64,
                        inviscid: env.config.inviscid,
                        viscous: env.config.viscous,
                        incidence: &work.mesh_cache.lsq_rhs_incidence,
                        solver_order: &work.mesh_cache.solver_order,
                        dt,
                        p_floor,
                        epsilon_rel: config.epsilon,
                    },
                )?,
            )
        }
        GmresPreconditionerKind::LusgsSweep => {
            if T::PRECISION != ComputePrecision::F64 {
                return Err(AsimuError::Config(
                    "非结构 gmres_preconditioner = \"lusgs_sweep\" 暂仅支持 compute_precision = \"f64\""
                        .to_string(),
                ));
            }
            UnstructuredGmresPreconditioner::LusgsSweep(Box::new(
                LusgsSweepUnstructuredGmresPreconditioner::new(
                    LusgsSweepUnstructuredGmresPreconditionerBuild {
                        eos: *env.config.eos,
                        couplings: work.lusgs_couplings.clone(),
                        base: fields.cast_real()?,
                        frozen_primitives: work.primitives.cast_real()?,
                        dt: dt.to_vec(),
                        sigma: sigma.to_vec(),
                        volumes: work.volumes.clone(),
                        solver_order: work.mesh_cache.solver_order.clone(),
                        solver_rank: work.mesh_cache.solver_rank.clone(),
                        omega: lu_sgs.omega,
                        backward_damping: lu_sgs.sweep_backward_damping,
                        inv_dt_phys: 0.0,
                    },
                )?,
            ))
        }
        GmresPreconditionerKind::BlockLusgs => {
            if T::PRECISION != ComputePrecision::F64 {
                return Err(AsimuError::Config(
                    "非结构 gmres_preconditioner = \"block_lusgs\" 暂仅支持 compute_precision = \"f64\""
                        .to_string(),
                ));
            }
            let fields_f64 = fields.cast_real()?;
            let mut primitives_f64 = work.primitives.cast_real()?;
            UnstructuredGmresPreconditioner::BlockLusgs(Box::new(
                build_block_lusgs_preconditioner_unstructured(
                    UnstructuredCellBlockPreconditionerBuild {
                        mesh: env.config.mesh,
                        eos: env.config.eos,
                        patches: env.config.patches,
                        topology: &work.mesh_cache.face_topology,
                        ghosts: &work.ghosts,
                        exec: &work.exec,
                        fields: &fields_f64,
                        primitives: &mut primitives_f64,
                        inviscid: env.config.inviscid,
                        viscous: env.config.viscous,
                        incidence: &work.mesh_cache.lsq_rhs_incidence,
                        solver_order: &work.mesh_cache.solver_order,
                        dt,
                        p_floor,
                        epsilon_rel: config.epsilon,
                    },
                )?,
            ))
        }
    })
}

struct MatrixFreeUnstructuredResidualOperatorTyped<'a, T: ComputeFloat> {
    env: &'a UnstructuredRunEnvTyped<'a>,
    work: &'a mut UnstructuredStepWorkTyped<T>,
    base: &'a ConservedFieldsT<T>,
    base_residual: &'a ConservedResidualT<T>,
    dt: &'a [Real],
    p_floor: Real,
    epsilon_rel: Real,
    diagnostics: GmresImplicitDiagnostics,
    perturbed: ConservedFieldsT<T>,
    perturbed_residual: ConservedResidualT<T>,
}

impl<T: ComputeFloat + UnstructuredComputeBackend> LinearOperator
    for MatrixFreeUnstructuredResidualOperatorTyped<'_, T>
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
            self.env.config.eos.gamma,
            self.p_floor,
        )?;
        self.record_perturbation_scale(eps / requested_eps);
        evaluate_unstructured_rhs(
            self.env,
            self.work,
            &self.perturbed,
            &mut self.perturbed_residual,
            self.p_floor,
        )?;
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

impl<T: ComputeFloat> MatrixFreeUnstructuredResidualOperatorTyped<'_, T> {
    fn record_perturbation_scale(&mut self, scale: Real) {
        self.diagnostics.perturbation_evals += 1;
        self.diagnostics.min_perturbation_scale =
            self.diagnostics.min_perturbation_scale.min(scale);
        if scale < 1.0 - 1.0e-12 {
            self.diagnostics.perturbation_limited_evals += 1;
        }
    }
}

fn evaluate_unstructured_rhs<T: ComputeFloat + UnstructuredComputeBackend>(
    env: &UnstructuredRunEnvTyped<'_>,
    work: &mut UnstructuredStepWorkTyped<T>,
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
    p_floor: Real,
) -> Result<()> {
    let mut rhs_work = UnstructuredTypedRhsWork {
        ghosts: &mut work.ghosts,
        primitives: &mut work.primitives,
        gradients: &mut work.gradients,
        viscous_scratch: &mut work.viscous_scratch,
        viscous_grad_scratch_f32: &mut work.viscous_grad_scratch_f32,
        mesh_cache: &work.mesh_cache,
        exec: &mut work.exec,
    };
    assemble_unstructured_typed_rhs(env, &mut rhs_work, fields, residual, true, p_floor)
}
