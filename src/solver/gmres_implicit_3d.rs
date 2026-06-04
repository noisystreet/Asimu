//! 3D 可压缩残差的 matrix-free GMRES 隐式线性化。

use crate::core::Real;
use crate::discretization::InviscidFluxConfig;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, primitive_from_conserved};
use crate::linalg::{
    GmresConfig, GmresReport, GmresSolver, LinearOperator, LusgsDiagonalPreconditioner,
};
use crate::physics::IdealGasEoS;

use super::{CompressibleAdvanceContext3d, CompressibleEulerSolver};

const CONSERVED_COMPONENTS_3D: usize = 5;

/// GMRES 隐式更新结果。
#[derive(Debug, Clone, PartialEq)]
pub struct GmresImplicitDelta {
    pub delta: Vec<Real>,
    pub report: GmresReport,
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
}

pub(crate) fn apply_delta_with_line_search(
    fields: &mut ConservedFields,
    stage: &mut ConservedFields,
    base: &ConservedFields,
    delta: &GmresImplicitDelta,
    eos: &IdealGasEoS,
    p_floor: Real,
) -> Result<Real> {
    const MIN_ALPHA: Real = 1.0 / 1024.0;
    let mut alpha = 1.0;
    loop {
        assign_delta_scaled(stage, base, &delta.delta, alpha)?;
        if fields_are_physical(stage, eos, p_floor) {
            fields.copy_from(stage)?;
            return Ok(alpha);
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
        validate_gmres_inputs(fields.num_cells(), dt, sigma, config.epsilon)?;
        let inviscid = self.config.inviscid;
        let mut base_residual = ConservedResidual::zeros(fields.num_cells())?;
        self.rhs_context_3d(ctx, &inviscid, p_floor)
            .run(fields, &mut base_residual)?;
        let rhs = residual_to_vector(&base_residual);
        let precond = LusgsDiagonalPreconditioner::from_lusgs_diagonal(
            dt,
            sigma,
            self.config.lu_sgs.omega,
            CONSERVED_COMPONENTS_3D,
        )?;
        let mut op = MatrixFreeResidualOperator3d {
            solver: self,
            ctx,
            base: fields,
            base_residual: &base_residual,
            inviscid: &inviscid,
            dt,
            p_floor,
            epsilon_rel: config.epsilon,
            perturbed: zero_conserved_fields(fields.num_cells())?,
            perturbed_residual: ConservedResidual::zeros(fields.num_cells())?,
        };
        let mut delta = vec![0.0; rhs.len()];
        let report = GmresSolver::new(config.gmres)?.solve(&mut op, &precond, &rhs, &mut delta)?;
        Ok(GmresImplicitDelta { delta, report })
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
        let eps = finite_difference_epsilon(x, self.epsilon_rel)?;
        assign_perturbed_fields(&mut self.perturbed, self.base, x, eps)?;
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

fn finite_difference_epsilon(direction: &[Real], epsilon_rel: Real) -> Result<Real> {
    let norm = direction.iter().map(|v| v * v).sum::<Real>().sqrt();
    if !norm.is_finite() {
        return Err(AsimuError::Solver(
            "GMRES 隐式更新：方向向量含非有限值".to_string(),
        ));
    }
    Ok(epsilon_rel / norm.max(1.0))
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

pub(crate) fn fields_are_physical(
    fields: &ConservedFields,
    eos: &crate::physics::IdealGasEoS,
    min_pressure: Real,
) -> bool {
    (0..fields.num_cells()).all(|cell| {
        let Ok(state) = fields.cell_state(cell) else {
            return false;
        };
        let Ok(prim) = primitive_from_conserved(eos, &state) else {
            return false;
        };
        prim.density.is_finite()
            && prim.density > 0.0
            && prim.pressure.is_finite()
            && prim.pressure > min_pressure
            && prim.velocity.iter().all(|v| v.is_finite())
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ConservedFields;
    use crate::physics::ConservedState;

    #[test]
    fn residual_vector_uses_cell_major_component_order() {
        let mut r = ConservedResidual::zeros(2).expect("r");
        r.density.values_mut()[1] = 1.0;
        r.momentum_x.values_mut()[1] = 2.0;
        r.momentum_y.values_mut()[1] = 3.0;
        r.momentum_z.values_mut()[1] = 4.0;
        r.total_energy.values_mut()[1] = 5.0;
        let v = residual_to_vector(&r);
        assert_eq!(&v[5..10], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn perturbation_assigns_all_conserved_components() {
        let state = ConservedState {
            density: 1.0,
            momentum: [2.0, 3.0, 4.0],
            total_energy: 10.0,
        };
        let base = ConservedFields::uniform(1, state).expect("base");
        let mut out = base.clone();
        assign_perturbed_fields(&mut out, &base, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.1).expect("perturb");
        assert!((out.density.values()[0] - 1.1).abs() < 1.0e-12);
        assert!((out.momentum_x.values()[0] - 2.2).abs() < 1.0e-12);
        assert!((out.momentum_y.values()[0] - 3.3).abs() < 1.0e-12);
        assert!((out.momentum_z.values()[0] - 4.4).abs() < 1.0e-12);
        assert!((out.total_energy.values()[0] - 10.5).abs() < 1.0e-12);
    }

    #[test]
    fn scaled_delta_assigns_all_conserved_components() {
        let state = ConservedState {
            density: 1.0,
            momentum: [2.0, 3.0, 4.0],
            total_energy: 10.0,
        };
        let base = ConservedFields::uniform(1, state).expect("base");
        let mut out = base.clone();
        assign_delta_scaled(&mut out, &base, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.25).expect("delta");
        assert!((out.density.values()[0] - 1.25).abs() < 1.0e-12);
        assert!((out.momentum_x.values()[0] - 2.5).abs() < 1.0e-12);
        assert!((out.momentum_y.values()[0] - 3.75).abs() < 1.0e-12);
        assert!((out.momentum_z.values()[0] - 5.0).abs() < 1.0e-12);
        assert!((out.total_energy.values()[0] - 11.25).abs() < 1.0e-12);
    }

    #[test]
    fn validates_gmres_timestep_inputs() {
        assert!(validate_gmres_inputs(2, &[0.1, 0.2], &[1.0, 2.0], 1.0e-7).is_ok());
        assert!(validate_gmres_inputs(2, &[0.1], &[1.0, 2.0], 1.0e-7).is_err());
        assert!(validate_gmres_inputs(1, &[0.0], &[1.0], 1.0e-7).is_err());
        assert!(validate_gmres_inputs(1, &[0.1], &[-1.0], 1.0e-7).is_err());
        assert!(validate_gmres_inputs(1, &[0.1], &[1.0], 0.0).is_err());
    }
}
