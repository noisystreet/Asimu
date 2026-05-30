//! 可压缩无粘 Euler 显式求解（RK4 + FVM 残差）。
//!
//! 理论：[`docs/theory/time_integration.md`](../../docs/theory/time_integration.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{
    BoundaryGhostBuffer, apply_compressible_boundary_conditions, assemble_inviscid_residual_1d,
    assemble_inviscid_residual_3d,
};
use crate::error::Result;
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::{BoundaryMesh3d, StructuredMesh1d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, PrimitiveState};
use crate::solver::state::SolverState;
use crate::solver::time::{
    Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrator, rk4_step,
};

/// 显式可压缩 Euler 瞬态求解器配置。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleEulerConfig {
    pub time: RungeKutta4Config,
    pub inviscid: crate::discretization::InviscidFluxConfig,
    pub cfl: Real,
}

impl Default for CompressibleEulerConfig {
    fn default() -> Self {
        Self {
            time: RungeKutta4Config::default(),
            inviscid: crate::discretization::InviscidFluxConfig::default(),
            cfl: 0.4,
        }
    }
}

/// 单步推进结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleStepInfo {
    pub dt: Real,
    pub physical_time: Real,
    pub step: u64,
    pub residual_l2: Real,
    pub is_final: bool,
}

/// 3D 单步推进上下文（减少参数个数）。
pub struct CompressibleAdvanceContext3d<'a> {
    pub mesh: &'a dyn BoundaryMesh3d,
    pub structured: &'a StructuredMesh3d,
    pub patches: &'a BoundarySet,
    pub ghosts: &'a mut BoundaryGhostBuffer,
    pub eos: &'a IdealGasEoS,
    pub freestream: &'a FreestreamParams,
}

/// 1D 多步推进上下文。
pub struct CompressibleAdvanceContext1d<'a> {
    pub mesh: &'a StructuredMesh1d,
    pub boundary: crate::discretization::InviscidBoundary1d,
    pub eos: &'a IdealGasEoS,
}

/// 可压缩 Euler 显式 RK4 求解器。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleEulerSolver {
    pub config: CompressibleEulerConfig,
}

impl CompressibleEulerSolver {
    #[must_use]
    pub const fn new(config: CompressibleEulerConfig) -> Self {
        Self { config }
    }

    /// 1D 瞬态推进：每步刷新边界 ghost、装配残差、RK4 更新守恒量。
    pub fn advance_step_1d(
        &self,
        ctx: &CompressibleAdvanceContext1d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let dt = self.suggest_dt_1d(ctx.mesh, fields, ctx.eos)?;
        integrator.config.dt = dt;
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            let boundaries = ctx.boundary.resolve(u)?;
            assemble_inviscid_residual_1d(
                ctx.mesh,
                u,
                r,
                ctx.eos,
                &self.config.inviscid,
                &boundaries,
            )
        };
        rk4_step(fields, storage, dt, evaluate)?;
        let boundaries = ctx.boundary.resolve(fields)?;
        assemble_inviscid_residual_1d(
            ctx.mesh,
            fields,
            &mut storage.k1,
            ctx.eos,
            &self.config.inviscid,
            &boundaries,
        )?;
        let last_residual = storage.k1.density_l2_norm();
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_l2: last_residual,
            is_final: time_info.is_final,
        })
    }

    /// 3D 瞬态推进：每 RK 阶段重算边界 ghost 与残差。
    pub fn advance_step_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let dt = self.suggest_dt_3d(ctx.structured, fields, ctx.eos)?;
        integrator.config.dt = dt;
        let inviscid = self.config.inviscid;
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            apply_compressible_boundary_conditions(
                ctx.mesh,
                ctx.patches,
                u,
                ctx.ghosts,
                ctx.eos,
                ctx.freestream,
            )?;
            assemble_inviscid_residual_3d(
                ctx.structured,
                u,
                r,
                ctx.eos,
                &inviscid,
                ctx.patches,
                ctx.ghosts,
            )
        };
        rk4_step(fields, storage, dt, evaluate)?;
        apply_compressible_boundary_conditions(
            ctx.mesh,
            ctx.patches,
            fields,
            ctx.ghosts,
            ctx.eos,
            ctx.freestream,
        )?;
        assemble_inviscid_residual_3d(
            ctx.structured,
            fields,
            &mut storage.k1,
            ctx.eos,
            &inviscid,
            ctx.patches,
            ctx.ghosts,
        )?;
        let last_residual = storage.k1.density_l2_norm();
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_l2: last_residual,
            is_final: time_info.is_final,
        })
    }

    /// 1D 多步瞬态积分直至 `max_steps`。
    pub fn run_transient_1d(
        &self,
        ctx: &CompressibleAdvanceContext1d<'_>,
        fields: &mut ConservedFields,
    ) -> Result<Vec<CompressibleStepInfo>> {
        let mut storage = Rk4Storage::new(ctx.mesh.num_cells())?;
        let mut state = SolverState::default();
        let mut integrator = RungeKutta4Integrator::new(self.config.time);
        let mut history = Vec::new();
        loop {
            let info =
                self.advance_step_1d(ctx, fields, &mut storage, &mut state, &mut integrator)?;
            let is_final = info.is_final;
            history.push(info);
            if is_final {
                break;
            }
        }
        Ok(history)
    }

    fn suggest_dt_1d(
        &self,
        mesh: &StructuredMesh1d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
    ) -> Result<Real> {
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(dt);
        }
        let max_speed = max_wave_speed(fields, eos)?;
        crate::solver::time::suggested_dt_cfl(mesh.dx(), max_speed, self.config.cfl)
    }

    fn suggest_dt_3d(
        &self,
        mesh: &StructuredMesh3d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
    ) -> Result<Real> {
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(dt);
        }
        let min_spacing = mesh.cell_dx().min(mesh.cell_dy()).min(mesh.cell_dz());
        let max_speed = max_wave_speed(fields, eos)?;
        crate::solver::time::suggested_dt_cfl(min_spacing, max_speed, self.config.cfl)
    }
}

fn positive_fixed_dt(dt: Real) -> Option<Real> {
    if dt > 0.0 { Some(dt) } else { None }
}

/// 全场最大波速 \(|u| + a\)（CFL 估计）。
pub fn max_wave_speed(fields: &ConservedFields, eos: &IdealGasEoS) -> Result<Real> {
    let mut max_speed = Real::EPSILON;
    for i in 0..fields.num_cells() {
        let prim = fields.primitive_at(i, eos)?;
        max_speed = max_speed.max(wave_speed_primitive(&prim, eos)?);
    }
    Ok(max_speed)
}

fn wave_speed_primitive(prim: &PrimitiveState, eos: &IdealGasEoS) -> Result<Real> {
    let speed = (prim.velocity[0] * prim.velocity[0]
        + prim.velocity[1] * prim.velocity[1]
        + prim.velocity[2] * prim.velocity[2])
        .sqrt();
    Ok(speed + eos.sound_speed(prim.temperature)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::physics::ConservedState;

    #[test]
    fn uniform_1d_field_remains_stationary_over_steps() {
        let mesh = StructuredMesh1d::new("line", 8, 0.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let mut fields = ConservedFields::from_freestream(8, &eos, &FreestreamParams::default())
            .expect("fields");
        let reference = fields.clone();
        let ctx = CompressibleAdvanceContext1d {
            mesh: &mesh,
            boundary: crate::discretization::InviscidBoundary1d::ZeroGradient,
            eos: &eos,
        };
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: RungeKutta4Config {
                dt: 1.0e-5,
                max_steps: 2,
            },
            ..CompressibleEulerConfig::default()
        });
        solver.run_transient_1d(&ctx, &mut fields).expect("run");
        for i in 0..mesh.num_cells() {
            assert!(approx_eq(
                fields.density.values()[i],
                reference.density.values()[i],
                1.0e-8,
            ));
        }
    }

    #[test]
    fn sod_like_disturbance_evolve_with_rk4() {
        let mesh = StructuredMesh1d::new("sod", 16, 0.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::new(1.4, 1.0).expect("eos");
        let left = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 1.0,
                velocity: [0.0, 0.0, 0.0],
                pressure: 1.0,
                temperature: 1.0,
            },
        )
        .expect("left");
        let right = ConservedState::from_primitive(
            &eos,
            &PrimitiveState {
                density: 0.125,
                velocity: [0.0, 0.0, 0.0],
                pressure: 0.1,
                temperature: 1.0,
            },
        )
        .expect("right");
        let mut fields = ConservedFields::uniform(mesh.num_cells(), left).expect("fields");
        let mid = mesh.num_cells() / 2;
        for i in mid..mesh.num_cells() {
            fields.density.values_mut()[i] = right.density;
            fields.total_energy.values_mut()[i] = right.total_energy;
        }
        let rho_before = fields.density.values()[mid - 1];
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: RungeKutta4Config {
                dt: 2.0e-4,
                max_steps: 5,
            },
            ..CompressibleEulerConfig::default()
        });
        solver
            .run_transient_1d(
                &CompressibleAdvanceContext1d {
                    mesh: &mesh,
                    boundary: crate::discretization::InviscidBoundary1d::ZeroGradient,
                    eos: &eos,
                },
                &mut fields,
            )
            .expect("run");
        assert!(fields.density.values()[mid - 1] != rho_before);
    }
}
