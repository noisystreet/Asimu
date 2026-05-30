//! 可压缩无粘 Euler 显式求解（RK4 / 一阶 Euler + FVM 残差）。
//!
//! 理论：[`docs/theory/time_integration.md`](../../docs/theory/time_integration.md)、
//! [`inviscid_flux.md`](../../docs/theory/inviscid_flux.md)

use crate::boundary::BoundarySet;
use crate::core::{Real, format_log_fixed4, format_log_sci4, log10_positive};
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
    CflSchedule, Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, TimeIntegrationScheme,
    TimeIntegrator, euler_step, euler_step_local, local_dt_cfl, min_positive_dt, rk4_step,
    rk4_step_local,
};
use tracing::info;

/// 稳态伪时间 / 瞬态物理时间。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleTimeMode {
    Steady,
    Transient,
}

/// 显式可压缩 Euler 求解器配置。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleEulerConfig {
    pub time: RungeKutta4Config,
    pub inviscid: crate::discretization::InviscidFluxConfig,
    pub cfl_schedule: CflSchedule,
    pub time_mode: CompressibleTimeMode,
    pub local_time_step: bool,
    /// 时间积分格式（`rk4` 默认；`euler` 用于排除 RK 多阶段 bug）。
    pub time_scheme: TimeIntegrationScheme,
}

impl Default for CompressibleEulerConfig {
    fn default() -> Self {
        Self {
            time: RungeKutta4Config::default(),
            inviscid: crate::discretization::InviscidFluxConfig::default(),
            cfl_schedule: CflSchedule::constant(0.4),
            time_mode: CompressibleTimeMode::Transient,
            local_time_step: false,
            time_scheme: TimeIntegrationScheme::Rk4,
        }
    }
}

/// 单步推进结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleStepInfo {
    pub dt: Real,
    pub physical_time: Real,
    pub step: u64,
    /// 全场 \(\mathrm{RMS}(\dot\rho)=\|\dot\rho\|_2/\sqrt{N}\)。
    pub residual_rms: Real,
    /// \(\log_{10}(\mathrm{RMS}(\dot\rho))\)，便于跨量级对比。
    pub residual_log10: Real,
    /// 本步使用的 CFL 数。
    pub cfl: Real,
    pub is_final: bool,
    /// 本步是否因 log₁₀(RMS(ρ̇)) ≤ `[time].tolerance` 触发早停（由算例编排层设置）。
    pub converged: bool,
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
        let cfl = self.cfl_for_step(state);
        let dt = self.suggest_dt_1d(ctx.mesh, fields, ctx.eos, cfl)?;
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
        self.advance_explicit_step(fields, storage, dt, None, evaluate, None)?;
        let boundaries = ctx.boundary.resolve(fields)?;
        assemble_inviscid_residual_1d(
            ctx.mesh,
            fields,
            &mut storage.k1,
            ctx.eos,
            &self.config.inviscid,
            &boundaries,
        )?;
        let last_residual = storage.k1.density_rms_norm();
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: last_residual,
            residual_log10: log10_positive(last_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
        })
    }

    /// 3D 推进：每 RK 阶段重算边界 ghost 与残差；支持全局/逐单元时间步。
    pub fn advance_step_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        state: &mut SolverState,
        integrator: &mut RungeKutta4Integrator,
    ) -> Result<CompressibleStepInfo> {
        let cfl = self.cfl_for_step(state);
        let p_floor = Self::positivity_pressure_floor(ctx.freestream);
        fields.enforce_positivity(ctx.eos, p_floor);
        let cell_dts = self.suggest_cell_dts_3d(ctx.structured, fields, ctx.eos, cfl, p_floor)?;
        let dt = min_positive_dt(&cell_dts);
        integrator.config.dt = dt;
        let inviscid = self.config.inviscid;
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
        let step_residual = storage.k1.density_rms_norm();
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
        let cell_dts_arg = if self.config.local_time_step {
            Some(cell_dts.as_slice())
        } else {
            None
        };
        self.advance_explicit_step(
            fields,
            storage,
            dt,
            cell_dts_arg,
            evaluate,
            Some((ctx.eos, p_floor)),
        )?;
        fields.enforce_positivity(ctx.eos, p_floor);
        let time_info = integrator.advance(state)?;
        Ok(CompressibleStepInfo {
            dt: time_info.dt,
            physical_time: time_info.physical_time,
            step: time_info.step,
            residual_rms: step_residual,
            residual_log10: log10_positive(step_residual),
            cfl,
            is_final: time_info.is_final,
            converged: false,
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
            info!(
                step = info.step,
                dt = %format_log_sci4(info.dt),
                t = %format_log_sci4(info.physical_time),
                log10_residual = %format_log_fixed4(info.residual_log10),
                cfl = info.cfl,
                is_final,
                "可压缩 Euler 1D 时间步"
            );
            history.push(info);
            if is_final {
                break;
            }
        }
        Ok(history)
    }

    /// 3D 多步瞬态积分直至 `max_steps`。
    pub fn run_transient_3d(
        &self,
        ctx: &mut CompressibleAdvanceContext3d<'_>,
        fields: &mut ConservedFields,
    ) -> Result<Vec<CompressibleStepInfo>> {
        let mut storage = Rk4Storage::new(ctx.structured.num_cells())?;
        let mut state = SolverState::default();
        let mut integrator = RungeKutta4Integrator::new(self.config.time);
        let mut history = Vec::new();
        loop {
            let info =
                self.advance_step_3d(ctx, fields, &mut storage, &mut state, &mut integrator)?;
            let is_final = info.is_final;
            info!(
                step = info.step,
                dt = %format_log_sci4(info.dt),
                t = %format_log_sci4(info.physical_time),
                log10_residual = %format_log_fixed4(info.residual_log10),
                cfl = info.cfl,
                is_final,
                "可压缩 Euler 3D 时间步"
            );
            history.push(info);
            if is_final {
                break;
            }
        }
        Ok(history)
    }

    fn cfl_for_step(&self, state: &SolverState) -> Real {
        let next_step = state.time_step.saturating_add(1);
        self.config
            .cfl_schedule
            .at_step(next_step, self.config.time.max_steps)
    }

    fn positivity_pressure_floor(freestream: &FreestreamParams) -> Real {
        crate::field::positivity_pressure_floor(freestream.pressure)
    }

    fn advance_explicit_step<F>(
        &self,
        fields: &mut ConservedFields,
        storage: &mut Rk4Storage,
        dt_global: Real,
        cell_dts: Option<&[Real]>,
        evaluate_rhs: F,
        positivity: Option<(&IdealGasEoS, Real)>,
    ) -> Result<()>
    where
        F: FnMut(&ConservedFields, &mut ConservedResidual) -> Result<()>,
    {
        let (eos, min_pressure) = match positivity {
            Some((eos, p)) => (Some(eos), p),
            None => (None, 1.0e-6),
        };
        match (self.config.time_scheme, cell_dts) {
            (TimeIntegrationScheme::Rk4, Some(dt)) => {
                rk4_step_local(fields, storage, dt, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::Rk4, None) => {
                rk4_step(fields, storage, dt_global, evaluate_rhs)
            }
            (TimeIntegrationScheme::Euler, Some(dt)) => {
                euler_step_local(fields, storage, dt, evaluate_rhs, eos, min_pressure)
            }
            (TimeIntegrationScheme::Euler, None) => {
                euler_step(fields, storage, dt_global, evaluate_rhs, eos, min_pressure)
            }
        }
    }

    fn suggest_dt_1d(
        &self,
        mesh: &StructuredMesh1d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
        cfl: Real,
    ) -> Result<Real> {
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(dt);
        }
        let max_speed = max_wave_speed(fields, eos, 1.0e-6)?;
        crate::solver::time::suggested_dt_cfl(mesh.dx(), max_speed, cfl)
    }

    fn suggest_dt_3d(
        &self,
        mesh: &StructuredMesh3d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
        cfl: Real,
        min_pressure: Real,
    ) -> Result<Real> {
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(dt);
        }
        let min_spacing = mesh.min_positive_spacing()?;
        let max_speed = max_wave_speed(fields, eos, min_pressure)?;
        crate::solver::time::suggested_dt_cfl(min_spacing, max_speed, cfl)
    }

    fn suggest_cell_dts_3d(
        &self,
        mesh: &StructuredMesh3d,
        fields: &ConservedFields,
        eos: &IdealGasEoS,
        cfl: Real,
        min_pressure: Real,
    ) -> Result<Vec<Real>> {
        let n = fields.num_cells();
        if let Some(dt) = positive_fixed_dt(self.config.time.dt) {
            return Ok(vec![dt; n]);
        }
        if self.config.local_time_step {
            let lengths = mesh.cell_cfl_lengths()?;
            let speeds = cell_wave_speeds(fields, eos, min_pressure)?;
            return local_dt_cfl(&lengths, &speeds, cfl);
        }
        Ok(vec![
            self.suggest_dt_3d(
                mesh,
                fields,
                eos,
                cfl,
                min_pressure
            )?;
            n
        ])
    }
}

fn positive_fixed_dt(dt: Real) -> Option<Real> {
    if dt > 0.0 { Some(dt) } else { None }
}

/// 全场最大波速 \(|u| + a\)（CFL 估计）。
pub fn max_wave_speed(
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<Real> {
    let mut max_speed = Real::EPSILON;
    for i in 0..fields.num_cells() {
        let prim = crate::field::primitive_from_conserved_relaxed(
            eos,
            &fields.cell_state(i)?,
            min_pressure,
        )?;
        max_speed = max_speed.max(wave_speed_primitive(&prim, eos)?);
    }
    Ok(max_speed)
}

fn cell_wave_speeds(
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: Real,
) -> Result<Vec<Real>> {
    let mut speeds = Vec::with_capacity(fields.num_cells());
    for i in 0..fields.num_cells() {
        let prim = crate::field::primitive_from_conserved_relaxed(
            eos,
            &fields.cell_state(i)?,
            min_pressure,
        )?;
        speeds.push(wave_speed_primitive(&prim, eos)?);
    }
    Ok(speeds)
}

fn wave_speed_primitive(prim: &PrimitiveState, eos: &IdealGasEoS) -> Result<Real> {
    let rho = prim.density.max(1.0e-12);
    let pressure = prim.pressure.max(1.0e-6);
    let speed = (prim.velocity[0] * prim.velocity[0]
        + prim.velocity[1] * prim.velocity[1]
        + prim.velocity[2] * prim.velocity[2])
        .sqrt();
    Ok(speed + (eos.gamma * pressure / rho).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::physics::ConservedState;
    use std::collections::HashSet;

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

    /// 圆柱网格、无边界 patch：均匀来流时间推进（`--nocapture` 打印逐步指标）。
    #[test]
    fn cylinder_uniform_freestream_no_bc_time_advance_when_present() {
        use std::path::PathBuf;

        use crate::boundary::BoundarySet;
        use crate::discretization::{BoundaryGhostBuffer, InviscidFluxConfig};
        use crate::io::{CaseMesh, load_case};
        use crate::solver::time::{CflSchedule, RungeKutta4Integrator};

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        let case = load_case(&case_path).expect("load case");
        let CaseMesh::Structured3d(mesh) = &case.mesh else {
            panic!("expected 3d");
        };
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("freestream");
        let mut fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let rho0 = fs.pressure / (eos.gas_constant * fs.temperature);
        let empty_bc = BoundarySet::default();
        let mut ghosts = BoundaryGhostBuffer::new();
        let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
        let mut state = SolverState::default();
        let steps: u64 = 200;
        let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
            dt: 0.0,
            max_steps: steps,
        });
        let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
            time: RungeKutta4Config {
                dt: 0.0,
                max_steps: steps,
            },
            inviscid: InviscidFluxConfig::roe_first_order(),
            cfl_schedule: CflSchedule {
                initial: 0.01,
                max: 0.1,
                ramp_steps: Some(500),
            },
            time_mode: CompressibleTimeMode::Steady,
            local_time_step: true,
            ..CompressibleEulerConfig::default()
        });
        let mut ctx = CompressibleAdvanceContext3d {
            mesh,
            structured: mesh,
            patches: &empty_bc,
            ghosts: &mut ghosts,
            eos: &eos,
            freestream: &fs,
        };

        eprintln!("=== 圆柱网格 无 BC 均匀来流 时间推进 ({} 步) ===", steps);
        eprintln!("来流 rho_ref = {rho0:.6e}");

        let report_steps = [1_u64, 10, 50, 100, 200];
        for _ in 0..steps {
            let info = solver
                .advance_step_3d(
                    &mut ctx,
                    &mut fields,
                    &mut storage,
                    &mut state,
                    &mut integrator,
                )
                .expect("step");
            if report_steps.contains(&info.step) {
                let rho = fields.density.values();
                let rmin = rho.iter().copied().fold(f64::INFINITY, f64::min);
                let rmax = rho.iter().copied().fold(0.0_f64, f64::max);
                let center = rho[mesh.cell_index(mesh.nx / 2, mesh.ny / 2, 0)];
                eprintln!(
                    "step {:4}: log10_res={:.4} rho=[{:.6e}, {:.6e}] center={:.6e}",
                    info.step, info.residual_log10, rmin, rmax, center
                );
            }
        }

        let rho = fields.density.values();
        let rmax = rho.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            rmax < rho0 * 100.0,
            "无 BC 推进后 rho_max={rmax:.6e} 异常 (>100×来流)"
        );
    }

    /// 圆柱网格：仅内部面通量 + 边界单元 RHS 清零（不参与更新）。
    #[test]
    fn cylinder_uniform_freestream_interior_only_advance_when_present() {
        use std::path::PathBuf;

        use crate::boundary::BoundarySet;
        use crate::core::log10_positive;
        use crate::discretization::{
            BoundaryGhostBuffer, InviscidFluxConfig, assemble_inviscid_residual_3d,
        };
        use crate::field::ConservedResidual;
        use crate::io::{CaseMesh, load_case};
        use crate::solver::time::{CflSchedule, RungeKutta4Integrator};

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        let case = load_case(&case_path).expect("load case");
        let CaseMesh::Structured3d(mesh) = &case.mesh else {
            panic!("expected 3d");
        };
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("freestream");
        let mut fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let reference = fields.clone();
        let rho0 = fs.pressure / (eos.gas_constant * fs.temperature);
        let boundary_cells = structured_3d_boundary_cell_indices(mesh);
        let boundary_set: HashSet<usize> = boundary_cells.iter().copied().collect();
        let empty_bc = BoundarySet::default();
        let ghosts = BoundaryGhostBuffer::new();
        let mut storage = Rk4Storage::new(mesh.num_cells()).expect("storage");
        let mut state = SolverState::default();
        let steps: u64 = 200;
        let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
            dt: 0.0,
            max_steps: steps,
        });
        let inviscid = InviscidFluxConfig::roe_first_order();
        let cfl_schedule = CflSchedule {
            initial: 0.01,
            max: 0.1,
            ramp_steps: Some(500),
        };

        eprintln!("=== 圆柱网格 仅内部面 + 边界单元冻结 ({} 步) ===", steps);
        eprintln!(
            "来流 rho_ref = {rho0:.6e}  边界单元数 = {} / {}",
            boundary_cells.len(),
            mesh.num_cells()
        );

        let report_steps = [1_u64, 10, 50, 100, 200];
        for _ in 0..steps {
            let cfl = cfl_schedule.at_step(state.time_step.saturating_add(1), steps);
            let lengths = mesh.cell_cfl_lengths().expect("lengths");
            let speeds = cell_wave_speeds(&fields, &eos, fs.pressure * 1.0e-3).expect("speeds");
            let cell_dts = local_dt_cfl(&lengths, &speeds, cfl).expect("dt");

            let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
                assemble_inviscid_residual_3d(mesh, u, r, &eos, &inviscid, &empty_bc, &ghosts)?;
                zero_residual_on_cells(r, &boundary_cells);
                Ok(())
            };

            rk4_step_local(
                &mut fields,
                &mut storage,
                &cell_dts,
                evaluate,
                Some(&eos),
                fs.pressure * 1.0e-3,
            )
            .expect("rk4");
            let _ = integrator.advance(&mut state).expect("advance");

            if report_steps.contains(&state.time_step) {
                let rho = fields.density.values();
                let interior_rho: Vec<f64> = rho
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !boundary_set.contains(i))
                    .map(|(_, v)| *v)
                    .collect();
                let rmin = interior_rho.iter().copied().fold(f64::INFINITY, f64::min);
                let rmax = interior_rho.iter().copied().fold(0.0_f64, f64::max);
                let center = rho[mesh.cell_index(mesh.nx / 2, mesh.ny / 2, 0)];

                evaluate(&fields, &mut storage.k1).expect("rhs");
                let int_res = interior_density_rms(&storage.k1, &boundary_set);
                let boundary_frozen = boundary_cells
                    .iter()
                    .all(|&c| fields.density.values()[c] == reference.density.values()[c]);

                eprintln!(
                    "step {:4}: log10_int_res={:.4} int_rho=[{:.6e}, {:.6e}] center={:.6e} boundary_frozen={boundary_frozen}",
                    state.time_step,
                    log10_positive(int_res),
                    rmin,
                    rmax,
                    center
                );
            }
        }

        let interior_max = fields
            .density
            .values()
            .iter()
            .enumerate()
            .filter(|(i, _)| !boundary_set.contains(i))
            .map(|(_, v)| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            interior_max < rho0 * 1.01,
            "内部单元 rho 偏离来流: max={interior_max:.6e}"
        );
    }

    /// 贴体结构化网格的边界层单元（准 2D：`nz==1` 时不计 K 面）。
    fn structured_3d_boundary_cell_indices(mesh: &StructuredMesh3d) -> Vec<usize> {
        let mut cells = Vec::new();
        let include_k = mesh.nz > 1;
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    let on_i = i == 0 || i + 1 == mesh.nx;
                    let on_j = j == 0 || j + 1 == mesh.ny;
                    let on_k = include_k && (k == 0 || k + 1 == mesh.nz);
                    if on_i || on_j || on_k {
                        cells.push(mesh.cell_index(i, j, k));
                    }
                }
            }
        }
        cells
    }

    fn zero_residual_on_cells(residual: &mut ConservedResidual, cells: &[usize]) {
        for &c in cells {
            residual.density.values_mut()[c] = 0.0;
            residual.momentum_x.values_mut()[c] = 0.0;
            residual.momentum_y.values_mut()[c] = 0.0;
            residual.momentum_z.values_mut()[c] = 0.0;
            residual.total_energy.values_mut()[c] = 0.0;
        }
    }

    fn interior_density_rms(residual: &ConservedResidual, boundary: &HashSet<usize>) -> f64 {
        let mut sum_sq = 0.0_f64;
        let mut count = 0_usize;
        for (i, &v) in residual.density.values().iter().enumerate() {
            if boundary.contains(&i) {
                continue;
            }
            sum_sq += v * v;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            (sum_sq / count as f64).sqrt()
        }
    }
}
