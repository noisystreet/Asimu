//! 四阶 Runge-Kutta 显式时间推进。
//!
//! 理论：[`docs/theory/time_integration.md`](../../../docs/theory/time_integration.md) §3

use tracing::info_span;

use crate::core::{ComputeFloat, Real};
use crate::error::Result;
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::solver::state::SolverState;
use crate::solver::time::common::maybe_enforce_positivity;
use crate::solver::time::{TimeIntegrator, TimeMode, TimeStepInfo};

/// RK4 时间步配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RungeKutta4Config {
    pub dt: Real,
    pub max_steps: u64,
}

impl Default for RungeKutta4Config {
    fn default() -> Self {
        Self {
            dt: 1.0e-4,
            max_steps: 1_000,
        }
    }
}

/// 经典四阶 Runge-Kutta 积分器（瞬态模式）。
#[derive(Debug, Clone, PartialEq)]
pub struct RungeKutta4Integrator {
    pub config: RungeKutta4Config,
}

impl RungeKutta4Integrator {
    #[must_use]
    pub const fn new(config: RungeKutta4Config) -> Self {
        Self { config }
    }
}

impl TimeIntegrator for RungeKutta4Integrator {
    fn mode(&self) -> TimeMode {
        TimeMode::Transient
    }

    fn advance(&mut self, state: &mut SolverState) -> Result<TimeStepInfo> {
        state.time_step = state.time_step.saturating_add(1);
        state.iteration = state.iteration.saturating_add(1);
        state.physical_time += self.config.dt;
        state.dt = self.config.dt;
        let is_final = state.time_step >= self.config.max_steps;
        Ok(TimeStepInfo {
            dt: self.config.dt,
            physical_time: state.physical_time,
            step: state.time_step,
            is_final,
        })
    }
}

/// 单步 RK4：\(\mathbf{U}^{n+1} = \mathbf{U}^n + \Delta t \cdot \mathrm{RK4}(\mathrm{d}\mathbf{U}/\mathrm{d}t)\)。
pub fn rk4_step<T, F>(
    fields: &mut ConservedFieldsT<T>,
    storage: &mut Rk4StorageT<T>,
    dt: Real,
    mut evaluate_rhs: F,
) -> Result<()>
where
    T: ComputeFloat,
    F: FnMut(&ConservedFieldsT<T>, &mut ConservedResidualT<T>) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    storage.u0.copy_from(fields)?;
    {
        let _span = info_span!("rk4_stage", stage = 1).entered();
        evaluate_rhs(fields, &mut storage.k1)?;
    }
    storage
        .stage
        .assign_axpy(&storage.u0, &storage.k1, 0.5 * dt)?;
    {
        let _span = info_span!("rk4_stage", stage = 2).entered();
        evaluate_rhs(&storage.stage, &mut storage.k2)?;
    }
    storage
        .stage
        .assign_axpy(&storage.u0, &storage.k2, 0.5 * dt)?;
    {
        let _span = info_span!("rk4_stage", stage = 3).entered();
        evaluate_rhs(&storage.stage, &mut storage.k3)?;
    }
    storage.stage.assign_axpy(&storage.u0, &storage.k3, dt)?;
    {
        let _span = info_span!("rk4_stage", stage = 4).entered();
        evaluate_rhs(&storage.stage, &mut storage.k4)?;
    }
    {
        let _span = info_span!("rk4_update").entered();
        storage.increment.assign_rk4_increment(
            &storage.k1,
            &storage.k2,
            &storage.k3,
            &storage.k4,
        )?;
        fields.assign_axpy(&storage.u0, &storage.increment, dt)
    }
}

/// 逐单元 \(\Delta t_i\) 的 RK4 步进（稳态当地时间步）。
pub fn rk4_step_local<T, F>(
    fields: &mut ConservedFieldsT<T>,
    storage: &mut Rk4StorageT<T>,
    dt: &[Real],
    mut evaluate_rhs: F,
    eos: Option<&crate::physics::IdealGasEoS>,
    min_pressure: Real,
) -> Result<()>
where
    T: ComputeFloat,
    F: FnMut(&ConservedFieldsT<T>, &mut ConservedResidualT<T>) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    if dt.len() != n {
        return Err(crate::error::AsimuError::Solver(format!(
            "rk4_step_local: dt 长度 {} 与单元数 {n} 不一致",
            dt.len()
        )));
    }
    storage.u0.copy_from(fields)?;
    maybe_enforce_positivity(fields, eos, min_pressure);
    let gamma = eos.map(|e| e.gamma).unwrap_or(1.4);
    {
        let _span = info_span!("rk4_stage", stage = 1).entered();
        evaluate_rhs(fields, &mut storage.k1)?;
    }
    storage
        .stage
        .assign_axpy_dt(&storage.u0, &storage.k1, dt, 0.5, gamma, min_pressure)?;
    maybe_enforce_positivity(&mut storage.stage, eos, min_pressure);
    {
        let _span = info_span!("rk4_stage", stage = 2).entered();
        evaluate_rhs(&storage.stage, &mut storage.k2)?;
    }
    storage
        .stage
        .assign_axpy_dt(&storage.u0, &storage.k2, dt, 0.5, gamma, min_pressure)?;
    maybe_enforce_positivity(&mut storage.stage, eos, min_pressure);
    {
        let _span = info_span!("rk4_stage", stage = 3).entered();
        evaluate_rhs(&storage.stage, &mut storage.k3)?;
    }
    storage
        .stage
        .assign_axpy_dt(&storage.u0, &storage.k3, dt, 1.0, gamma, min_pressure)?;
    maybe_enforce_positivity(&mut storage.stage, eos, min_pressure);
    {
        let _span = info_span!("rk4_stage", stage = 4).entered();
        evaluate_rhs(&storage.stage, &mut storage.k4)?;
    }
    {
        let _span = info_span!("rk4_update").entered();
        storage.increment.assign_rk4_increment(
            &storage.k1,
            &storage.k2,
            &storage.k3,
            &storage.k4,
        )?;
        fields.assign_axpy_dt(
            &storage.u0,
            &storage.increment,
            dt,
            1.0,
            gamma,
            min_pressure,
        )?;
        maybe_enforce_positivity(fields, eos, min_pressure);
    }
    Ok(())
}

/// 逐单元 \(\Delta t_i\) 的 RK4 步进（f32 当地时间步）。
pub fn rk4_step_local_f32<F>(
    fields: &mut ConservedFieldsT<f32>,
    storage: &mut Rk4StorageT<f32>,
    dt: &[f32],
    mut evaluate_rhs: F,
    eos: Option<&crate::physics::IdealGasEoS>,
    min_pressure: Real,
) -> Result<()>
where
    F: FnMut(&ConservedFieldsT<f32>, &mut ConservedResidualT<f32>) -> Result<()>,
{
    let n = fields.num_cells();
    storage.ensure_capacity(n)?;
    if dt.len() != n {
        return Err(crate::error::AsimuError::Solver(format!(
            "rk4_step_local_f32: dt 长度 {} 与单元数 {n} 不一致",
            dt.len()
        )));
    }
    storage.u0.copy_from(fields)?;
    maybe_enforce_positivity(fields, eos, min_pressure);
    let gamma = eos.map(|e| e.gamma as f32).unwrap_or(1.4_f32);
    let min_p = min_pressure as f32;
    {
        let _span = info_span!("rk4_stage", stage = 1).entered();
        evaluate_rhs(fields, &mut storage.k1)?;
    }
    storage
        .stage
        .assign_axpy_dt_f32(&storage.u0, &storage.k1, dt, 0.5, gamma, min_p)?;
    maybe_enforce_positivity(&mut storage.stage, eos, min_pressure);
    {
        let _span = info_span!("rk4_stage", stage = 2).entered();
        evaluate_rhs(&storage.stage, &mut storage.k2)?;
    }
    storage
        .stage
        .assign_axpy_dt_f32(&storage.u0, &storage.k2, dt, 0.5, gamma, min_p)?;
    maybe_enforce_positivity(&mut storage.stage, eos, min_pressure);
    {
        let _span = info_span!("rk4_stage", stage = 3).entered();
        evaluate_rhs(&storage.stage, &mut storage.k3)?;
    }
    storage
        .stage
        .assign_axpy_dt_f32(&storage.u0, &storage.k3, dt, 1.0, gamma, min_p)?;
    maybe_enforce_positivity(&mut storage.stage, eos, min_pressure);
    {
        let _span = info_span!("rk4_stage", stage = 4).entered();
        evaluate_rhs(&storage.stage, &mut storage.k4)?;
    }
    {
        let _span = info_span!("rk4_update").entered();
        storage.increment.assign_rk4_increment(
            &storage.k1,
            &storage.k2,
            &storage.k3,
            &storage.k4,
        )?;
        fields.assign_axpy_dt_f32(&storage.u0, &storage.increment, dt, 1.0, gamma, min_p)?;
        maybe_enforce_positivity(fields, eos, min_pressure);
    }
    Ok(())
}

/// RK4 工作区（阶段态与四个斜率）；Euler 步进复用 `k1`/`u0`。
#[derive(Debug, Clone, PartialEq)]
pub struct Rk4StorageT<T: ComputeFloat> {
    pub u0: ConservedFieldsT<T>,
    pub stage: ConservedFieldsT<T>,
    pub k1: ConservedResidualT<T>,
    pub k2: ConservedResidualT<T>,
    pub k3: ConservedResidualT<T>,
    pub k4: ConservedResidualT<T>,
    pub increment: ConservedResidualT<T>,
}

/// 默认工程标量 RK4 工作区（`f64`）。
pub type Rk4Storage = Rk4StorageT<Real>;

impl<T: ComputeFloat> Rk4StorageT<T> {
    pub fn new(num_cells: usize) -> Result<Self> {
        Ok(Self {
            u0: zero_fields(num_cells)?,
            stage: zero_fields(num_cells)?,
            k1: ConservedResidualT::zeros(num_cells)?,
            k2: ConservedResidualT::zeros(num_cells)?,
            k3: ConservedResidualT::zeros(num_cells)?,
            k4: ConservedResidualT::zeros(num_cells)?,
            increment: ConservedResidualT::zeros(num_cells)?,
        })
    }

    pub(crate) fn ensure_capacity(&mut self, num_cells: usize) -> Result<()> {
        if self.u0.num_cells() != num_cells {
            *self = Self::new(num_cells)?;
        }
        Ok(())
    }
}

fn zero_fields<T: ComputeFloat>(num_cells: usize) -> Result<ConservedFieldsT<T>> {
    ConservedFieldsT::uniform(
        num_cells,
        crate::physics::ConservedState {
            density: 0.0,
            momentum: [0.0, 0.0, 0.0],
            total_energy: 0.0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::field::{ConservedFields, ConservedResidual};
    use crate::physics::ConservedState;

    #[test]
    fn rk4_integrates_linear_decay() {
        let n = 1;
        let mut fields = ConservedFields::uniform(
            n,
            ConservedState {
                density: 1.0,
                momentum: [0.0, 0.0, 0.0],
                total_energy: 0.0,
            },
        )
        .expect("fields");
        let mut storage = Rk4Storage::new(n).expect("storage");
        let lambda = 2.0;
        let dt = 0.5;
        let evaluate = |u: &ConservedFields, r: &mut ConservedResidual| {
            r.clear();
            for (rv, &val) in r.density.values_mut().iter_mut().zip(u.density.values()) {
                *rv = -lambda * val;
            }
            Ok(())
        };
        rk4_step(&mut fields, &mut storage, dt, evaluate).expect("rk4");
        let expected = 0.375;
        assert!(approx_eq(fields.density.values()[0], expected, 1.0e-12));
    }

    #[test]
    fn integrator_advances_physical_time() {
        let mut integrator = RungeKutta4Integrator::new(RungeKutta4Config {
            dt: 0.01,
            max_steps: 3,
        });
        let mut state = SolverState::default();
        let info = integrator.advance(&mut state).expect("advance");
        assert!(approx_eq(info.dt, 0.01, 1.0e-12));
        assert!(!info.is_final);
        integrator.advance(&mut state).expect("advance");
        let final_info = integrator.advance(&mut state).expect("advance");
        assert!(final_info.is_final);
        assert!(approx_eq(state.physical_time, 0.03, 1.0e-12));
    }
}
