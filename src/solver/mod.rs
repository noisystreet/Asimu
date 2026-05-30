//! CFD 求解器入口（占位实现：返回收敛占位结果）。

pub mod compressible;
pub mod sod;
pub mod state;
pub mod time;

use tracing::{info, instrument};

use crate::config::SolverConfig;
use crate::core::Real;
use crate::error::Result;
use crate::mesh::Mesh;

pub use compressible::{
    CompressibleAdvanceContext1d, CompressibleAdvanceContext3d, CompressibleEulerConfig,
    CompressibleEulerSolver, CompressibleStepInfo, CompressibleTimeMode, max_wave_speed,
};
pub use sod::{
    SodBenchmarkConfig, SodBenchmarkResult, run_sod_benchmark, sod_initial_fields,
    write_sod_compare_profile, write_sod_profile,
};
pub use state::SolverState;
pub use time::{
    CflSchedule, Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, SteadyStateIntegrator,
    TimeIntegrationScheme, TimeIntegrator, TimeMode, TimeStepInfo, euler_step, euler_step_local,
    local_dt_cfl, min_positive_dt, rk4_step, rk4_step_local,
};

/// 求解结果摘要。
#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub iterations: u32,
    pub residual: Real,
    pub converged: bool,
}

/// 占位求解器：验证配置与网格管线，不含真实数值离散。
pub struct Solver {
    config: SolverConfig,
}

impl Solver {
    #[must_use]
    pub const fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    #[instrument(skip(self, mesh), fields(mesh = %mesh.name, cells = mesh.cell_count))]
    pub fn run(&self, mesh: &Mesh) -> Result<SolveResult> {
        info!(max_steps = self.config.max_steps, "开始占位求解");

        const PLACEHOLDER_TOLERANCE: f64 = 1.0e-6;
        let iterations = self.config.max_steps.min(10) as u32;
        let residual = PLACEHOLDER_TOLERANCE * 0.1;
        let converged = residual <= PLACEHOLDER_TOLERANCE;

        Ok(SolveResult {
            iterations,
            residual,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SolverConfig;

    #[test]
    fn placeholder_solver_converges() {
        let mesh = Mesh::new("unit-cube", 8).expect("mesh");
        let solver = Solver::new(SolverConfig { max_steps: 5 });
        let result = solver.run(&mesh).expect("run");
        assert!(result.converged);
        assert_eq!(result.iterations, 5);
    }
}
