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
    CompressibleEulerSolver, CompressibleStepInfo, max_wave_speed,
};
pub use sod::{
    SodBenchmarkConfig, SodBenchmarkResult, run_sod_benchmark, sod_initial_fields,
    write_sod_compare_profile, write_sod_profile,
};
pub use state::SolverState;
pub use time::{
    Rk4Storage, RungeKutta4Config, RungeKutta4Integrator, SteadyStateIntegrator, TimeIntegrator,
    TimeMode, TimeStepInfo, rk4_step,
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
        info!(
            max_iterations = self.config.max_iterations,
            tolerance = self.config.tolerance,
            "开始占位求解"
        );

        let iterations = self.config.max_iterations.min(10);
        let residual = self.config.tolerance * 0.1;
        let converged = residual <= self.config.tolerance;

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
        let solver = Solver::new(SolverConfig {
            max_iterations: 5,
            tolerance: 1.0e-3,
        });
        let result = solver.run(&mesh).expect("run");
        assert!(result.converged);
        assert_eq!(result.iterations, 5);
    }
}
