//! CFD 求解器入口（占位实现：返回收敛占位结果）。

pub mod compressible;
pub mod compressible_helpers;
pub mod incompressible;
pub mod lu_sgs_common;
pub mod lu_sgs_sweep_unstructured;
pub mod sod;
pub mod spectral_radius;
pub mod spectral_radius_unstructured;
pub mod state;
pub mod time;
pub mod wave_speed;

use tracing::{info, instrument};

use crate::config::SolverConfig;
use crate::core::Real;
use crate::error::Result;
use crate::mesh::Mesh;

pub use compressible::{
    CompressibleAdvanceContext1d, CompressibleAdvanceContext3d, CompressibleEulerConfig,
    CompressibleEulerSolver, CompressibleStepInfo, CompressibleTimeMode, GmresImplicitConfig,
    GmresImplicitDelta, GmresPreconditionerKind,
};
pub use compressible_helpers::{
    EvaluateRhsUnstructured, RefreshCompressibleStateInput, finalize_cell_dts_from_sigma,
    refresh_compressible_ghosts_and_primitives,
};
pub use incompressible::{
    IncompressibleSimplecConfig, IncompressibleSimplecDiagnostic, run_incompressible_simplec,
};
pub use lu_sgs_sweep_unstructured::{
    LuSgsSweepUnstructuredInput, LuSgsSweepUnstructuredParams, LuSgsUnstructuredCouplings,
    lu_sgs_sweep_unstructured,
};
pub use sod::{
    SodBenchmarkConfig, SodBenchmarkResult, run_sod_benchmark, sod_initial_fields,
    write_sod_compare_profile, write_sod_profile,
};
pub use spectral_radius::{
    SpectralRadius3dParams, cell_local_dt_cfl_3d, cell_local_dt_spectral, cell_spectral_radius_3d,
    cell_viscous_diffusivity_max, local_pseudo_dt_lusgs,
};
pub use spectral_radius_unstructured::{
    SpectralRadiusUnstructuredParams, cell_spectral_radius_unstructured,
};
pub use state::SolverState;
pub use time::{
    CflSchedule, LuSgsConfig, Rk4Storage, RungeKutta4Config, RungeKutta4Integrator,
    SteadyStateIntegrator, TimeIntegrationScheme, TimeIntegrator, TimeMode, TimeStepInfo,
    euler_step, euler_step_local, local_dt_cfl, lu_sgs_step, lu_sgs_step_local, min_positive_dt,
    rk4_step, rk4_step_local,
};
pub use wave_speed::max_wave_speed;

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
