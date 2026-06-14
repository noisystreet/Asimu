//! CFD 求解器入口（占位实现：返回收敛占位结果）。

pub mod compressible;
pub mod compressible_helpers;
mod compressible_multiblock;
mod compressible_multiblock_driver;
mod compressible_multiblock_driver_typed;
mod compressible_multiblock_interface;
mod compressible_unstructured_driver;
mod compressible_unstructured_driver_typed;
pub mod incompressible;
mod incompressible_diagnostics;
mod incompressible_linear;
mod incompressible_pressure_reference;
pub mod lu_sgs_common;
pub mod lu_sgs_sweep_unstructured;
mod lu_sgs_sweep_unstructured_typed;
pub mod sod;
pub mod spectral_radius;
pub mod spectral_radius_f32;
pub mod spectral_radius_unstructured;
mod spectral_radius_unstructured_f32;
pub mod state;
pub mod time;
pub mod wave_speed;

use tracing::{info, instrument};

use crate::config::SolverConfig;
use crate::core::Real;
use crate::error::Result;
use crate::mesh::Mesh;

pub use compressible::{
    CompressibleAdvanceContext1d, CompressibleAdvanceContext3d, CompressibleAdvanceContext3dTyped,
    CompressibleEulerConfig, CompressibleEulerSolver, CompressibleStepInfo, CompressibleTimeMode,
    GmresImplicitConfig, GmresImplicitDelta, GmresPreconditionerKind,
};
pub use compressible_helpers::{
    EvaluateRhsUnstructured, RefreshCompressibleStateInput, RefreshCompressibleStateTypedInput,
    finalize_cell_dts_from_sigma, refresh_compressible_ghosts_and_primitives,
    refresh_compressible_ghosts_and_primitives_typed,
};
pub use compressible_multiblock_driver::{
    CompressibleMultiblockStepView, MultiblockStructuredDriverInput,
    run_multiblock_structured_with_observer,
};
pub use compressible_multiblock_driver_typed::run_multiblock_structured_typed_with_observer;
pub use compressible_unstructured_driver::{
    CompressibleUnstructuredStepView, UnstructuredDriverConfig, run_unstructured_with_observer,
};
pub(crate) use compressible_unstructured_driver_typed::UnstructuredComputeBackend;
pub use compressible_unstructured_driver_typed::run_unstructured_typed_with_observer;
pub use incompressible::{
    IncompressibleLinearSolverConfig, IncompressiblePressureLinearSolverConfig,
    IncompressiblePressureLinearSolverKind, IncompressiblePressureVelocityAlgorithm,
    IncompressiblePressureVelocityConfig, IncompressiblePressureVelocityDiagnostic,
    IncompressiblePressureVelocitySnapshot, IncompressiblePressureVelocityStepInfo,
    IncompressiblePressureVelocityStepView, IncompressibleSimplecConfig,
    IncompressibleSimplecDiagnostic, run_incompressible_pressure_velocity,
    run_incompressible_pressure_velocity_with_observer, run_incompressible_simplec,
};
pub use lu_sgs_sweep_unstructured::{
    LuSgsSweepUnstructuredInput, LuSgsSweepUnstructuredParams, LuSgsUnstructuredCouplings,
    LuSgsUnstructuredCouplingsRef, lu_sgs_sweep_unstructured,
};
pub use lu_sgs_sweep_unstructured_typed::{
    LuSgsSweepUnstructuredTypedParams, LuSgsUnstructuredSweepTyped, lu_sgs_sweep_unstructured_f32,
    lu_sgs_sweep_unstructured_typed,
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
    SpectralRadiusUnstructuredParams, SpectralRadiusUnstructuredTypedParams,
    UnstructuredSpectralRadiusTyped, cell_spectral_radius_unstructured,
};
pub use spectral_radius_unstructured_f32::{
    SpectralRadiusUnstructuredF32Params, cell_spectral_radius_unstructured_f32,
};
pub use state::SolverState;
pub use time::{
    CflSchedule, LuSgsConfig, Rk4Storage, Rk4StorageT, RungeKutta4Config, RungeKutta4Integrator,
    SteadyStateIntegrator, TimeIntegrationScheme, TimeIntegrator, TimeMode, TimeStepInfo,
    TransientStepControl, euler_step, euler_step_local, local_dt_cfl, lu_sgs_step,
    lu_sgs_step_local, min_positive_dt, rk4_step, rk4_step_local,
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
