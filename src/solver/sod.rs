//! Sod 激波管 benchmark 辅助（初值、误差、运行至指定时刻）。

use crate::core::Real;
use crate::error::Result;
use crate::field::ConservedFields;
use crate::mesh::StructuredMesh1d;
use crate::physics::{
    ConservedState, IdealGasEoS, PrimitiveState, RiemannPrimitive1d, SodProblem, sample_exact,
};
use crate::solver::compressible::{
    CompressibleAdvanceContext1d, CompressibleEulerConfig, CompressibleEulerSolver,
    CompressibleStepInfo,
};
use crate::solver::time::{RungeKutta4Config, RungeKutta4Integrator};

/// Sod benchmark 配置（与 `tests/benchmarks/sod_1d/expected.json` 对齐）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SodBenchmarkConfig {
    pub ncells: usize,
    pub length: Real,
    pub diaphragm: Real,
    pub final_time: Real,
    pub cfl: Real,
    pub sod: SodProblem,
}

impl Default for SodBenchmarkConfig {
    fn default() -> Self {
        Self {
            ncells: 100,
            length: 1.0,
            diaphragm: 0.5,
            final_time: 0.2,
            cfl: 0.4,
            sod: SodProblem::CLASSIC,
        }
    }
}

/// Sod 运行结果摘要。
#[derive(Debug, Clone, PartialEq)]
pub struct SodBenchmarkResult {
    pub l1_density: Real,
    pub l2_density: Real,
    pub final_time: Real,
    pub steps: u64,
    pub cell_centers: Vec<Real>,
    pub density_numeric: Vec<Real>,
    pub density_exact: Vec<Real>,
}

/// 在 1D 网格上设置 Sod 间断初值。
pub fn sod_initial_fields(
    mesh: &StructuredMesh1d,
    eos: &IdealGasEoS,
    problem: &SodProblem,
    diaphragm: Real,
) -> Result<ConservedFields> {
    let n = mesh.num_cells();
    let dx = mesh.dx();
    let origin = mesh.origin;
    let mut density = Vec::with_capacity(n);
    let mut momentum_x = Vec::with_capacity(n);
    let mut momentum_y = Vec::with_capacity(n);
    let mut momentum_z = Vec::with_capacity(n);
    let mut total_energy = Vec::with_capacity(n);
    for i in 0..n {
        let x = origin + (i as Real + 0.5) * dx;
        let prim = if x < diaphragm {
            problem.left
        } else {
            problem.right
        };
        let cons = riemann_to_conserved(eos, prim)?;
        density.push(cons.density);
        momentum_x.push(cons.momentum[0]);
        momentum_y.push(cons.momentum[1]);
        momentum_z.push(cons.momentum[2]);
        total_energy.push(cons.total_energy);
    }
    Ok(ConservedFields {
        density: crate::field::ScalarField::from_values(density)?,
        momentum_x: crate::field::ScalarField::from_values(momentum_x)?,
        momentum_y: crate::field::ScalarField::from_values(momentum_y)?,
        momentum_z: crate::field::ScalarField::from_values(momentum_z)?,
        total_energy: crate::field::ScalarField::from_values(total_energy)?,
    })
}

/// 运行 Sod benchmark 并与精确解对比密度 L1/L2。
pub fn run_sod_benchmark(config: &SodBenchmarkConfig) -> Result<SodBenchmarkResult> {
    let mesh = StructuredMesh1d::new("sod", config.ncells, 0.0, config.length)?;
    let eos = IdealGasEoS::new(config.sod.gamma, 1.0)?;
    let mut fields = sod_initial_fields(&mesh, &eos, &config.sod, config.diaphragm)?;
    let ctx = CompressibleAdvanceContext1d {
        mesh: &mesh,
        boundary: crate::discretization::InviscidBoundary1d::ZeroGradient,
        eos: &eos,
    };
    let solver = CompressibleEulerSolver::new(CompressibleEulerConfig {
        time: RungeKutta4Config {
            dt: config.final_time / (config.ncells as Real * 2.0),
            max_steps: u64::MAX,
        },
        cfl: config.cfl,
        ..CompressibleEulerConfig::default()
    });
    let history = run_until_time(&solver, &ctx, &mut fields, config.final_time)?;
    let final_time = history.last().map(|s| s.physical_time).unwrap_or(0.0);
    let steps = history.last().map(|s| s.step).unwrap_or(0);
    let problem = config.sod.riemann_problem();
    let mut cell_centers = Vec::with_capacity(mesh.num_cells());
    let mut density_numeric = Vec::with_capacity(mesh.num_cells());
    let mut density_exact = Vec::with_capacity(mesh.num_cells());
    let dx = mesh.dx();
    for i in 0..mesh.num_cells() {
        let x = mesh.origin + (i as Real + 0.5) * dx;
        cell_centers.push(x);
        density_numeric.push(fields.density.values()[i]);
        density_exact.push(sample_exact(&problem, x - config.diaphragm, final_time)?.density);
    }
    let l1 = l1_error(&density_numeric, &density_exact);
    let l2 = l2_error(&density_numeric, &density_exact);
    Ok(SodBenchmarkResult {
        l1_density: l1,
        l2_density: l2,
        final_time,
        steps,
        cell_centers,
        density_numeric,
        density_exact,
    })
}

/// 将 Sod 数值/精确解剖面写入文本文件（`#` 元数据 + 表头 + 数据列）。
pub fn write_sod_profile(
    path: &std::path::Path,
    config: &SodBenchmarkConfig,
    result: &SodBenchmarkResult,
) -> Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path).map_err(crate::error::AsimuError::from)?;
    writeln!(file, "# asimu sod_1d benchmark profile")?;
    writeln!(
        file,
        "# ncells={} length={} diaphragm={} final_time={} steps={} l1_density={:.8} l2_density={:.8}",
        config.ncells,
        config.length,
        config.diaphragm,
        result.final_time,
        result.steps,
        result.l1_density,
        result.l2_density
    )?;
    writeln!(file, "# columns: x rho_numeric rho_exact rho_error")?;
    writeln!(file, "x rho_numeric rho_exact rho_error")?;
    for i in 0..result.cell_centers.len() {
        let err = result.density_numeric[i] - result.density_exact[i];
        writeln!(
            file,
            "{:.8} {:.8} {:.8} {:.8}",
            result.cell_centers[i], result.density_numeric[i], result.density_exact[i], err
        )?;
    }
    Ok(())
}

fn run_until_time(
    solver: &CompressibleEulerSolver,
    ctx: &CompressibleAdvanceContext1d<'_>,
    fields: &mut ConservedFields,
    final_time: Real,
) -> Result<Vec<CompressibleStepInfo>> {
    let mut storage = crate::solver::time::Rk4Storage::new(ctx.mesh.num_cells())?;
    let mut state = crate::solver::state::SolverState::default();
    let mut integrator = RungeKutta4Integrator::new(solver.config.time);
    let mut history = Vec::new();
    while state.physical_time < final_time {
        let info =
            solver.advance_step_1d(ctx, fields, &mut storage, &mut state, &mut integrator)?;
        let done = info.is_final || info.physical_time >= final_time - 1.0e-14;
        history.push(info);
        if done {
            break;
        }
    }
    Ok(history)
}

fn riemann_to_conserved(eos: &IdealGasEoS, prim: RiemannPrimitive1d) -> Result<ConservedState> {
    let primitive = PrimitiveState {
        density: prim.density,
        velocity: [prim.velocity, 0.0, 0.0],
        pressure: prim.pressure,
        temperature: prim.pressure / (prim.density * eos.gas_constant),
    };
    ConservedState::from_primitive(eos, &primitive)
}

fn l1_error(numeric: &[Real], exact: &[Real]) -> Real {
    debug_assert_eq!(numeric.len(), exact.len());
    let n = numeric.len() as Real;
    numeric
        .iter()
        .zip(exact.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<Real>()
        / n
}

fn l2_error(numeric: &[Real], exact: &[Real]) -> Real {
    debug_assert_eq!(numeric.len(), exact.len());
    let n = numeric.len() as Real;
    let sum = numeric
        .iter()
        .zip(exact.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<Real>();
    (sum / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_sod_profile_roundtrip_header() {
        let config = SodBenchmarkConfig::default();
        let result = run_sod_benchmark(&config).expect("benchmark");
        let path = std::env::temp_dir().join("asimu_sod_profile_test.txt");
        write_sod_profile(&path, &config, &result).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.starts_with("# asimu sod_1d benchmark profile\n"));
        assert!(text.contains("l1_density="));
        assert!(text.contains("x rho_numeric rho_exact rho_error\n"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sod_benchmark_100_cells_l1_below_threshold() {
        let result = run_sod_benchmark(&SodBenchmarkConfig {
            ncells: 100,
            ..SodBenchmarkConfig::default()
        })
        .expect("benchmark");
        assert!(result.l1_density < 0.04);
        assert!(result.l2_density < 0.06);
        assert!((result.final_time - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn sod_benchmark_400_cells_l1_below_threshold() {
        let result = run_sod_benchmark(&SodBenchmarkConfig {
            ncells: 400,
            ..SodBenchmarkConfig::default()
        })
        .expect("benchmark");
        assert!(result.l1_density < 0.025);
        assert!(result.l2_density < 0.04);
    }
}
