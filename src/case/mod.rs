//! 算例编排：case.toml → 求解器 dispatch → 结果摘要。
//!
//! 应用层（`app`）与集成测试共用本模块，避免在 CLI 中重复装配逻辑。

mod compressible_3d;
mod compressible_unstructured_3d;
#[cfg(test)]
mod compressible_unstructured_3d_tests;
mod diffusion;
mod incompressible_3d;
mod output_3d;
mod sod;

use std::path::Path;

use tracing::{info, info_span, instrument};

use crate::config::init_tracing;
use crate::error::{AsimuError, Result};
use crate::io::{CaseMesh, CaseSpec, load_case};

pub use compressible_3d::Compressible3dRunMetrics;
pub use diffusion::DiffusionRunMetrics;
pub use incompressible_3d::Incompressible3dRunMetrics;
pub use sod::SodRunMetrics;

/// 算例运行模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseRunKind {
    Diffusion1dSteady,
    Sod1dTransient,
    Compressible3dTransient,
    Incompressible3dSteady,
}

/// 算例运行结果摘要（写入日志 / 后续 manifest）。
#[derive(Debug, Clone, PartialEq)]
pub struct CaseRunResult {
    pub name: String,
    pub benchmark_id: Option<String>,
    pub kind: CaseRunKind,
    pub summary: String,
    pub diffusion: Option<DiffusionRunMetrics>,
    pub sod: Option<SodRunMetrics>,
    pub compressible_3d: Option<Compressible3dRunMetrics>,
    pub incompressible_3d: Option<Incompressible3dRunMetrics>,
}

/// 从 `case.toml` 路径加载并运行（默认日志级别 `info`）。
pub fn run_case_path(path: &Path) -> Result<CaseRunResult> {
    run_case_path_logged(path, "info", None)
}

/// 从 `case.toml` 加载并运行；按 `log_level` 与 Chrome trace 配置初始化 tracing。
///
/// `chrome_trace_cli`：`None` 仅用算例 `[observability]`；`Some(path)` 为 CLI `--chrome-trace [PATH]`。
pub fn run_case_path_logged(
    path: &Path,
    log_level: &str,
    chrome_trace_cli: Option<&str>,
) -> Result<CaseRunResult> {
    let case = load_case(path)?;
    let chrome = case.effective_chrome_trace_path(chrome_trace_cli)?;
    let _tracing = init_tracing(log_level, chrome.as_deref())?;
    run_case(&case)
}

/// 运行已解析算例。
#[instrument(skip(case), fields(name = %case.name))]
pub fn run_case(case: &CaseSpec) -> Result<CaseRunResult> {
    let kind = {
        let _span = info_span!("detect_run_kind").entered();
        detect_run_kind(case)?
    };
    info!(name = %case.name, ?kind, "开始算例编排");
    {
        let _span = info_span!(
            "log_boundary_patches",
            patches = case.boundary.patches().len()
        )
        .entered();
        case.boundary.log_patches();
    }
    let _span = info_span!("dispatch_case_run", ?kind).entered();
    match kind {
        CaseRunKind::Diffusion1dSteady => diffusion::run(case),
        CaseRunKind::Sod1dTransient => sod::run(case),
        CaseRunKind::Compressible3dTransient => match case.mesh {
            CaseMesh::Unstructured3d(_) => compressible_unstructured_3d::run(case),
            _ => compressible_3d::run(case),
        },
        CaseRunKind::Incompressible3dSteady => incompressible_3d::run(case),
    }
}

fn detect_run_kind(case: &CaseSpec) -> Result<CaseRunKind> {
    if case.sod.is_some() {
        case.mesh.as_1d()?;
        return Ok(CaseRunKind::Sod1dTransient);
    }
    if case.is_compressible() {
        let is_3d = matches!(
            case.mesh,
            CaseMesh::MultiBlockStructured3d(_) | CaseMesh::Unstructured3d(_)
        );
        if is_3d && (case.euler.is_some() || case.navier_stokes.is_some()) {
            return Ok(CaseRunKind::Compressible3dTransient);
        }
        return Err(AsimuError::Config(
            "3D 可压缩算例须包含 [euler] 或 [navier_stokes] 段且 mesh 为 3D".to_string(),
        ));
    }
    if case.incompressible.is_some() {
        case.mesh.as_3d()?;
        return Ok(CaseRunKind::Incompressible3dSteady);
    }
    case.mesh.as_1d()?;
    if case.time.mode == crate::io::CaseTimeMode::Transient {
        return Err(AsimuError::Config(
            "标量 1D 瞬态尚未实现；请使用 mode = \"steady\"".to_string(),
        ));
    }
    Ok(CaseRunKind::Diffusion1dSteady)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn runs_diffusion_benchmark_case() {
        let path = Path::new("tests/benchmarks/1d_diffusion_analytical/case.toml");
        let result = run_case_path(path).expect("run");
        assert_eq!(result.kind, CaseRunKind::Diffusion1dSteady);
        let metrics = result.diffusion.expect("metrics");
        assert!(metrics.max_abs_error > 0.0);
        assert!(metrics.max_abs_error < 0.05);
    }

    #[test]
    fn runs_sod_benchmark_case() {
        let path = Path::new("tests/benchmarks/sod_1d/case.toml");
        let result = run_case_path(path).expect("run");
        assert_eq!(result.kind, CaseRunKind::Sod1dTransient);
        let metrics = result.sod.expect("metrics");
        assert_eq!(metrics.scheme, "muscl_roe");
        assert_eq!(metrics.limiter, "van_albada");
        assert!(metrics.l1_density < 0.02);
    }

    #[test]
    fn runs_multiblock_compressible_smoke_case() {
        let case = crate::io::parse_case_str(
            r#"
name = "multi_cns"
[mesh]
kind = "multi_block_structured_3d"

[[mesh.blocks]]
name = "a"
nx = 1
ny = 1
nz = 1

[[mesh.blocks]]
name = "b"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[euler]
flux = "hllc"

[time]
scheme = "lu_sgs"
local_time_step = true
max_steps = 1
"#,
        )
        .expect("parse");
        let result = run_case(&case).expect("run");
        assert_eq!(result.kind, CaseRunKind::Compressible3dTransient);
        let metrics = result.compressible_3d.expect("metrics");
        assert_eq!(metrics.steps, 1);
        assert!(metrics.residual_rms.is_finite());
    }

    #[test]
    fn runs_incompressible_3d_i0_placeholder_case() {
        let case = crate::io::parse_case_str(
            r#"
name = "inc_i0"

[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 1
lx = 1.0
ly = 1.0
lz = 0.5

[physics]

[incompressible]
pressure = 0.0
velocity = [1.0, 0.0, 0.0]
density = 1.0
kinematic_viscosity = 0.01

[incompressible.reference]
length = 1.0
velocity = 1.0

[time]
mode = "steady"
max_steps = 1

[output]
dir = "output"
solution_cgns = "flow.cgns"
"#,
        )
        .expect("parse");
        let result = run_case(&case).expect("run");
        assert_eq!(result.kind, CaseRunKind::Incompressible3dSteady);
        let metrics = result.incompressible_3d.expect("metrics");
        assert_eq!(metrics.steps, 1);
        assert!(metrics.max_abs_divergence.is_finite());
        assert!(crate::core::approx_eq(
            metrics.max_abs_predicted_divergence,
            0.0,
            1.0e-12
        ));
        assert!(crate::core::approx_eq(
            metrics.max_abs_corrected_divergence,
            0.0,
            1.0e-12
        ));
        assert_eq!(metrics.pressure_system_rows, 4);
        assert!(metrics.pressure_system_nnz >= metrics.pressure_system_rows);
        assert!(metrics.pressure_solve_converged);
        assert_eq!(metrics.pressure_solve_iterations, 0);
        assert!(crate::core::approx_eq(
            metrics.max_abs_pressure_correction,
            0.0,
            1.0e-12
        ));
        assert_eq!(metrics.momentum_system_rows, 4);
        assert!(metrics.momentum_system_nnz >= metrics.momentum_system_rows);
        assert!(crate::core::approx_eq(
            metrics.max_momentum_d_coefficient,
            1.0,
            1.0e-12
        ));
        assert!(metrics.momentum_solve_converged);
        assert!(metrics.momentum_solve_iterations <= 1);
        assert!(crate::core::approx_eq(
            metrics.max_abs_predicted_velocity_delta,
            0.0,
            1.0e-12
        ));
        assert!(crate::core::approx_eq(
            metrics.max_abs_corrected_velocity_delta,
            0.0,
            1.0e-12
        ));
    }

    #[test]
    fn runs_multiblock_compressible_restart_case() {
        use crate::io::write_multiblock_conserved_fields;
        use crate::physics::{FreestreamParams, IdealGasEoS};

        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.3,
            ..FreestreamParams::default()
        };
        let block_a = crate::field::ConservedFields::from_freestream(1, &eos, &fs).expect("a");
        let block_b = crate::field::ConservedFields::from_freestream(1, &eos, &fs).expect("b");
        let path = std::env::temp_dir().join("asimu_multiblock_case_restart.toml");
        write_multiblock_conserved_fields(&path, &[("a", &block_a), ("b", &block_b)])
            .expect("write restart");

        let case = crate::io::parse_case_str(&format!(
            r#"
name = "multi_cns_restart"
[mesh]
kind = "multi_block_structured_3d"

[[mesh.blocks]]
name = "a"
nx = 1
ny = 1
nz = 1

[[mesh.blocks]]
name = "b"
nx = 1
ny = 1
nz = 1

[physics]
gamma = 1.4
gas_constant = 287.0

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15

[restart]
path = "{}"

[euler]
flux = "hllc"

[time]
scheme = "lu_sgs"
local_time_step = true
max_steps = 1
"#,
            path.display()
        ))
        .expect("parse");
        let mesh = match &case.mesh {
            crate::io::CaseMesh::MultiBlockStructured3d(mesh) => mesh,
            _ => panic!("expected multiblock mesh"),
        };
        let loaded = case
            .build_multiblock_conserved_fields(mesh.blocks())
            .expect("restart fields");
        assert_eq!(loaded[0].density.values(), block_a.density.values());
        assert_eq!(loaded[1].density.values(), block_b.density.values());
        let result = run_case(&case).expect("run");
        assert_eq!(result.kind, CaseRunKind::Compressible3dTransient);
        let _ = std::fs::remove_file(path);
    }

    #[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
    #[test]
    fn runs_cylinder_mach8_when_cgns_present() {
        let mesh_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("cylinder.cgns");
        if !mesh_path.is_file() {
            return;
        }
        let path = Path::new("tests/benchmarks/cylinder_mach8/case.toml");
        let result = run_case_path(path).expect("run");
        assert_eq!(result.kind, CaseRunKind::Compressible3dTransient);
        let metrics = result.compressible_3d.expect("metrics");
        assert_eq!(metrics.steps, 10);
        assert!(metrics.final_time > 0.0);
        assert!(metrics.final_time < 5.0);
        assert!(metrics.residual_rms.is_finite() && metrics.residual_rms > 0.0);
        assert_eq!(metrics.scheme, "first_order_hllc");
    }

    #[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
    #[test]
    fn debug_cylinder_step1_nan_root_cause() {
        use crate::core::Real;
        use crate::discretization::{
            BoundaryGhostBuffer, apply_compressible_boundary_conditions,
            assemble_inviscid_residual_3d,
        };
        use crate::field::ConservedResidual;
        use crate::solver::max_wave_speed;

        let mesh_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("cylinder.cgns");
        if !mesh_path.is_file() {
            return;
        }
        let case = load_case(Path::new("tests/benchmarks/cylinder_mach8/case.toml")).expect("case");
        let mesh = case.mesh.as_3d().expect("3d");
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("fs");
        let fields = case.build_conserved_fields().expect("fields");
        let i0 = mesh.node_index(0, 0, 0);
        let i1 = mesh.node_index(1, 0, 0);
        eprintln!(
            "dx(0,0,0)={} cartesian_cell_volume={} metric={:?} min_spacing={:?} max_wave={:?}",
            (mesh.points_x[i1] - mesh.points_x[i0]).abs(),
            mesh.cell_volume(),
            mesh.metric_mode(),
            mesh.min_positive_spacing(),
            max_wave_speed(&fields, &eos, 1.0e-6)
        );
        assert!(
            mesh.uses_curvilinear_metrics(),
            "CGNS cylinder 算例应默认启用 curvilinear metric"
        );
        assert!(
            mesh.metric_cache().is_some(),
            "CGNS 算例加载后应预构建 MetricCache"
        );
        let mut zero_vol_cart = 0usize;
        let mut zero_vol_curv = 0usize;
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    if mesh.cell_volume_at(i, j, k) <= Real::EPSILON {
                        zero_vol_cart += 1;
                    }
                    if mesh.cell_metric(i, j, k).volume <= Real::EPSILON {
                        zero_vol_curv += 1;
                    }
                }
            }
        }
        eprintln!("zero_volume_cells cartesian={zero_vol_cart} curvilinear={zero_vol_curv}");
        assert_eq!(zero_vol_curv, 0, "贴体 metric 不应产生零体积单元");
        let mut ghosts = BoundaryGhostBuffer::new();
        let fs_ctx = crate::physics::FreestreamContext::new(&eos, case.reference.as_ref(), None);
        apply_compressible_boundary_conditions(
            mesh,
            &case.boundary,
            &fields,
            &mut ghosts,
            &fs_ctx,
            &fs,
            None,
        )
        .expect("bc");
        let inviscid = case.euler.as_ref().expect("euler").inviscid();
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let mut primitives = crate::field::PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(
                &fields,
                &eos,
                crate::field::positivity_pressure_floor(fs.pressure),
            )
            .expect("fill");
        let p_floor = crate::field::positivity_pressure_floor(fs.pressure);
        let assembly = crate::discretization::residual::InviscidAssembly3dParams {
            mesh,
            eos: &eos,
            config: &inviscid,
            boundaries: &case.boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            min_pressure: p_floor,
        };
        assemble_inviscid_residual_3d(&fields, &mut rhs, &assembly).expect("asm");
        let nan_cells = rhs
            .density
            .values()
            .iter()
            .filter(|v| !v.is_finite())
            .count();
        eprintln!(
            "initial rhs RMS(rho_dot)={} log10={} nan_cells={}",
            rhs.density_rms_norm(),
            crate::core::log10_positive(rhs.density_rms_norm()),
            nan_cells
        );
        assert!(
            rhs.density_rms_norm().is_finite(),
            "uniform freestream initial residual must be finite"
        );

        let solver =
            crate::solver::CompressibleEulerSolver::new(crate::solver::CompressibleEulerConfig {
                time: crate::solver::RungeKutta4Config {
                    dt: 0.0,
                    max_steps: 1,
                },
                inviscid: inviscid.clone(),
                cfl_schedule: crate::solver::time::CflSchedule::constant(0.05),
                ..crate::solver::CompressibleEulerConfig::default()
            });
        let mut fields_step = fields.clone();
        let mut storage = crate::solver::Rk4Storage::new(mesh.num_cells()).expect("storage");
        let mut state = crate::solver::SolverState::default();
        let mut integrator = crate::solver::RungeKutta4Integrator::new(solver.config.time);
        let mut ctx = crate::solver::CompressibleAdvanceContext3d {
            mesh,
            structured: mesh,
            patches: &case.boundary,
            ghosts: &mut ghosts,
            eos: &eos,
            freestream: &fs,
            reference: case.reference.as_ref(),
            primitive_scratch: crate::field::PrimitiveFields::zeros(mesh.num_cells())
                .expect("primitives"),
            gradient_scratch: crate::discretization::GradientFields::zeros(mesh.num_cells())
                .expect("gradients"),
            viscous: None,
            residual_correction: None,
        };
        let step1 = solver
            .advance_step_3d(
                &mut ctx,
                &mut fields_step,
                &mut storage,
                &mut state,
                &mut integrator,
            )
            .expect("step1");
        assert!(step1.residual_rms.is_finite());
        assert_step1_fields_finite(mesh, &fields_step, &eos, fs.pressure);
    }

    #[cfg(all(feature = "io-cgns", feature = "slow-tests"))]
    fn assert_step1_fields_finite(
        mesh: &crate::mesh::StructuredMesh3d,
        fields: &crate::field::ConservedFields,
        eos: &crate::physics::IdealGasEoS,
        reference_pressure: crate::core::Real,
    ) {
        let p_floor = crate::field::positivity_pressure_floor(reference_pressure);
        for i in 0..mesh.num_cells() {
            assert!(fields.density.values()[i].is_finite() && fields.density.values()[i] > 0.0);
            let _ = fields
                .primitive_at(i, eos, p_floor)
                .expect("primitive after step1");
        }
    }
}
