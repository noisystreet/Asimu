//! 可压缩算例段解析（`[euler]` / `[output]` / CFL 调度）。

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::warn;

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::CaseSpec;
use super::validate_input_path;

/// 算例可观测性配置（`[observability]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct CaseObservabilityConfig {
    /// Chrome trace JSON 相对路径（相对 `[output].dir`，默认 `output/`）。
    pub chrome_trace: Option<String>,
}

/// 算例输出配置（`[output]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct CaseOutputConfig {
    pub dir: PathBuf,
    pub residual_csv: Option<String>,
    pub residual_plot: Option<String>,
    pub solution_cgns: Option<String>,
    pub solution_every: Option<u64>,
    /// 与 CGNS 同主文件名写出 `.vtu` / `.vts`（默认关闭）。
    pub solution_vtk: bool,
}

impl CaseOutputConfig {
    #[must_use]
    pub fn wants_interval_flow(&self) -> bool {
        self.solution_cgns.is_some() && self.solution_every.is_some_and(|n| n > 0)
    }
}

/// 3D 可压缩 Euler 算例段（`[euler]`，离散格式；CFL 见 `[time].cfl`）。
#[derive(Debug, Clone, PartialEq)]
pub struct EulerCaseConfig {
    pub final_time: Option<Real>,
    pub max_steps: Option<u64>,
    pub reconstruction: Option<String>,
    pub flux: Option<String>,
    pub limiter: Option<String>,
}

impl EulerCaseConfig {
    pub fn inviscid(&self) -> crate::discretization::InviscidFluxConfig {
        inviscid_from_toml(
            self.reconstruction.as_deref(),
            self.flux.as_deref(),
            self.limiter.as_deref(),
        )
    }
}

pub(super) fn inviscid_from_toml(
    reconstruction: Option<&str>,
    flux: Option<&str>,
    limiter: Option<&str>,
) -> crate::discretization::InviscidFluxConfig {
    use crate::discretization::{
        FluxScheme, InviscidFluxConfig, ReconstructionKind, RoeFluxConfig, SlopeLimiter,
    };
    let mut config = InviscidFluxConfig::default();
    if let Some(name) = reconstruction {
        config.reconstruction = match name {
            "first_order" | "first-order" => ReconstructionKind::FirstOrder,
            "muscl" => ReconstructionKind::Muscl,
            _ => ReconstructionKind::FirstOrder,
        };
    }
    if let Some(name) = limiter {
        if config.reconstruction == ReconstructionKind::Muscl {
            config.limiter = match name {
                "minmod" => SlopeLimiter::Minmod,
                "van_leer" | "vanleer" => SlopeLimiter::VanLeer,
                "van_albada" | "vanalbada" | "van-albada" => SlopeLimiter::VanAlbada,
                _ => SlopeLimiter::Minmod,
            };
        }
    }
    if let Some(name) = flux {
        config.scheme = match name {
            "roe" => FluxScheme::Roe(RoeFluxConfig::default()),
            "hllc" => FluxScheme::Hllc,
            "van_leer" | "vanleer" => FluxScheme::VanLeer,
            "hanel_van_leer" | "hanel-van-leer" | "hanelvanleer" => FluxScheme::HanelVanLeer,
            "slau2" | "slau_2" | "slau-2" => FluxScheme::Slau2,
            _ => FluxScheme::Roe(RoeFluxConfig::default()),
        };
        if reconstruction.is_none() {
            config.reconstruction = ReconstructionKind::Muscl;
        }
    }
    config
}

impl CaseSpec {
    /// 时间推进步数上限：`[time].max_steps`，其次 `[euler].max_steps`，默认 100。
    pub fn resolved_max_steps(&self) -> u64 {
        self.time
            .max_steps
            .or_else(|| self.euler.as_ref().and_then(|e| e.max_steps))
            .unwrap_or(100)
    }

    /// log₁₀(RMS(ρ̇)) 早停容差（`[time].tolerance`）。
    pub fn resolved_tolerance(&self) -> Option<Real> {
        self.time.tolerance.filter(|t| t.is_finite())
    }

    /// CFL 初值：仅 `[time].cfl`，默认 0.4（Sod 算例见 `[sod].cfl`）。
    pub fn resolved_cfl(&self) -> Real {
        self.time.cfl.unwrap_or(0.4)
    }

    /// CFL 调度：从 `[time].cfl` 线性增至 `[time].cfl_max`（未设则恒定）。
    pub fn cfl_schedule(&self) -> Result<crate::solver::time::CflSchedule> {
        use crate::solver::time::CflSchedule;
        let initial = self.time.cfl.unwrap_or(0.4);
        let max = self.time.cfl_max.unwrap_or(initial);
        if initial <= 0.0 || max <= 0.0 {
            return Err(AsimuError::Config(
                "[time].cfl 与 cfl_max 须为正".to_string(),
            ));
        }
        Ok(CflSchedule {
            initial,
            max,
            ramp_steps: self.time.cfl_ramp_steps,
        })
    }

    /// 解析后配置自检（告警，不中断加载）。
    pub fn warn_config_inconsistencies(&self) {
        warn_if_output_interval_exceeds_max_steps(self);
        warn_if_limiter_ignored_with_first_order(self);
    }

    /// 解析 `[observability].chrome_trace` 为绝对路径（未配置则 `None`）。
    pub fn resolved_chrome_trace_path(&self) -> Result<Option<PathBuf>> {
        let Some(obs) = &self.observability else {
            return Ok(None);
        };
        let Some(rel) = obs.chrome_trace.as_ref() else {
            return Ok(None);
        };
        Ok(Some(self.resolve_chrome_trace_relative(rel)?))
    }

    /// 将相对路径解析为 Chrome trace 绝对路径（相对 `[output].dir`；绝对路径原样返回）。
    pub fn resolve_chrome_trace_relative(&self, rel: &str) -> Result<PathBuf> {
        let path = Path::new(rel);
        if path.is_absolute() {
            return Ok(path.to_path_buf());
        }
        let output_dir = self
            .output
            .as_ref()
            .map(|o| o.dir.as_path())
            .unwrap_or(Path::new("output"));
        resolve_case_output_path(self.case_dir.as_deref(), output_dir, rel)
    }

    /// 合并 CLI 与算例配置：`cli` 为 `Some` 时优先。
    ///
    /// - `None`：仅用 `[observability]`
    /// - `Some("")`：`--chrome-trace` 无路径 → `profiling/trace.json`（相对 `[output].dir`）
    /// - `Some(path)`：`--chrome-trace PATH` → 相对**当前工作目录**或绝对路径
    pub fn effective_chrome_trace_path(&self, cli: Option<&str>) -> Result<Option<PathBuf>> {
        match cli {
            None => self.resolved_chrome_trace_path(),
            Some("") => Ok(Some(
                self.resolve_chrome_trace_relative("profiling/trace.json")?,
            )),
            Some(rel) => Ok(Some(resolve_chrome_trace_cli(rel)?)),
        }
    }
}

/// CLI `--chrome-trace PATH`：相对路径基于进程当前工作目录。
fn resolve_chrome_trace_cli(rel: &str) -> Result<PathBuf> {
    let path = Path::new(rel);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir()
        .map_err(|err| AsimuError::Config(format!("无法获取当前工作目录: {err}")))?;
    Ok(cwd.join(path))
}

#[derive(Debug, Deserialize)]
pub(super) struct EulerToml {
    cfl: Option<Real>,
    final_time: Option<Real>,
    max_steps: Option<u64>,
    reconstruction: Option<String>,
    flux: Option<String>,
    limiter: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ObservabilityToml {
    chrome_trace: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct OutputToml {
    dir: Option<PathBuf>,
    residual_csv: Option<String>,
    residual_plot: Option<String>,
    solution_cgns: Option<String>,
    solution_every: Option<u64>,
    solution_vtk: Option<bool>,
}

pub(super) fn parse_euler_config(raw: &EulerToml) -> Result<EulerCaseConfig> {
    if raw.cfl.is_some() {
        return Err(AsimuError::Config(
            "[euler].cfl 已移除，请在 [time] 段设置 cfl".to_string(),
        ));
    }
    Ok(EulerCaseConfig {
        final_time: raw.final_time,
        max_steps: raw.max_steps,
        reconstruction: raw.reconstruction.clone(),
        flux: raw.flux.clone(),
        limiter: raw.limiter.clone(),
    })
}

pub(super) fn parse_observability(raw: &ObservabilityToml) -> CaseObservabilityConfig {
    CaseObservabilityConfig {
        chrome_trace: raw.chrome_trace.clone().filter(|s| !s.trim().is_empty()),
    }
}

pub(super) fn parse_output(raw: &OutputToml) -> CaseOutputConfig {
    CaseOutputConfig {
        dir: raw.dir.clone().unwrap_or_else(|| PathBuf::from("output")),
        residual_csv: raw.residual_csv.clone(),
        residual_plot: raw.residual_plot.clone(),
        solution_cgns: raw.solution_cgns.clone(),
        solution_every: raw.solution_every,
        solution_vtk: raw.solution_vtk.unwrap_or(false),
    }
}

/// 将 `[output]` 相对路径解析为绝对路径（相对算例目录 + output.dir）。
pub fn resolve_case_output_path(
    case_dir: Option<&Path>,
    output_dir: &Path,
    rel: &str,
) -> Result<PathBuf> {
    let rel_path = PathBuf::from(rel);
    validate_input_path(&rel_path)?;
    let base = case_dir.unwrap_or_else(|| Path::new(".")).join(output_dir);
    Ok(base.join(rel_path))
}

fn is_first_order_reconstruction(name: &str) -> bool {
    matches!(name, "first_order" | "first-order")
}

fn warn_if_limiter_ignored_with_first_order(case: &CaseSpec) {
    let Ok(euler) = case.compressible_discretization() else {
        return;
    };
    if euler.limiter.is_none() {
        return;
    }
    let recon = euler.reconstruction.as_deref().unwrap_or("first_order");
    if is_first_order_reconstruction(recon) {
        warn!(
            limiter = ?euler.limiter,
            reconstruction = %recon,
            "[euler].limiter 在一阶重构下无效（分段常数已单调），可省略该字段"
        );
    }
}

fn warn_if_output_interval_exceeds_max_steps(case: &CaseSpec) {
    let Some(output) = &case.output else {
        return;
    };
    if !output.wants_interval_flow() {
        return;
    }
    let every = output.solution_every.expect("wants_interval_flow");
    let max_steps = case.resolved_max_steps();
    if every > max_steps {
        warn!(
            solution_every = every,
            max_steps, "[output].solution_every 大于 [time].max_steps，间隔流场 CGNS 不会写出"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::inviscid_from_toml;

    #[test]
    fn first_order_ignores_limiter_from_toml() {
        let cfg = inviscid_from_toml(Some("first_order"), Some("hllc"), Some("van_albada"));
        assert!(!cfg.uses_limiter());
        assert_eq!(cfg.limiter_label(), "none");
        assert_eq!(cfg.short_label(), "first_order_hllc");
    }

    #[test]
    fn muscl_applies_limiter_from_toml() {
        let cfg = inviscid_from_toml(Some("muscl"), Some("hllc"), Some("van_albada"));
        assert!(cfg.uses_limiter());
        assert_eq!(cfg.limiter_label(), "van_albada");
    }

    #[test]
    fn parse_navier_stokes_enables_viscous_physics() {
        let case = crate::io::parse_case_str(
            r#"
name = "ns_box"
benchmark_id = "ns_box"

[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
lx = 1.0
ly = 1.0
lz = 1.0

[physics]
gamma = 1.4
gas_constant = 287.0
prandtl = 0.72

[freestream]
mach = 0.1
pressure = 101325.0
temperature = 300.0

[time]
mode = "steady"
max_steps = 1

[navier_stokes]
flux = "roe"
reconstruction = "first_order"
"#,
        )
        .expect("parse");
        assert!(case.is_navier_stokes());
        assert!(case.physics.viscous.is_some());
    }
}
