//! TOML 算例解析（扩散 1D + 可压缩 3D NS）。

#[path = "case_boundary.rs"]
mod case_boundary;
#[path = "case_compressible.rs"]
mod case_compressible;
#[path = "case_incompressible.rs"]
mod case_incompressible;
#[path = "mesh_load.rs"]
mod mesh_load;

pub use case_compressible::{
    CaseObservabilityConfig, CaseOutputConfig, EulerCaseConfig, resolve_case_output_path,
};
pub use case_incompressible::IncompressibleCaseConfig;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::warn;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{
    Fields, FluidInitialConfig, InitialKind, InitialSet, ScalarField, ScalarInitial,
};
use crate::mesh::{
    MultiBlockStructuredMesh3d, StructuredMesh1d, StructuredMesh3d, UnstructuredMesh3d,
};
use crate::physics::{
    FreestreamParams, IdealGasEoS, IncompressibleReferenceScales, PhysicsConfig, ReferenceScales,
};
use case_boundary::resolve_case_boundary;

use super::validate_input_path;
use case_compressible::{
    EulerToml, ObservabilityToml, OutputToml, parse_euler_config, parse_observability, parse_output,
};
use case_incompressible::{IncompressibleToml, parse_incompressible_config};

/// 算例网格（1D 扩散 / 3D 多块可压缩）。
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum CaseMesh {
    Structured1d(StructuredMesh1d),
    MultiBlockStructured3d(MultiBlockStructuredMesh3d),
    Unstructured3d(UnstructuredMesh3d),
}

impl CaseMesh {
    #[must_use]
    pub fn num_cells(&self) -> usize {
        match self {
            Self::Structured1d(m) => m.num_cells(),
            Self::MultiBlockStructured3d(m) => m.num_cells(),
            Self::Unstructured3d(m) => m.num_cells(),
        }
    }

    pub fn as_1d(&self) -> Result<&StructuredMesh1d> {
        match self {
            Self::Structured1d(m) => Ok(m),
            _ => Err(AsimuError::Mesh("算例非 1D 网格".to_string())),
        }
    }

    /// 单 block 3D 网格的便捷访问；多块时返回错误。
    pub fn as_3d(&self) -> Result<&StructuredMesh3d> {
        match self {
            Self::MultiBlockStructured3d(m) => {
                if m.num_blocks() == 1 {
                    Ok(&m.blocks()[0].mesh)
                } else {
                    Err(AsimuError::Mesh(format!(
                        "多块 3D 网格含 {} 个 block，请使用 as_multiblock_3d()",
                        m.num_blocks()
                    )))
                }
            }
            _ => Err(AsimuError::Mesh("算例非 3D 网格".to_string())),
        }
    }

    pub fn as_multiblock_3d(&self) -> Result<&MultiBlockStructuredMesh3d> {
        match self {
            Self::MultiBlockStructured3d(m) => Ok(m),
            _ => Err(AsimuError::Mesh("算例非 3D 多块网格".to_string())),
        }
    }

    pub fn as_unstructured_3d(&self) -> Result<&UnstructuredMesh3d> {
        match self {
            Self::Unstructured3d(m) => Ok(m),
            _ => Err(AsimuError::Mesh("算例非 3D 非结构网格".to_string())),
        }
    }

    /// 将所有节点坐标乘以 `factor`。
    pub fn scale_coordinates(&mut self, factor: Real) -> Result<()> {
        if factor <= 0.0 {
            return Err(AsimuError::Config("mesh.scale 必须大于 0".to_string()));
        }
        match self {
            Self::Structured1d(mesh) => {
                mesh.origin *= factor;
                mesh.length *= factor;
            }
            Self::MultiBlockStructured3d(mesh) => mesh.scale_coordinates(factor),
            Self::Unstructured3d(mesh) => mesh.scale_coordinates(factor)?,
        }
        if let CaseMesh::MultiBlockStructured3d(mesh) = self {
            mesh.rebuild_metric_cache_if_needed()?;
        }
        Ok(())
    }
}

/// 完整算例描述。
#[derive(Debug, Clone, PartialEq)]
pub struct CaseSpec {
    pub name: String,
    pub benchmark_id: Option<String>,
    pub mesh: CaseMesh,
    pub physics: PhysicsConfig,
    pub boundary: BoundarySet,
    pub initial: InitialSet,
    pub fluid_initial: FluidInitialConfig,
    pub freestream: Option<FreestreamParams>,
    pub restart: Option<PathBuf>,
    pub time: CaseTimeConfig,
    pub sod: Option<SodCaseConfig>,
    pub euler: Option<EulerCaseConfig>,
    /// 3D 可压缩 Navier-Stokes（`[navier_stokes]`，与 `[euler]` 二选一）。
    pub navier_stokes: Option<EulerCaseConfig>,
    pub incompressible: Option<IncompressibleCaseConfig>,
    pub output: Option<CaseOutputConfig>,
    pub observability: Option<CaseObservabilityConfig>,
    pub case_dir: Option<PathBuf>,
    /// 无量纲参考量；可压缩算例解析后恒为 `Some`。
    pub reference: Option<ReferenceScales>,
    /// 不可压缩无量纲参考量；不可压缩算例解析后恒为 `Some`。
    pub incompressible_reference: Option<IncompressibleReferenceScales>,
}

/// 算例时间推进配置（`[time]`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseTimeMode {
    Steady,
    Transient,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseTimeConfig {
    pub mode: CaseTimeMode,
    pub dt: Option<Real>,
    pub cfl: Option<Real>,
    pub cfl_max: Option<Real>,
    pub final_time: Option<Real>,
    pub max_steps: Option<u64>,
    /// 允许收敛早停前必须完成的最小步数。
    pub min_steps: Option<u64>,
    /// log₁₀(RMS(ρ̇)) 收敛容差；设有限值时与 `max_steps` 成对用于残差早停。
    pub tolerance: Option<Real>,
    pub local_time_step: bool,
    pub cfl_ramp_steps: Option<u64>,
    /// 时间积分格式：`rk4`（默认）、`euler` 或 `lu_sgs`。
    pub scheme: Option<crate::solver::time::TimeIntegrationScheme>,
    /// `lu_sgs` 松弛因子 \(\omega\in(0,1]\)（默认 1）。
    pub lusgs_omega: Option<Real>,
    /// `lu_sgs` 双扫（默认 false）；`true` 启用 i/j/k 双扫。
    pub lusgs_sweep: Option<bool>,
    /// `lu_sgs` 后扫耦合阻尼（默认 0.5）。
    pub lusgs_sweep_backward_damping: Option<Real>,
    /// GMRES 左预条件器（默认 scalar_diagonal）。
    pub gmres_preconditioner: Option<crate::solver::GmresPreconditionerKind>,
    /// 方向分裂隐式残差光顺（仅稳态 3D 伪时间）。
    pub residual_smoothing: crate::solver::time::ResidualSmoothingConfig,
}

impl CaseTimeConfig {
    #[must_use]
    pub fn uses_local_time_step(&self) -> bool {
        self.local_time_step
    }

    #[must_use]
    pub fn resolved_time_scheme(&self) -> crate::solver::time::TimeIntegrationScheme {
        self.scheme.unwrap_or_default()
    }

    pub fn resolved_lusgs_config(&self) -> Result<crate::solver::time::LuSgsConfig> {
        crate::solver::time::LuSgsConfig::parse(
            self.lusgs_omega,
            self.lusgs_sweep,
            self.lusgs_sweep_backward_damping,
        )
    }

    #[must_use]
    pub const fn residual_smoothing_config(&self) -> crate::solver::time::ResidualSmoothingConfig {
        self.residual_smoothing
    }

    #[must_use]
    pub fn resolved_gmres_config(&self) -> crate::solver::GmresImplicitConfig {
        crate::solver::GmresImplicitConfig {
            preconditioner: self
                .gmres_preconditioner
                .unwrap_or(crate::solver::GmresPreconditionerKind::ScalarDiagonal),
            ..crate::solver::GmresImplicitConfig::default()
        }
    }
}

impl Default for CaseTimeConfig {
    fn default() -> Self {
        Self {
            mode: CaseTimeMode::Steady,
            dt: None,
            cfl: None,
            cfl_max: None,
            final_time: None,
            max_steps: None,
            min_steps: None,
            tolerance: None,
            local_time_step: false,
            cfl_ramp_steps: None,
            scheme: None,
            lusgs_omega: None,
            lusgs_sweep: None,
            lusgs_sweep_backward_damping: None,
            gmres_preconditioner: None,
            residual_smoothing: crate::solver::time::ResidualSmoothingConfig::disabled(),
        }
    }
}

/// Sod 激波管专用段（`[sod]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct SodCaseConfig {
    pub diaphragm: Real,
    pub final_time: Real,
    pub cfl: Real,
    pub reconstruction: Option<String>,
    pub flux: Option<String>,
    pub limiter: Option<String>,
}

impl SodCaseConfig {
    pub fn inviscid(&self) -> crate::discretization::InviscidFluxConfig {
        case_compressible::inviscid_from_toml(
            self.reconstruction.as_deref(),
            self.flux.as_deref(),
            self.limiter.as_deref(),
            None,
        )
    }
}

impl CaseSpec {
    pub fn build_initial_fields(&self) -> Result<Fields> {
        let mesh_1d = self.mesh.as_1d()?;
        Fields::from_initial_set(mesh_1d, &self.initial)
    }

    pub fn initial_scalar(&self, name: &str) -> Result<ScalarField> {
        let mesh_1d = self.mesh.as_1d()?;
        self.initial.build_scalar_or_zero(name, mesh_1d)
    }

    pub fn build_conserved_fields(&self) -> Result<crate::field::ConservedFields> {
        if let Some(path) = &self.restart {
            return super::restart::load_conserved_fields(path);
        }
        super::restart::initial_freestream_conserved_fields(
            self.mesh.num_cells(),
            &self.physics.eos()?,
            self.reference.as_ref(),
            self.physics.viscous.as_ref(),
            self.freestream.or(self.fluid_initial.freestream),
        )
    }

    /// 按 block 顺序构建多块守恒初场；`[restart]` 使用 version=2 多块 TOML。
    pub fn build_multiblock_conserved_fields(
        &self,
        blocks: &[crate::mesh::StructuredBlock3d],
    ) -> Result<Vec<crate::field::ConservedFields>> {
        super::restart::initial_multiblock_conserved_fields(
            self.restart.as_deref(),
            blocks,
            &self.physics.eos()?,
            self.reference.as_ref(),
            self.physics.viscous.as_ref(),
            self.freestream.or(self.fluid_initial.freestream),
        )
    }

    pub fn is_compressible(&self) -> bool {
        self.physics.eos.is_some()
    }

    /// 有 1-to-1 接口的多块可压缩算例须满足 LU-SGS 对角隐式约束。
    pub fn validate_multiblock_compressible(&self) -> Result<()> {
        if !self.is_compressible() || (self.euler.is_none() && self.navier_stokes.is_none()) {
            return Ok(());
        }
        let CaseMesh::MultiBlockStructured3d(mesh) = &self.mesh else {
            return Ok(());
        };
        if mesh.interfaces().is_empty() {
            return Ok(());
        }
        if self.time.resolved_time_scheme() != crate::solver::TimeIntegrationScheme::LuSgs {
            return Err(AsimuError::Config(
                "有 1-to-1 接口的多块 3D 可压缩算例当前仅支持 time.scheme = \"lu_sgs\"".to_string(),
            ));
        }
        if self.time.resolved_lusgs_config()?.sweep {
            return Err(AsimuError::Config(
                "有 1-to-1 接口的多块 3D 可压缩算例暂不支持 lusgs_sweep = true".to_string(),
            ));
        }
        Ok(())
    }

    /// 3D 可压缩离散段：`[navier_stokes]` 优先，否则 `[euler]`。
    pub fn compressible_discretization(&self) -> Result<&EulerCaseConfig> {
        if let Some(ns) = &self.navier_stokes {
            return Ok(ns);
        }
        self.euler.as_ref().ok_or_else(|| {
            AsimuError::Config("3D 可压缩算例须包含 [euler] 或 [navier_stokes]".to_string())
        })
    }

    pub fn is_navier_stokes(&self) -> bool {
        self.physics.is_navier_stokes()
    }

    #[must_use]
    pub fn is_nondimensional(&self) -> bool {
        self.reference.is_some()
    }

    /// 输出还原用的有量纲 EOS（可压缩算例由 `reference` 还原 SI 气体常数）。
    pub fn dimensional_eos(&self) -> Result<IdealGasEoS> {
        let reference = self.reference.as_ref().ok_or_else(|| {
            AsimuError::Config("可压缩算例缺少无量纲参考量 reference".to_string())
        })?;
        reference.dimensional_eos(self.physics.eos()?.gamma)
    }

    pub fn diffusivity(&self) -> Real {
        self.physics.diffusivity.unwrap_or(1.0)
    }
}

#[derive(Debug, Deserialize)]
struct CaseToml {
    name: String,
    benchmark_id: Option<String>,
    mesh: MeshToml,
    physics: PhysicsToml,
    #[serde(default)]
    boundary: BTreeMap<String, BoundaryToml>,
    #[serde(default)]
    initial: BTreeMap<String, InitialToml>,
    freestream: Option<FreestreamToml>,
    restart: Option<RestartToml>,
    time: Option<TimeToml>,
    sod: Option<SodToml>,
    euler: Option<EulerToml>,
    navier_stokes: Option<EulerToml>,
    incompressible: Option<IncompressibleToml>,
    output: Option<OutputToml>,
    observability: Option<ObservabilityToml>,
}

#[derive(Debug, Deserialize)]
struct MeshToml {
    kind: String,
    cells: Option<usize>,
    ncells: Option<usize>,
    length: Option<Real>,
    origin: Option<Real>,
    nx: Option<usize>,
    ny: Option<usize>,
    nz: Option<usize>,
    lx: Option<Real>,
    ly: Option<Real>,
    lz: Option<Real>,
    #[cfg_attr(not(feature = "io-cgns"), allow(dead_code))]
    path: Option<PathBuf>,
    scale: Option<Real>,
    metric: Option<String>,
    /// 已废弃：`cgns` 读入现自动加载全部 structured zone。
    zone: Option<usize>,
    #[serde(default)]
    blocks: Vec<mesh_load::MeshBlockTomlFields>,
}

#[derive(Debug, Deserialize)]
struct PhysicsToml {
    diffusivity: Option<Real>,
    gamma: Option<Real>,
    gas_constant: Option<Real>,
    /// 动力粘度 Pa·s（`viscosity = "constant"` 时必填）。
    mu: Option<Real>,
    /// Prandtl 数（默认 0.72）。
    prandtl: Option<Real>,
    /// `sutherland`（默认）或 `constant`。
    viscosity: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct BoundaryToml {
    kind: String,
    value: Option<Real>,
    flux: Option<Real>,
    mach: Option<Real>,
    pressure: Option<Real>,
    temperature: Option<Real>,
    alpha: Option<Real>,
    beta: Option<Real>,
    total_pressure: Option<Real>,
    total_temperature: Option<Real>,
    static_pressure: Option<Real>,
    velocity_direction: Option<[Real; 3]>,
    velocity: Option<[Real; 3]>,
    no_slip: Option<bool>,
    heat: Option<String>,
    wall_temperature: Option<Real>,
    heat_flux: Option<Real>,
    partner: Option<String>,
    turbulent_k: Option<Real>,
    turbulent_omega: Option<Real>,
    supersonic: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct InitialToml {
    kind: String,
    value: Option<Real>,
    left: Option<Real>,
    right: Option<Real>,
    data: Option<Vec<Real>>,
}

#[derive(Debug, Deserialize)]
struct FreestreamToml {
    mach: Option<Real>,
    pressure: Option<Real>,
    temperature: Option<Real>,
    velocity_direction: Option<[Real; 3]>,
    alpha: Option<Real>,
    beta: Option<Real>,
}

#[derive(Debug, Deserialize)]
struct RestartToml {
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct TimeToml {
    mode: Option<String>,
    dt: Option<Real>,
    cfl: Option<Real>,
    cfl_max: Option<Real>,
    final_time: Option<Real>,
    max_steps: Option<u64>,
    min_steps: Option<u64>,
    tolerance: Option<Real>,
    local_time_step: Option<bool>,
    cfl_ramp_steps: Option<u64>,
    scheme: Option<String>,
    lusgs_omega: Option<Real>,
    lusgs_sweep: Option<bool>,
    lusgs_sweep_backward_damping: Option<Real>,
    gmres_preconditioner: Option<String>,
    residual_smoothing: Option<bool>,
    residual_smoothing_epsilon: Option<Real>,
    residual_smoothing_sweeps: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SodToml {
    diaphragm: Option<Real>,
    final_time: Option<Real>,
    cfl: Option<Real>,
    reconstruction: Option<String>,
    flux: Option<String>,
    limiter: Option<String>,
}

struct FlowModelConfigs {
    sod: Option<SodCaseConfig>,
    euler: Option<EulerCaseConfig>,
    navier_stokes: Option<EulerCaseConfig>,
    incompressible: Option<IncompressibleCaseConfig>,
}

/// 从字符串解析算例（测试与集成用）。
pub fn parse_case_str(content: &str) -> Result<CaseSpec> {
    parse_case_toml(content, None)
}

/// 从 `case.toml` 加载算例。
pub fn load_case(path: &Path) -> Result<CaseSpec> {
    validate_input_path(path)?;
    let content = std::fs::read_to_string(path)?;
    parse_case_toml(&content, path.parent())
}

fn parse_case_toml(content: &str, case_dir: Option<&Path>) -> Result<CaseSpec> {
    let raw: CaseToml = toml::from_str(content)?;
    let physics = parse_physics(&raw.physics, raw.navier_stokes.is_some())?;
    let parsed =
        mesh_load::parse_mesh_from_toml(&mesh_toml_fields(&raw.mesh), &raw.name, case_dir)?;
    let boundary = resolve_case_boundary(
        &parsed.mesh,
        parsed.cgns_boundary,
        &raw.boundary,
        raw.freestream.as_ref().map(parse_freestream),
        &physics,
        raw.euler.is_some() && raw.navier_stokes.is_none(),
    )?;
    let initial = resolve_initial_set(&raw.initial)?;
    let freestream = raw.freestream.as_ref().map(parse_freestream);
    let flow = parse_flow_model_configs(&raw)?;
    let fluid_initial = FluidInitialConfig {
        freestream,
        scalars: initial.clone(),
    };
    let restart = raw
        .restart
        .map(|r| super::restart::resolve_restart_path(r.path, case_dir));

    let time = parse_time_config(raw.time.as_ref(), raw.sod.is_some())?;

    let deprecated_mesh_zone = raw.mesh.zone;
    let mut case = CaseSpec {
        name: raw.name,
        benchmark_id: raw.benchmark_id,
        mesh: parsed.mesh,
        physics,
        boundary,
        initial,
        fluid_initial,
        freestream,
        restart,
        time,
        sod: flow.sod,
        euler: flow.euler,
        navier_stokes: flow.navier_stokes,
        incompressible: flow.incompressible,
        output: raw.output.as_ref().map(parse_output),
        observability: raw.observability.as_ref().map(parse_observability),
        case_dir: case_dir.map(Path::to_path_buf),
        reference: None,
        incompressible_reference: None,
    };
    finalize_parsed_case(deprecated_mesh_zone, &mut case)?;
    Ok(case)
}

fn parse_flow_model_configs(raw: &CaseToml) -> Result<FlowModelConfigs> {
    if raw.euler.is_some() && raw.navier_stokes.is_some() {
        return Err(AsimuError::Config(
            "算例不能同时包含 [euler] 与 [navier_stokes]".to_string(),
        ));
    }
    if raw.incompressible.is_some()
        && (raw.euler.is_some() || raw.navier_stokes.is_some() || raw.sod.is_some())
    {
        return Err(AsimuError::Config(
            "[incompressible] 不能与 [sod] / [euler] / [navier_stokes] 同时使用".to_string(),
        ));
    }
    Ok(FlowModelConfigs {
        sod: raw.sod.as_ref().map(parse_sod_config),
        euler: raw.euler.as_ref().map(parse_euler_config).transpose()?,
        navier_stokes: raw
            .navier_stokes
            .as_ref()
            .map(parse_euler_config)
            .transpose()?,
        incompressible: raw
            .incompressible
            .as_ref()
            .map(parse_incompressible_config)
            .transpose()?,
    })
}

fn finalize_parsed_case(deprecated_mesh_zone: Option<usize>, case: &mut CaseSpec) -> Result<()> {
    case.warn_config_inconsistencies();
    if deprecated_mesh_zone.is_some() {
        warn!(
            zone = ?deprecated_mesh_zone,
            "mesh.zone 已废弃：cgns 读入现自动加载全部 structured zone，该字段被忽略"
        );
    }
    super::nondimensional::apply_nondimensionalization_for_compressible(case)?;
    super::nondimensional::apply_nondimensionalization_for_incompressible(case)?;
    case.validate_multiblock_compressible()
}

fn mesh_toml_fields(raw: &MeshToml) -> mesh_load::MeshTomlFields {
    mesh_load::MeshTomlFields {
        kind: raw.kind.clone(),
        cells: raw.cells,
        ncells: raw.ncells,
        length: raw.length,
        origin: raw.origin,
        nx: raw.nx,
        ny: raw.ny,
        nz: raw.nz,
        lx: raw.lx,
        ly: raw.ly,
        lz: raw.lz,
        path: raw.path.clone(),
        scale: raw.scale,
        metric: raw.metric.clone(),
        blocks: raw.blocks.clone(),
    }
}

fn parse_physics(raw: &PhysicsToml, navier_stokes: bool) -> Result<PhysicsConfig> {
    use crate::physics::{ViscosityModel, ViscousPhysicsConfig};
    let eos = match (raw.gamma, raw.gas_constant) {
        (Some(gamma), Some(gas_constant)) => Some(IdealGasEoS::new(gamma, gas_constant)?),
        (None, None) => None,
        _ => {
            return Err(AsimuError::Config(
                "gamma 与 gas_constant 须同时指定".to_string(),
            ));
        }
    };
    if let Some(d) = raw.diffusivity {
        if d < 0.0 {
            return Err(AsimuError::Config("diffusivity 不能为负".to_string()));
        }
    }
    let viscous = if navier_stokes {
        let prandtl = raw.prandtl.unwrap_or(0.72);
        let model = match raw.viscosity.as_deref() {
            Some("constant") => {
                let mu = raw.mu.ok_or_else(|| {
                    AsimuError::Config(
                        "viscosity = \"constant\" 时须在 [physics] 指定 mu".to_string(),
                    )
                })?;
                ViscosityModel::constant(mu)?
            }
            Some("sutherland") | None => ViscosityModel::AIR_SUTHERLAND,
            Some(other) => {
                return Err(AsimuError::Config(format!(
                    "未知粘度模型 \"{other}\"（支持 sutherland / constant）"
                )));
            }
        };
        Some(ViscousPhysicsConfig::new(model, prandtl)?)
    } else {
        None
    };
    Ok(PhysicsConfig {
        diffusivity: raw.diffusivity,
        eos,
        viscous,
    })
}

fn parse_freestream(raw: &FreestreamToml) -> FreestreamParams {
    FreestreamParams {
        mach: raw.mach.unwrap_or(0.0),
        pressure: raw.pressure.unwrap_or(101_325.0),
        temperature: raw.temperature.unwrap_or(288.15),
        velocity_direction: raw.velocity_direction.unwrap_or([1.0, 0.0, 0.0]),
        alpha: raw.alpha.unwrap_or(0.0),
        beta: raw.beta.unwrap_or(0.0),
    }
}

fn resolve_initial_set(initials: &BTreeMap<String, InitialToml>) -> Result<InitialSet> {
    let mut scalars = Vec::with_capacity(initials.len());
    for (name, ic) in initials {
        let kind = InitialKind::from_toml(&ic.kind, ic.value, ic.left, ic.right, ic.data.clone())
            .ok_or_else(|| {
            AsimuError::Field(format!("初始条件 \"{name}\" 无效：kind=\"{}\"", ic.kind))
        })?;
        scalars.push(ScalarInitial::new(name.clone(), kind));
    }
    let set = InitialSet::new(scalars);
    set.validate()?;
    Ok(set)
}

fn parse_time_config(raw: Option<&TimeToml>, has_sod: bool) -> Result<CaseTimeConfig> {
    let Some(raw) = raw else {
        return Ok(if has_sod {
            CaseTimeConfig {
                mode: CaseTimeMode::Transient,
                ..CaseTimeConfig::default()
            }
        } else {
            CaseTimeConfig::default()
        });
    };
    let mode = match raw.mode.as_deref().unwrap_or("steady") {
        "steady" => CaseTimeMode::Steady,
        "transient" => CaseTimeMode::Transient,
        other => {
            return Err(AsimuError::Config(format!(
                "不支持的 time.mode \"{other}\""
            )));
        }
    };
    let scheme = raw
        .scheme
        .as_deref()
        .map(crate::solver::time::TimeIntegrationScheme::parse)
        .transpose()?;
    let lusgs_omega = raw.lusgs_omega;
    let lusgs_sweep = raw.lusgs_sweep;
    let lusgs_sweep_backward_damping = raw.lusgs_sweep_backward_damping;
    let _ = crate::solver::time::LuSgsConfig::parse(
        lusgs_omega,
        lusgs_sweep,
        lusgs_sweep_backward_damping,
    )?;
    let residual_smoothing = crate::solver::time::ResidualSmoothingConfig::parse(
        raw.residual_smoothing.unwrap_or(false),
        raw.residual_smoothing_epsilon,
        raw.residual_smoothing_sweeps,
    )?;
    let gmres_preconditioner = raw
        .gmres_preconditioner
        .as_deref()
        .map(crate::solver::GmresPreconditionerKind::parse)
        .transpose()?;
    Ok(CaseTimeConfig {
        mode,
        dt: raw.dt,
        cfl: raw.cfl,
        cfl_max: raw.cfl_max,
        final_time: raw.final_time,
        max_steps: raw.max_steps,
        min_steps: raw.min_steps,
        tolerance: raw.tolerance,
        local_time_step: raw.local_time_step.unwrap_or(false),
        cfl_ramp_steps: raw.cfl_ramp_steps,
        scheme,
        lusgs_omega,
        lusgs_sweep,
        lusgs_sweep_backward_damping,
        gmres_preconditioner,
        residual_smoothing,
    })
}

fn parse_sod_config(raw: &SodToml) -> SodCaseConfig {
    SodCaseConfig {
        diaphragm: raw.diaphragm.unwrap_or(0.5),
        final_time: raw.final_time.unwrap_or(0.2),
        cfl: raw.cfl.unwrap_or(0.4),
        reconstruction: raw.reconstruction.clone(),
        flux: raw.flux.clone(),
        limiter: raw.limiter.clone(),
    }
}

#[cfg(test)]
#[path = "case_tests.rs"]
mod case_tests;
