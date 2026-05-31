//! TOML 算例解析（扩散 1D + 可压缩 3D NS）。

#[path = "case_compressible.rs"]
mod case_compressible;
#[path = "mesh_load.rs"]
mod mesh_load;

pub use case_compressible::{
    CaseObservabilityConfig, CaseOutputConfig, EulerCaseConfig, resolve_case_output_path,
};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::boundary::{
    BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, BoundaryTomlFields,
};
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{
    Fields, FluidInitialConfig, InitialKind, InitialSet, ScalarField, ScalarInitial,
};
use crate::mesh::{BoundaryMesh, StructuredMesh1d, StructuredMesh3d};
use crate::physics::{FreestreamParams, IdealGasEoS, PhysicsConfig};

use super::validate_input_path;
use case_compressible::{
    EulerToml, ObservabilityToml, OutputToml, parse_euler_config, parse_observability, parse_output,
};

/// 算例网格（1D 扩散 / 3D 可压缩）。
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum CaseMesh {
    Structured1d(StructuredMesh1d),
    Structured3d(StructuredMesh3d),
}

impl CaseMesh {
    #[must_use]
    pub fn num_cells(&self) -> usize {
        match self {
            Self::Structured1d(m) => m.num_cells(),
            Self::Structured3d(m) => m.num_cells(),
        }
    }

    pub fn as_1d(&self) -> Result<&StructuredMesh1d> {
        match self {
            Self::Structured1d(m) => Ok(m),
            _ => Err(AsimuError::Mesh("算例非 1D 网格".to_string())),
        }
    }

    pub fn as_3d(&self) -> Result<&StructuredMesh3d> {
        match self {
            Self::Structured3d(m) => Ok(m),
            _ => Err(AsimuError::Mesh("算例非 3D 网格".to_string())),
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
            Self::Structured3d(mesh) => mesh.scale_coordinates(factor),
        }
        if let CaseMesh::Structured3d(mesh) = self {
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
    pub output: Option<CaseOutputConfig>,
    pub observability: Option<CaseObservabilityConfig>,
    pub case_dir: Option<PathBuf>,
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
    /// log₁₀(RMS(ρ̇)) 收敛容差；设有限值时与 `max_steps` 成对用于残差早停。
    pub tolerance: Option<Real>,
    pub local_time_step: bool,
    pub cfl_ramp_steps: Option<u64>,
    /// 时间积分格式：`rk4`（默认）或 `euler`（一阶前向 Euler，排错对照）。
    pub scheme: Option<crate::solver::time::TimeIntegrationScheme>,
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
            tolerance: None,
            local_time_step: false,
            cfl_ramp_steps: None,
            scheme: None,
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
        let eos = self.physics.eos()?;
        let fs = self
            .freestream
            .or(self.fluid_initial.freestream)
            .ok_or_else(|| AsimuError::Field("须指定 [freestream] 或 [restart]".to_string()))?;
        crate::field::ConservedFields::from_freestream(self.mesh.num_cells(), &eos, &fs)
    }

    pub fn is_compressible(&self) -> bool {
        self.physics.eos.is_some()
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
    #[cfg_attr(not(feature = "io-cgns"), allow(dead_code))]
    zone: Option<usize>,
    scale: Option<Real>,
    metric: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PhysicsToml {
    diffusivity: Option<Real>,
    gamma: Option<Real>,
    gas_constant: Option<Real>,
}

#[derive(Debug, Deserialize)]
struct BoundaryToml {
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
    tolerance: Option<Real>,
    local_time_step: Option<bool>,
    cfl_ramp_steps: Option<u64>,
    scheme: Option<String>,
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
    let physics = parse_physics(&raw.physics)?;
    let parsed =
        mesh_load::parse_mesh_from_toml(&mesh_toml_fields(&raw.mesh), &raw.name, case_dir)?;
    let boundary = resolve_case_boundary(
        &parsed.mesh,
        parsed.cgns_boundary,
        &raw.boundary,
        raw.freestream.as_ref().map(parse_freestream),
        &physics,
        raw.euler.is_some(),
    )?;
    let initial = resolve_initial_set(&raw.initial)?;
    let freestream = raw.freestream.as_ref().map(parse_freestream);
    let fluid_initial = FluidInitialConfig {
        freestream,
        scalars: initial.clone(),
    };
    let restart = raw.restart.map(|r| resolve_restart_path(r.path, case_dir));

    let euler = raw.euler.as_ref().map(parse_euler_config).transpose()?;
    let time = parse_time_config(raw.time.as_ref(), raw.sod.is_some())?;
    let sod = raw.sod.as_ref().map(parse_sod_config);

    let case = CaseSpec {
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
        sod,
        euler,
        output: raw.output.as_ref().map(parse_output),
        observability: raw.observability.as_ref().map(parse_observability),
        case_dir: case_dir.map(Path::to_path_buf),
    };
    case.warn_config_inconsistencies();
    Ok(case)
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
        zone: raw.zone,
        scale: raw.scale,
        metric: raw.metric.clone(),
    }
}

fn resolve_case_boundary(
    mesh: &CaseMesh,
    cgns_boundary: Option<BoundarySet>,
    toml_boundary: &BTreeMap<String, BoundaryToml>,
    freestream: Option<FreestreamParams>,
    physics: &PhysicsConfig,
    euler: bool,
) -> Result<BoundarySet> {
    let has_cgns_boundary = cgns_boundary.is_some();
    let mut boundary = if let Some(cgns) = cgns_boundary {
        cgns
    } else if !toml_boundary.is_empty() {
        resolve_boundary_set(mesh, toml_boundary)?
    } else {
        BoundarySet::default()
    };
    if has_cgns_boundary && !toml_boundary.is_empty() {
        apply_boundary_overrides(&mut boundary, mesh, toml_boundary)?;
    }
    if let Some(fs) = freestream {
        let eos = physics.eos()?;
        boundary.apply_freestream(&fs, &eos)?;
    }
    if euler {
        boundary.apply_inviscid_euler_walls();
    }
    Ok(boundary)
}

fn parse_physics(raw: &PhysicsToml) -> Result<PhysicsConfig> {
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
    Ok(PhysicsConfig {
        diffusivity: raw.diffusivity,
        eos,
    })
}

fn resolve_restart_path(path: PathBuf, case_dir: Option<&Path>) -> PathBuf {
    if path.is_absolute() {
        path
    } else if let Some(dir) = case_dir {
        dir.join(path)
    } else {
        path
    }
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

fn resolve_boundary_set(
    mesh: &CaseMesh,
    boundaries: &BTreeMap<String, BoundaryToml>,
) -> Result<BoundarySet> {
    let mut patches = Vec::with_capacity(boundaries.len());
    for (logical_name, bc) in boundaries {
        let fields = BoundaryTomlFields {
            kind: &bc.kind,
            value: bc.value,
            flux: bc.flux,
            mach: bc.mach,
            pressure: bc.pressure,
            temperature: bc.temperature,
            alpha: bc.alpha,
            beta: bc.beta,
            total_pressure: bc.total_pressure,
            total_temperature: bc.total_temperature,
            static_pressure: bc.static_pressure,
            velocity_direction: bc.velocity_direction,
            no_slip: bc.no_slip,
            heat: bc.heat.as_deref(),
            wall_temperature: bc.wall_temperature,
            heat_flux: bc.heat_flux,
            partner: bc.partner.as_deref(),
            turbulent_k: bc.turbulent_k,
            turbulent_omega: bc.turbulent_omega,
            supersonic: bc.supersonic,
        };
        let kind = BoundaryKind::from_toml(&fields).ok_or_else(|| {
            AsimuError::Boundary(format!(
                "边界 \"{logical_name}\" 无效：kind=\"{}\"",
                bc.kind
            ))
        })?;
        let face_ids = match mesh {
            CaseMesh::Structured1d(m) => m.resolve_logical_boundary(logical_name)?,
            CaseMesh::Structured3d(m) => m.resolve_logical_boundary(logical_name)?,
        };
        patches.push(BoundaryPatch::new(logical_name.clone(), face_ids, kind));
    }
    BoundaryRegistry::validate_patches(&patches)?;
    Ok(BoundarySet::new(patches))
}

/// 用 `[boundary]` 表覆盖 CGNS 已加载 patch（按 patch 名匹配）。
fn apply_boundary_overrides(
    boundary: &mut BoundarySet,
    mesh: &CaseMesh,
    overrides: &BTreeMap<String, BoundaryToml>,
) -> Result<()> {
    for (name, bc) in overrides {
        let fields = BoundaryTomlFields {
            kind: &bc.kind,
            value: bc.value,
            flux: bc.flux,
            mach: bc.mach,
            pressure: bc.pressure,
            temperature: bc.temperature,
            alpha: bc.alpha,
            beta: bc.beta,
            total_pressure: bc.total_pressure,
            total_temperature: bc.total_temperature,
            static_pressure: bc.static_pressure,
            velocity_direction: bc.velocity_direction,
            no_slip: bc.no_slip,
            heat: bc.heat.as_deref(),
            wall_temperature: bc.wall_temperature,
            heat_flux: bc.heat_flux,
            partner: bc.partner.as_deref(),
            turbulent_k: bc.turbulent_k,
            turbulent_omega: bc.turbulent_omega,
            supersonic: bc.supersonic,
        };
        let kind = BoundaryKind::from_toml(&fields).ok_or_else(|| {
            AsimuError::Boundary(format!("边界覆盖 \"{name}\" 无效：kind=\"{}\"", bc.kind))
        })?;
        if let Some(patch) = boundary.patches_mut().iter_mut().find(|p| p.name == *name) {
            patch.kind = kind;
        } else {
            return Err(AsimuError::Boundary(format!(
                "边界覆盖 \"{name}\" 在 CGNS patch 列表中不存在"
            )));
        }
    }
    let _ = mesh;
    Ok(())
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
    Ok(CaseTimeConfig {
        mode,
        dt: raw.dt,
        cfl: raw.cfl,
        cfl_max: raw.cfl_max,
        final_time: raw.final_time,
        max_steps: raw.max_steps,
        tolerance: raw.tolerance,
        local_time_step: raw.local_time_step.unwrap_or(false),
        cfl_ramp_steps: raw.cfl_ramp_steps,
        scheme,
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
mod tests {
    use super::*;

    const BENCHMARK_CASE: &str =
        include_str!("../../tests/benchmarks/1d_diffusion_analytical/case.toml");

    #[test]
    fn parses_diffusion_benchmark() {
        let case = parse_case_toml(BENCHMARK_CASE, None).expect("parse");
        assert_eq!(case.name, "1d_diffusion_analytical");
        assert_eq!(case.mesh.as_1d().expect("1d").num_cells(), 32);
        assert!(!case.is_compressible());
    }

    #[test]
    fn parses_compressible_3d_case() {
        let content = r#"
name = "box_cns"
[mesh]
kind = "structured_3d"
nx = 4
ny = 4
nz = 4
lx = 1.0
ly = 1.0
lz = 1.0
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.i_min]
kind = "wall"
no_slip = true
heat = "adiabatic"
[boundary.i_max]
kind = "farfield"
mach = 0.3
pressure = 101325.0
temperature = 288.15
[boundary.j_min]
kind = "symmetry"
[boundary.j_max]
kind = "symmetry"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "outlet"
static_pressure = 100000.0
"#;
        let case = parse_case_toml(content, None).expect("parse");
        assert!(case.is_compressible());
        assert_eq!(case.mesh.num_cells(), 64);
        assert_eq!(case.boundary.patches().len(), 6);
        let fields = case.build_conserved_fields().expect("ic");
        assert_eq!(fields.num_cells(), 64);
    }

    #[test]
    fn parses_sod_benchmark_case() {
        let content = include_str!("../../tests/benchmarks/sod_1d/case.toml");
        let case = parse_case_toml(content, None).expect("parse");
        assert_eq!(case.benchmark_id.as_deref(), Some("sod_1d"));
        assert_eq!(case.mesh.as_1d().expect("1d").num_cells(), 100);
        assert!(case.is_compressible());
        let sod = case.sod.expect("sod");
        assert_eq!(case.time.mode, CaseTimeMode::Transient);
        let inviscid = sod.inviscid();
        assert_eq!(inviscid.short_label(), "muscl_roe");
        assert_eq!(inviscid.limiter_label(), "van_albada");
    }

    #[test]
    fn parses_sod_muscl_hllc_case() {
        let content = include_str!("../../tests/benchmarks/sod_1d/case_muscl_hllc.toml");
        let case = parse_case_toml(content, None).expect("parse");
        let sod = case.sod.expect("sod");
        let inviscid = sod.inviscid();
        assert_eq!(inviscid.short_label(), "muscl_hllc");
        assert_eq!(inviscid.limiter_label(), "van_albada");
    }

    #[test]
    fn parses_inlet_and_turbulent_inlet() {
        let content = r#"
name = "inlet_test"
[mesh]
kind = "structured_3d"
nx = 2
ny = 2
nz = 2
[physics]
gamma = 1.4
gas_constant = 287.0
[freestream]
mach = 0.1
[boundary.i_min]
kind = "turbulent_inlet"
total_pressure = 110000.0
total_temperature = 320.0
turbulent_k = 1.0
turbulent_omega = 100.0
velocity_direction = [1.0, 0.0, 0.0]
[boundary.i_max]
kind = "inlet"
total_pressure = 105000.0
total_temperature = 300.0
[boundary.j_min]
kind = "wall"
[boundary.j_max]
kind = "wall"
[boundary.k_min]
kind = "wall"
[boundary.k_max]
kind = "wall"
"#;
        let case = parse_case_toml(content, None).expect("parse");
        assert!(case.boundary.find("i_min").is_some());
    }
}
