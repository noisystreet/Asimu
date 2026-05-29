//! TOML 算例解析（扩散 1D + 可压缩 3D NS）。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::boundary::{
    BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, BoundaryTomlFields,
};
use crate::config::SolverConfig;
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{Fields, FluidInitialConfig, InitialKind, InitialSet, ScalarField, ScalarInitial};
use crate::mesh::{BoundaryMesh, StructuredMesh1d, StructuredMesh3d};
use crate::physics::{IdealGasEoS, PhysicsConfig, FreestreamParams};

use super::validate_input_path;

/// 算例网格（1D 扩散 / 3D 可压缩）。
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
    pub solver: SolverConfig,
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
    solver: Option<SolverToml>,
}

#[derive(Debug, Deserialize)]
struct MeshToml {
    kind: String,
    cells: Option<usize>,
    length: Option<Real>,
    origin: Option<Real>,
    nx: Option<usize>,
    ny: Option<usize>,
    nz: Option<usize>,
    lx: Option<Real>,
    ly: Option<Real>,
    lz: Option<Real>,
    path: Option<PathBuf>,
    zone: Option<usize>,
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
struct SolverToml {
    max_iterations: u32,
    tolerance: Real,
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
    let mesh = parse_mesh(&raw.mesh, &raw.name, case_dir)?;
    let boundary = resolve_boundary_set(&mesh, &raw.boundary)?;
    let initial = resolve_initial_set(&raw.initial)?;
    let freestream = raw.freestream.as_ref().map(parse_freestream);
    let fluid_initial = FluidInitialConfig {
        freestream,
        scalars: initial.clone(),
    };
    let restart = raw.restart.map(|r| resolve_restart_path(r.path, case_dir));

    let solver = raw
        .solver
        .map(|s| SolverConfig {
            max_iterations: s.max_iterations,
            tolerance: s.tolerance,
        })
        .unwrap_or(SolverConfig {
            max_iterations: 1000,
            tolerance: 1.0e-8,
        });

    Ok(CaseSpec {
        name: raw.name,
        benchmark_id: raw.benchmark_id,
        mesh,
        physics,
        boundary,
        initial,
        fluid_initial,
        freestream,
        restart,
        solver,
    })
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

fn parse_mesh(raw: &MeshToml, name: &str, case_dir: Option<&Path>) -> Result<CaseMesh> {
    match raw.kind.as_str() {
        "structured_1d" => {
            let cells = raw.cells.ok_or_else(|| {
                AsimuError::Config("structured_1d 缺少 cells".to_string())
            })?;
            let mesh = StructuredMesh1d::new(
                name,
                cells,
                raw.origin.unwrap_or(0.0),
                raw.length.ok_or_else(|| {
                    AsimuError::Config("structured_1d 缺少 length".to_string())
                })?,
            )?;
            Ok(CaseMesh::Structured1d(mesh))
        }
        "structured_3d" => {
            let nx = raw.nx.ok_or_else(|| AsimuError::Config("structured_3d 缺少 nx".to_string()))?;
            let ny = raw.ny.ok_or_else(|| AsimuError::Config("structured_3d 缺少 ny".to_string()))?;
            let nz = raw.nz.ok_or_else(|| AsimuError::Config("structured_3d 缺少 nz".to_string()))?;
            let mesh = StructuredMesh3d::uniform_box(
                name,
                nx,
                ny,
                nz,
                raw.lx.unwrap_or(1.0),
                raw.ly.unwrap_or(1.0),
                raw.lz.unwrap_or(1.0),
            )?;
            Ok(CaseMesh::Structured3d(mesh))
        }
        "cgns" => load_cgns_mesh(raw, name, case_dir),
        other => Err(AsimuError::Config(format!("不支持的 mesh.kind \"{other}\""))),
    }
}

fn load_cgns_mesh(raw: &MeshToml, _name: &str, case_dir: Option<&Path>) -> Result<CaseMesh> {
    #[cfg(feature = "io-cgns")]
    {
        let rel = raw
            .path
            .as_ref()
            .ok_or_else(|| AsimuError::Config("cgns 网格缺少 path".to_string()))?;
        let path = resolve_restart_path(rel.clone(), case_dir);
        let zone = raw.zone.unwrap_or(1);
        let result = crate::io::load_cgns_zone(&path, zone)?;
        match result.mesh {
            crate::mesh::StructuredMesh::D3(mesh) => Ok(CaseMesh::Structured3d(mesh)),
            _ => Err(AsimuError::Mesh("CGNS zone 须为 3D structured".to_string())),
        }
    }
    #[cfg(not(feature = "io-cgns"))]
    {
        let _ = (raw, _name, case_dir);
        Err(AsimuError::Config(
            "cgns 网格须启用 feature io-cgns".to_string(),
        ))
    }
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
        let kind = InitialKind::from_toml(
            &ic.kind,
            ic.value,
            ic.left,
            ic.right,
            ic.data.clone(),
        )
        .ok_or_else(|| {
            AsimuError::Field(format!(
                "初始条件 \"{name}\" 无效：kind=\"{}\"",
                ic.kind
            ))
        })?;
        scalars.push(ScalarInitial::new(name.clone(), kind));
    }
    let set = InitialSet::new(scalars);
    set.validate()?;
    Ok(set)
}

fn resolve_boundary_set(mesh: &CaseMesh, boundaries: &BTreeMap<String, BoundaryToml>) -> Result<BoundarySet> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const BENCHMARK_CASE: &str = include_str!("../../tests/benchmarks/1d_diffusion_analytical/case.toml");

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
