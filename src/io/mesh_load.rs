//! case.toml 网格段解析（结构化 / CGNS、scale、metric）。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::mesh::{MeshMetricMode, MultiBlockStructuredMesh3d, StructuredMesh1d, StructuredMesh3d};

use super::CaseMesh;

#[derive(Debug, Clone)]
pub(super) struct MeshTomlFields {
    pub kind: String,
    pub cells: Option<usize>,
    pub ncells: Option<usize>,
    pub length: Option<Real>,
    pub origin: Option<Real>,
    pub nx: Option<usize>,
    pub ny: Option<usize>,
    pub nz: Option<usize>,
    pub lx: Option<Real>,
    pub ly: Option<Real>,
    pub lz: Option<Real>,
    pub path: Option<PathBuf>,
    pub scale: Option<Real>,
    pub metric: Option<String>,
    pub blocks: Vec<MeshBlockTomlFields>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct MeshBlockTomlFields {
    pub name: String,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub lx: Option<Real>,
    pub ly: Option<Real>,
    pub lz: Option<Real>,
}

#[derive(Debug)]
pub(super) struct ParsedMesh {
    pub mesh: CaseMesh,
    pub cgns_boundary: Option<BoundarySet>,
}

pub(super) fn parse_mesh_from_toml(
    raw: &MeshTomlFields,
    name: &str,
    case_dir: Option<&Path>,
) -> Result<ParsedMesh> {
    let metric_mode = parse_metric_mode(raw)?;
    let mut parsed = match raw.kind.as_str() {
        "structured_1d" => parse_structured_1d(raw, name),
        "structured_3d" => parse_structured_3d(raw, name),
        "multi_block_structured_3d" => parse_multiblock_structured_3d(raw, name),
        "cgns" => load_cgns_mesh(raw, name, case_dir),
        other => Err(AsimuError::Config(format!(
            "不支持的 mesh.kind \"{other}\""
        ))),
    }?;
    if let Some(scale) = raw.scale {
        parsed.mesh.scale_coordinates(scale)?;
    }
    if let CaseMesh::Structured3d(mesh) = &mut parsed.mesh {
        mesh.set_metric_mode(metric_mode);
        mesh.rebuild_metric_cache_if_needed()?;
    }
    if let CaseMesh::MultiBlockStructured3d(mesh) = &mut parsed.mesh {
        mesh.set_metric_mode(metric_mode);
        mesh.rebuild_metric_cache_if_needed()?;
    }
    Ok(parsed)
}

fn parse_structured_1d(raw: &MeshTomlFields, name: &str) -> Result<ParsedMesh> {
    let cells = raw
        .cells
        .or(raw.ncells)
        .ok_or_else(|| AsimuError::Config("structured_1d 缺少 cells（或 ncells）".to_string()))?;
    let mesh = StructuredMesh1d::new(
        name,
        cells,
        raw.origin.unwrap_or(0.0),
        raw.length
            .ok_or_else(|| AsimuError::Config("structured_1d 缺少 length".to_string()))?,
    )?;
    Ok(ParsedMesh {
        mesh: CaseMesh::Structured1d(mesh),
        cgns_boundary: None,
    })
}

fn parse_structured_3d(raw: &MeshTomlFields, name: &str) -> Result<ParsedMesh> {
    let nx = raw
        .nx
        .ok_or_else(|| AsimuError::Config("structured_3d 缺少 nx".to_string()))?;
    let ny = raw
        .ny
        .ok_or_else(|| AsimuError::Config("structured_3d 缺少 ny".to_string()))?;
    let nz = raw
        .nz
        .ok_or_else(|| AsimuError::Config("structured_3d 缺少 nz".to_string()))?;
    let mesh = StructuredMesh3d::uniform_box(
        name,
        nx,
        ny,
        nz,
        raw.lx.unwrap_or(1.0),
        raw.ly.unwrap_or(1.0),
        raw.lz.unwrap_or(1.0),
    )?;
    Ok(ParsedMesh {
        mesh: CaseMesh::Structured3d(mesh),
        cgns_boundary: None,
    })
}

fn parse_multiblock_structured_3d(raw: &MeshTomlFields, name: &str) -> Result<ParsedMesh> {
    if raw.blocks.is_empty() {
        return Err(AsimuError::Config(
            "multi_block_structured_3d 缺少 [[mesh.blocks]]".to_string(),
        ));
    }
    let mut blocks = Vec::with_capacity(raw.blocks.len());
    for block in &raw.blocks {
        blocks.push(StructuredMesh3d::uniform_box(
            &block.name,
            block.nx,
            block.ny,
            block.nz,
            block.lx.unwrap_or(1.0),
            block.ly.unwrap_or(1.0),
            block.lz.unwrap_or(1.0),
        )?);
    }
    Ok(ParsedMesh {
        mesh: CaseMesh::MultiBlockStructured3d(MultiBlockStructuredMesh3d::new(name, blocks)?),
        cgns_boundary: None,
    })
}

fn parse_metric_mode(raw: &MeshTomlFields) -> Result<MeshMetricMode> {
    match raw.metric.as_deref() {
        None if raw.kind == "cgns" => Ok(MeshMetricMode::Curvilinear),
        None => Ok(MeshMetricMode::Cartesian),
        Some("cartesian") => Ok(MeshMetricMode::Cartesian),
        Some("curvilinear") => Ok(MeshMetricMode::Curvilinear),
        Some(other) => Err(AsimuError::Config(format!(
            "不支持的 mesh.metric \"{other}\"（支持 cartesian | curvilinear）"
        ))),
    }
}

#[cfg(feature = "io-cgns")]
fn load_cgns_mesh(raw: &MeshTomlFields, name: &str, case_dir: Option<&Path>) -> Result<ParsedMesh> {
    let rel = raw
        .path
        .as_ref()
        .ok_or_else(|| AsimuError::Config("cgns 网格缺少 path".to_string()))?;
    let path = resolve_mesh_path(rel.clone(), case_dir)?;
    let loaded = crate::io::load_cgns_all_zones(&path)?;
    let multiblock = loaded.zones.len() > 1;
    let mut blocks = Vec::with_capacity(loaded.zones.len());
    let mut patches = Vec::new();

    for zone in loaded.zones {
        let crate::io::CgnsLoadResult { mesh, boundary, .. } = zone;
        let crate::mesh::StructuredMesh::D3(mesh) = mesh else {
            return Err(AsimuError::Mesh("CGNS zone 须为 3D structured".to_string()));
        };
        for patch in boundary.patches() {
            let mut patch = patch.clone();
            if multiblock {
                patch.name = format!("{}/{}", mesh.name, patch.name);
            }
            patches.push(patch);
        }
        blocks.push(mesh);
    }

    match blocks.len() {
        0 => Err(AsimuError::Mesh(
            "CGNS 文件不含 structured zone".to_string(),
        )),
        1 => Ok(ParsedMesh {
            mesh: CaseMesh::Structured3d(blocks.remove(0)),
            cgns_boundary: Some(BoundarySet::new(patches)),
        }),
        _ => Ok(ParsedMesh {
            mesh: CaseMesh::MultiBlockStructured3d(MultiBlockStructuredMesh3d::with_interfaces(
                name,
                blocks,
                loaded
                    .interfaces
                    .into_iter()
                    .map(|interface| crate::mesh::StructuredBlockInterface3d {
                        owner_block: interface.zone_name,
                        donor_block: interface.donor_name,
                        owner_range: interface.range,
                        donor_range: interface.donor_range,
                        transform: interface.transform,
                    })
                    .collect(),
            )?),
            cgns_boundary: Some(BoundarySet::new(patches)),
        }),
    }
}

#[cfg(not(feature = "io-cgns"))]
fn load_cgns_mesh(
    raw: &MeshTomlFields,
    _name: &str,
    _case_dir: Option<&Path>,
) -> Result<ParsedMesh> {
    let _ = raw.path.as_ref();
    Err(AsimuError::Config(
        "cgns 网格须启用 feature io-cgns".to_string(),
    ))
}

#[cfg(feature = "io-cgns")]
fn resolve_mesh_path(rel: PathBuf, case_dir: Option<&Path>) -> Result<PathBuf> {
    let label = rel.display().to_string();
    let mut candidates = Vec::new();
    if rel.is_absolute() {
        candidates.push(rel);
    } else {
        if let Some(dir) = case_dir {
            candidates.push(dir.join(&rel));
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        candidates.push(manifest.join(&rel));
        if let Some(parent) = manifest.parent() {
            candidates.push(parent.join(&rel));
        }
    }
    for candidate in candidates {
        if candidate.is_file() {
            return candidate.canonicalize().map_err(|err| {
                AsimuError::Io(std::io::Error::new(
                    err.kind(),
                    format!("无法解析网格路径 {}: {err}", candidate.display()),
                ))
            });
        }
    }
    Err(AsimuError::Config(format!(
        "找不到 CGNS 网格文件 \"{label}\""
    )))
}
