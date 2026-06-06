//! case.toml `[boundary]` 解析与 CGNS 覆盖。

use std::collections::BTreeMap;

use crate::boundary::{
    BoundaryKind, BoundaryPatch, BoundaryRegistry, BoundarySet, BoundaryTomlFields,
};
use crate::core::FaceId;
use crate::error::{AsimuError, Result};
use crate::mesh::BoundaryMesh;
use crate::physics::{FreestreamParams, PhysicsConfig};

use super::BoundaryToml;
use super::CaseMesh;

pub(super) fn resolve_case_boundary(
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
        apply_boundary_overrides(&mut boundary, toml_boundary)?;
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

fn resolve_boundary_set(
    mesh: &CaseMesh,
    boundaries: &BTreeMap<String, BoundaryToml>,
) -> Result<BoundarySet> {
    let mut patches = Vec::with_capacity(boundaries.len());
    for (logical_name, bc) in boundaries {
        let kind = parse_boundary_kind(logical_name, bc)?;
        let face_ids = resolve_mesh_logical_boundary(mesh, logical_name)?;
        patches.push(BoundaryPatch::new(logical_name.clone(), face_ids, kind));
    }
    BoundaryRegistry::validate_patches(&patches)?;
    Ok(BoundarySet::new(patches))
}

fn apply_boundary_overrides(
    boundary: &mut BoundarySet,
    overrides: &BTreeMap<String, BoundaryToml>,
) -> Result<()> {
    for (name, bc) in overrides {
        let kind = parse_boundary_kind(name, bc)?;
        if let Some(patch) = boundary.patches_mut().iter_mut().find(|p| p.name == *name) {
            patch.kind = kind;
        } else {
            return Err(AsimuError::Boundary(format!(
                "边界覆盖 \"{name}\" 在 CGNS patch 列表中不存在"
            )));
        }
    }
    Ok(())
}

fn parse_boundary_kind(name: &str, bc: &BoundaryToml) -> Result<BoundaryKind> {
    let fields = boundary_toml_fields(bc);
    BoundaryKind::from_toml(&fields)
        .ok_or_else(|| AsimuError::Boundary(format!("边界 \"{name}\" 无效：kind=\"{}\"", bc.kind)))
}

fn boundary_toml_fields(bc: &BoundaryToml) -> BoundaryTomlFields<'_> {
    BoundaryTomlFields {
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
    }
}

fn resolve_mesh_logical_boundary(mesh: &CaseMesh, logical_name: &str) -> Result<Vec<FaceId>> {
    match mesh {
        CaseMesh::Structured1d(m) => m.resolve_logical_boundary(logical_name),
        CaseMesh::MultiBlockStructured3d(m) => {
            if m.num_blocks() == 1 && m.interfaces().is_empty() {
                m.blocks()[0].mesh.resolve_logical_boundary(logical_name)
            } else {
                Err(AsimuError::Boundary(
                    "多块 3D 网格须使用 block_name/patch 前缀解析 [boundary]".to_string(),
                ))
            }
        }
    }
}
