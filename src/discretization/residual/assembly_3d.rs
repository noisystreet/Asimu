//! 3D 结构化网格无粘残差装配。

use crate::boundary::BoundarySet;
use crate::core::Vector3;
use crate::discretization::{BoundaryGhostBuffer, RoeFluxConfig, face_inviscid_flux};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

use super::{accumulate_boundary_face, accumulate_interior_face};

struct BoundaryAssembly3d<'a> {
    mesh: &'a dyn BoundaryMesh3d,
    eos: &'a IdealGasEoS,
    config: &'a RoeFluxConfig,
    boundaries: &'a BoundarySet,
    ghosts: &'a BoundaryGhostBuffer,
    volume: crate::core::Real,
}

/// 装配 3D 均匀结构化网格无粘 Euler 残差（内部面 + 边界 ghost）。
pub fn assemble_inviscid_residual_3d(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
    boundaries: &BoundarySet,
    ghosts: &BoundaryGhostBuffer,
) -> Result<()> {
    let n = mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "场/残差尺寸 {} 与网格单元数 {n} 不一致",
            fields.num_cells()
        )));
    }
    residual.clear();
    let volume = mesh.cell_volume();
    assemble_i_faces(mesh, fields, residual, eos, config, volume)?;
    assemble_j_faces(mesh, fields, residual, eos, config, volume)?;
    assemble_k_faces(mesh, fields, residual, eos, config, volume)?;
    assemble_boundary_faces_3d(
        fields,
        residual,
        &BoundaryAssembly3d {
            mesh,
            eos,
            config,
            boundaries,
            ghosts,
            volume,
        },
    )?;
    Ok(())
}

fn assemble_i_faces(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
    volume: crate::core::Real,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    let area = mesh.cell_dy() * mesh.cell_dz();
    let normal = Vector3::new(1.0, 0.0, 0.0);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let left = fields.cell_state(owner)?;
                let right = fields.cell_state(neighbor)?;
                let flux = face_inviscid_flux(&left, &right, normal, eos, config)?;
                accumulate_interior_face(residual, owner, neighbor, &flux, area, volume, volume)?;
            }
        }
    }
    Ok(())
}

fn assemble_j_faces(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
    volume: crate::core::Real,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    let area = mesh.cell_dx() * mesh.cell_dz();
    let normal = Vector3::new(0.0, 1.0, 0.0);
    for k in 0..nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j + 1, k);
                let left = fields.cell_state(owner)?;
                let right = fields.cell_state(neighbor)?;
                let flux = face_inviscid_flux(&left, &right, normal, eos, config)?;
                accumulate_interior_face(residual, owner, neighbor, &flux, area, volume, volume)?;
            }
        }
    }
    Ok(())
}

fn assemble_k_faces(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
    volume: crate::core::Real,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    let area = mesh.cell_dx() * mesh.cell_dy();
    let normal = Vector3::new(0.0, 0.0, 1.0);
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j, k + 1);
                let left = fields.cell_state(owner)?;
                let right = fields.cell_state(neighbor)?;
                let flux = face_inviscid_flux(&left, &right, normal, eos, config)?;
                accumulate_interior_face(residual, owner, neighbor, &flux, area, volume, volume)?;
            }
        }
    }
    Ok(())
}

fn assemble_boundary_faces_3d(
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    ctx: &BoundaryAssembly3d<'_>,
) -> Result<()> {
    for patch in ctx.boundaries.patches() {
        for &face in &patch.face_ids {
            let owner_id = ctx.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let owner_state = fields.cell_state(owner)?;
            let geom = ctx.mesh.face_geometry_3d(face)?;
            let ghost = ctx.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
            })?;
            let flux = face_inviscid_flux(
                &owner_state,
                &ghost.conserved,
                geom.normal,
                ctx.eos,
                ctx.config,
            )?;
            accumulate_boundary_face(residual, owner, &flux, geom.area, ctx.volume)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
    use crate::mesh::BoundaryMesh;
    use crate::physics::FreestreamParams;

    #[test]
    fn uniform_freestream_with_farfield_has_near_zero_rhs() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 3, 3, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut patches = Vec::new();
        for name in ["i_min", "i_max", "j_min", "j_max", "k_min", "k_max"] {
            let faces = mesh.resolve_logical_boundary(name).expect("faces");
            patches.push(BoundaryPatch::new(
                name,
                faces,
                BoundaryKind::Farfield {
                    mach: fs.mach,
                    pressure: fs.pressure,
                    temperature: fs.temperature,
                    alpha: 0.0,
                    beta: 0.0,
                },
            ));
        }
        let boundary_set = BoundarySet::new(patches);
        let mut ghosts = BoundaryGhostBuffer::new();
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary_set,
            &fields,
            &mut ghosts,
            &eos,
            &fs,
        )
        .expect("bc");
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        assemble_inviscid_residual_3d(
            &mesh,
            &fields,
            &mut rhs,
            &eos,
            &RoeFluxConfig::default(),
            &boundary_set,
            &ghosts,
        )
        .expect("assemble");
        assert!(rhs.density.values().iter().all(|&v| v.abs() < 1.0e-8));
        assert!(rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-6));
        assert!(rhs.total_energy.values().iter().all(|&v| v.abs() < 1.0e-6));
    }
}
