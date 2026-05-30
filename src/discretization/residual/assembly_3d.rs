//! 3D 结构化网格无粘残差装配。

use crate::boundary::BoundarySet;
use crate::discretization::{BoundaryGhostBuffer, InviscidFluxConfig};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

use super::muscl_stencil_3d::{
    BoundaryFaceFlux3d, InteriorFaceFlux3d, flux_at_boundary_face, flux_at_i_face, flux_at_j_face,
    flux_at_k_face,
};
use super::{accumulate_boundary_face, accumulate_interior_face, is_degenerate_volume};

struct BoundaryAssembly3d<'a> {
    mesh: &'a dyn BoundaryMesh3d,
    structured: &'a StructuredMesh3d,
    eos: &'a IdealGasEoS,
    config: &'a InviscidFluxConfig,
    boundaries: &'a BoundarySet,
    ghosts: &'a BoundaryGhostBuffer,
}

/// 装配 3D 均匀结构化网格无粘 Euler 残差（内部面 + 边界 ghost）。
pub fn assemble_inviscid_residual_3d(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
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
    assemble_i_faces(mesh, fields, residual, eos, config)?;
    assemble_j_faces(mesh, fields, residual, eos, config)?;
    assemble_k_faces(mesh, fields, residual, eos, config)?;
    assemble_boundary_faces_3d(
        fields,
        residual,
        &BoundaryAssembly3d {
            mesh,
            structured: mesh,
            eos,
            config,
            boundaries,
            ghosts,
        },
    )?;
    Ok(())
}

fn assemble_i_faces(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                let ctx = InteriorFaceFlux3d {
                    fields,
                    mesh,
                    eos,
                    config,
                    normal: face.normal,
                };
                let flux = flux_at_i_face(&ctx, i, j, k)?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i + 1, j, k).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    owner_volume,
                    neighbor_volume,
                )?;
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
    config: &InviscidFluxConfig,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j + 1, k);
                let face = mesh.j_face_metric(i, j, k);
                let ctx = InteriorFaceFlux3d {
                    fields,
                    mesh,
                    eos,
                    config,
                    normal: face.normal,
                };
                let flux = flux_at_j_face(&ctx, i, j, k)?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i, j + 1, k).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    owner_volume,
                    neighbor_volume,
                )?;
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
    config: &InviscidFluxConfig,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j, k + 1);
                let face = mesh.k_face_metric(i, j, k);
                let ctx = InteriorFaceFlux3d {
                    fields,
                    mesh,
                    eos,
                    config,
                    normal: face.normal,
                };
                let flux = flux_at_k_face(&ctx, i, j, k)?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i, j, k + 1).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    owner_volume,
                    neighbor_volume,
                )?;
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
    let mesh = ctx.structured;
    let flux_ctx = BoundaryFaceFlux3d {
        fields,
        mesh,
        eos: ctx.eos,
        config: ctx.config,
    };
    for patch in ctx.boundaries.patches() {
        for &face in &patch.face_ids {
            let owner_id = ctx.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let geom = ctx.mesh.face_geometry_3d(face)?;
            let ghost = ctx.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
            })?;
            let flux = flux_at_boundary_face(&flux_ctx, face, ghost.conserved, geom.normal)?;
            let (logical, local) = crate::mesh::LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            let owner_volume = mesh.cell_metric(i, j, k).volume;
            if is_degenerate_volume(owner_volume) {
                continue;
            }
            accumulate_boundary_face(residual, owner, &flux, geom.area, owner_volume)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::core::approx_eq;
    use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
    use crate::mesh::{BoundaryMesh, MeshMetricMode};
    use crate::physics::FreestreamParams;

    fn assemble_uniform_freestream(
        config: &InviscidFluxConfig,
        metric_mode: MeshMetricMode,
    ) -> ConservedResidual {
        let mut mesh = StructuredMesh3d::uniform_box("box", 3, 3, 3, 1.0, 1.0, 1.0).expect("mesh");
        mesh.set_metric_mode(metric_mode);
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
            config,
            &boundary_set,
            &ghosts,
        )
        .expect("assemble");
        rhs
    }

    #[test]
    fn uniform_freestream_with_farfield_has_near_zero_rhs() {
        let rhs =
            assemble_uniform_freestream(&InviscidFluxConfig::default(), MeshMetricMode::Cartesian);
        assert!(rhs.density.values().iter().all(|&v| v.abs() < 1.0e-8));
        assert!(rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-6));
        assert!(rhs.total_energy.values().iter().all(|&v| v.abs() < 1.0e-6));
    }

    #[test]
    fn uniform_freestream_with_muscl_hllc_has_near_zero_rhs() {
        let rhs = assemble_uniform_freestream(
            &InviscidFluxConfig::muscl_hllc(),
            MeshMetricMode::Cartesian,
        );
        assert!(rhs.density.values().iter().all(|&v| v.abs() < 1.0e-8));
        assert!(rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-6));
        assert!(rhs.total_energy.values().iter().all(|&v| v.abs() < 1.0e-6));
    }

    #[test]
    fn curvilinear_metrics_match_cartesian_rhs_on_uniform_box() {
        let config = InviscidFluxConfig::muscl_hllc();
        let cart = assemble_uniform_freestream(&config, MeshMetricMode::Cartesian);
        let curv = assemble_uniform_freestream(&config, MeshMetricMode::Curvilinear);
        for i in 0..cart.num_cells() {
            assert!(approx_eq(
                cart.density.values()[i],
                curv.density.values()[i],
                1.0e-10
            ));
            assert!(approx_eq(
                cart.momentum_x.values()[i],
                curv.momentum_x.values()[i],
                1.0e-10
            ));
            assert!(approx_eq(
                cart.total_energy.values()[i],
                curv.total_energy.values()[i],
                1.0e-10
            ));
        }
    }

    /// 圆柱 CGNS 网格 + case 边界：全场均匀来流下装配 RHS（`--nocapture` 打印分区统计）。
    #[test]
    fn cylinder_uniform_freestream_residual_is_reported_when_present() {
        use std::path::PathBuf;

        use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
        use crate::io::{CaseMesh, load_case};
        use crate::mesh::BoundaryMesh;

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        let case = load_case(&case_path).expect("load case");
        let CaseMesh::Structured3d(mesh) = &case.mesh else {
            panic!("expected 3d mesh");
        };
        let eos = case.physics.eos().expect("eos");
        let fs = case.freestream.expect("freestream");
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut ghosts = BoundaryGhostBuffer::new();
        apply_compressible_boundary_conditions(
            mesh,
            &case.boundary,
            &fields,
            &mut ghosts,
            &eos,
            &fs,
        )
        .expect("bc");
        let config = InviscidFluxConfig::roe_first_order();
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        assemble_inviscid_residual_3d(
            mesh,
            &fields,
            &mut rhs,
            &eos,
            &config,
            &case.boundary,
            &ghosts,
        )
        .expect("assemble");

        let rho_rms = rhs.density_rms_norm();
        let rho_max = rhs
            .density
            .values()
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        eprintln!("=== cylinder 均匀来流 RHS 诊断 ===");
        eprintln!("log10(RMS(rho_dot)) = {:.4}", rho_rms.log10());
        eprintln!("max |rho_dot| = {rho_max:.6e}");

        for patch in case.boundary.patches() {
            let mut patch_max = 0.0_f64;
            for &face in &patch.face_ids {
                let owner = mesh.face_owner(face).expect("owner").index() as usize;
                patch_max = patch_max.max(rhs.density.values()[owner].abs());
            }
            eprintln!("  patch {:<8} max |rho_dot| = {patch_max:.6e}", patch.name);
        }

        // 出口 patch：均匀来流 + 零梯度超声速出口，通量应近零（法向修正后）。
        if let Some(outlet) = case.boundary.find("dom-3") {
            let mut outlet_max = 0.0_f64;
            for &face in &outlet.face_ids {
                let owner = mesh.face_owner(face).expect("owner").index() as usize;
                outlet_max = outlet_max.max(rhs.density.values()[owner].abs());
            }
            let sample = outlet.face_ids[81.min(outlet.face_ids.len() - 1)];
            let g = mesh.face_geometry_3d(sample).expect("geom");
            eprintln!(
                "  dom-3 出口法向(样本 i=81): n=({:.3},{:.3},{:.3})",
                g.normal.x, g.normal.y, g.normal.z
            );
            eprintln!("  dom-3 判定: outlet_max={outlet_max:.6e} (均匀流期望 < 1e-2)");
        }

        // 对称面/壁面在均匀来流下本就不满足 BC，RHS 大属预期；出口应近零。
    }
}

#[cfg(test)]
#[path = "assembly_3d_decomposition_tests.rs"]
mod decomposition_tests;
