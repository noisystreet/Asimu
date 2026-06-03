//! 3D 结构化网格无粘残差装配。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::{BoundaryGhostBuffer, InviscidFluxConfig};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

use super::face_flux_3d::{BoundaryInviscidFluxInput, inviscid_boundary_face_flux};
use super::muscl_stencil_3d::{InteriorFaceFlux3d, flux_at_i_face, flux_at_j_face, flux_at_k_face};
use super::{accumulate_boundary_face, accumulate_interior_face, is_degenerate_volume};

/// 3D 无粘残差装配上下文（控制参数个数）。
pub struct InviscidAssembly3dParams<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub min_pressure: Real,
}

struct BoundaryAssembly3d<'a> {
    mesh: &'a dyn BoundaryMesh3d,
    structured: &'a StructuredMesh3d,
    params: &'a InviscidAssembly3dParams<'a>,
}

/// 装配 3D 均匀结构化网格无粘 Euler 残差（内部面 + 边界 ghost）。
pub fn assemble_inviscid_residual_3d(
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    params: &InviscidAssembly3dParams<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "场/残差尺寸 {} 与网格单元数 {n} 不一致",
            fields.num_cells()
        )));
    }
    residual.clear();
    {
        let _span = info_span!("assemble_faces", dim = "i").entered();
        assemble_i_faces(mesh, residual, params)?;
    }
    {
        let _span = info_span!("assemble_faces", dim = "j").entered();
        assemble_j_faces(mesh, residual, params)?;
    }
    {
        let _span = info_span!("assemble_faces", dim = "k").entered();
        assemble_k_faces(mesh, residual, params)?;
    }
    {
        let _span = info_span!("assemble_faces", dim = "boundary").entered();
        assemble_boundary_faces_3d(
            residual,
            &BoundaryAssembly3d {
                mesh,
                structured: mesh,
                params,
            },
        )?;
    }
    Ok(())
}

fn assemble_i_faces(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidual,
    params: &InviscidAssembly3dParams<'_>,
) -> Result<()> {
    let eos = params.eos;
    let config = params.config;
    let primitives = params.primitives;
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
                    primitives,
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
    residual: &mut ConservedResidual,
    params: &InviscidAssembly3dParams<'_>,
) -> Result<()> {
    let eos = params.eos;
    let config = params.config;
    let primitives = params.primitives;
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
                    primitives,
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
    residual: &mut ConservedResidual,
    params: &InviscidAssembly3dParams<'_>,
) -> Result<()> {
    let eos = params.eos;
    let config = params.config;
    let primitives = params.primitives;
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
                    primitives,
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
    residual: &mut ConservedResidual,
    ctx: &BoundaryAssembly3d<'_>,
) -> Result<()> {
    let mesh = ctx.structured;
    let params = ctx.params;
    for patch in params.boundaries.patches() {
        for &face in &patch.face_ids {
            let owner_id = ctx.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let geom = ctx.mesh.face_geometry_3d(face)?;
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
            })?;
            let flux = inviscid_boundary_face_flux(BoundaryInviscidFluxInput {
                mesh: ctx.mesh,
                structured: mesh,
                primitives: params.primitives,
                eos: params.eos,
                config: params.config,
                min_pressure: params.min_pressure,
                face,
                exterior: ghost.conserved,
            })?;
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
    use crate::core::approx_eq;
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
    use crate::mesh::MeshMetricMode;

    fn assemble_uniform_freestream(
        side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
        config: &InviscidFluxConfig,
        metric_mode: MeshMetricMode,
    ) -> ConservedResidual {
        let (mut mesh, boundary_set, fields, ghosts) =
            uniform_farfield_box(3, 3, 3, 1.0, 1.0, 1.0, side);
        mesh.set_metric_mode(metric_mode);
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, side.eos, side.min_pressure)
            .expect("fill");
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let params = InviscidAssembly3dParams {
            mesh: &mesh,
            eos: side.eos,
            config,
            boundaries: &boundary_set,
            ghosts: &ghosts,
            primitives: &primitives,
            min_pressure: side.min_pressure,
        };
        assemble_inviscid_residual_3d(&fields, &mut rhs, &params).expect("assemble");
        rhs
    }

    #[test]
    fn uniform_freestream_with_farfield_has_near_zero_rhs() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        pair.for_each_inviscid_side(|side| {
            let rhs = assemble_uniform_freestream(
                side,
                &InviscidFluxConfig::default(),
                MeshMetricMode::Cartesian,
            );
            assert!(
                rhs.density.values().iter().all(|&v| v.abs() < 1.0e-8),
                "{} density rhs",
                side.label
            );
            assert!(
                rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-6),
                "{} momentum rhs",
                side.label
            );
            assert!(
                rhs.total_energy.values().iter().all(|&v| v.abs() < 1.0e-6),
                "{} energy rhs",
                side.label
            );
        });
    }

    #[test]
    fn uniform_freestream_with_muscl_hllc_has_near_zero_rhs() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        pair.for_each_inviscid_side(|side| {
            let rhs = assemble_uniform_freestream(
                side,
                &InviscidFluxConfig::muscl_hllc(),
                MeshMetricMode::Cartesian,
            );
            assert!(
                rhs.density.values().iter().all(|&v| v.abs() < 1.0e-8),
                "{} density rhs",
                side.label
            );
            assert!(
                rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-6),
                "{} momentum rhs",
                side.label
            );
            assert!(
                rhs.total_energy.values().iter().all(|&v| v.abs() < 1.0e-6),
                "{} energy rhs",
                side.label
            );
        });
    }

    #[test]
    fn curvilinear_metrics_match_cartesian_rhs_on_uniform_box() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_dimensional();
        let config = InviscidFluxConfig::muscl_hllc();
        let cart = assemble_uniform_freestream(&side, &config, MeshMetricMode::Cartesian);
        let curv = assemble_uniform_freestream(&side, &config, MeshMetricMode::Curvilinear);
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
        use crate::physics::FreestreamContext;

        let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
        if !case_path.is_file() {
            return;
        }
        if !cfg!(feature = "io-cgns") {
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
        let fs_ctx = FreestreamContext::new(&eos, case.reference.as_ref(), None);
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
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let config = InviscidFluxConfig::roe_first_order();
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let params = InviscidAssembly3dParams {
            mesh,
            eos: &eos,
            config: &config,
            boundaries: &case.boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            min_pressure: 1.0e-6,
        };
        assemble_inviscid_residual_3d(&fields, &mut rhs, &params).expect("assemble");

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
