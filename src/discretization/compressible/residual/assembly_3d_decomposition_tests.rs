//! 圆柱边界面通量分解诊断（独立测试模块，满足复杂度门禁）。

use super::super::muscl_stencil_3d::{BoundaryFaceFlux3d, flux_at_boundary_face};
use super::super::{accumulate_boundary_face, is_degenerate_volume};
use super::assemble_inviscid_residual_3d;
use crate::core::Vector3;
use crate::discretization::InviscidFluxConfig;
use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
use crate::field::{ConservedFields, ConservedResidual};
use crate::io::load_case;
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, FaceMetric, LogicalFace3d, StructuredMesh3d};
use crate::physics::FreestreamContext;
use crate::physics::IdealGasEoS;

#[test]
fn cylinder_boundary_flux_decomposition_when_present() {
    use std::path::PathBuf;

    let case_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("case_cylinder/case.toml");
    if !case_path.is_file() {
        return;
    }
    if !cfg!(feature = "io-cgns") {
        return;
    }
    let case = load_case(&case_path).expect("load case");
    let mesh = case.mesh.as_3d().expect("expected 3d mesh");
    let eos = case.physics.eos().expect("eos");
    let fs = case.freestream.expect("freestream");
    let fs_ctx = FreestreamContext::new(&eos, case.reference.as_ref(), None);
    let fields =
        ConservedFields::from_freestream_context(mesh.num_cells(), &fs_ctx, &fs).expect("fields");
    let mut ghosts = BoundaryGhostBuffer::new();
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
    let mut primitives = crate::field::PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
    primitives
        .fill_from_conserved(&fields, &eos, 1.0e-6)
        .expect("fill");
    let config = InviscidFluxConfig::roe_first_order();
    let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
    let params = super::InviscidAssembly3dParams {
        mesh,
        eos: &eos,
        config: &config,
        boundaries: &case.boundary,
        ghosts: &ghosts,
        primitives: &primitives,
        min_pressure: 1.0e-6,
    };
    assemble_inviscid_residual_3d(&fields, &mut rhs, &params).expect("assemble");

    report_internal_face_alignment(mesh);
    report_boundary_patch_flux_samples(&BoundaryFluxReport {
        mesh,
        boundary: &case.boundary,
        fields: &fields,
        eos: &eos,
        config: &config,
        ghosts: &ghosts,
        primitives: &primitives,
        rhs: &rhs,
    });
    report_outlet_area_closure(mesh, &case.boundary);
}

fn report_internal_face_alignment(mesh: &StructuredMesh3d) {
    let mut misaligned_i = 0usize;
    let mut misaligned_j = 0usize;
    let mut misaligned_k = 0usize;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                if face_normal_misaligned(
                    mesh,
                    (i, j, k),
                    (i + 1, j, k),
                    mesh.i_face_metric(i, j, k),
                ) {
                    misaligned_i += 1;
                }
            }
        }
    }
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                if face_normal_misaligned(
                    mesh,
                    (i, j, k),
                    (i, j + 1, k),
                    mesh.j_face_metric(i, j, k),
                ) {
                    misaligned_j += 1;
                }
            }
        }
    }
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                if face_normal_misaligned(
                    mesh,
                    (i, j, k),
                    (i, j, k + 1),
                    mesh.k_face_metric(i, j, k),
                ) {
                    misaligned_k += 1;
                }
            }
        }
    }
    eprintln!("=== 内界面法向 vs owner→neighbor ===");
    eprintln!("  反向 i/j/k 面数: {misaligned_i} / {misaligned_j} / {misaligned_k}");
}

fn face_normal_misaligned(
    mesh: &StructuredMesh3d,
    a: (usize, usize, usize),
    b: (usize, usize, usize),
    face: FaceMetric,
) -> bool {
    let (i0, j0, k0) = a;
    let (i1, j1, k1) = b;
    let c0 = mesh.cell_metric(i0, j0, k0).center;
    let c1 = mesh.cell_metric(i1, j1, k1).center;
    let dc = Vector3::new(c1.x - c0.x, c1.y - c0.y, c1.z - c0.z);
    dc.x * face.normal.x + dc.y * face.normal.y + dc.z * face.normal.z < 0.0
}

struct BoundaryFluxReport<'a> {
    mesh: &'a StructuredMesh3d,
    boundary: &'a crate::boundary::BoundarySet,
    fields: &'a ConservedFields,
    eos: &'a IdealGasEoS,
    config: &'a InviscidFluxConfig,
    ghosts: &'a BoundaryGhostBuffer,
    primitives: &'a crate::field::PrimitiveFields,
    rhs: &'a ConservedResidual,
}

fn report_boundary_patch_flux_samples(ctx: &BoundaryFluxReport<'_>) {
    let flux_ctx = BoundaryFaceFlux3d {
        primitives: ctx.primitives,
        mesh: ctx.mesh,
        eos: ctx.eos,
        config: ctx.config,
        min_pressure: 1.0e-6,
    };
    eprintln!("=== 各 patch 边界面通量样本 (mid face) ===");
    for patch in ctx.boundary.patches() {
        let mid = patch.face_ids[patch.face_ids.len() / 2];
        let owner_id = ctx.mesh.face_owner(mid).expect("owner");
        let (logical, local) = LogicalFace3d::decode(mid).expect("decode");
        let (i, j, k) = ctx.mesh.face_ij(logical, local).expect("ij");
        let geom = ctx.mesh.face_geometry_3d(mid).expect("geom");
        let ghost = ctx.ghosts.get_face(mid).expect("ghost");
        let owner = ctx
            .fields
            .cell_state(owner_id.index() as usize)
            .expect("cell");
        let prim = crate::field::primitive_from_conserved(ctx.eos, &owner).expect("prim");
        let un = prim.velocity[0] * geom.normal.x
            + prim.velocity[1] * geom.normal.y
            + prim.velocity[2] * geom.normal.z;
        let flux =
            flux_at_boundary_face(&flux_ctx, mid, ghost.conserved, geom.normal).expect("flux");
        let vol = ctx.mesh.cell_metric(i, j, k).volume;
        let mut bnd_rhs = ConservedResidual::zeros(ctx.mesh.num_cells()).expect("bnd");
        if !is_degenerate_volume(vol) {
            accumulate_boundary_face(
                &mut bnd_rhs,
                owner_id.index() as usize,
                &flux,
                geom.area,
                vol,
            )
            .expect("acc");
        }
        let bnd_rho = bnd_rhs.density.values()[owner_id.index() as usize];
        let total_rho = ctx.rhs.density.values()[owner_id.index() as usize];
        eprintln!(
            "  {:<8} logical={logical:?} un={un:.4} |bnd_rho_dot|={:.3e} |total_rho_dot|={:.3e} n=({:.3},{:.3},{:.3})",
            patch.name,
            bnd_rho.abs(),
            total_rho.abs(),
            geom.normal.x,
            geom.normal.y,
            geom.normal.z
        );
    }
}

fn report_outlet_area_closure(mesh: &StructuredMesh3d, boundary: &crate::boundary::BoundarySet) {
    let Some(outlet) = boundary.find("dom-3") else {
        return;
    };
    let sample = outlet.face_ids[outlet.face_ids.len() / 2];
    let (logical, local) = LogicalFace3d::decode(sample).expect("decode");
    let (i, j, k) = mesh.face_ij(logical, local).expect("ij");
    let mut area_sum = Vector3::new(0.0, 0.0, 0.0);
    if i > 0 {
        let f = mesh.i_face_metric(i - 1, j, k);
        area_sum.x += f.area_vector.x;
        area_sum.y += f.area_vector.y;
        area_sum.z += f.area_vector.z;
    }
    if j > 0 {
        let f = mesh.j_face_metric(i, j - 1, k);
        area_sum.x -= f.area_vector.x;
        area_sum.y -= f.area_vector.y;
        area_sum.z -= f.area_vector.z;
    }
    if j + 1 < mesh.ny {
        let f = mesh.j_face_metric(i, j, k);
        area_sum.x += f.area_vector.x;
        area_sum.y += f.area_vector.y;
        area_sum.z += f.area_vector.z;
    }
    let bg = mesh.face_geometry_3d(sample).expect("geom");
    area_sum.x += bg.normal.x * bg.area;
    area_sum.y += bg.normal.y * bg.area;
    area_sum.z += bg.normal.z * bg.area;
    eprintln!(
        "=== dom-3 owner ({i},{j},{k}) 面积向量和 (i/j/bnd 部分) = ({:.3e},{:.3e},{:.3e}) mag={:.3e}",
        area_sum.x,
        area_sum.y,
        area_sum.z,
        (area_sum.x * area_sum.x + area_sum.y * area_sum.y + area_sum.z * area_sum.z).sqrt()
    );
}
