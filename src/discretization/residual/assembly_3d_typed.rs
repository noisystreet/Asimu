//! 3D 结构化网格无粘残差装配（typed 场；P2 首版仅一阶）。

use tracing::info_span;

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::{ComputeFloat, Real};
use crate::discretization::face_flux_typed::InviscidFaceFluxTyped;
use crate::discretization::{BoundaryGhostBuffer, InviscidFluxConfig, ReconstructionKind};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

use super::{accumulate_boundary_face_typed, accumulate_interior_face_typed, is_degenerate_volume};

/// typed 无粘残差装配上下文。
pub struct InviscidAssembly3dTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub min_pressure: Real,
}

struct BoundaryAssembly3dTyped<'a, T: ComputeFloat> {
    mesh: &'a dyn BoundaryMesh3d,
    structured: &'a StructuredMesh3d,
    params: &'a InviscidAssembly3dTypedParams<'a, T>,
}

/// 装配 3D 结构化网格无粘 Euler 残差（一阶；`T=f32`/`f64`）。
pub fn assemble_inviscid_residual_3d_typed<T: ComputeFloat + InviscidFaceFluxTyped>(
    fields: &ConservedFieldsT<T>,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
) -> Result<()> {
    if params.config.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(format!(
            "compute_precision = \"{}\" 的结构化 typed 路径暂仅支持一阶重构",
            T::PRECISION.label()
        )));
    }
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
        let _span = info_span!("assemble_faces_typed", dim = "i").entered();
        assemble_i_faces_typed(mesh, residual, params)?;
    }
    {
        let _span = info_span!("assemble_faces_typed", dim = "j").entered();
        assemble_j_faces_typed(mesh, residual, params)?;
    }
    {
        let _span = info_span!("assemble_faces_typed", dim = "k").entered();
        assemble_k_faces_typed(mesh, residual, params)?;
    }
    {
        let _span = info_span!("assemble_faces_typed", dim = "boundary").entered();
        assemble_boundary_faces_3d_typed(
            residual,
            &BoundaryAssembly3dTyped {
                mesh,
                structured: mesh,
                params,
            },
        )?;
    }
    Ok(())
}

fn first_order_interior_flux<T: InviscidFaceFluxTyped>(
    primitives: &PrimitiveFieldsT<T>,
    owner: usize,
    neighbor: usize,
    normal: crate::core::Vector3,
    eos: &IdealGasEoS,
    config: &InviscidFluxConfig,
) -> Result<crate::discretization::InviscidFlux> {
    T::first_order_interior_soa(primitives, owner, neighbor, normal, eos, config)
}

fn assemble_i_faces_typed<T: InviscidFaceFluxTyped>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
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
                let flux = first_order_interior_flux(
                    params.primitives,
                    owner,
                    neighbor,
                    face.normal,
                    params.eos,
                    params.config,
                )?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i + 1, j, k).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face_typed(
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

fn assemble_j_faces_typed<T: InviscidFaceFluxTyped>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
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
                let flux = first_order_interior_flux(
                    params.primitives,
                    owner,
                    neighbor,
                    face.normal,
                    params.eos,
                    params.config,
                )?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i, j + 1, k).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face_typed(
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

fn assemble_k_faces_typed<T: InviscidFaceFluxTyped>(
    mesh: &StructuredMesh3d,
    residual: &mut ConservedResidualT<T>,
    params: &InviscidAssembly3dTypedParams<'_, T>,
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
                let flux = first_order_interior_flux(
                    params.primitives,
                    owner,
                    neighbor,
                    face.normal,
                    params.eos,
                    params.config,
                )?;
                let owner_volume = mesh.cell_metric(i, j, k).volume;
                let neighbor_volume = mesh.cell_metric(i, j, k + 1).volume;
                if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
                    continue;
                }
                accumulate_interior_face_typed(
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

fn assemble_boundary_faces_3d_typed<T: InviscidFaceFluxTyped>(
    residual: &mut ConservedResidualT<T>,
    ctx: &BoundaryAssembly3dTyped<'_, T>,
) -> Result<()> {
    let mesh = ctx.structured;
    let params = ctx.params;
    for patch in params.boundaries.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner_id = ctx.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let geom = ctx.mesh.face_geometry_3d(face)?;
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
            })?;
            let flux = T::first_order_boundary_soa(
                params.primitives,
                owner,
                &ghost,
                geom.normal,
                params.eos,
                params.config,
                params.min_pressure,
            )?;
            let (logical, local) = crate::mesh::LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            let owner_volume = mesh.cell_metric(i, j, k).volume;
            if is_degenerate_volume(owner_volume) {
                continue;
            }
            accumulate_boundary_face_typed(residual, owner, &flux, geom.area, owner_volume)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
    use crate::field::PrimitiveFillFromConserved;
    use crate::mesh::MeshMetricMode;

    fn assemble_uniform_freestream_typed<
        T: ComputeFloat + InviscidFaceFluxTyped + PrimitiveFillFromConserved,
    >(
        side: &crate::discretization::freestream_pair::UniformFarfieldSide<'_>,
        metric_mode: MeshMetricMode,
    ) -> ConservedResidualT<T> {
        let (mut mesh, boundary_set, fields, ghosts) =
            uniform_farfield_box(3, 3, 3, 1.0, 1.0, 1.0, side);
        mesh.set_metric_mode(metric_mode);
        let state = fields.cell_state(0).expect("state");
        let fields_t = ConservedFieldsT::<T>::uniform(mesh.num_cells(), state).expect("fields");
        let mut primitives = PrimitiveFieldsT::<T>::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields_t, side.eos, side.min_pressure)
            .expect("fill");
        let mut rhs = ConservedResidualT::<T>::zeros(mesh.num_cells()).expect("rhs");
        let config = InviscidFluxConfig::default();
        let params = InviscidAssembly3dTypedParams {
            mesh: &mesh,
            eos: side.eos,
            config: &config,
            boundaries: &boundary_set,
            ghosts: &ghosts,
            primitives: &primitives,
            min_pressure: side.min_pressure,
        };
        assemble_inviscid_residual_3d_typed(&fields_t, &mut rhs, &params).expect("assemble");
        rhs
    }

    #[test]
    fn f32_uniform_freestream_has_near_zero_rhs() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        pair.for_each_inviscid_side(|side| {
            let rhs = assemble_uniform_freestream_typed::<f32>(side, MeshMetricMode::Cartesian);
            assert!(
                rhs.density
                    .values()
                    .iter()
                    .all(|v| v.to_real().abs() < 1.0e-6),
                "{} f32 density rhs",
                side.label
            );
        });
    }
}
