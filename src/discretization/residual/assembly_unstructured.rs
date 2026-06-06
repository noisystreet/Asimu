//! 非结构 3D 网格无粘残差装配（一阶面循环）。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::FaceId;
use crate::discretization::{
    BoundaryGhostBuffer, FaceFluxInput, InviscidFluxConfig, ReconstructionKind, face_inviscid_flux,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use super::{accumulate_boundary_face, accumulate_interior_face, is_degenerate_volume};

pub struct InviscidAssemblyUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub config: &'a InviscidFluxConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
}

/// 非结构一阶 Euler 残差：遍历显式 face owner/neighbor 拓扑。
pub fn assemble_inviscid_residual_unstructured(
    fields: &ConservedFields,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if fields.num_cells() != n || residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构场/残差/primitive 长度须等于网格单元数 {n}"
        )));
    }
    if params.config.reconstruction != ReconstructionKind::FirstOrder {
        return Err(AsimuError::Config(
            "非结构网格当前仅支持 reconstruction = \"first_order\"".to_string(),
        ));
    }
    residual.clear();
    assemble_interior_faces(mesh, residual, params)?;
    assemble_boundary_faces(mesh, residual, params)
}

fn assemble_interior_faces(
    mesh: &UnstructuredMesh3d,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    for face in 0..mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let Some(neighbor_id) = mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner_id = mesh.face_owner(face_id)?;
        let owner = owner_id.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        let metric = mesh.face_metric(face_id);
        let owner_volume = mesh.cell_metric(owner_id).volume;
        let neighbor_volume = mesh.cell_metric(neighbor_id).volume;
        if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
            continue;
        }
        let owner_prim = params.primitives.cell_primitive(owner);
        let neighbor_prim = params.primitives.cell_primitive(neighbor);
        let flux = face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &neighbor_prim),
            metric.normal,
            params.eos,
            params.config,
        )?;
        accumulate_interior_face(
            residual,
            owner,
            neighbor,
            &flux,
            metric.area,
            owner_volume,
            neighbor_volume,
        )?;
    }
    Ok(())
}

fn assemble_boundary_faces(
    mesh: &UnstructuredMesh3d,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    for patch in params.boundaries.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        for &face in &patch.face_ids {
            let owner_id = mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let metric = mesh.face_metric(face);
            let owner_volume = mesh.cell_metric(owner_id).volume;
            if is_degenerate_volume(owner_volume) {
                continue;
            }
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
            })?;
            let owner_prim = params.primitives.cell_primitive(owner);
            let ghost_prim = crate::field::primitive_from_conserved_relaxed(
                params.eos,
                &ghost.conserved,
                1.0e-12,
            )?;
            let flux = face_inviscid_flux(
                FaceFluxInput::first_order(&owner_prim, &ghost_prim),
                metric.normal,
                params.eos,
                params.config,
            )?;
            accumulate_boundary_face(residual, owner, &flux, metric.area, owner_volume)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::discretization::InviscidFluxConfig;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::{FreestreamParams, IdealGasEoS};

    #[test]
    fn uniform_field_on_closed_tet_has_near_zero_rhs() {
        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell")],
        )
        .expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.3,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("primitive");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-8)
            .expect("fill");
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        for &face in &faces {
            ghosts.insert_face(
                face,
                crate::discretization::GhostCellState { conserved: state },
            );
        }
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: fs.mach,
                pressure: fs.pressure,
                temperature: fs.temperature,
                alpha: fs.alpha,
                beta: fs.beta,
            },
        )]);
        let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("residual");
        let params = InviscidAssemblyUnstructuredParams {
            mesh: &mesh,
            eos: &eos,
            config: &InviscidFluxConfig::roe_first_order(),
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
        };
        assemble_inviscid_residual_unstructured(&fields, &mut residual, &params).expect("rhs");
        assert!(residual.density_rms_norm() < 1.0e-10);
    }
}
