//! 非结构单元谱半径（local time step / 对角 LU-SGS）。

use crate::boundary::BoundarySet;
use crate::core::{FaceId, Real};
use crate::discretization::BoundaryGhostBuffer;
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::spectral_radius::{
    add_viscous_parabolic_face_sigma, cell_viscous_diffusivity_max, face_spectral_radius,
};

pub struct SpectralRadiusUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    /// 若启用 Navier-Stokes，叠加非结构 face-sum 粘性/热传导抛物型谱半径。
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// 非结构面循环谱半径：\(\sigma_i = V_i^{-1}\sum_f (|u_n|+a)_f A_f + \sigma_i^v\)。
pub fn cell_spectral_radius_unstructured(
    params: &SpectralRadiusUnstructuredParams<'_>,
) -> Result<Vec<Real>> {
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if params.primitives.num_cells() != n {
        return Err(AsimuError::Solver(format!(
            "cell_spectral_radius_unstructured: PrimitiveFields 长度 {} 与网格 {n} 不一致",
            params.primitives.num_cells()
        )));
    }
    let mut sigma = vec![0.0; n];
    for face in 0..mesh.num_faces() {
        accumulate_face_sigma(params, FaceId(face as u32), &mut sigma)?;
    }
    if let Some(viscous) = params.viscous {
        let diff = cell_viscous_diffusivity_max(params.primitives, params.eos, viscous)?;
        add_viscous_parabolic_sigma(params.mesh, &diff, &mut sigma)?;
    }
    for s in &mut sigma {
        *s = s.max(Real::EPSILON);
    }
    Ok(sigma)
}

fn accumulate_face_sigma(
    params: &SpectralRadiusUnstructuredParams<'_>,
    face: FaceId,
    sigma: &mut [Real],
) -> Result<()> {
    let mesh = params.mesh;
    let owner_id = mesh.face_owner(face)?;
    let owner = owner_id.index() as usize;
    let metric = mesh.face_metric(face);
    let owner_prim = params.primitives.cell_primitive(owner);
    let radius = if let Some(neighbor_id) = mesh.face_neighbor(face)? {
        let neighbor = neighbor_id.index() as usize;
        let neighbor_prim = params.primitives.cell_primitive(neighbor);
        face_spectral_radius(&owner_prim, &neighbor_prim, metric.normal, params.eos.gamma)
    } else if boundary_face_is_patched(params.boundaries, face) {
        let ghost = params.ghosts.get_face(face).ok_or_else(|| {
            AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost 状态", face.index()))
        })?;
        let ghost_prim =
            primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
        face_spectral_radius(&owner_prim, &ghost_prim, metric.normal, params.eos.gamma)
    } else {
        return Ok(());
    };
    add_sigma(mesh, owner_id, radius, metric.area, sigma);
    if let Some(neighbor_id) = mesh.face_neighbor(face)? {
        add_sigma(mesh, neighbor_id, radius, metric.area, sigma);
    }
    Ok(())
}

fn add_viscous_parabolic_sigma(
    mesh: &UnstructuredMesh3d,
    diffusivity: &[Real],
    sigma: &mut [Real],
) -> Result<()> {
    debug_assert_eq!(sigma.len(), diffusivity.len());
    for face in 0..mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let owner_id = mesh.face_owner(face_id)?;
        let metric = mesh.face_metric(face_id);
        add_viscous_parabolic_face_sigma(
            sigma,
            diffusivity,
            owner_id.index() as usize,
            metric.area,
            mesh.cell_metric(owner_id).volume,
        );
        if let Some(neighbor_id) = mesh.face_neighbor(face_id)? {
            add_viscous_parabolic_face_sigma(
                sigma,
                diffusivity,
                neighbor_id.index() as usize,
                metric.area,
                mesh.cell_metric(neighbor_id).volume,
            );
        }
    }
    Ok(())
}

fn add_sigma(
    mesh: &UnstructuredMesh3d,
    cell: crate::core::CellId,
    radius: Real,
    area: Real,
    sigma: &mut [Real],
) {
    let index = cell.index() as usize;
    let volume = mesh.cell_metric(cell).volume.max(1.0e-30);
    sigma[index] += radius * area / volume;
}

fn boundary_face_is_patched(boundaries: &BoundarySet, face: FaceId) -> bool {
    boundaries
        .patches()
        .iter()
        .any(|patch| patch.face_ids.contains(&face))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::GhostCellState;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::{FreestreamParams, ViscousPhysicsConfig};
    use crate::solver::spectral_radius::cell_local_dt_spectral;

    #[test]
    fn viscous_term_increases_unstructured_sigma_and_reduces_dt() {
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
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-8)
            .expect("fill");
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
            .collect::<Vec<_>>();
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        for &face in &faces {
            ghosts.insert_face(face, GhostCellState { conserved: state });
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
        let inviscid = SpectralRadiusUnstructuredParams {
            mesh: &mesh,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: 1.0e-8,
            viscous: None,
        };
        let viscous_cfg = ViscousPhysicsConfig::default();
        let viscous = SpectralRadiusUnstructuredParams {
            mesh: &mesh,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            eos: &eos,
            min_pressure: 1.0e-8,
            viscous: Some(&viscous_cfg),
        };
        let sigma_inv = cell_spectral_radius_unstructured(&inviscid).expect("sigma inv");
        let sigma_visc = cell_spectral_radius_unstructured(&viscous).expect("sigma visc");
        assert!(sigma_visc[0] > sigma_inv[0]);

        let volumes = mesh.cell_volumes();
        let dt_inv = cell_local_dt_spectral(&volumes, &sigma_inv, 1.0).expect("dt inv");
        let dt_visc = cell_local_dt_spectral(&volumes, &sigma_visc, 1.0).expect("dt visc");
        assert!(dt_visc[0] < dt_inv[0]);
    }
}
