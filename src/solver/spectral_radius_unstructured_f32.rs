//! 非结构单元谱半径 f32 路径（local time step / LU-SGS 伪时间）。

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::unstructured_face_cache::{
    LsqRhsCellIncidence, UnstructuredSolverMeshCache,
};
use crate::discretization::unstructured_face_cache_f32::{
    UnstructuredBoundaryFaceF32, UnstructuredFaceTopologyF32, UnstructuredInteriorFaceF32,
};
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed_f32_from_state};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::spectral_radius::add_viscous_parabolic_face_sigma;
use super::spectral_radius_f32::{
    FacePrimitiveLaneF32, cell_viscous_diffusivity_max_f32, face_spectral_radius_f32,
};

const DEGENERATE_VOLUME: Real = 1.0e-30;

/// f32 非结构谱半径求值上下文。
pub struct SpectralRadiusUnstructuredF32Params<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// 非结构面循环谱半径（f32 primitive；\(\sigma_i\) 输出为 `Real`）。
pub fn cell_spectral_radius_unstructured_f32(
    params: &SpectralRadiusUnstructuredF32Params<'_>,
) -> Result<Vec<Real>> {
    let n = params.mesh.num_cells();
    if params.primitives.num_cells() != n {
        return Err(AsimuError::Solver(format!(
            "cell_spectral_radius_unstructured_f32: PrimitiveFields 长度 {} 与网格 {n} 不一致",
            params.primitives.num_cells()
        )));
    }
    let diffusivity = if let Some(viscous) = params.viscous {
        Some(cell_viscous_diffusivity_max_f32(
            params.primitives,
            params.eos,
            viscous,
        )?)
    } else {
        None
    };
    let mut sigma = vec![0.0; n];
    let topology = &params.mesh_cache.face_topology_f32;
    let incidence = &params.mesh_cache.lsq_rhs_incidence;
    let gamma = params.eos.gamma as f32;
    let prim = params.primitives;
    for (cell, sigma_cell) in sigma.iter_mut().enumerate().take(n) {
        accumulate_hyperbolic_sigma_one_cell_f32(
            params, prim, topology, incidence, cell, gamma, sigma_cell,
        )?;
        if let Some(diff) = &diffusivity {
            accumulate_parabolic_sigma_one_cell(topology, incidence, cell, diff, sigma_cell);
        }
    }
    for s in &mut sigma {
        *s = s.max(Real::EPSILON);
    }
    Ok(sigma)
}

fn accumulate_hyperbolic_sigma_one_cell_f32(
    params: &SpectralRadiusUnstructuredF32Params<'_>,
    prim: &PrimitiveFieldsT<f32>,
    topology: &UnstructuredFaceTopologyF32,
    incidence: &LsqRhsCellIncidence,
    cell: usize,
    gamma: f32,
    sigma_cell: &mut Real,
) -> Result<()> {
    for &face_idx in &incidence.interior_as_owner[cell] {
        accumulate_interior_hyperbolic_as_owner_f32(
            prim,
            &topology.interior[face_idx],
            gamma,
            sigma_cell,
        );
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
        accumulate_interior_hyperbolic_as_neighbor_f32(
            prim,
            &topology.interior[face_idx],
            gamma,
            sigma_cell,
        );
    }
    for &boundary_idx in &incidence.boundary_faces[cell] {
        accumulate_boundary_hyperbolic_f32(
            params,
            prim,
            &topology.boundary[boundary_idx],
            gamma,
            sigma_cell,
        )?;
    }
    Ok(())
}

fn prim_lane_f32(prim: &PrimitiveFieldsT<f32>, cell: usize) -> FacePrimitiveLaneF32 {
    FacePrimitiveLaneF32 {
        rho: prim.density.values()[cell],
        pressure: prim.pressure.values()[cell],
        velocity: [
            prim.velocity_x.values()[cell],
            prim.velocity_y.values()[cell],
            prim.velocity_z.values()[cell],
        ],
    }
}

fn accumulate_interior_hyperbolic_as_owner_f32(
    prim: &PrimitiveFieldsT<f32>,
    face: &UnstructuredInteriorFaceF32,
    gamma: f32,
    sigma_cell: &mut Real,
) {
    let left = prim_lane_f32(prim, face.owner);
    let right = prim_lane_f32(prim, face.neighbor);
    let radius = face_spectral_radius_f32(left, right, face.normal, gamma);
    add_hyperbolic_contribution_f32(sigma_cell, radius, face.area, face.inv_owner_volume);
}

fn accumulate_interior_hyperbolic_as_neighbor_f32(
    prim: &PrimitiveFieldsT<f32>,
    face: &UnstructuredInteriorFaceF32,
    gamma: f32,
    sigma_cell: &mut Real,
) {
    let left = prim_lane_f32(prim, face.owner);
    let right = prim_lane_f32(prim, face.neighbor);
    let radius = face_spectral_radius_f32(left, right, face.normal, gamma);
    add_hyperbolic_contribution_f32(sigma_cell, radius, face.area, face.inv_neighbor_volume);
}

fn accumulate_boundary_hyperbolic_f32(
    params: &SpectralRadiusUnstructuredF32Params<'_>,
    prim: &PrimitiveFieldsT<f32>,
    face: &UnstructuredBoundaryFaceF32,
    gamma: f32,
    sigma_cell: &mut Real,
) -> Result<()> {
    let ghost = params.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "谱半径边界面 FaceId({}) 缺少 ghost 状态",
            face.face.index()
        ))
    })?;
    let ghost_prim = primitive_from_conserved_relaxed_f32_from_state(
        params.eos,
        &ghost.conserved,
        params.min_pressure,
    )?;
    let left = prim_lane_f32(prim, face.owner);
    let radius = face_spectral_radius_f32(
        left,
        FacePrimitiveLaneF32 {
            rho: ghost_prim.density,
            pressure: ghost_prim.pressure,
            velocity: ghost_prim.velocity,
        },
        face.normal,
        gamma,
    );
    add_hyperbolic_contribution_f32(
        sigma_cell,
        radius,
        face.area,
        inv_volume_f32(face.owner_volume),
    );
    Ok(())
}

fn add_hyperbolic_contribution_f32(sigma_cell: &mut Real, radius: f32, area: f32, inv_volume: f32) {
    if inv_volume > 0.0 {
        *sigma_cell += (radius as Real) * (area as Real) * (inv_volume as Real);
    }
}

fn inv_volume_f32(volume: f32) -> f32 {
    if volume > DEGENERATE_VOLUME as f32 {
        1.0 / volume
    } else {
        0.0
    }
}

fn accumulate_parabolic_sigma_one_cell(
    topology: &UnstructuredFaceTopologyF32,
    incidence: &LsqRhsCellIncidence,
    cell: usize,
    diffusivity: &[Real],
    sigma_cell: &mut Real,
) {
    let diff = diffusivity[cell];
    if diff <= 0.0 {
        return;
    }
    for &face_idx in &incidence.interior_as_owner[cell] {
        let face = &topology.interior[face_idx];
        add_parabolic_contribution_f32(sigma_cell, diff, face.area, face.owner_volume);
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
        let face = &topology.interior[face_idx];
        add_parabolic_contribution_f32(sigma_cell, diff, face.area, face.neighbor_volume);
    }
    for &boundary_idx in &incidence.boundary_faces[cell] {
        let face = &topology.boundary[boundary_idx];
        add_parabolic_contribution_f32(sigma_cell, diff, face.area, face.owner_volume);
    }
}

fn add_parabolic_contribution_f32(sigma_cell: &mut Real, diff: Real, area: f32, volume: f32) {
    add_viscous_parabolic_face_sigma(
        std::slice::from_mut(sigma_cell),
        &[diff],
        0,
        area as Real,
        volume as Real,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::core::approx_eq;
    use crate::discretization::{BoundaryGhostBuffer, GhostCellState};
    use crate::field::{ConservedFields, PrimitiveFields};
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::{FreestreamParams, ViscousPhysicsConfig};

    fn tet_mesh_and_boundary() -> (UnstructuredMesh3d, BoundarySet) {
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
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let boundary = BoundarySet::new(vec![BoundaryPatch::new(
            "farfield",
            faces,
            BoundaryKind::Farfield {
                mach: 0.2,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        (mesh, boundary)
    }

    #[test]
    fn f32_unstructured_spectral_radius_matches_f64_on_freestream_tet() {
        let (mesh, boundary) = tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.2,
            ..FreestreamParams::default()
        };
        let fields_f64 =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let fields_f32 =
            crate::field::ConservedFieldsT::<f32>::from_real_fields(&fields_f64).expect("f32");
        let mut ghosts_f64 = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let mut ghosts_f32 = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let state = fields_f64.cell_state(0).expect("state");
        for face in 0..mesh.num_faces() {
            let face = crate::core::FaceId(face as u32);
            ghosts_f64.insert_face(face, GhostCellState { conserved: state });
            ghosts_f32.insert_face(face, GhostCellState { conserved: state });
        }
        let mut prim_f64 = PrimitiveFields::zeros(mesh.num_cells()).expect("prim f64");
        let mut prim_f32 =
            crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim f32");
        prim_f64
            .fill_from_conserved(&fields_f64, &eos, 1.0e-8)
            .expect("fill f64");
        prim_f32
            .fill_from_conserved(&fields_f32, &eos, 1.0e-8)
            .expect("fill f32");
        let sigma_f64 =
            super::super::spectral_radius_unstructured::cell_spectral_radius_unstructured(
                &super::super::spectral_radius_unstructured::SpectralRadiusUnstructuredParams {
                    mesh: &mesh,
                    mesh_cache: &mesh_cache,
                    boundaries: &boundary,
                    ghosts: &ghosts_f64,
                    primitives: &prim_f64,
                    eos: &eos,
                    min_pressure: 1.0e-8,
                    viscous: None,
                },
            )
            .expect("sigma f64");
        let sigma_f32 =
            cell_spectral_radius_unstructured_f32(&SpectralRadiusUnstructuredF32Params {
                mesh: &mesh,
                mesh_cache: &mesh_cache,
                boundaries: &boundary,
                ghosts: &ghosts_f32,
                primitives: &prim_f32,
                eos: &eos,
                min_pressure: 1.0e-8,
                viscous: None,
            })
            .expect("sigma f32");
        assert!(approx_eq(sigma_f32[0], sigma_f64[0], 1.0e-3));
    }

    #[test]
    fn f32_viscous_spectral_radius_exceeds_inviscid() {
        let (mesh, boundary) = tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields_f32 = crate::field::ConservedFieldsT::<f32>::from_real_fields(
            &ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields"),
        )
        .expect("f32");
        let mut ghosts = BoundaryGhostBuffer::with_face_capacity(mesh.num_faces());
        let state = fields_f32.cell_state(0).expect("state");
        for face in 0..mesh.num_faces() {
            ghosts.insert_face(
                crate::core::FaceId(face as u32),
                GhostCellState { conserved: state },
            );
        }
        let mut prim_f32 =
            crate::field::PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        prim_f32
            .fill_from_conserved(&fields_f32, &eos, 1.0e-8)
            .expect("fill");
        let viscous = ViscousPhysicsConfig::default();
        let base = cell_spectral_radius_unstructured_f32(&SpectralRadiusUnstructuredF32Params {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &prim_f32,
            eos: &eos,
            min_pressure: 1.0e-8,
            viscous: None,
        })
        .expect("inv");
        let visc = cell_spectral_radius_unstructured_f32(&SpectralRadiusUnstructuredF32Params {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &prim_f32,
            eos: &eos,
            min_pressure: 1.0e-8,
            viscous: Some(&viscous),
        })
        .expect("visc");
        assert!(visc[0] > base[0]);
    }
}
