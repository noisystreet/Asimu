//! 非结构单元谱半径（local time step / 对角 LU-SGS）。

use crate::boundary::BoundarySet;
use crate::core::{ComputeFloat, Real};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::unstructured_face_cache::{
    LsqRhsCellIncidence, UnstructuredBoundaryFace, UnstructuredFaceTopology,
    UnstructuredInteriorFace, UnstructuredSolverMeshCache,
};
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFields, PrimitiveFieldsT, primitive_from_conserved_relaxed};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::low_mach_face_spectral::face_spectral_radius_with_low_mach;
use super::spectral_radius::{add_viscous_parabolic_face_sigma, cell_viscous_diffusivity_max};

const DEGENERATE_VOLUME: Real = 1.0e-30;

pub struct SpectralRadiusUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    /// 若启用 Navier-Stokes，叠加非结构 face-sum 粘性/热传导抛物型谱半径。
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    /// 低马赫预处理（P1）：声速项按局部 Mach 缩放。
    pub low_mach_preconditioning: Option<crate::solver::time::LowMachPreconditioningConfig>,
}

/// 非结构面循环谱半径：\(\sigma_i = V_i^{-1}\sum_f (|u_n|+a)_f A_f + \sigma_i^v\)。
pub fn cell_spectral_radius_unstructured(
    params: &SpectralRadiusUnstructuredParams<'_>,
) -> Result<Vec<Real>> {
    let n = params.mesh.num_cells();
    if params.primitives.num_cells() != n {
        return Err(AsimuError::Solver(format!(
            "cell_spectral_radius_unstructured: PrimitiveFields 长度 {} 与网格 {n} 不一致",
            params.primitives.num_cells()
        )));
    }
    let diffusivity = if let Some(viscous) = params.viscous {
        Some(cell_viscous_diffusivity_max(
            params.primitives,
            params.eos,
            viscous,
        )?)
    } else {
        None
    };
    let mut sigma = vec![0.0; n];
    let topology = &params.mesh_cache.face_topology;
    let incidence = &params.mesh_cache.lsq_rhs_incidence;
    #[cfg(feature = "parallel-fvm")]
    {
        crate::exec::parallel::par_try_for_each_enumerated_result(
            &mut sigma,
            |cell, sigma_cell| {
                accumulate_hyperbolic_sigma_one_cell(
                    params, topology, incidence, cell, sigma_cell,
                )?;
                if let Some(diff) = &diffusivity {
                    accumulate_parabolic_sigma_one_cell(
                        topology, incidence, cell, diff, sigma_cell,
                    );
                }
                Ok(())
            },
        )?;
    }
    #[cfg(not(feature = "parallel-fvm"))]
    {
        for cell in 0..n {
            accumulate_hyperbolic_sigma_one_cell(
                params,
                topology,
                incidence,
                cell,
                &mut sigma[cell],
            )?;
            if let Some(diff) = &diffusivity {
                accumulate_parabolic_sigma_one_cell(
                    topology,
                    incidence,
                    cell,
                    diff,
                    &mut sigma[cell],
                );
            }
        }
    }
    for s in &mut sigma {
        *s = s.max(Real::EPSILON);
    }
    Ok(sigma)
}

/// typed 非结构谱半径求值上下文。
pub struct SpectralRadiusUnstructuredTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
    pub low_mach_preconditioning: Option<crate::solver::time::LowMachPreconditioningConfig>,
}

/// typed 非结构谱半径分发（f32 原生 `Vec<f32>` / f64 `Vec<Real>`）。
pub trait UnstructuredSpectralRadiusTyped: ComputeFloat {
    type Sigma: Sized;
    fn cell_spectral_radius_unstructured_typed(
        params: &SpectralRadiusUnstructuredTypedParams<'_, Self>,
    ) -> Result<Self::Sigma>;
}

impl UnstructuredSpectralRadiusTyped for f64 {
    type Sigma = Vec<Real>;
    fn cell_spectral_radius_unstructured_typed(
        params: &SpectralRadiusUnstructuredTypedParams<'_, f64>,
    ) -> Result<Vec<Real>> {
        cell_spectral_radius_unstructured(&SpectralRadiusUnstructuredParams {
            mesh: params.mesh,
            mesh_cache: params.mesh_cache,
            boundaries: params.boundaries,
            ghosts: params.ghosts,
            primitives: params.primitives,
            eos: params.eos,
            min_pressure: params.min_pressure,
            viscous: params.viscous,
            low_mach_preconditioning: params.low_mach_preconditioning,
        })
    }
}

impl UnstructuredSpectralRadiusTyped for f32 {
    type Sigma = Vec<f32>;
    fn cell_spectral_radius_unstructured_typed(
        params: &SpectralRadiusUnstructuredTypedParams<'_, f32>,
    ) -> Result<Vec<f32>> {
        super::spectral_radius_unstructured_f32::cell_spectral_radius_unstructured_f32(
            &super::spectral_radius_unstructured_f32::SpectralRadiusUnstructuredF32Params {
                mesh: params.mesh,
                mesh_cache: params.mesh_cache,
                boundaries: params.boundaries,
                ghosts: params.ghosts,
                primitives: params.primitives,
                eos: params.eos,
                min_pressure: params.min_pressure,
                viscous: params.viscous,
                low_mach_preconditioning: params.low_mach_preconditioning,
            },
        )
    }
}

fn accumulate_hyperbolic_sigma_one_cell(
    params: &SpectralRadiusUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
    incidence: &LsqRhsCellIncidence,
    cell: usize,
    sigma_cell: &mut Real,
) -> Result<()> {
    let prim = params.primitives;
    let gamma = params.eos.gamma;
    for &face_idx in &incidence.interior_as_owner[cell] {
        accumulate_interior_hyperbolic_as_owner(
            params,
            prim,
            &topology.interior[face_idx],
            gamma,
            sigma_cell,
        );
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
        accumulate_interior_hyperbolic_as_neighbor(
            params,
            prim,
            &topology.interior[face_idx],
            gamma,
            sigma_cell,
        );
    }
    for &boundary_idx in &incidence.boundary_faces[cell] {
        accumulate_boundary_hyperbolic(
            params,
            &topology.boundary[boundary_idx],
            gamma,
            sigma_cell,
        )?;
    }
    Ok(())
}

fn accumulate_interior_hyperbolic_as_owner(
    params: &SpectralRadiusUnstructuredParams<'_>,
    prim: &PrimitiveFields,
    face: &UnstructuredInteriorFace,
    gamma: Real,
    sigma_cell: &mut Real,
) {
    let radius = face_spectral_radius_with_low_mach(
        &prim.cell_primitive(face.owner),
        &prim.cell_primitive(face.neighbor),
        face.normal,
        gamma,
        params.low_mach_preconditioning,
    );
    add_hyperbolic_contribution(sigma_cell, radius, face.area, face.inv_owner_volume);
}

fn accumulate_interior_hyperbolic_as_neighbor(
    params: &SpectralRadiusUnstructuredParams<'_>,
    prim: &PrimitiveFields,
    face: &UnstructuredInteriorFace,
    gamma: Real,
    sigma_cell: &mut Real,
) {
    let radius = face_spectral_radius_with_low_mach(
        &prim.cell_primitive(face.owner),
        &prim.cell_primitive(face.neighbor),
        face.normal,
        gamma,
        params.low_mach_preconditioning,
    );
    add_hyperbolic_contribution(sigma_cell, radius, face.area, face.inv_neighbor_volume);
}

fn accumulate_boundary_hyperbolic(
    params: &SpectralRadiusUnstructuredParams<'_>,
    face: &UnstructuredBoundaryFace,
    gamma: Real,
    sigma_cell: &mut Real,
) -> Result<()> {
    let ghost = params.ghosts.get_face(face.face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "谱半径边界面 FaceId({}) 缺少 ghost 状态",
            face.face.index()
        ))
    })?;
    let ghost_prim =
        primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
    let radius = face_spectral_radius_with_low_mach(
        &params.primitives.cell_primitive(face.owner),
        &ghost_prim,
        face.normal,
        gamma,
        params.low_mach_preconditioning,
    );
    let inv_volume = inv_volume(face.owner_volume);
    add_hyperbolic_contribution(sigma_cell, radius, face.area, inv_volume);
    Ok(())
}

fn add_hyperbolic_contribution(sigma_cell: &mut Real, radius: Real, area: Real, inv_volume: Real) {
    if inv_volume > 0.0 {
        *sigma_cell += radius * area * inv_volume;
    }
}

fn inv_volume(volume: Real) -> Real {
    if volume > DEGENERATE_VOLUME {
        1.0 / volume
    } else {
        0.0
    }
}

fn accumulate_parabolic_sigma_one_cell(
    topology: &UnstructuredFaceTopology,
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
        add_parabolic_contribution(sigma_cell, diff, face.area, face.owner_volume);
    }
    for &face_idx in &incidence.interior_as_neighbor[cell] {
        let face = &topology.interior[face_idx];
        add_parabolic_contribution(sigma_cell, diff, face.area, face.neighbor_volume);
    }
    for &boundary_idx in &incidence.boundary_faces[cell] {
        let face = &topology.boundary[boundary_idx];
        add_parabolic_contribution(sigma_cell, diff, face.area, face.owner_volume);
    }
}

fn add_parabolic_contribution(sigma_cell: &mut Real, diff: Real, area: Real, volume: Real) {
    add_viscous_parabolic_face_sigma(std::slice::from_mut(sigma_cell), &[diff], 0, area, volume);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::GhostCellState;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::{FreestreamParams, ViscousPhysicsConfig};
    use crate::solver::compressible::spectral_radius::cell_local_dt_spectral;

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
                mach: 0.0,
                pressure: 101_325.0,
                temperature: 300.0,
                alpha: 0.0,
                beta: 0.0,
            },
        )]);
        (mesh, boundary)
    }

    fn spectral_params<'a>(
        mesh: &'a UnstructuredMesh3d,
        mesh_cache: &'a UnstructuredSolverMeshCache,
        boundary: &'a BoundarySet,
        ghosts: &'a BoundaryGhostBuffer,
        primitives: &'a PrimitiveFields,
        viscous: Option<&'a ViscousPhysicsConfig>,
        low_mach: Option<crate::solver::time::LowMachPreconditioningConfig>,
    ) -> SpectralRadiusUnstructuredParams<'a> {
        SpectralRadiusUnstructuredParams {
            mesh,
            mesh_cache,
            boundaries: boundary,
            ghosts,
            primitives,
            eos: &IdealGasEoS::AIR_STANDARD,
            min_pressure: 1.0e-8,
            viscous,
            low_mach_preconditioning: low_mach,
        }
    }

    #[test]
    fn viscous_term_increases_unstructured_sigma_and_reduces_dt() {
        let (mesh, boundary) = tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
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
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        for &face in &faces {
            ghosts.insert_face(face, GhostCellState { conserved: state });
        }
        let inviscid = spectral_params(
            &mesh,
            &mesh_cache,
            &boundary,
            &ghosts,
            &primitives,
            None,
            None,
        );
        let viscous_cfg = ViscousPhysicsConfig::default();
        let viscous = spectral_params(
            &mesh,
            &mesh_cache,
            &boundary,
            &ghosts,
            &primitives,
            Some(&viscous_cfg),
            None,
        );
        let sigma_inv = cell_spectral_radius_unstructured(&inviscid).expect("sigma inv");
        let sigma_visc = cell_spectral_radius_unstructured(&viscous).expect("sigma visc");
        assert!(sigma_visc[0] > sigma_inv[0]);

        let volumes = mesh.cell_volumes();
        let dt_inv = cell_local_dt_spectral(&volumes, &sigma_inv, 1.0).expect("dt inv");
        let dt_visc = cell_local_dt_spectral(&volumes, &sigma_visc, 1.0).expect("dt visc");
        assert!(dt_visc[0] < dt_inv[0]);
    }

    #[cfg(feature = "parallel-fvm")]
    #[test]
    fn cached_spectral_radius_matches_face_topology_serial() {
        use crate::core::approx_eq;

        let (mesh, boundary) = tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives.density.values_mut()[0] = 1.1;
        primitives.pressure.values_mut()[0] = 120_000.0;
        primitives.velocity_x.values_mut()[0] = 120.0;
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
            .collect::<Vec<_>>();
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = crate::physics::ConservedState {
            density: 1.0,
            momentum: [10.0, 0.0, 0.0],
            total_energy: 250_000.0,
        };
        for &face in &faces {
            ghosts.insert_face(face, GhostCellState { conserved: state });
        }
        let params = spectral_params(
            &mesh,
            &mesh_cache,
            &boundary,
            &ghosts,
            &primitives,
            None,
            None,
        );
        let parallel = cell_spectral_radius_unstructured(&params).expect("parallel");
        let mut serial = vec![0.0; mesh.num_cells()];
        let topology = &mesh_cache.face_topology;
        let incidence = &mesh_cache.lsq_rhs_incidence;
        for (cell, sigma_cell) in serial.iter_mut().enumerate() {
            accumulate_hyperbolic_sigma_one_cell(&params, topology, incidence, cell, sigma_cell)
                .expect("serial");
        }
        for (lhs, rhs) in parallel.iter().zip(serial.iter()) {
            assert!(approx_eq(*lhs, *rhs, 1.0e-12));
        }
    }

    #[test]
    fn low_mach_preconditioning_reduces_hyperbolic_sigma() {
        let (mesh, boundary) = tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.05,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-8)
            .expect("fill");
        let mut ghosts = BoundaryGhostBuffer::new();
        let state = fields.cell_state(0).expect("state");
        for face in 0..mesh.num_faces() {
            ghosts.insert_face(
                crate::core::FaceId(face as u32),
                GhostCellState { conserved: state },
            );
        }
        let base = spectral_params(
            &mesh,
            &mesh_cache,
            &boundary,
            &ghosts,
            &primitives,
            None,
            None,
        );
        let low_mach = spectral_params(
            &mesh,
            &mesh_cache,
            &boundary,
            &ghosts,
            &primitives,
            None,
            Some(crate::solver::time::LowMachPreconditioningConfig { mach_cutoff: 0.1 }),
        );
        let sigma_base = cell_spectral_radius_unstructured(&base).expect("base");
        let sigma_lm = cell_spectral_radius_unstructured(&low_mach).expect("low mach");
        assert!(
            sigma_lm[0] < sigma_base[0],
            "sigma_lm={} sigma_base={}",
            sigma_lm[0],
            sigma_base[0]
        );
    }
}
