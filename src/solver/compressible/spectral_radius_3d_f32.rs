//! 结构化 3D 单元谱半径 f32 路径（ADR 0019 S1-b）。

use crate::boundary::BoundarySet;
use crate::core::{ComputeFloat, FaceId, Real};
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::structured_face_cache_f32::{
    StructuredFaceCacheF32, StructuredInteriorFaceF32,
};
use crate::error::{AsimuError, Result};
use crate::field::{PrimitiveFieldsT, primitive_from_conserved_relaxed_f32_from_state};
use crate::mesh::{BoundaryMesh3d, LogicalFace3d, StructuredMesh3d};
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::spectral_radius::{SpectralRadius3dParams, cell_spectral_radius_3d};
use super::spectral_radius_f32::{
    FacePrimitiveLaneF32, cell_viscous_diffusivity_max_f32, face_spectral_radius_f32,
};

const PARABOLIC_SPECTRAL_FACTOR_3D_F32: f32 = 6.0;
const DEGENERATE_VOLUME_F32: f32 = 1.0e-30;

/// f32 结构化谱半径求值上下文。
pub struct SpectralRadius3dF32Params<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub boundary_mesh: &'a dyn BoundaryMesh3d,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub face_cache: &'a StructuredFaceCacheF32,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// 结构化面循环谱半径（f32 primitive；\(\sigma_i\) 输出 `Vec<f32>`）。
pub fn cell_spectral_radius_3d_f32(params: &SpectralRadius3dF32Params<'_>) -> Result<Vec<f32>> {
    let n = params.mesh.num_cells();
    if params.primitives.num_cells() != n {
        return Err(AsimuError::Solver(format!(
            "cell_spectral_radius_3d_f32: PrimitiveFields 长度 {} 与网格 {n} 不一致",
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
    let mut sigma_acc = vec![0.0_f64; n];
    let gamma = params.eos.gamma as f32;
    let prim = params.primitives;
    let cache = params.face_cache;
    accumulate_interior_hyperbolic_f32(prim, &cache.i_faces, gamma, &mut sigma_acc);
    accumulate_interior_hyperbolic_f32(prim, &cache.j_faces, gamma, &mut sigma_acc);
    accumulate_interior_hyperbolic_f32(prim, &cache.k_faces, gamma, &mut sigma_acc);
    accumulate_boundary_hyperbolic_f32(params, prim, gamma, &mut sigma_acc)?;
    if let Some(diff) = &diffusivity {
        accumulate_interior_parabolic_f32(&cache.i_faces, diff, &mut sigma_acc);
        accumulate_interior_parabolic_f32(&cache.j_faces, diff, &mut sigma_acc);
        accumulate_interior_parabolic_f32(&cache.k_faces, diff, &mut sigma_acc);
        accumulate_boundary_parabolic_f32(params, diff, &mut sigma_acc)?;
    }
    let mut sigma = Vec::with_capacity(n);
    for acc in sigma_acc {
        sigma.push((acc.max(f64::EPSILON)) as f32);
    }
    Ok(sigma)
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

fn accumulate_interior_hyperbolic_f32(
    prim: &PrimitiveFieldsT<f32>,
    faces: &[StructuredInteriorFaceF32],
    gamma: f32,
    sigma_acc: &mut [f64],
) {
    for face in faces {
        if face.owner_volume <= DEGENERATE_VOLUME_F32
            || face.neighbor_volume <= DEGENERATE_VOLUME_F32
        {
            continue;
        }
        let left = prim_lane_f32(prim, face.owner);
        let right = prim_lane_f32(prim, face.neighbor);
        let radius = face_spectral_radius_f32(left, right, face.normal, gamma);
        add_hyperbolic_contribution_f32(
            &mut sigma_acc[face.owner],
            radius,
            face.area,
            1.0 / face.owner_volume,
        );
        add_hyperbolic_contribution_f32(
            &mut sigma_acc[face.neighbor],
            radius,
            face.area,
            1.0 / face.neighbor_volume,
        );
    }
}

fn accumulate_boundary_hyperbolic_f32(
    params: &SpectralRadius3dF32Params<'_>,
    prim: &PrimitiveFieldsT<f32>,
    gamma: f32,
    sigma_acc: &mut [f64],
) -> Result<()> {
    let mesh = params.mesh;
    for patch in params.boundaries.patches() {
        for &face in &patch.face_ids {
            add_boundary_face_sigma_f32(params, face, mesh, prim, gamma, sigma_acc)?;
        }
    }
    Ok(())
}

fn add_boundary_face_sigma_f32(
    params: &SpectralRadius3dF32Params<'_>,
    face: FaceId,
    mesh: &StructuredMesh3d,
    prim: &PrimitiveFieldsT<f32>,
    gamma: f32,
    sigma_acc: &mut [f64],
) -> Result<()> {
    let owner = params.boundary_mesh.face_owner(face)?.index() as usize;
    let geom = params.boundary_mesh.face_geometry_3d(face)?;
    let ghost = params.ghosts.get_face(face).ok_or_else(|| {
        AsimuError::Boundary(format!(
            "cell_spectral_radius_3d_f32: 边界面 FaceId({}) 缺少 ghost",
            face.index()
        ))
    })?;
    let ghost_prim = primitive_from_conserved_relaxed_f32_from_state(
        params.eos,
        &ghost.conserved,
        params.min_pressure,
    )?;
    let left = prim_lane_f32(prim, owner);
    let radius = face_spectral_radius_f32(
        left,
        FacePrimitiveLaneF32 {
            rho: ghost_prim.density,
            pressure: ghost_prim.pressure,
            velocity: ghost_prim.velocity,
        },
        [
            geom.normal.x as f32,
            geom.normal.y as f32,
            geom.normal.z as f32,
        ],
        gamma,
    );
    let (logical, local) = LogicalFace3d::decode(face)?;
    let (i, j, k) = mesh.face_ij(logical, local)?;
    let volume = mesh.cell_metric(i, j, k).volume as f32;
    if volume > DEGENERATE_VOLUME_F32 {
        add_hyperbolic_contribution_f32(
            &mut sigma_acc[owner],
            radius,
            geom.area as f32,
            1.0 / volume,
        );
    }
    Ok(())
}

fn accumulate_interior_parabolic_f32(
    faces: &[StructuredInteriorFaceF32],
    diffusivity: &[f32],
    sigma_acc: &mut [f64],
) {
    for face in faces {
        add_parabolic_contribution_f32(
            &mut sigma_acc[face.owner],
            diffusivity[face.owner],
            face.area,
            face.owner_volume,
        );
        add_parabolic_contribution_f32(
            &mut sigma_acc[face.neighbor],
            diffusivity[face.neighbor],
            face.area,
            face.neighbor_volume,
        );
    }
}

fn accumulate_boundary_parabolic_f32(
    params: &SpectralRadius3dF32Params<'_>,
    diffusivity: &[f32],
    sigma_acc: &mut [f64],
) -> Result<()> {
    let mesh = params.mesh;
    for patch in params.boundaries.patches() {
        for &face in &patch.face_ids {
            let owner = params.boundary_mesh.face_owner(face)?.index() as usize;
            let geom = params.boundary_mesh.face_geometry_3d(face)?;
            let (logical, local) = LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            add_parabolic_contribution_f32(
                &mut sigma_acc[owner],
                diffusivity[owner],
                geom.area as f32,
                mesh.cell_metric(i, j, k).volume as f32,
            );
        }
    }
    Ok(())
}

fn add_hyperbolic_contribution_f32(sigma_cell: &mut f64, radius: f32, area: f32, inv_volume: f32) {
    if inv_volume > 0.0 {
        *sigma_cell += f64::from(radius * area * inv_volume);
    }
}

fn add_parabolic_contribution_f32(sigma_cell: &mut f64, diff: f32, area: f32, volume: f32) {
    if diff > 0.0 && area > f32::EPSILON && volume > DEGENERATE_VOLUME_F32 {
        *sigma_cell +=
            f64::from(PARABOLIC_SPECTRAL_FACTOR_3D_F32) * f64::from(diff) * f64::from(area).powi(2)
                / f64::from(volume).powi(2);
    }
}

// --- typed 分发（ADR 0019 S1-b）---

/// typed 结构化谱半径求值上下文。
pub struct SpectralRadius3dTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a StructuredMesh3d,
    pub boundary_mesh: &'a dyn BoundaryMesh3d,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<T>,
    pub face_cache_f32: Option<&'a StructuredFaceCacheF32>,
    pub eos: &'a IdealGasEoS,
    pub min_pressure: Real,
    pub viscous: Option<&'a ViscousPhysicsConfig>,
}

/// typed 结构化谱半径分发（f32 原生 `Vec<f32>` / f64 `Vec<Real>`）。
pub trait StructuredSpectralRadiusTyped: ComputeFloat {
    type Sigma: Clone;
    fn cell_spectral_radius_3d_typed(
        params: &SpectralRadius3dTypedParams<'_, Self>,
    ) -> Result<Self::Sigma>;
    fn sigma_to_real(sigma: Self::Sigma) -> Vec<Real>;
}

impl StructuredSpectralRadiusTyped for f64 {
    type Sigma = Vec<Real>;

    fn cell_spectral_radius_3d_typed(
        params: &SpectralRadius3dTypedParams<'_, f64>,
    ) -> Result<Vec<Real>> {
        cell_spectral_radius_3d(&SpectralRadius3dParams {
            mesh: params.mesh,
            boundary_mesh: params.boundary_mesh,
            boundaries: params.boundaries,
            ghosts: params.ghosts,
            primitives: params.primitives,
            eos: params.eos,
            min_pressure: params.min_pressure,
            viscous: params.viscous,
        })
    }

    fn sigma_to_real(sigma: Self::Sigma) -> Vec<Real> {
        sigma
    }
}

impl StructuredSpectralRadiusTyped for f32 {
    type Sigma = Vec<f32>;

    fn cell_spectral_radius_3d_typed(
        params: &SpectralRadius3dTypedParams<'_, f32>,
    ) -> Result<Vec<f32>> {
        let cache = params.face_cache_f32.ok_or_else(|| {
            AsimuError::Solver(
                "compute_precision = f32 结构化谱半径须传入 face_cache_f32".to_string(),
            )
        })?;
        cell_spectral_radius_3d_f32(&SpectralRadius3dF32Params {
            mesh: params.mesh,
            boundary_mesh: params.boundary_mesh,
            boundaries: params.boundaries,
            ghosts: params.ghosts,
            primitives: params.primitives,
            face_cache: cache,
            eos: params.eos,
            min_pressure: params.min_pressure,
            viscous: params.viscous,
        })
    }

    fn sigma_to_real(sigma: Self::Sigma) -> Vec<Real> {
        sigma.iter().map(|s| f64::from(*s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;
    use crate::discretization::StructuredFaceCacheF32;
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
    use crate::field::{ConservedFieldsT, PrimitiveFields};
    use crate::mesh::MeshMetricMode;
    use crate::solver::compressible::spectral_radius::{
        SpectralRadius3dParams, cell_spectral_radius_3d,
    };

    #[test]
    fn f32_structured_spectral_radius_matches_f64_on_freestream_box() {
        let pair = FreestreamPairFixture::air_sutherland(0.2);
        let side = pair.inviscid_side();
        let (mut mesh, boundary, fields_f64, ghosts_f64) =
            uniform_farfield_box(3, 3, 3, 1.0, 1.0, 1.0, &side);
        mesh.set_metric_mode(MeshMetricMode::Cartesian);
        let fields_f32 =
            ConservedFieldsT::<f32>::from_real_fields(&fields_f64).expect("fields f32");
        let mut prim_f64 = PrimitiveFields::zeros(mesh.num_cells()).expect("prim f64");
        let mut prim_f32 = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim f32");
        prim_f64
            .fill_from_conserved(&fields_f64, side.eos, side.min_pressure)
            .expect("fill f64");
        prim_f32
            .fill_from_conserved(&fields_f32, side.eos, side.min_pressure)
            .expect("fill f32");
        let cache = StructuredFaceCacheF32::from_mesh(&mesh);
        let sigma_f64 = cell_spectral_radius_3d(&SpectralRadius3dParams {
            mesh: &mesh,
            boundary_mesh: &mesh,
            boundaries: &boundary,
            ghosts: &ghosts_f64,
            primitives: &prim_f64,
            eos: side.eos,
            min_pressure: side.min_pressure,
            viscous: None,
        })
        .expect("sigma f64");
        let sigma_f32 = cell_spectral_radius_3d_f32(&SpectralRadius3dF32Params {
            mesh: &mesh,
            boundary_mesh: &mesh,
            boundaries: &boundary,
            ghosts: &ghosts_f64,
            primitives: &prim_f32,
            face_cache: &cache,
            eos: side.eos,
            min_pressure: side.min_pressure,
            viscous: None,
        })
        .expect("sigma f32");
        for (i, (&s64, &s32)) in sigma_f64.iter().zip(sigma_f32.iter()).enumerate() {
            let rel = (f64::from(s32) - s64).abs() / s64.max(1.0e-12);
            assert!(
                rel < 1.0e-3 || approx_eq(f64::from(s32), s64, 1.0e-3),
                "cell {i} rel={rel}"
            );
        }
    }
}
