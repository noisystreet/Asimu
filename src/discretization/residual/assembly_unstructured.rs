//! 非结构 3D 网格无粘残差装配（一阶面循环）。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::{FaceId, Real};
use crate::discretization::unstructured_face_cache::{
    UnstructuredFaceTopology, UnstructuredSolverMeshCache,
};
use crate::discretization::{
    BoundaryGhostBuffer, FaceFluxInput, GradientFields, InviscidFlux, InviscidFluxConfig,
    ReconstructionKind, UnstructuredGradientLimiter, UnstructuredMusclReconstructionCtx,
    face_inviscid_flux, face_inviscid_flux_from_interface, reconstruct_unstructured_boundary_face,
    reconstruct_unstructured_interior_face,
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
    /// 若提供，内面走缓存拓扑 + 着色桶顺序（与粘性共用 `InteriorFaceColoring`）。
    pub face_topology: Option<&'a UnstructuredFaceTopology>,
    /// 二阶重构：完整 mesh cache（含限制器样本与面心偏移）。
    pub mesh_cache: Option<&'a UnstructuredSolverMeshCache>,
    /// 二阶重构：IDWLS 原始变量梯度。
    pub gradients: Option<&'a GradientFields>,
    pub min_pressure: Real,
}

/// scatter 阶段所需的内面几何（与 `UnstructuredInteriorFace` 子集一致）。
#[derive(Debug, Clone, Copy)]
struct InteriorInviscidScatterGeom {
    owner: usize,
    neighbor: usize,
    area: Real,
    owner_volume: Real,
    neighbor_volume: Real,
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
    if params.config.reconstruction == ReconstructionKind::Muscl {
        validate_unstructured_muscl_params(params)?;
    }
    residual.clear();
    if let Some(topology) = params.face_topology {
        assemble_interior_faces_cached(residual, params, topology)?;
    } else {
        assemble_interior_faces(mesh, residual, params)?;
    }
    assemble_boundary_faces(residual, params)
}

fn validate_unstructured_muscl_params(
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    if params.mesh_cache.is_none() || params.face_topology.is_none() || params.gradients.is_none() {
        return Err(AsimuError::Config(
            "非结构 MUSCL 须同时提供 mesh_cache、face_topology 与 gradients".to_string(),
        ));
    }
    if params.config.unstructured_gradient_limiter.is_none() {
        return Err(AsimuError::Config(
            "非结构 MUSCL 须设置 unstructured_limiter（barth_jespersen 或 venkatakrishnan）"
                .to_string(),
        ));
    }
    Ok(())
}

fn unstructured_limiter(
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> UnstructuredGradientLimiter {
    params
        .config
        .unstructured_gradient_limiter
        .unwrap_or(UnstructuredGradientLimiter::BarthJespersen)
}

fn compute_interior_inviscid_face_contribution(
    face_idx: usize,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<Option<(InteriorInviscidScatterGeom, InviscidFlux)>> {
    let face = &topology.interior[face_idx];
    if is_degenerate_volume(face.owner_volume) || is_degenerate_volume(face.neighbor_volume) {
        return Ok(None);
    }
    let flux = if params.config.reconstruction == ReconstructionKind::Muscl {
        let mesh_cache = params.mesh_cache.expect("muscl cache");
        let gradients = params.gradients.expect("muscl gradients");
        let limiter = unstructured_limiter(params);
        let ctx = UnstructuredMusclReconstructionCtx {
            mesh_cache,
            primitives: params.primitives,
            ghosts: params.ghosts,
            eos: params.eos,
            min_pressure: params.min_pressure,
            limiter,
        };
        let iface = reconstruct_unstructured_interior_face(
            face,
            ctx,
            gradients.inviscid_primitive_grad_at(face.owner),
            gradients.inviscid_primitive_grad_at(face.neighbor),
        )?;
        face_inviscid_flux_from_interface(iface, face.normal, params.eos, params.config)?
    } else {
        let owner_prim = params.primitives.cell_primitive(face.owner);
        let neighbor_prim = params.primitives.cell_primitive(face.neighbor);
        face_inviscid_flux(
            FaceFluxInput::first_order(&owner_prim, &neighbor_prim),
            face.normal,
            params.eos,
            params.config,
        )?
    };
    let geom = InteriorInviscidScatterGeom {
        owner: face.owner,
        neighbor: face.neighbor,
        area: face.area,
        owner_volume: face.owner_volume,
        neighbor_volume: face.neighbor_volume,
    };
    Ok(Some((geom, flux)))
}

fn scatter_interior_inviscid_face(
    residual: &mut ConservedResidual,
    geom: &InteriorInviscidScatterGeom,
    flux: &InviscidFlux,
) -> Result<()> {
    accumulate_interior_face(
        residual,
        geom.owner,
        geom.neighbor,
        flux,
        geom.area,
        geom.owner_volume,
        geom.neighbor_volume,
    )
}

#[cfg(any(not(feature = "parallel-fvm"), test))]
fn accumulate_one_interior_inviscid_face(
    face_idx: usize,
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    if let Some((geom, flux)) =
        compute_interior_inviscid_face_contribution(face_idx, params, topology)?
    {
        scatter_interior_inviscid_face(residual, &geom, &flux)?;
    }
    Ok(())
}

fn assemble_interior_faces_cached(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
    topology: &UnstructuredFaceTopology,
) -> Result<()> {
    #[cfg(not(feature = "parallel-fvm"))]
    {
        for bucket in &topology.interior_coloring.buckets {
            for &face_idx in bucket {
                accumulate_one_interior_inviscid_face(face_idx, residual, params, topology)?;
            }
        }
    }

    #[cfg(feature = "parallel-fvm")]
    {
        let bucket_results = topology.interior_coloring.par_map_buckets(|face_idx| {
            compute_interior_inviscid_face_contribution(face_idx, params, topology)
        });
        for bucket in bucket_results {
            for item in bucket {
                if let Some((geom, flux)) = item? {
                    scatter_interior_inviscid_face(residual, &geom, &flux)?;
                }
            }
        }
    }
    Ok(())
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
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    if params.config.reconstruction == ReconstructionKind::Muscl {
        assemble_boundary_faces_muscl(residual, params)
    } else {
        assemble_boundary_faces_first_order(residual, params)
    }
}

fn assemble_boundary_faces_muscl(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mesh_cache = params.mesh_cache.expect("muscl cache");
    let gradients = params.gradients.expect("muscl gradients");
    let ctx = UnstructuredMusclReconstructionCtx {
        mesh_cache,
        primitives: params.primitives,
        ghosts: params.ghosts,
        eos: params.eos,
        min_pressure: params.min_pressure,
        limiter: unstructured_limiter(params),
    };
    for bface in &mesh_cache.face_topology.boundary {
        if is_degenerate_volume(bface.owner_volume) {
            continue;
        }
        let iface = reconstruct_unstructured_boundary_face(
            bface,
            ctx,
            gradients.inviscid_primitive_grad_at(bface.owner),
        )?;
        let flux =
            face_inviscid_flux_from_interface(iface, bface.normal, params.eos, params.config)?;
        accumulate_boundary_face(residual, bface.owner, &flux, bface.area, bface.owner_volume)?;
    }
    Ok(())
}

fn assemble_boundary_faces_first_order(
    residual: &mut ConservedResidual,
    params: &InviscidAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mesh = params.mesh;
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
                params.min_pressure,
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
    use crate::core::approx_eq;
    use crate::discretization::{
        GradientFields, InviscidFluxConfig, UnstructuredGradientLimiter,
        UnstructuredGradientLsqInput, UnstructuredGradientScratch, UnstructuredSolverMeshCache,
        compute_unstructured_inviscid_muscl_gradients_idw_lsq,
    };
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::{FreestreamParams, IdealGasEoS};

    fn two_tet_mesh_and_boundary() -> (UnstructuredMesh3d, BoundarySet) {
        let mesh = UnstructuredMesh3d::new(
            "two_tets",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![
                UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell"),
                UnstructuredCell::new(CellKind::Tet, vec![1, 2, 3, 4]).expect("cell"),
            ],
        )
        .expect("mesh");
        let faces = (0..mesh.num_faces())
            .map(|face| FaceId(face as u32))
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

    fn perturbed_two_tet_primitives(mesh: &UnstructuredMesh3d) -> PrimitiveFields {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields = ConservedFields::from_freestream(
            mesh.num_cells(),
            &eos,
            &FreestreamParams {
                mach: 0.3,
                ..FreestreamParams::default()
            },
        )
        .expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, 1.0e-8)
            .expect("fill");
        for (cell, ux) in primitives.velocity_x.values_mut().iter_mut().enumerate() {
            *ux = 100.0 + cell as f64 * 50.0;
        }
        primitives
    }

    fn inviscid_interior_only_residual(
        params: &InviscidAssemblyUnstructuredParams<'_>,
        linear_order: bool,
    ) -> ConservedResidual {
        let mut residual = ConservedResidual::zeros(params.mesh.num_cells()).expect("rhs");
        let topology = params.face_topology.expect("topology");
        let coloring = &topology.interior_coloring;
        if linear_order {
            coloring.for_each_face_index_linear(topology.interior.len(), |face_idx| {
                accumulate_one_interior_inviscid_face(face_idx, &mut residual, params, topology)
                    .expect("face");
            });
        } else {
            coloring.for_each_face_index(|face_idx| {
                accumulate_one_interior_inviscid_face(face_idx, &mut residual, params, topology)
                    .expect("face");
            });
        }
        residual
    }

    fn assert_residuals_match(a: &ConservedResidual, b: &ConservedResidual) {
        for (va, vb) in a.density.values().iter().zip(b.density.values()) {
            assert!(approx_eq(*va, *vb, 1.0e-12));
        }
        for (va, vb) in a.momentum_x.values().iter().zip(b.momentum_x.values()) {
            assert!(approx_eq(*va, *vb, 1.0e-12));
        }
        for (va, vb) in a.total_energy.values().iter().zip(b.total_energy.values()) {
            assert!(approx_eq(*va, *vb, 1.0e-12));
        }
    }

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
            face_topology: None,
            mesh_cache: None,
            gradients: None,
            min_pressure: 1.0e-8,
        };
        assemble_inviscid_residual_unstructured(&fields, &mut residual, &params).expect("rhs");
        assert!(residual.density_rms_norm() < 1.0e-10);
    }

    fn closed_tet_freestream_muscl_rhs(limiter: UnstructuredGradientLimiter) -> Real {
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
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let mut gradients = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut scratch = UnstructuredGradientScratch::new(mesh.num_cells());
        compute_unstructured_inviscid_muscl_gradients_idw_lsq(
            UnstructuredGradientLsqInput {
                mesh: &mesh,
                mesh_cache: &mesh_cache,
                primitives: &primitives,
                eos: &eos,
                ghosts: &ghosts,
                min_pressure: 1.0e-8,
                viscous: None,
            },
            &mut gradients,
            &mut scratch,
        )
        .expect("grad");
        let config = InviscidFluxConfig::muscl_roe().with_unstructured_gradient_limiter(limiter);
        let mut residual = ConservedResidual::zeros(mesh.num_cells()).expect("residual");
        let params = InviscidAssemblyUnstructuredParams {
            mesh: &mesh,
            eos: &eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            face_topology: Some(&mesh_cache.face_topology),
            mesh_cache: Some(&mesh_cache),
            gradients: Some(&gradients),
            min_pressure: 1.0e-8,
        };
        assemble_inviscid_residual_unstructured(&fields, &mut residual, &params).expect("rhs");
        residual.density_rms_norm()
    }

    #[test]
    fn uniform_freestream_muscl_bj_rhs_near_zero() {
        assert!(
            closed_tet_freestream_muscl_rhs(UnstructuredGradientLimiter::BarthJespersen) < 1.0e-9
        );
    }

    #[test]
    fn uniform_freestream_muscl_venk_rhs_near_zero() {
        assert!(
            closed_tet_freestream_muscl_rhs(UnstructuredGradientLimiter::Venkatakrishnan) < 1.0e-9
        );
    }

    #[test]
    fn cached_interior_inviscid_matches_mesh_face_loop() {
        let (mesh, boundary) = two_tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.3,
            ..FreestreamParams::default()
        };
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let primitives = perturbed_two_tet_primitives(&mesh);
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
        let config = InviscidFluxConfig::roe_first_order();
        let params_mesh = InviscidAssemblyUnstructuredParams {
            mesh: &mesh,
            eos: &eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            face_topology: None,
            mesh_cache: None,
            gradients: None,
            min_pressure: 1.0e-8,
        };
        let params_cached = InviscidAssemblyUnstructuredParams {
            face_topology: Some(&mesh_cache.face_topology),
            mesh_cache: Some(&mesh_cache),
            ..params_mesh
        };
        let mut mesh_loop = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let mut cached = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        assemble_inviscid_residual_unstructured(&fields, &mut mesh_loop, &params_mesh).expect("m");
        assemble_inviscid_residual_unstructured(&fields, &mut cached, &params_cached).expect("c");
        assert_residuals_match(&mesh_loop, &cached);
    }

    #[test]
    fn colored_interior_inviscid_matches_linear_face_order() {
        let (mesh, boundary) = two_tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let primitives = perturbed_two_tet_primitives(&mesh);
        let config = InviscidFluxConfig::roe_first_order();
        let params = InviscidAssemblyUnstructuredParams {
            mesh: &mesh,
            eos: &eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &BoundaryGhostBuffer::new(),
            primitives: &primitives,
            face_topology: Some(&mesh_cache.face_topology),
            mesh_cache: Some(&mesh_cache),
            gradients: None,
            min_pressure: 1.0e-8,
        };
        let linear = inviscid_interior_only_residual(&params, true);
        let colored = inviscid_interior_only_residual(&params, false);
        assert_residuals_match(&linear, &colored);
    }

    #[cfg(feature = "parallel-fvm")]
    #[test]
    fn parallel_interior_inviscid_matches_colored_serial() {
        let (mesh, boundary) = two_tet_mesh_and_boundary();
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let eos = IdealGasEoS::AIR_STANDARD;
        let primitives = perturbed_two_tet_primitives(&mesh);
        let config = InviscidFluxConfig::roe_first_order();
        let params = InviscidAssemblyUnstructuredParams {
            mesh: &mesh,
            eos: &eos,
            config: &config,
            boundaries: &boundary,
            ghosts: &BoundaryGhostBuffer::new(),
            primitives: &primitives,
            face_topology: Some(&mesh_cache.face_topology),
            mesh_cache: Some(&mesh_cache),
            gradients: None,
            min_pressure: 1.0e-8,
        };
        let serial = inviscid_interior_only_residual(&params, false);
        let mut parallel = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        assemble_interior_faces_cached(&mut parallel, &params, &mesh_cache.face_topology)
            .expect("par");
        assert_residuals_match(&serial, &parallel);
    }
}
