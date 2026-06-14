//! 非结构 3D 粘性残差 f32 装配（梯度、内面与边界面通量均为 f32；串行路径）。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::UnstructuredGradientScratchF32;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::{
    UnstructuredGradientLsqInputF32, compute_unstructured_gradients_idw_lsq_f32,
};
use crate::discretization::unstructured_face_cache::{
    UnstructuredFaceTopology, UnstructuredSolverMeshCache,
};
use crate::discretization::unstructured_face_cache_f32::UnstructuredFaceTopologyF32;
use crate::discretization::viscous_f32::{
    average_face_lane_f32, fused_interior_viscous_face_flux_averaged_f32,
    scatter_fused_interior_viscous_face_f32,
};
use crate::error::Result;
use crate::exec::ColoredViscousFaceGeom;
use crate::exec::ExecutionContext;
use crate::field::{ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::assembly_unstructured_viscous::assemble_boundary_faces_f32;
use super::assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredScratch, prepare_unstructured_viscous_transport_f32,
};
use crate::discretization::viscous_boundary_f32::ViscousBoundaryFluxParamsF32;

struct ViscousInteriorAssemblyF32<'a> {
    face_topology: &'a UnstructuredFaceTopologyF32,
    transport_topology: &'a UnstructuredFaceTopology,
    eos: &'a IdealGasEoS,
    viscous: &'a ViscousPhysicsConfig,
    primitives: &'a PrimitiveFieldsT<f32>,
    gradients: &'a GradientFieldsT<f32>,
    temperatures: &'a [f32],
}

/// f32 非结构粘性装配输入。
pub struct ViscousAssemblyUnstructuredF32Input<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a crate::discretization::BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFieldsT<f32>,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFieldsT<f32>,
    pub exec: &'a mut ExecutionContext,
}

/// 计算 f32 IDWLS 梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured_f32(
    residual: &mut ConservedResidualT<f32>,
    input: &mut ViscousAssemblyUnstructuredF32Input<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
    grad_scratch: &mut UnstructuredGradientScratchF32,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_viscous_idw_lsq_gradient_f32",
            cells = input.mesh.num_cells(),
        )
        .entered();
        compute_unstructured_gradients_idw_lsq_f32(
            UnstructuredGradientLsqInputF32 {
                mesh: input.mesh,
                mesh_cache: input.mesh_cache,
                primitives: input.primitives,
                eos: input.eos,
                ghosts: input.ghosts,
                min_pressure: input.min_pressure,
                viscous: Some(input.viscous),
            },
            input.gradient_scratch,
            grad_scratch,
            input.exec,
        )?;
    }
    assemble_viscous_residual_f32_interior(
        residual,
        &ViscousInteriorAssemblyF32 {
            face_topology: &input.mesh_cache.face_topology_f32,
            transport_topology: &input.mesh_cache.face_topology,
            eos: input.eos,
            viscous: input.viscous,
            primitives: input.primitives,
            gradients: input.gradient_scratch,
            temperatures: &grad_scratch.temperatures,
        },
        scratch,
    )?;
    let boundary_params = ViscousBoundaryFluxParamsF32 {
        eos: input.eos,
        viscous: input.viscous,
        primitives: input.primitives,
        gradients: input.gradient_scratch,
    };
    assemble_boundary_faces_f32(
        residual,
        &input.mesh_cache.face_topology_f32,
        input.ghosts,
        &boundary_params,
        input.min_pressure,
        &grad_scratch.temperatures,
    )
}

fn assemble_viscous_residual_f32_interior(
    residual: &mut ConservedResidualT<f32>,
    params: &ViscousInteriorAssemblyF32<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let constant = prepare_unstructured_viscous_transport_f32(
        params.transport_topology,
        params.primitives.num_cells(),
        params.viscous,
        params.eos,
        params.temperatures,
        scratch,
    )?;
    let grad_slices = params.gradients.velocity_gradient_slices();
    let _span = info_span!(
        "unstructured_viscous_interior_flux_f32",
        faces = params.face_topology.interior.len(),
    )
    .entered();
    for (i, face) in params.face_topology.interior.iter().enumerate() {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        let (mu, lambda) = if let Some(c) = constant {
            (c.0 as f32, c.1 as f32)
        } else {
            let (m, l) = scratch.face_transport_at(i);
            (m as f32, l as f32)
        };
        let normal = face.normal;
        let geom = ColoredViscousFaceGeom {
            owner: face.owner,
            neighbor: face.neighbor,
            nx: normal[0] as Real,
            ny: normal[1] as Real,
            nz: normal[2] as Real,
            mu: mu as Real,
            lambda: lambda as Real,
            owner_scale: face.owner_rhs_scale as Real,
            neighbor_scale: face.neighbor_rhs_scale as Real,
        };
        let lane =
            average_face_lane_f32(face.owner, face.neighbor, params.primitives, &grad_slices);
        let flux = fused_interior_viscous_face_flux_averaged_f32(
            lane, normal[0], normal[1], normal[2], mu, lambda,
        );
        scatter_fused_interior_viscous_face_f32(residual, &geom, &flux);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
    use crate::core::ComputeFloat;
    use crate::discretization::BoundaryGhostBuffer;
    use crate::discretization::GhostCellState;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
    use crate::physics::FreestreamParams;
    use crate::physics::ViscosityModel;

    fn uniform_closed_tet_viscous_rhs() -> (
        UnstructuredMesh3d,
        IdealGasEoS,
        BoundarySet,
        UnstructuredSolverMeshCache,
        BoundaryGhostBuffer,
        PrimitiveFieldsT<f32>,
        FreestreamParams,
    ) {
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
        let mut primitives_f32 = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        let fields_f32 =
            crate::field::ConservedFieldsT::<f32>::from_real_fields(&fields).expect("f32");
        primitives_f32
            .fill_from_conserved(&fields_f32, &eos, 1.0e-8)
            .expect("fill");
        let faces = (0..mesh.num_faces())
            .map(|face| crate::core::FaceId(face as u32))
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
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32, fs)
    }

    #[test]
    fn f32_sutherland_uniform_closed_tet_has_near_zero_viscous_rhs() {
        let (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32, _fs) =
            uniform_closed_tet_viscous_rhs();
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::AIR_SUTHERLAND, 0.72).expect("viscous");
        let mut grad = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let mut exec = ExecutionContext::for_unit_test();
        let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut grad_scratch = UnstructuredGradientScratchF32::new(mesh.num_cells());
        let mut input = ViscousAssemblyUnstructuredF32Input {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives_f32,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
            exec: &mut exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_f32(
            &mut rhs,
            &mut input,
            &mut scratch,
            &mut grad_scratch,
        )
        .expect("visc");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-12)
        );
        assert!(
            rhs.momentum_x
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
        assert!(
            rhs.total_energy
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
    }

    #[test]
    fn f32_native_boundary_uniform_closed_tet_has_near_zero_viscous_rhs() {
        let (mesh, eos, boundary, mesh_cache, ghosts, primitives_f32, _fs) =
            uniform_closed_tet_viscous_rhs();
        let viscous = ViscousPhysicsConfig::default();
        let mut grad = GradientFieldsT::<f32>::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let mut exec = ExecutionContext::for_unit_test();
        let mut scratch = ViscousAssemblyUnstructuredScratch::new(mesh.num_cells());
        let mut grad_scratch = UnstructuredGradientScratchF32::new(mesh.num_cells());
        let mut input = ViscousAssemblyUnstructuredF32Input {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives_f32,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
            exec: &mut exec,
        };
        compute_gradients_and_assemble_viscous_unstructured_f32(
            &mut rhs,
            &mut input,
            &mut scratch,
            &mut grad_scratch,
        )
        .expect("visc");
        assert!(
            rhs.density
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-12)
        );
        assert!(
            rhs.momentum_x
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
        assert!(
            rhs.total_energy
                .values()
                .iter()
                .all(|v| v.to_real().abs() < 1.0e-8)
        );
    }
}
