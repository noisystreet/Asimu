//! 非结构 3D 粘性残差 f32 装配（梯度与通量均为 f32；串行内面路径）。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::gradient_typed::GradientFieldsT;
use crate::discretization::gradient_unstructured_f32::{
    UnstructuredGradientLsqInputF32, UnstructuredGradientScratchF32,
    compute_unstructured_gradients_idw_lsq_f32,
};
use crate::discretization::unstructured_face_cache::UnstructuredSolverMeshCache;
use crate::discretization::viscous_f32::{
    average_face_lane_f32, fused_interior_viscous_face_flux_averaged_f32,
    scatter_fused_interior_viscous_face_f32,
};
use crate::error::Result;
use crate::exec::ColoredViscousFaceGeom;
use crate::exec::ExecutionContext;
use crate::field::{ConservedResidualT, PrimitiveFields, PrimitiveFieldsT, ScalarField};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, ViscousPhysicsConfig};

use super::assembly_unstructured_viscous::assemble_boundary_faces_typed;
use super::assembly_unstructured_viscous::{
    ViscousAssemblyUnstructuredParams, ViscousAssemblyUnstructuredScratch,
    prepare_unstructured_viscous_transport,
};

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
    scratch
        .gradient
        .temperatures
        .resize(grad_scratch.temperatures.len(), 0.0);
    for (dst, &src) in scratch
        .gradient
        .temperatures
        .iter_mut()
        .zip(grad_scratch.temperatures.iter())
    {
        *dst = src as Real;
    }
    let f64_primitives = bridge_primitives_f32(input.primitives)?;
    let f64_gradients = input.gradient_scratch.to_real_fields()?;
    let params = ViscousAssemblyUnstructuredParams {
        mesh: input.mesh,
        face_topology: &input.mesh_cache.face_topology,
        eos: input.eos,
        viscous: input.viscous,
        ghosts: input.ghosts,
        primitives: &f64_primitives,
        gradients: &f64_gradients,
        min_pressure: input.min_pressure,
    };
    assemble_viscous_residual_f32_interior(
        residual,
        &params,
        input.primitives,
        input.gradient_scratch,
        scratch,
    )?;
    assemble_boundary_faces_typed(residual, &params, scratch)
}

fn bridge_primitives_f32(prim: &PrimitiveFieldsT<f32>) -> Result<PrimitiveFields> {
    Ok(PrimitiveFields {
        density: ScalarField::from_real_values(prim.density.to_real_values())?,
        pressure: ScalarField::from_real_values(prim.pressure.to_real_values())?,
        velocity_x: ScalarField::from_real_values(prim.velocity_x.to_real_values())?,
        velocity_y: ScalarField::from_real_values(prim.velocity_y.to_real_values())?,
        velocity_z: ScalarField::from_real_values(prim.velocity_z.to_real_values())?,
    })
}

fn assemble_viscous_residual_f32_interior(
    residual: &mut ConservedResidualT<f32>,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    primitives: &PrimitiveFieldsT<f32>,
    gradients: &GradientFieldsT<f32>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let constant = prepare_unstructured_viscous_transport(params, scratch)?;
    let grad_slices = gradients.velocity_gradient_slices();
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
            nx: normal.x,
            ny: normal.y,
            nz: normal.z,
            mu: mu as Real,
            lambda: lambda as Real,
            owner_scale: face.owner_rhs_scale,
            neighbor_scale: face.neighbor_rhs_scale,
        };
        let lane = average_face_lane_f32(face.owner, face.neighbor, primitives, &grad_slices);
        let flux = fused_interior_viscous_face_flux_averaged_f32(
            lane,
            normal.x as f32,
            normal.y as f32,
            normal.z as f32,
            mu,
            lambda,
        );
        scatter_fused_interior_viscous_face_f32(residual, &geom, &flux);
    }
    Ok(())
}
