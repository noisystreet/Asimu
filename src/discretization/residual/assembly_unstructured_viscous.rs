//! 非结构 3D 网格粘性残差装配。

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::gradient::GradientFields;
use crate::discretization::gradient_unstructured::{
    UnstructuredGradientLsqInput, UnstructuredGradientScratch,
    compute_unstructured_gradients_idw_lsq_with_scratch,
};
use crate::discretization::unstructured_face_cache::{
    UnstructuredBoundaryViscousKind, UnstructuredFaceTopology, UnstructuredSolverMeshCache,
};
use crate::discretization::viscous::{
    InteriorViscousFaceGeom, InteriorViscousFaceInputs, InteriorViscousResidualMut, ViscousFlux,
    accumulate_fused_interior_viscous_face, face_transport_coefficients, viscous_face_flux,
};
use crate::discretization::wall_thermal::wall_heat_flux_into_fluid;
use crate::discretization::{BoundaryGhostBuffer, InviscidFlux};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, PrimitiveState, ViscosityModel, ViscousPhysicsConfig};

use super::{accumulate_boundary_face, is_degenerate_volume};

/// 非结构粘性残差装配输入。
pub struct ViscousAssemblyUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub face_topology: &'a UnstructuredFaceTopology,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub gradients: &'a GradientFields,
    pub min_pressure: Real,
}

/// 在已有残差上叠加非结构粘性通量贡献（不清零 residual）。
pub fn assemble_viscous_residual_unstructured(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
) -> Result<()> {
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(params.mesh.num_cells());
    crate::discretization::gradient::cell_temperatures_into(
        params.primitives,
        params.eos,
        Some(params.viscous),
        &mut scratch.gradient.temperatures,
    )?;
    assemble_viscous_residual_unstructured_with_scratch(residual, params, &mut scratch)
}

fn assemble_viscous_residual_unstructured_with_scratch(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let n = params.mesh.num_cells();
    if residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构粘性装配：场/残差长度须等于网格单元数 {n}"
        )));
    }
    if scratch.gradient.temperatures.len() != n {
        return Err(AsimuError::Field(format!(
            "非结构粘性装配：温度缓冲长度 {} 与单元数 {n} 不一致",
            scratch.gradient.temperatures.len()
        )));
    }
    {
        let _span = info_span!(
            "unstructured_viscous_assemble_interior_faces",
            faces = params.face_topology.interior.len(),
        )
        .entered();
        assemble_interior_faces(residual, params, scratch)?;
    }
    {
        let _span = info_span!(
            "unstructured_viscous_assemble_boundary_faces",
            faces = params.face_topology.boundary.len(),
        )
        .entered();
        assemble_boundary_faces(residual, params, scratch)?;
    }
    Ok(())
}

/// 非结构粘性梯度 + 装配输入。
pub struct ViscousAssemblyUnstructuredInput<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub mesh_cache: &'a UnstructuredSolverMeshCache,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFields,
}

/// 非结构粘性 RHS 复用缓冲。
pub struct ViscousAssemblyUnstructuredScratch {
    pub gradient: UnstructuredGradientScratch,
    cell_mu: Vec<Real>,
    cell_lambda: Vec<Real>,
    face_mu: Vec<Real>,
    face_lambda: Vec<Real>,
    constant_transport: Option<(Real, Real)>,
}

impl ViscousAssemblyUnstructuredScratch {
    #[must_use]
    pub fn new(num_cells: usize) -> Self {
        Self {
            gradient: UnstructuredGradientScratch::new(num_cells),
            cell_mu: Vec::new(),
            cell_lambda: Vec::new(),
            face_mu: Vec::new(),
            face_lambda: Vec::new(),
            constant_transport: None,
        }
    }

    fn ensure_cell_transport(&mut self, num_cells: usize) {
        self.cell_mu.resize(num_cells, 0.0);
        self.cell_lambda.resize(num_cells, 0.0);
    }

    fn ensure_face_transport(&mut self, num_faces: usize) {
        self.face_mu.resize(num_faces, 0.0);
        self.face_lambda.resize(num_faces, 0.0);
    }
}

/// 计算非结构 IDWLS 梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured(
    residual: &mut ConservedResidual,
    input: &mut ViscousAssemblyUnstructuredInput<'_>,
) -> Result<()> {
    let mut scratch = ViscousAssemblyUnstructuredScratch::new(input.mesh.num_cells());
    compute_gradients_and_assemble_viscous_unstructured_with_scratch(residual, input, &mut scratch)
}

/// 使用调用方提供的 scratch 计算非结构梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured_with_scratch(
    residual: &mut ConservedResidual,
    input: &mut ViscousAssemblyUnstructuredInput<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    {
        let _span = info_span!(
            "unstructured_viscous_idw_lsq_gradient",
            cells = input.mesh.num_cells(),
            interior_faces = input.mesh_cache.face_topology.interior.len(),
            boundary_faces = input.mesh_cache.face_topology.boundary.len(),
        )
        .entered();
        compute_unstructured_gradients_idw_lsq_with_scratch(
            UnstructuredGradientLsqInput {
                mesh: input.mesh,
                mesh_cache: input.mesh_cache,
                primitives: input.primitives,
                eos: input.eos,
                ghosts: input.ghosts,
                min_pressure: input.min_pressure,
                viscous: Some(input.viscous),
            },
            input.gradient_scratch,
            &mut scratch.gradient,
        )?;
    }
    let params = ViscousAssemblyUnstructuredParams {
        mesh: input.mesh,
        face_topology: &input.mesh_cache.face_topology,
        eos: input.eos,
        viscous: input.viscous,
        ghosts: input.ghosts,
        primitives: input.primitives,
        gradients: input.gradient_scratch,
        min_pressure: input.min_pressure,
    };
    assemble_viscous_residual_unstructured_with_scratch(residual, &params, scratch)
}

fn assemble_interior_faces(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let num_faces = params.face_topology.interior.len();
    scratch.ensure_face_transport(num_faces);
    if matches!(params.viscous.model, ViscosityModel::Constant { .. }) {
        scratch.constant_transport = Some(face_transport_coefficients(
            1.0,
            1.0,
            params.viscous,
            params.eos,
        )?);
    } else {
        scratch.constant_transport = None;
        let num_cells = params.mesh.num_cells();
        scratch.ensure_cell_transport(num_cells);
        {
            let _span =
                info_span!("unstructured_viscous_interior_transport", cells = num_cells).entered();
            fill_cell_transport_coefficients(params, scratch)?;
            fill_face_transport_coefficients(params, scratch)?;
        }
    }
    {
        let _span = info_span!("unstructured_viscous_interior_flux", faces = num_faces,).entered();
        accumulate_interior_faces_fused(residual, params, scratch)?;
    }
    Ok(())
}

fn fill_face_transport_coefficients(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    for (i, face) in params.face_topology.interior.iter().enumerate() {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        let owner = face.owner;
        let neighbor = face.neighbor;
        scratch.face_mu[i] = 0.5 * (scratch.cell_mu[owner] + scratch.cell_mu[neighbor]);
        scratch.face_lambda[i] = 0.5 * (scratch.cell_lambda[owner] + scratch.cell_lambda[neighbor]);
    }
    Ok(())
}

fn accumulate_interior_faces_fused(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let prim = params.primitives;
    let grad_slices = params.gradients.velocity_gradient_slices();
    let ux = prim.velocity_x.values();
    let uy = prim.velocity_y.values();
    let uz = prim.velocity_z.values();
    let inputs = InteriorViscousFaceInputs {
        grad: &grad_slices,
        ux,
        uy,
        uz,
    };
    let mut residual_mut = InteriorViscousResidualMut {
        mx: residual.momentum_x.values_mut(),
        my: residual.momentum_y.values_mut(),
        mz: residual.momentum_z.values_mut(),
        energy: residual.total_energy.values_mut(),
    };
    let constant = scratch.constant_transport;

    for (i, face) in params.face_topology.interior.iter().enumerate() {
        if face.owner_rhs_scale == 0.0 && face.neighbor_rhs_scale == 0.0 {
            continue;
        }
        let (mu, lambda) = if let Some(coeffs) = constant {
            coeffs
        } else {
            (scratch.face_mu[i], scratch.face_lambda[i])
        };
        let normal = face.normal;
        accumulate_fused_interior_viscous_face(
            &inputs,
            &mut residual_mut,
            InteriorViscousFaceGeom {
                owner: face.owner,
                neighbor: face.neighbor,
                nx: normal.x,
                ny: normal.y,
                nz: normal.z,
                mu,
                lambda,
                owner_scale: face.owner_rhs_scale,
                neighbor_scale: face.neighbor_rhs_scale,
            },
        );
    }
    Ok(())
}

fn fill_cell_transport_coefficients(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &mut ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let temperatures = &scratch.gradient.temperatures;
    for (cell, t) in temperatures.iter().enumerate() {
        let (mu, lambda) = face_transport_coefficients(*t, *t, params.viscous, params.eos)?;
        scratch.cell_mu[cell] = mu;
        scratch.cell_lambda[cell] = lambda;
    }
    Ok(())
}

fn assemble_boundary_faces(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    scratch: &ViscousAssemblyUnstructuredScratch,
) -> Result<()> {
    let temperatures = &scratch.gradient.temperatures;
    for face in &params.face_topology.boundary {
        if is_degenerate_volume(face.owner_volume) {
            continue;
        }
        let ghost = params.ghosts.get_face(face.face).ok_or_else(|| {
            AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.face.index()))
        })?;
        let ghost_prim =
            primitive_from_conserved_relaxed(params.eos, &ghost.conserved, params.min_pressure)?;
        let flux = viscous_flux_at_boundary(
            params,
            ViscousBoundaryFluxInput {
                owner: face.owner,
                ghost_prim,
                normal: face.normal,
                spacing: face.spacing,
                kind: face.viscous,
            },
            temperatures,
        )?;
        accumulate_viscous_boundary(residual, face.owner, &flux, face.area, face.owner_volume)?;
    }
    Ok(())
}

struct ViscousBoundaryFluxInput {
    owner: usize,
    ghost_prim: PrimitiveState,
    normal: crate::core::Vector3,
    spacing: Real,
    kind: UnstructuredBoundaryViscousKind,
}

fn viscous_flux_at_boundary(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    input: ViscousBoundaryFluxInput,
    temperatures: &[Real],
) -> Result<ViscousFlux> {
    let prim_o = primitive_at(params.primitives, temperatures, input.owner);
    let t_ghost = params.viscous.static_temperature(
        input.ghost_prim.pressure,
        input.ghost_prim.density.max(1.0e-30),
        params.eos,
    );
    let mut ghost = input.ghost_prim;
    ghost.temperature = t_ghost;
    let grad_o = params.gradients.velocity_grad_at(input.owner);
    let grad_g = if input.kind.is_wall {
        wall_extrapolated_gradient(&grad_o, &prim_o, &ghost, input.normal, input.spacing)
    } else {
        grad_o
    };
    let (mu, lambda) = face_transport_coefficients(
        temperatures[input.owner],
        t_ghost,
        params.viscous,
        params.eos,
    )?;
    let mut flux = viscous_face_flux(&prim_o, &grad_o, &ghost, &grad_g, input.normal, mu, lambda);
    if input.kind.no_slip {
        let grad = crate::discretization::viscous::average_gradient_for_wall(&grad_o, &grad_g);
        flux.energy = lambda
            * (grad.dt[0] * input.normal.x
                + grad.dt[1] * input.normal.y
                + grad.dt[2] * input.normal.z);
    }
    if let Some(heat) = input.kind.wall_heat {
        flux.energy = wall_heat_flux_into_fluid(
            prim_o.temperature,
            ghost.temperature,
            input.spacing,
            lambda,
            heat,
        );
    }
    Ok(flux)
}

fn primitive_at(
    primitives: &PrimitiveFields,
    temperatures: &[Real],
    cell: usize,
) -> PrimitiveState {
    PrimitiveState {
        density: primitives.density.values()[cell],
        velocity: [
            primitives.velocity_x.values()[cell],
            primitives.velocity_y.values()[cell],
            primitives.velocity_z.values()[cell],
        ],
        pressure: primitives.pressure.values()[cell],
        temperature: temperatures[cell],
    }
}

fn wall_extrapolated_gradient(
    grad_cell: &crate::discretization::VelocityGradient,
    prim_owner: &PrimitiveState,
    prim_ghost: &PrimitiveState,
    normal: crate::core::Vector3,
    spacing: Real,
) -> crate::discretization::VelocityGradient {
    if spacing <= Real::EPSILON {
        return *grad_cell;
    }
    let inv_two_delta = 1.0 / (2.0 * spacing);
    let mut grad = *grad_cell;
    for (grad_comp, u_o, u_g) in [
        (&mut grad.du, prim_owner.velocity[0], prim_ghost.velocity[0]),
        (&mut grad.dv, prim_owner.velocity[1], prim_ghost.velocity[1]),
        (&mut grad.dw, prim_owner.velocity[2], prim_ghost.velocity[2]),
    ] {
        let dudn = (u_g - u_o) * inv_two_delta;
        let grad_n = grad_comp[0] * normal.x + grad_comp[1] * normal.y + grad_comp[2] * normal.z;
        let corr = dudn - grad_n;
        grad_comp[0] += corr * normal.x;
        grad_comp[1] += corr * normal.y;
        grad_comp[2] += corr * normal.z;
    }
    let dtdn = (prim_ghost.temperature - prim_owner.temperature) * inv_two_delta;
    let grad_t_n = grad.dt[0] * normal.x + grad.dt[1] * normal.y + grad.dt[2] * normal.z;
    let corr_t = dtdn - grad_t_n;
    grad.dt[0] += corr_t * normal.x;
    grad.dt[1] += corr_t * normal.y;
    grad.dt[2] += corr_t * normal.z;
    grad
}

fn viscous_flux_for_accumulation(flux: &ViscousFlux) -> InviscidFlux {
    InviscidFlux {
        mass: flux.mass,
        momentum: [-flux.momentum[0], -flux.momentum[1], -flux.momentum[2]],
        energy: flux.energy,
    }
}

fn accumulate_viscous_boundary(
    residual: &mut ConservedResidual,
    owner: usize,
    flux: &ViscousFlux,
    area: Real,
    owner_volume: Real,
) -> Result<()> {
    let inv = viscous_flux_for_accumulation(flux);
    accumulate_boundary_face(residual, owner, &inv, area, owner_volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryPatch;
    use crate::discretization::GhostCellState;
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::{FreestreamParams, ViscousPhysicsConfig};

    #[test]
    fn uniform_closed_tet_has_near_zero_unstructured_viscous_rhs() {
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
            crate::boundary::BoundaryKind::Farfield {
                mach: fs.mach,
                pressure: fs.pressure,
                temperature: fs.temperature,
                alpha: fs.alpha,
                beta: fs.beta,
            },
        )]);
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &boundary).expect("cache");
        let viscous = ViscousPhysicsConfig::default();
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let mut input = ViscousAssemblyUnstructuredInput {
            mesh: &mesh,
            mesh_cache: &mesh_cache,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &primitives,
            min_pressure: 1.0e-8,
            gradient_scratch: &mut grad,
        };
        compute_gradients_and_assemble_viscous_unstructured(&mut rhs, &mut input).expect("visc");
        assert!(rhs.density.values().iter().all(|v| v.abs() < 1.0e-12));
        assert!(rhs.momentum_x.values().iter().all(|v| v.abs() < 1.0e-8));
        assert!(rhs.total_energy.values().iter().all(|v| v.abs() < 1.0e-8));
    }
}
