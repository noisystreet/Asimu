//! 非结构 3D 网格粘性残差装配。

use crate::boundary::{BoundaryKind, BoundarySet, WallHeat};
use crate::core::{FaceId, Real};
use crate::discretization::gradient::GradientFields;
use crate::discretization::gradient::cell_temperatures;
use crate::discretization::gradient_unstructured::{
    UnstructuredGradientLsqInput, compute_unstructured_gradients_idw_lsq,
};
use crate::discretization::viscous::{ViscousFlux, face_transport_coefficients, viscous_face_flux};
use crate::discretization::wall_thermal::wall_heat_flux_into_fluid;
use crate::discretization::{BoundaryGhostBuffer, InviscidFlux};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::{IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

use super::{accumulate_boundary_face, accumulate_interior_face, is_degenerate_volume};

/// 非结构粘性残差装配输入。
pub struct ViscousAssemblyUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
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
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(format!(
            "非结构粘性装配：场/残差长度须等于网格单元数 {n}"
        )));
    }
    let temperatures = cell_temperatures(params.primitives, params.eos, Some(params.viscous))?;
    assemble_interior_faces(residual, params, &temperatures)?;
    assemble_boundary_faces(residual, params, &temperatures)
}

/// 非结构粘性梯度 + 装配输入。
pub struct ViscousAssemblyUnstructuredInput<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFields,
}

/// 计算非结构 IDWLS 梯度并装配粘性残差。
pub fn compute_gradients_and_assemble_viscous_unstructured(
    residual: &mut ConservedResidual,
    input: &mut ViscousAssemblyUnstructuredInput<'_>,
) -> Result<()> {
    compute_unstructured_gradients_idw_lsq(
        UnstructuredGradientLsqInput {
            mesh: input.mesh,
            primitives: input.primitives,
            eos: input.eos,
            boundaries: input.boundaries,
            ghosts: input.ghosts,
            min_pressure: input.min_pressure,
            viscous: Some(input.viscous),
        },
        input.gradient_scratch,
    )?;
    let params = ViscousAssemblyUnstructuredParams {
        mesh: input.mesh,
        eos: input.eos,
        viscous: input.viscous,
        boundaries: input.boundaries,
        ghosts: input.ghosts,
        primitives: input.primitives,
        gradients: input.gradient_scratch,
        min_pressure: input.min_pressure,
    };
    assemble_viscous_residual_unstructured(residual, &params)
}

fn assemble_interior_faces(
    residual: &mut ConservedResidual,
    params: &ViscousAssemblyUnstructuredParams<'_>,
    temperatures: &[Real],
) -> Result<()> {
    for face in 0..params.mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let Some(neighbor_id) = params.mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner_id = params.mesh.face_owner(face_id)?;
        let owner = owner_id.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        let metric = params.mesh.face_metric(face_id);
        let owner_volume = params.mesh.cell_metric(owner_id).volume;
        let neighbor_volume = params.mesh.cell_metric(neighbor_id).volume;
        if is_degenerate_volume(owner_volume) || is_degenerate_volume(neighbor_volume) {
            continue;
        }
        let flux = viscous_flux_at_cells(params, owner, neighbor, temperatures, metric.normal)?;
        accumulate_viscous_interior(
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
    params: &ViscousAssemblyUnstructuredParams<'_>,
    temperatures: &[Real],
) -> Result<()> {
    for patch in params.boundaries.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        let boundary_kind = viscous_boundary_kind(&patch.kind);
        for &face in &patch.face_ids {
            let owner_id = params.mesh.face_owner(face)?;
            let owner = owner_id.index() as usize;
            let metric = params.mesh.face_metric(face);
            let volume = params.mesh.cell_metric(owner_id).volume;
            if is_degenerate_volume(volume) {
                continue;
            }
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.index()))
            })?;
            let ghost_prim = primitive_from_conserved_relaxed(
                params.eos,
                &ghost.conserved,
                params.min_pressure,
            )?;
            let flux = viscous_flux_at_boundary(
                params,
                ViscousBoundaryFluxInput {
                    owner,
                    ghost_prim,
                    normal: metric.normal,
                    spacing: boundary_spacing(params.mesh, owner_id, face),
                    kind: boundary_kind,
                },
                temperatures,
            )?;
            accumulate_viscous_boundary(residual, owner, &flux, metric.area, volume)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct BoundaryViscousKind {
    wall_heat: Option<WallHeat>,
    no_slip: bool,
    is_wall: bool,
}

fn viscous_boundary_kind(kind: &BoundaryKind) -> BoundaryViscousKind {
    match kind {
        BoundaryKind::Wall { heat, no_slip, .. } => BoundaryViscousKind {
            wall_heat: Some(*heat),
            no_slip: *no_slip,
            is_wall: true,
        },
        _ => BoundaryViscousKind {
            wall_heat: None,
            no_slip: false,
            is_wall: false,
        },
    }
}

fn viscous_flux_at_cells(
    params: &ViscousAssemblyUnstructuredParams<'_>,
    left: usize,
    right: usize,
    temperatures: &[Real],
    normal: crate::core::Vector3,
) -> Result<ViscousFlux> {
    let prim_l = primitive_at(params.primitives, temperatures, left);
    let prim_r = primitive_at(params.primitives, temperatures, right);
    let grad_l = params.gradients.velocity_grad_at(left);
    let grad_r = params.gradients.velocity_grad_at(right);
    let (mu, lambda) = face_transport_coefficients(
        temperatures[left],
        temperatures[right],
        params.viscous,
        params.eos,
    )?;
    Ok(viscous_face_flux(
        &prim_l, &grad_l, &prim_r, &grad_r, normal, mu, lambda,
    ))
}

struct ViscousBoundaryFluxInput {
    owner: usize,
    ghost_prim: PrimitiveState,
    normal: crate::core::Vector3,
    spacing: Real,
    kind: BoundaryViscousKind,
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

fn boundary_spacing(mesh: &UnstructuredMesh3d, owner: crate::core::CellId, face: FaceId) -> Real {
    let cell = mesh.cell_metric(owner).center;
    let face = mesh.face_metric(face).center;
    crate::core::Vector3::new(cell.x - face.x, cell.y - face.y, cell.z - face.z).magnitude()
}

fn viscous_flux_for_accumulation(flux: &ViscousFlux) -> InviscidFlux {
    InviscidFlux {
        mass: flux.mass,
        momentum: [-flux.momentum[0], -flux.momentum[1], -flux.momentum[2]],
        energy: flux.energy,
    }
}

fn accumulate_viscous_interior(
    residual: &mut ConservedResidual,
    owner: usize,
    neighbor: usize,
    flux: &ViscousFlux,
    area: Real,
    owner_volume: Real,
    neighbor_volume: Real,
) -> Result<()> {
    let inv = viscous_flux_for_accumulation(flux);
    accumulate_interior_face(
        residual,
        owner,
        neighbor,
        &inv,
        area,
        owner_volume,
        neighbor_volume,
    )
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
        let viscous = ViscousPhysicsConfig::default();
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let mut input = ViscousAssemblyUnstructuredInput {
            mesh: &mesh,
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
