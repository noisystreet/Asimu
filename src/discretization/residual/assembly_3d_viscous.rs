//! 3D 结构化网格粘性残差装配。

use crate::boundary::{BoundaryKind, BoundarySet};
use crate::core::Real;
use crate::discretization::BoundaryGhostBuffer;
use crate::discretization::gradient::{
    GradientFields, cell_temperatures_into, compute_structured_gradients_3d,
};
use crate::discretization::residual::is_degenerate_volume;
use crate::discretization::viscous::{
    InteriorViscousFaceGeom, InteriorViscousFaceInputs, InteriorViscousResidualMut,
    accumulate_fused_interior_viscous_face, face_transport_coefficients,
};
use crate::discretization::viscous_assembly::{
    ViscousBoundaryFaceKind, ViscousBoundaryFluxParams, accumulate_viscous_boundary,
    viscous_flux_at_boundary,
};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, PrimitiveFields, primitive_from_conserved_relaxed};
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, LogicalFace3d, StructuredMesh3d};
use crate::physics::{IdealGasEoS, ViscosityModel, ViscousPhysicsConfig};

/// 3D 粘性残差装配参数。
pub struct ViscousAssembly3dParams<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub gradients: &'a GradientFields,
    pub min_pressure: Real,
}

struct ViscousAssembly3dScratch {
    temperatures: Vec<Real>,
    cell_mu: Vec<Real>,
    cell_lambda: Vec<Real>,
    constant_transport: Option<(Real, Real)>,
}

impl ViscousAssembly3dScratch {
    fn new(num_cells: usize) -> Self {
        Self {
            temperatures: vec![0.0; num_cells],
            cell_mu: Vec::new(),
            cell_lambda: Vec::new(),
            constant_transport: None,
        }
    }

    fn ensure_cell_transport(&mut self, num_cells: usize) {
        self.cell_mu.resize(num_cells, 0.0);
        self.cell_lambda.resize(num_cells, 0.0);
    }
}

/// 在已有残差上叠加粘性通量贡献（不清零 residual）。
pub fn assemble_viscous_residual_3d(
    residual: &mut ConservedResidual,
    params: &ViscousAssembly3dParams<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    let n = mesh.num_cells();
    if residual.num_cells() != n || params.primitives.num_cells() != n {
        return Err(AsimuError::Field(
            "粘性装配：场尺寸与网格不一致".to_string(),
        ));
    }
    let mut scratch = ViscousAssembly3dScratch::new(n);
    cell_temperatures_into(
        params.primitives,
        params.eos,
        Some(params.viscous),
        &mut scratch.temperatures,
    )?;
    assemble_viscous_i_faces(residual, mesh, params, &mut scratch)?;
    assemble_viscous_j_faces(residual, mesh, params, &mut scratch)?;
    assemble_viscous_k_faces(residual, mesh, params, &mut scratch)?;
    assemble_viscous_boundary_faces(residual, mesh, params, &scratch.temperatures)
}

fn prepare_interior_transport(
    params: &ViscousAssembly3dParams<'_>,
    scratch: &mut ViscousAssembly3dScratch,
) -> Result<()> {
    if matches!(params.viscous.model, ViscosityModel::Constant { .. }) {
        scratch.constant_transport = Some(face_transport_coefficients(
            1.0,
            1.0,
            params.viscous,
            params.eos,
        )?);
        return Ok(());
    }
    scratch.constant_transport = None;
    let num_cells = params.mesh.num_cells();
    scratch.ensure_cell_transport(num_cells);
    for (cell, t) in scratch.temperatures.iter().enumerate().take(num_cells) {
        let (mu, lambda) = face_transport_coefficients(*t, *t, params.viscous, params.eos)?;
        scratch.cell_mu[cell] = mu;
        scratch.cell_lambda[cell] = lambda;
    }
    Ok(())
}

fn face_transport_at_cells(
    scratch: &ViscousAssembly3dScratch,
    owner: usize,
    neighbor: usize,
) -> (Real, Real) {
    if let Some(coeffs) = scratch.constant_transport {
        coeffs
    } else {
        (
            0.5 * (scratch.cell_mu[owner] + scratch.cell_mu[neighbor]),
            0.5 * (scratch.cell_lambda[owner] + scratch.cell_lambda[neighbor]),
        )
    }
}

struct StructuredInteriorViscousFace {
    owner: usize,
    neighbor: usize,
    normal: crate::core::Vector3,
    area: Real,
    owner_volume: Real,
    neighbor_volume: Real,
}

fn accumulate_fused_interior_structured(
    residual: &mut ConservedResidual,
    params: &ViscousAssembly3dParams<'_>,
    scratch: &ViscousAssembly3dScratch,
    face: StructuredInteriorViscousFace,
) {
    if is_degenerate_volume(face.owner_volume) || is_degenerate_volume(face.neighbor_volume) {
        return;
    }
    let (mu, lambda) = face_transport_at_cells(scratch, face.owner, face.neighbor);
    let prim = params.primitives;
    let grad_slices = params.gradients.velocity_gradient_slices();
    let inputs = InteriorViscousFaceInputs {
        grad: &grad_slices,
        ux: prim.velocity_x.values(),
        uy: prim.velocity_y.values(),
        uz: prim.velocity_z.values(),
    };
    let mut residual_mut = InteriorViscousResidualMut {
        mx: residual.momentum_x.values_mut(),
        my: residual.momentum_y.values_mut(),
        mz: residual.momentum_z.values_mut(),
        energy: residual.total_energy.values_mut(),
    };
    accumulate_fused_interior_viscous_face(
        &inputs,
        &mut residual_mut,
        InteriorViscousFaceGeom {
            owner: face.owner,
            neighbor: face.neighbor,
            nx: face.normal.x,
            ny: face.normal.y,
            nz: face.normal.z,
            mu,
            lambda,
            owner_scale: -face.area / face.owner_volume,
            neighbor_scale: face.area / face.neighbor_volume,
        },
    );
}

fn assemble_viscous_i_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    scratch: &mut ViscousAssembly3dScratch,
) -> Result<()> {
    prepare_interior_transport(params, scratch)?;
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                accumulate_fused_interior_structured(
                    residual,
                    params,
                    scratch,
                    StructuredInteriorViscousFace {
                        owner,
                        neighbor,
                        normal: face.normal,
                        area: face.area,
                        owner_volume: mesh.cell_metric(i, j, k).volume,
                        neighbor_volume: mesh.cell_metric(i + 1, j, k).volume,
                    },
                );
            }
        }
    }
    Ok(())
}

fn assemble_viscous_j_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    scratch: &mut ViscousAssembly3dScratch,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j + 1, k);
                let face = mesh.j_face_metric(i, j, k);
                accumulate_fused_interior_structured(
                    residual,
                    params,
                    scratch,
                    StructuredInteriorViscousFace {
                        owner,
                        neighbor,
                        normal: face.normal,
                        area: face.area,
                        owner_volume: mesh.cell_metric(i, j, k).volume,
                        neighbor_volume: mesh.cell_metric(i, j + 1, k).volume,
                    },
                );
            }
        }
    }
    Ok(())
}

fn assemble_viscous_k_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    scratch: &mut ViscousAssembly3dScratch,
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny {
            for i in 0..nx {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i, j, k + 1);
                let face = mesh.k_face_metric(i, j, k);
                accumulate_fused_interior_structured(
                    residual,
                    params,
                    scratch,
                    StructuredInteriorViscousFace {
                        owner,
                        neighbor,
                        normal: face.normal,
                        area: face.area,
                        owner_volume: mesh.cell_metric(i, j, k).volume,
                        neighbor_volume: mesh.cell_metric(i, j, k + 1).volume,
                    },
                );
            }
        }
    }
    Ok(())
}

fn assemble_viscous_boundary_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    temperatures: &[Real],
) -> Result<()> {
    let boundary_params = ViscousBoundaryFluxParams {
        eos: params.eos,
        viscous: params.viscous,
        primitives: params.primitives,
        gradients: params.gradients,
    };
    for patch in params.boundaries.patches() {
        if matches!(patch.kind, BoundaryKind::Periodic { .. }) {
            continue;
        }
        let (wall_heat, no_slip, is_wall) = match &patch.kind {
            BoundaryKind::Wall { heat, no_slip, .. } => (Some(*heat), *no_slip, true),
            _ => (None, false, false),
        };
        for &face in &patch.face_ids {
            let owner = BoundaryMesh::face_owner(mesh, face)?.index() as usize;
            let (logical, local) = LogicalFace3d::decode(face)?;
            let (i, j, k) = mesh.face_ij(logical, local)?;
            let geom = mesh.face_geometry_3d(face)?;
            let ghost = params.ghosts.get_face(face).ok_or_else(|| {
                AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.index()))
            })?;
            let ghost_prim = primitive_from_conserved_relaxed(
                params.eos,
                &ghost.conserved,
                params.min_pressure,
            )?;
            let flux = viscous_flux_at_boundary(
                &boundary_params,
                owner,
                ghost_prim,
                geom.normal,
                geom.spacing,
                ViscousBoundaryFaceKind {
                    is_wall,
                    no_slip,
                    wall_heat,
                },
                temperatures,
            )?;
            let volume = mesh.cell_metric(i, j, k).volume;
            if is_degenerate_volume(volume) {
                continue;
            }
            accumulate_viscous_boundary(residual, owner, &flux, geom.area, volume)?;
        }
    }
    Ok(())
}

/// 粘性梯度 + 装配输入（合并多参数，满足复杂度门禁）。
pub struct ViscousAssembly3dInput<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub viscous: &'a ViscousPhysicsConfig,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a PrimitiveFields,
    pub min_pressure: Real,
    pub gradient_scratch: &'a mut GradientFields,
}

/// 计算梯度并装配粘性残差（叠加到已有残差，通常紧随无粘项）。
pub fn compute_gradients_and_assemble_viscous_3d(
    residual: &mut ConservedResidual,
    input: &mut ViscousAssembly3dInput<'_>,
) -> Result<()> {
    compute_structured_gradients_3d(
        input.mesh,
        input.primitives,
        input.eos,
        input.boundaries,
        input.ghosts,
        input.min_pressure,
        Some(input.viscous),
        input.gradient_scratch,
    )?;
    let params = ViscousAssembly3dParams {
        mesh: input.mesh,
        eos: input.eos,
        viscous: input.viscous,
        boundaries: input.boundaries,
        ghosts: input.ghosts,
        primitives: input.primitives,
        gradients: input.gradient_scratch,
        min_pressure: input.min_pressure,
    };
    assemble_viscous_residual_3d(residual, &params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
    use crate::discretization::viscous::{face_transport_coefficients, viscous_face_flux};
    use crate::discretization::viscous_assembly::accumulate_viscous_interior;
    use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
    use crate::field::{ConservedFields, PrimitiveFields};
    use crate::mesh::StructuredMesh3d;
    use crate::physics::{FreestreamContext, FreestreamParams, ViscosityModel};

    #[test]
    fn uniform_freestream_viscous_rhs_near_zero() {
        let pair = FreestreamPairFixture::air_sutherland(0.1);
        pair.for_each_viscous_side(|side| {
            let (mesh, boundary, fields, ghosts) =
                uniform_farfield_box(3, 3, 3, 1.0, 1.0, 1.0, side);
            let viscous = side.viscous.expect("viscous side");
            let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
            prim.fill_from_conserved(&fields, side.eos, side.min_pressure)
                .expect("fill");
            let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
            let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
            let mut input = ViscousAssembly3dInput {
                mesh: &mesh,
                eos: side.eos,
                viscous,
                boundaries: &boundary,
                ghosts: &ghosts,
                primitives: &prim,
                min_pressure: side.min_pressure,
                gradient_scratch: &mut grad,
            };
            compute_gradients_and_assemble_viscous_3d(&mut rhs, &mut input).expect("viscous");
            assert!(
                rhs.density.values().iter().all(|&v| v.abs() < 1.0e-10),
                "{} density viscous rhs",
                side.label
            );
            assert!(
                rhs.momentum_x.values().iter().all(|&v| v.abs() < 1.0e-8),
                "{} momentum viscous rhs",
                side.label
            );
        });
    }

    #[test]
    fn shear_layer_viscous_work_heats_slow_side() {
        use crate::core::Vector3;
        use crate::discretization::gradient::VelocityGradient;

        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(1.0e-5).expect("mu"), 0.72)
                .expect("cfg");
        let slow = eos
            .freestream_primitive(0.0, 101_325.0, 300.0, [10.0, 0.0, 0.0])
            .expect("prim");
        let mut fast = slow;
        fast.velocity[0] = 110.0;
        let grad_l = VelocityGradient {
            du: [0.0; 3],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let grad_r = VelocityGradient {
            du: [100.0, 0.0, 0.0],
            dv: [0.0; 3],
            dw: [0.0; 3],
            dt: [0.0; 3],
        };
        let (mu, lambda) = face_transport_coefficients(300.0, 300.0, &viscous, &eos).expect("tc");
        let flux = viscous_face_flux(
            &slow,
            &grad_l,
            &fast,
            &grad_r,
            Vector3::new(1.0, 0.0, 0.0),
            mu,
            lambda,
        );
        let mut rhs = ConservedResidual::zeros(2).expect("rhs");
        accumulate_viscous_interior(&mut rhs, 0, 1, &flux, 1.0, 1.0, 1.0).expect("acc");
        assert!(
            rhs.total_energy.values()[0] > 1.0e-12,
            "shear dissipation should heat slower owner cell, got {}",
            rhs.total_energy.values()[0]
        );
        let _ = lambda;
    }

    #[test]
    fn viscous_diffusion_reduces_streamwise_velocity_spike_at_wall() {
        use crate::boundary::WallHeat;

        let nx = 4;
        let mesh = StructuredMesh3d::uniform_box("box", nx, 2, 2, 1.0, 0.5, 0.5).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(2.0e-5).expect("mu"), 0.72)
                .expect("visc");
        let p = 101_325.0;
        let t = 300.0;
        let rho = p / (eos.gas_constant * t);
        let u_bulk = 100.0;
        let u_spike = 150.0;
        let mut fields = ConservedFields::from_freestream(
            mesh.num_cells(),
            &eos,
            &FreestreamParams {
                mach: 0.0,
                pressure: p,
                temperature: t,
                ..FreestreamParams::default()
            },
        )
        .expect("fields");
        for v in fields.density.values_mut() {
            *v = rho;
        }
        let e_int = eos.specific_internal_energy(t, rho).expect("e");
        for k in 0..mesh.nz {
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    let idx = mesh.cell_index(i, j, k);
                    let u = if i == 0 { u_spike } else { u_bulk };
                    fields.momentum_x.values_mut()[idx] = rho * u;
                    fields.total_energy.values_mut()[idx] = rho * (e_int + 0.5 * u * u);
                }
            }
        }
        let mut patches = Vec::new();
        for name in ["i_max", "j_min", "j_max", "k_min", "k_max"] {
            patches.push(BoundaryPatch::new(
                name,
                mesh.resolve_logical_boundary(name).expect("faces"),
                BoundaryKind::Farfield {
                    mach: 0.0,
                    pressure: p,
                    temperature: t,
                    alpha: 0.0,
                    beta: 0.0,
                },
            ));
        }
        patches.push(BoundaryPatch::new(
            "i_min",
            mesh.resolve_logical_boundary("i_min").expect("i_min"),
            BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
        ));
        let boundary = crate::boundary::BoundarySet::new(patches);
        let mut ghosts = BoundaryGhostBuffer::new();
        let fs_ctx = FreestreamContext::new(&eos, None, Some(&viscous));
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary,
            &fields,
            &mut ghosts,
            &fs_ctx,
            &FreestreamParams {
                mach: 0.0,
                pressure: p,
                temperature: t,
                ..FreestreamParams::default()
            },
            Some(&viscous),
        )
        .expect("bc");
        let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        prim.fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let mut input = ViscousAssembly3dInput {
            mesh: &mesh,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &prim,
            min_pressure: 1.0e-6,
            gradient_scratch: &mut grad,
        };
        compute_gradients_and_assemble_viscous_3d(&mut rhs, &mut input).expect("viscous");
        let wall_cell = mesh.cell_index(0, 0, 0);
        let mx = rhs.momentum_x.values()[wall_cell];
        assert!(
            mx < -1.0e-8,
            "viscous diffusion should reduce wall-adjacent velocity spike, got momentum rhs {mx}"
        );
    }

    #[test]
    fn isothermal_cold_wall_cools_adjacent_cell() {
        use crate::boundary::WallHeat;

        let mesh = StructuredMesh3d::uniform_box("box", 4, 2, 2, 1.0, 0.5, 0.5).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let viscous =
            ViscousPhysicsConfig::new(ViscosityModel::constant(2.0e-5).expect("mu"), 0.72)
                .expect("visc");
        let t_hot = 400.0;
        let p = 101_325.0;
        let rho = p / (eos.gas_constant * t_hot);
        let mut fields = ConservedFields::from_freestream(
            mesh.num_cells(),
            &eos,
            &FreestreamParams {
                mach: 0.0,
                pressure: p,
                temperature: t_hot,
                ..FreestreamParams::default()
            },
        )
        .expect("fields");
        for v in fields.density.values_mut() {
            *v = rho;
        }
        let e_int = eos.specific_internal_energy(t_hot, rho).expect("e");
        for v in fields.total_energy.values_mut() {
            *v = rho * e_int;
        }
        let faces = mesh.resolve_logical_boundary("i_min").expect("i_min");
        let boundary = crate::boundary::BoundarySet::new(vec![BoundaryPatch::new(
            "i_min",
            faces,
            BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Isothermal { temperature: 280.0 },
            },
        )]);
        let mut ghosts = BoundaryGhostBuffer::new();
        let fs_ctx = FreestreamContext::new(&eos, None, Some(&viscous));
        apply_compressible_boundary_conditions(
            &mesh,
            &boundary,
            &fields,
            &mut ghosts,
            &fs_ctx,
            &FreestreamParams::default(),
            Some(&viscous),
        )
        .expect("bc");
        let mut prim = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        prim.fill_from_conserved(&fields, &eos, 1.0e-6)
            .expect("fill");
        let mut grad = GradientFields::zeros(mesh.num_cells()).expect("grad");
        let mut rhs = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let mut input = ViscousAssembly3dInput {
            mesh: &mesh,
            eos: &eos,
            viscous: &viscous,
            boundaries: &boundary,
            ghosts: &ghosts,
            primitives: &prim,
            min_pressure: 1.0e-6,
            gradient_scratch: &mut grad,
        };
        compute_gradients_and_assemble_viscous_3d(&mut rhs, &mut input).expect("viscous");
        let wall_cell = mesh.cell_index(0, 0, 0);
        let energy_rhs = rhs.total_energy.values()[wall_cell];
        assert!(
            energy_rhs < -1.0e-6,
            "cold isothermal wall should remove energy from hot fluid cell, got {energy_rhs}"
        );
    }
}
