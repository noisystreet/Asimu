//! 3D 结构化网格粘性残差装配。

use crate::boundary::{BoundaryKind, BoundarySet, WallHeat};
use crate::core::Real;
use crate::discretization::gradient::{
    GradientFields, VelocityGradient, compute_structured_gradients_3d,
};
use crate::discretization::residual::{
    accumulate_boundary_face, accumulate_interior_face, is_degenerate_volume,
};
use crate::discretization::viscous::{ViscousFlux, face_transport_coefficients, viscous_face_flux};
use crate::discretization::wall_thermal::wall_heat_flux_into_fluid;
use crate::discretization::{BoundaryGhostBuffer, InviscidFlux};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh, BoundaryMesh3d, LogicalFace3d, StructuredMesh3d};
use crate::physics::{IdealGasEoS, PrimitiveState, ViscousPhysicsConfig};

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
    let temperatures = cell_temperatures(params.primitives, params.eos, params.viscous)?;
    assemble_viscous_i_faces(residual, mesh, params, &temperatures)?;
    assemble_viscous_j_faces(residual, mesh, params, &temperatures)?;
    assemble_viscous_k_faces(residual, mesh, params, &temperatures)?;
    assemble_viscous_boundary_faces(residual, mesh, params, &temperatures)
}

fn assemble_viscous_i_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    temperatures: &[Real],
) -> Result<()> {
    let nx = mesh.nx;
    let ny = mesh.ny;
    let nz = mesh.nz;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx.saturating_sub(1) {
                let owner = mesh.cell_index(i, j, k);
                let neighbor = mesh.cell_index(i + 1, j, k);
                let face = mesh.i_face_metric(i, j, k);
                let flux =
                    viscous_flux_at_cells(params, owner, neighbor, temperatures, face.normal)?;
                accumulate_viscous_interior(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    mesh.cell_metric(i, j, k).volume,
                    mesh.cell_metric(i + 1, j, k).volume,
                )?;
            }
        }
    }
    Ok(())
}

fn assemble_viscous_j_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    temperatures: &[Real],
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
                let flux =
                    viscous_flux_at_cells(params, owner, neighbor, temperatures, face.normal)?;
                accumulate_viscous_interior(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    mesh.cell_metric(i, j, k).volume,
                    mesh.cell_metric(i, j + 1, k).volume,
                )?;
            }
        }
    }
    Ok(())
}

fn assemble_viscous_k_faces(
    residual: &mut ConservedResidual,
    mesh: &StructuredMesh3d,
    params: &ViscousAssembly3dParams<'_>,
    temperatures: &[Real],
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
                let flux =
                    viscous_flux_at_cells(params, owner, neighbor, temperatures, face.normal)?;
                accumulate_viscous_interior(
                    residual,
                    owner,
                    neighbor,
                    &flux,
                    face.area,
                    mesh.cell_metric(i, j, k).volume,
                    mesh.cell_metric(i, j, k + 1).volume,
                )?;
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
            let ghost_prim = crate::field::primitive_from_conserved_relaxed(
                params.eos,
                &ghost.conserved,
                params.min_pressure,
            )?;
            let flux = viscous_flux_at_boundary(
                params,
                ViscousBoundaryFluxInput {
                    owner,
                    ghost_prim,
                    normal: geom.normal,
                    spacing: geom.spacing,
                    wall_heat,
                    no_slip,
                    is_wall,
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

fn viscous_flux_at_cells(
    params: &ViscousAssembly3dParams<'_>,
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
    wall_heat: Option<WallHeat>,
    no_slip: bool,
    is_wall: bool,
}

fn viscous_flux_at_boundary(
    params: &ViscousAssembly3dParams<'_>,
    input: ViscousBoundaryFluxInput,
    temperatures: &[Real],
) -> Result<ViscousFlux> {
    let ViscousBoundaryFluxInput {
        owner,
        ghost_prim,
        normal,
        spacing,
        wall_heat,
        no_slip,
        is_wall,
    } = input;
    let prim_o = primitive_at(params.primitives, temperatures, owner);
    let t_ghost = params.viscous.static_temperature(
        ghost_prim.pressure,
        ghost_prim.density.max(1.0e-30),
        params.eos,
    );
    let mut ghost = ghost_prim;
    ghost.temperature = t_ghost;
    let grad_o = params.gradients.velocity_grad_at(owner);
    let grad_g = if is_wall {
        wall_extrapolated_gradient(&grad_o, &prim_o, &ghost, normal, spacing)
    } else {
        grad_o
    };
    let (mu, lambda) =
        face_transport_coefficients(temperatures[owner], t_ghost, params.viscous, params.eos)?;
    let mut flux = viscous_face_flux(&prim_o, &grad_o, &ghost, &grad_g, normal, mu, lambda);
    if no_slip {
        // 无滑移壁 u=0：u·(τ·n)=0，能量通量仅剩热传导
        let grad = crate::discretization::viscous::average_gradient_for_wall(&grad_o, &grad_g);
        flux.energy =
            lambda * (grad.dt[0] * normal.x + grad.dt[1] * normal.y + grad.dt[2] * normal.z);
    }
    if let Some(heat) = wall_heat {
        let q_into =
            wall_heat_flux_into_fluid(prim_o.temperature, ghost.temperature, spacing, lambda, heat);
        // 壁面粘性通量已按 `accumulate_viscous_boundary` 的符号约定传入；
        // `q_into > 0` 表示壁面向 owner 单元加热。
        flux.energy = q_into;
    }
    Ok(flux)
}

/// 壁面 ghost：法向分量用 \((\phi_g-\phi_o)/(2\delta)\)，切向保留单元差分梯度。
fn wall_extrapolated_gradient(
    grad_cell: &VelocityGradient,
    prim_owner: &PrimitiveState,
    prim_ghost: &PrimitiveState,
    normal: crate::core::Vector3,
    spacing: Real,
) -> VelocityGradient {
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

fn cell_temperatures(
    primitives: &PrimitiveFields,
    eos: &IdealGasEoS,
    viscous: &ViscousPhysicsConfig,
) -> Result<Vec<Real>> {
    let n = primitives.num_cells();
    let mut t = vec![0.0; n];
    for (i, ti) in t.iter_mut().enumerate().take(n) {
        let rho = primitives.density.values()[i];
        let p = primitives.pressure.values()[i];
        *ti = viscous.static_temperature(p, rho, eos);
    }
    Ok(t)
}

fn viscous_flux_for_accumulation(flux: &ViscousFlux) -> InviscidFlux {
    // NS 动量式右端为 +∇·τ，而 FVM 装配为 dU/dt = -1/V Σ F·A（见 inviscid_flux.md §3）。
    // 能量：viscous_face_flux 中 work 已取负；热传导项保持原约定。
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
    use crate::boundary::{BoundaryKind, BoundaryPatch};
    use crate::discretization::freestream_pair::{FreestreamPairFixture, uniform_farfield_box};
    use crate::discretization::{BoundaryGhostBuffer, apply_compressible_boundary_conditions};
    use crate::field::{ConservedFields, PrimitiveFields};
    use crate::mesh::StructuredMesh3d;
    use crate::physics::{FreestreamContext, FreestreamParams, ViscousPhysicsConfig};

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
        use crate::physics::ViscosityModel;

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
        use crate::physics::ViscosityModel;

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
        use crate::physics::ViscosityModel;

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
