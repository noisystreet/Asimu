//! 3D LU-SGS 扫掠（阶段 D）：i/j/k 前扫 + 后扫，面通量增量隐式更新。

#![allow(clippy::too_many_arguments)]

use tracing::info_span;

use crate::boundary::BoundarySet;
use crate::core::Real;
use crate::discretization::residual::{
    InviscidAssembly3dParams, inviscid_boundary_face_flux, inviscid_i_face_flux,
    inviscid_j_face_flux, inviscid_k_face_flux,
};
use crate::discretization::{BoundaryGhostBuffer, InviscidFlux, InviscidFluxConfig};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::{BoundaryMesh3d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

use crate::solver::spectral_radius::face_spectral_radius;

struct LuSgsSweepScalars<'a> {
    dt: &'a [Real],
    sigma: &'a [Real],
    volumes: &'a [Real],
    omega: Real,
    gamma: Real,
}

struct ImplicitFaceCoupling<'a> {
    flux: &'a InviscidFlux,
    area: Real,
    volume: Real,
    dt: Real,
    sigma: Real,
    omega: Real,
    gamma: Real,
    min_pressure: Real,
}

/// LU-SGS 扫掠参数。
pub struct LuSgsSweep3dParams<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub boundary_mesh: &'a dyn BoundaryMesh3d,
    pub eos: &'a IdealGasEoS,
    pub boundaries: &'a BoundarySet,
    pub ghosts: &'a BoundaryGhostBuffer,
    pub primitives: &'a mut PrimitiveFields,
    pub min_pressure: Real,
    pub sweep_config: InviscidFluxConfig,
}

/// 扫掠用一阶通量配置（从算例配置复制，强制一阶重构）。
#[must_use]
pub fn sweep_first_order_config(base: &InviscidFluxConfig) -> InviscidFluxConfig {
    let mut cfg = *base;
    cfg.reconstruction = crate::discretization::ReconstructionKind::FirstOrder;
    cfg
}

/// 真 LU-SGS 双扫：前向 (+i,+j,+k) 与后向 (−i,−j,−k)。
pub fn lu_sgs_sweep_3d(
    fields: &mut ConservedFields,
    residual: &ConservedResidual,
    params: &mut LuSgsSweep3dParams<'_>,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let n = fields.num_cells();
    if residual.num_cells() != n || dt.len() != n || sigma.len() != n || volumes.len() != n {
        return Err(AsimuError::Solver(
            "lu_sgs_sweep_3d: 场/残差/dt/sigma/volume 长度不一致".to_string(),
        ));
    }
    let u0 = fields.clone();
    let sweep_config = params.sweep_config;
    let scalars = LuSgsSweepScalars {
        dt,
        sigma,
        volumes,
        omega,
        gamma,
    };
    {
        let _span = info_span!("lu_sgs_sweep_forward").entered();
        forward_cell_coupling_sweep(fields, &u0, residual, params, &scalars)?;
        sweep_i_faces_forward(
            fields,
            params,
            &sweep_config,
            dt,
            sigma,
            volumes,
            omega,
            gamma,
        )?;
        sweep_j_faces_forward(
            fields,
            params,
            &sweep_config,
            dt,
            sigma,
            volumes,
            omega,
            gamma,
        )?;
        sweep_k_faces_forward(
            fields,
            params,
            &sweep_config,
            dt,
            sigma,
            volumes,
            omega,
            gamma,
        )?;
        sweep_boundary_faces_forward(fields, params, &sweep_config, &scalars)?;
    }
    {
        let _span = info_span!("lu_sgs_sweep_backward").entered();
        backward_cell_coupling_sweep(fields, &u0, params, &scalars)?;
        sweep_i_faces_backward(
            fields,
            params,
            &sweep_config,
            dt,
            sigma,
            volumes,
            omega,
            gamma,
        )?;
        sweep_j_faces_backward(
            fields,
            params,
            &sweep_config,
            dt,
            sigma,
            volumes,
            omega,
            gamma,
        )?;
        sweep_k_faces_backward(
            fields,
            params,
            &sweep_config,
            dt,
            sigma,
            volumes,
            omega,
            gamma,
        )?;
        sweep_boundary_faces_backward(fields, params, &sweep_config, &scalars)?;
    }
    Ok(())
}

/// 与对角 LU-SGS 一致：\(\Delta\mathbf{U}=\omega\,\Delta t\,\mathbf{R}/(1+\Delta t\,\sigma)\)，\(\sigma=(|u|+a)/h\)。
fn implicit_scale(dt: Real, sigma: Real, _volume: Real, omega: Real) -> Real {
    let denom = 1.0 + dt * sigma;
    if !(dt > 0.0 && omega > 0.0 && denom > 0.0) {
        return 0.0;
    }
    omega * dt / denom
}

fn forward_cell_coupling_sweep(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    residual: &ConservedResidual,
    params: &mut LuSgsSweep3dParams<'_>,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let idx = mesh.cell_index(i, j, k);
                let scale = implicit_scale(
                    scalars.dt[idx],
                    scalars.sigma[idx],
                    scalars.volumes[idx],
                    scalars.omega,
                );
                let mut source = residual_cell_vector(residual, idx);
                if i > 0 {
                    add_coupling_delta(
                        &mut source,
                        mesh.cell_index(i - 1, j, k),
                        i_face_lambda(mesh, params.primitives, params.eos.gamma, i - 1, j, k)?,
                        scalars.dt[idx],
                        scalars.volumes[idx],
                        fields,
                        u0,
                    );
                }
                if j > 0 {
                    add_coupling_delta(
                        &mut source,
                        mesh.cell_index(i, j - 1, k),
                        j_face_lambda(mesh, params.primitives, params.eos.gamma, i, j - 1, k)?,
                        scalars.dt[idx],
                        scalars.volumes[idx],
                        fields,
                        u0,
                    );
                }
                if k > 0 {
                    add_coupling_delta(
                        &mut source,
                        mesh.cell_index(i, j, k - 1),
                        k_face_lambda(mesh, params.primitives, params.eos.gamma, i, j, k - 1)?,
                        scalars.dt[idx],
                        scalars.volumes[idx],
                        fields,
                        u0,
                    );
                }
                fields.add_conserved_increment(
                    idx,
                    scale,
                    source,
                    scalars.gamma,
                    params.min_pressure,
                )?;
                refresh_primitive(params, fields, idx)?;
            }
        }
    }
    Ok(())
}

fn backward_cell_coupling_sweep(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    let mesh = params.mesh;
    for k in (0..mesh.nz).rev() {
        for j in (0..mesh.ny).rev() {
            for i in (0..mesh.nx).rev() {
                let idx = mesh.cell_index(i, j, k);
                let scale = implicit_scale(
                    scalars.dt[idx],
                    scalars.sigma[idx],
                    scalars.volumes[idx],
                    scalars.omega,
                );
                let mut source = [0.0; 5];
                if i + 1 < mesh.nx {
                    add_coupling_delta(
                        &mut source,
                        mesh.cell_index(i + 1, j, k),
                        i_face_lambda(mesh, params.primitives, params.eos.gamma, i, j, k)?,
                        scalars.dt[idx],
                        scalars.volumes[idx],
                        fields,
                        u0,
                    );
                }
                if j + 1 < mesh.ny {
                    add_coupling_delta(
                        &mut source,
                        mesh.cell_index(i, j + 1, k),
                        j_face_lambda(mesh, params.primitives, params.eos.gamma, i, j, k)?,
                        scalars.dt[idx],
                        scalars.volumes[idx],
                        fields,
                        u0,
                    );
                }
                if k + 1 < mesh.nz {
                    add_coupling_delta(
                        &mut source,
                        mesh.cell_index(i, j, k + 1),
                        k_face_lambda(mesh, params.primitives, params.eos.gamma, i, j, k)?,
                        scalars.dt[idx],
                        scalars.volumes[idx],
                        fields,
                        u0,
                    );
                }
                if source.iter().any(|c| c.abs() > Real::EPSILON) {
                    fields.add_conserved_increment(
                        idx,
                        scale,
                        source,
                        scalars.gamma,
                        params.min_pressure,
                    )?;
                    refresh_primitive(params, fields, idx)?;
                }
            }
        }
    }
    Ok(())
}

fn add_coupling_delta(
    source: &mut [Real; 5],
    neighbor: usize,
    lambda: Real,
    dt: Real,
    volume: Real,
    fields: &ConservedFields,
    u0: &ConservedFields,
) {
    let coef = dt * lambda / volume;
    let cur = conserved_vector(fields, neighbor);
    let old = conserved_vector(u0, neighbor);
    for (s, (&c, &o)) in source.iter_mut().zip(cur.iter().zip(old.iter())) {
        *s += coef * (c - o);
    }
}

fn sweep_i_faces_forward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let mesh = params.mesh;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx.saturating_sub(1) {
                let neighbor = mesh.cell_index(i + 1, j, k);
                let area = mesh.i_face_metric(i, j, k).area;
                let flux = {
                    let assembly = face_assembly(params, sweep_config);
                    inviscid_i_face_flux(&assembly, i, j, k)?
                };
                apply_face_implicit(
                    fields,
                    neighbor,
                    ImplicitFaceCoupling {
                        flux: &flux,
                        area,
                        volume: volumes[neighbor],
                        dt: dt[neighbor],
                        sigma: sigma[neighbor],
                        omega,
                        gamma,
                        min_pressure: params.min_pressure,
                    },
                    1.0,
                )?;
                refresh_primitive(params, fields, neighbor)?;
            }
        }
    }
    Ok(())
}

fn sweep_i_faces_backward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let mesh = params.mesh;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in (0..mesh.nx.saturating_sub(1)).rev() {
                let owner = mesh.cell_index(i, j, k);
                let area = mesh.i_face_metric(i, j, k).area;
                let flux = {
                    let assembly = face_assembly(params, sweep_config);
                    inviscid_i_face_flux(&assembly, i, j, k)?
                };
                apply_face_implicit(
                    fields,
                    owner,
                    ImplicitFaceCoupling {
                        flux: &flux,
                        area,
                        volume: volumes[owner],
                        dt: dt[owner],
                        sigma: sigma[owner],
                        omega,
                        gamma,
                        min_pressure: params.min_pressure,
                    },
                    -1.0,
                )?;
                refresh_primitive(params, fields, owner)?;
            }
        }
    }
    Ok(())
}

fn sweep_j_faces_forward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let mesh = params.mesh;
    for k in 0..mesh.nz {
        for j in 0..mesh.ny.saturating_sub(1) {
            for i in 0..mesh.nx {
                let neighbor = mesh.cell_index(i, j + 1, k);
                let area = mesh.j_face_metric(i, j, k).area;
                let flux = {
                    let assembly = face_assembly(params, sweep_config);
                    inviscid_j_face_flux(&assembly, i, j, k)?
                };
                apply_face_implicit(
                    fields,
                    neighbor,
                    ImplicitFaceCoupling {
                        flux: &flux,
                        area,
                        volume: volumes[neighbor],
                        dt: dt[neighbor],
                        sigma: sigma[neighbor],
                        omega,
                        gamma,
                        min_pressure: params.min_pressure,
                    },
                    1.0,
                )?;
                refresh_primitive(params, fields, neighbor)?;
            }
        }
    }
    Ok(())
}

fn sweep_j_faces_backward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let mesh = params.mesh;
    for k in 0..mesh.nz {
        for j in (0..mesh.ny.saturating_sub(1)).rev() {
            for i in 0..mesh.nx {
                let owner = mesh.cell_index(i, j, k);
                let area = mesh.j_face_metric(i, j, k).area;
                let flux = {
                    let assembly = face_assembly(params, sweep_config);
                    inviscid_j_face_flux(&assembly, i, j, k)?
                };
                apply_face_implicit(
                    fields,
                    owner,
                    ImplicitFaceCoupling {
                        flux: &flux,
                        area,
                        volume: volumes[owner],
                        dt: dt[owner],
                        sigma: sigma[owner],
                        omega,
                        gamma,
                        min_pressure: params.min_pressure,
                    },
                    -1.0,
                )?;
                refresh_primitive(params, fields, owner)?;
            }
        }
    }
    Ok(())
}

fn sweep_k_faces_forward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let mesh = params.mesh;
    for k in 0..mesh.nz.saturating_sub(1) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let neighbor = mesh.cell_index(i, j, k + 1);
                let area = mesh.k_face_metric(i, j, k).area;
                let flux = {
                    let assembly = face_assembly(params, sweep_config);
                    inviscid_k_face_flux(&assembly, i, j, k)?
                };
                apply_face_implicit(
                    fields,
                    neighbor,
                    ImplicitFaceCoupling {
                        flux: &flux,
                        area,
                        volume: volumes[neighbor],
                        dt: dt[neighbor],
                        sigma: sigma[neighbor],
                        omega,
                        gamma,
                        min_pressure: params.min_pressure,
                    },
                    1.0,
                )?;
                refresh_primitive(params, fields, neighbor)?;
            }
        }
    }
    Ok(())
}

fn sweep_k_faces_backward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    dt: &[Real],
    sigma: &[Real],
    volumes: &[Real],
    omega: Real,
    gamma: Real,
) -> Result<()> {
    let mesh = params.mesh;
    for k in (0..mesh.nz.saturating_sub(1)).rev() {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let owner = mesh.cell_index(i, j, k);
                let area = mesh.k_face_metric(i, j, k).area;
                let flux = {
                    let assembly = face_assembly(params, sweep_config);
                    inviscid_k_face_flux(&assembly, i, j, k)?
                };
                apply_face_implicit(
                    fields,
                    owner,
                    ImplicitFaceCoupling {
                        flux: &flux,
                        area,
                        volume: volumes[owner],
                        dt: dt[owner],
                        sigma: sigma[owner],
                        omega,
                        gamma,
                        min_pressure: params.min_pressure,
                    },
                    -1.0,
                )?;
                refresh_primitive(params, fields, owner)?;
            }
        }
    }
    Ok(())
}

fn face_assembly<'a>(
    params: &'a LuSgsSweep3dParams<'a>,
    sweep_config: &'a InviscidFluxConfig,
) -> InviscidAssembly3dParams<'a> {
    InviscidAssembly3dParams {
        mesh: params.mesh,
        eos: params.eos,
        config: sweep_config,
        boundaries: params.boundaries,
        ghosts: params.ghosts,
        primitives: params.primitives,
        min_pressure: params.min_pressure,
    }
}

fn sweep_boundary_faces_forward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for patch in params.boundaries.patches() {
        for &face in &patch.face_ids {
            apply_boundary_face(fields, params, sweep_config, face, scalars)?;
        }
    }
    Ok(())
}

fn sweep_boundary_faces_backward(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    let mut faces: Vec<_> = params
        .boundaries
        .patches()
        .iter()
        .flat_map(|p| p.face_ids.iter().copied())
        .collect();
    faces.sort_by_key(|f| f.index());
    faces.reverse();
    for face in faces {
        apply_boundary_face(fields, params, sweep_config, face, scalars)?;
    }
    Ok(())
}

fn apply_boundary_face(
    fields: &mut ConservedFields,
    params: &mut LuSgsSweep3dParams<'_>,
    sweep_config: &InviscidFluxConfig,
    face: crate::core::FaceId,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    let owner = params.boundary_mesh.face_owner(face)?.index() as usize;
    let ghost = params.ghosts.get_face(face).ok_or_else(|| {
        AsimuError::Boundary(format!("边界面 FaceId({}) 缺少 ghost", face.index()))
    })?;
    let flux = inviscid_boundary_face_flux(
        params.boundary_mesh,
        params.mesh,
        params.primitives,
        params.eos,
        sweep_config,
        params.min_pressure,
        face,
        ghost.conserved,
    )?;
    let area = params.boundary_mesh.face_geometry_3d(face)?.area;
    apply_face_implicit(
        fields,
        owner,
        ImplicitFaceCoupling {
            flux: &flux,
            area,
            volume: scalars.volumes[owner],
            dt: scalars.dt[owner],
            sigma: scalars.sigma[owner],
            omega: scalars.omega,
            gamma: scalars.gamma,
            min_pressure: params.min_pressure,
        },
        -1.0,
    )?;
    refresh_primitive(params, fields, owner)
}

fn apply_face_implicit(
    fields: &mut ConservedFields,
    cell: usize,
    coupling: ImplicitFaceCoupling<'_>,
    sign: Real,
) -> Result<()> {
    let ImplicitFaceCoupling {
        flux,
        area,
        volume,
        dt,
        sigma,
        omega,
        gamma,
        min_pressure,
    } = coupling;
    let scale = sign * implicit_scale(dt, sigma, volume, omega) * area / volume;
    let inc = [
        scale * flux.mass,
        scale * flux.momentum[0],
        scale * flux.momentum[1],
        scale * flux.momentum[2],
        scale * flux.energy,
    ];
    fields.add_conserved_increment(cell, 1.0, inc, gamma, min_pressure)
}

fn residual_cell_vector(residual: &ConservedResidual, cell: usize) -> [Real; 5] {
    [
        residual.density.values()[cell],
        residual.momentum_x.values()[cell],
        residual.momentum_y.values()[cell],
        residual.momentum_z.values()[cell],
        residual.total_energy.values()[cell],
    ]
}

fn conserved_vector(fields: &ConservedFields, cell: usize) -> [Real; 5] {
    [
        fields.density.values()[cell],
        fields.momentum_x.values()[cell],
        fields.momentum_y.values()[cell],
        fields.momentum_z.values()[cell],
        fields.total_energy.values()[cell],
    ]
}

fn i_face_lambda(
    mesh: &StructuredMesh3d,
    prim: &PrimitiveFields,
    gamma: Real,
    i: usize,
    j: usize,
    k: usize,
) -> Result<Real> {
    let owner = mesh.cell_index(i, j, k);
    let neighbor = mesh.cell_index(i + 1, j, k);
    Ok(face_spectral_radius(
        &prim.cell_primitive(owner),
        &prim.cell_primitive(neighbor),
        mesh.i_face_metric(i, j, k).normal,
        gamma,
    ))
}

fn j_face_lambda(
    mesh: &StructuredMesh3d,
    prim: &PrimitiveFields,
    gamma: Real,
    i: usize,
    j: usize,
    k: usize,
) -> Result<Real> {
    let owner = mesh.cell_index(i, j, k);
    let neighbor = mesh.cell_index(i, j + 1, k);
    Ok(face_spectral_radius(
        &prim.cell_primitive(owner),
        &prim.cell_primitive(neighbor),
        mesh.j_face_metric(i, j, k).normal,
        gamma,
    ))
}

fn k_face_lambda(
    mesh: &StructuredMesh3d,
    prim: &PrimitiveFields,
    gamma: Real,
    i: usize,
    j: usize,
    k: usize,
) -> Result<Real> {
    let owner = mesh.cell_index(i, j, k);
    let neighbor = mesh.cell_index(i, j, k + 1);
    Ok(face_spectral_radius(
        &prim.cell_primitive(owner),
        &prim.cell_primitive(neighbor),
        mesh.k_face_metric(i, j, k).normal,
        gamma,
    ))
}

fn refresh_primitive(
    params: &mut LuSgsSweep3dParams<'_>,
    fields: &ConservedFields,
    cell: usize,
) -> Result<()> {
    let cons = fields.cell_state(cell)?;
    let prim =
        crate::field::primitive_from_conserved_relaxed(params.eos, &cons, params.min_pressure)?;
    params.primitives.density.values_mut()[cell] = prim.density;
    params.primitives.pressure.values_mut()[cell] = prim.pressure;
    params.primitives.velocity_x.values_mut()[cell] = prim.velocity[0];
    params.primitives.velocity_y.values_mut()[cell] = prim.velocity[1];
    params.primitives.velocity_z.values_mut()[cell] = prim.velocity[2];
    Ok(())
}
