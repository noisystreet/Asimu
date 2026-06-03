//! 3D LU-SGS 实验性扫掠：前向 (+i,+j,+k) 与后向 (−i,−j,−k) 单元耦合扫掠。
//!
//! 当前实现用标量谱半径近似块 Jacobian 邻居耦合项；强激波算例应优先使用默认对角模式。
//! **不含**面通量增量步——残差 R 已包含全部面通量贡献；face sweep 会重复计入导致发散。

#![allow(clippy::too_many_arguments)]

use tracing::info_span;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::StructuredMesh3d;
use crate::physics::IdealGasEoS;

use crate::solver::spectral_radius::face_spectral_radius;

struct LuSgsSweepScalars<'a> {
    dt: &'a [Real],
    sigma: &'a [Real],
    volumes: &'a [Real],
    omega: Real,
    gamma: Real,
}

/// LU-SGS 扫掠参数。
pub struct LuSgsSweep3dParams<'a> {
    pub mesh: &'a StructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub primitives: &'a mut PrimitiveFields,
    pub min_pressure: Real,
}

/// 实验性 LU-SGS 双扫：前向 (+i,+j,+k) 与后向 (−i,−j,−k)。
///
/// 前扫：ΔU_i = ω·Δt/(1+Δt·σ_i) · (R_i − Σ A·λ/V_i · ΔU_j)，j 为已访问邻居
/// 后扫：ΔU_i = −ω·Δt/(1+Δt·σ_i) · Σ A·λ/V_i · ΔU_j，j 为未访问邻居
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
    }
    {
        let _span = info_span!("lu_sgs_sweep_backward").entered();
        backward_cell_coupling_sweep(fields, &u0, params, &scalars)?;
    }
    Ok(())
}

/// 与对角 LU-SGS 一致：\(\Delta\mathbf{U}=\omega\,\Delta t\,\mathbf{R}/(1+\Delta t\,\sigma)\)，\(\sigma\) 为 face-sum 谱半径。
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
                        mesh.i_face_metric(i - 1, j, k).area,
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
                        mesh.j_face_metric(i, j - 1, k).area,
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
                        mesh.k_face_metric(i, j, k - 1).area,
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
                        mesh.i_face_metric(i, j, k).area,
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
                        mesh.j_face_metric(i, j, k).area,
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
                        mesh.k_face_metric(i, j, k).area,
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
    area: Real,
    volume: Real,
    fields: &ConservedFields,
    u0: &ConservedFields,
) {
    let coef = area * lambda / volume;
    let cur = conserved_vector(fields, neighbor);
    let old = conserved_vector(u0, neighbor);
    for (s, (&c, &o)) in source.iter_mut().zip(cur.iter().zip(old.iter())) {
        *s -= coef * (c - o);
    }
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
