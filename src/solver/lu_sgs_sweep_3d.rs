//! 3D LU-SGS 双扫：前向 (+i,+j,+k) 与后向 (−i,−j,−k) 单元耦合扫掠。
//!
//! 标量谱半径近似邻居耦合；逐单元正性限制、后扫阻尼与全场线搜索用于强激波稳定化。
//! **不含**面通量增量步——残差 R 已包含全部面通量贡献；face sweep 会重复计入导致发散。

#![allow(clippy::too_many_arguments)]

use tracing::info_span;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFields, ConservedResidual, PrimitiveFields, is_physical_conserved,
    max_physical_increment_scale, state_after_increment,
};
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
    /// 后扫邻居耦合阻尼 \(\in(0,1]\)。
    pub backward_damping: Real,
}

/// LU-SGS 双扫：前向 (+i,+j,+k) 与后向 (−i,−j,−k)，含稳定化。
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
    let u_sweep = fields.clone();
    stabilize_sweep_update(
        fields,
        &u0,
        &u_sweep,
        residual,
        params.min_pressure,
        params.eos.gamma,
        &scalars,
    )?;
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
                apply_limited_cell_increment(
                    fields,
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
                    let damped = scale_source(source, params.backward_damping);
                    apply_limited_cell_increment(
                        fields,
                        idx,
                        scale,
                        damped,
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

fn scale_source(source: [Real; 5], factor: Real) -> [Real; 5] {
    [
        source[0] * factor,
        source[1] * factor,
        source[2] * factor,
        source[3] * factor,
        source[4] * factor,
    ]
}

fn apply_limited_cell_increment(
    fields: &mut ConservedFields,
    cell: usize,
    scale: Real,
    increment: [Real; 5],
    gamma: Real,
    min_pressure: Real,
) -> Result<()> {
    let base = fields.cell_state(cell)?;
    let effective = max_physical_increment_scale(&base, increment, scale, gamma, min_pressure);
    if effective <= 0.0 {
        return Ok(());
    }
    let updated = state_after_increment(&base, increment, effective);
    write_cell_state(fields, cell, &updated);
    Ok(())
}

fn write_cell_state(
    fields: &mut ConservedFields,
    cell: usize,
    state: &crate::physics::ConservedState,
) {
    fields.density.values_mut()[cell] = state.density;
    fields.momentum_x.values_mut()[cell] = state.momentum[0];
    fields.momentum_y.values_mut()[cell] = state.momentum[1];
    fields.momentum_z.values_mut()[cell] = state.momentum[2];
    fields.total_energy.values_mut()[cell] = state.total_energy;
}

fn fields_are_physical(fields: &ConservedFields, gamma: Real, min_pressure: Real) -> Result<bool> {
    for cell in 0..fields.num_cells() {
        let state = fields.cell_state(cell)?;
        if !is_physical_conserved(&state, gamma, min_pressure) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn blend_fields(
    out: &mut ConservedFields,
    base: &ConservedFields,
    target: &ConservedFields,
    alpha: Real,
) -> Result<()> {
    for cell in 0..base.num_cells() {
        let b = base.cell_state(cell)?;
        let t = target.cell_state(cell)?;
        let delta = [
            t.density - b.density,
            t.momentum[0] - b.momentum[0],
            t.momentum[1] - b.momentum[1],
            t.momentum[2] - b.momentum[2],
            t.total_energy - b.total_energy,
        ];
        write_cell_state(out, cell, &state_after_increment(&b, delta, alpha));
    }
    Ok(())
}

fn stabilize_sweep_update(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    u_sweep: &ConservedFields,
    residual: &ConservedResidual,
    min_pressure: Real,
    gamma: Real,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    if fields_are_physical(u_sweep, gamma, min_pressure)? {
        return Ok(());
    }
    const MIN_ALPHA: Real = 1.0 / 1024.0;
    let mut alpha = 1.0;
    loop {
        blend_fields(fields, u0, u_sweep, alpha)?;
        if fields_are_physical(fields, gamma, min_pressure)? {
            return Ok(());
        }
        alpha *= 0.5;
        if alpha < MIN_ALPHA {
            apply_diagonal_fallback(fields, u0, residual, gamma, min_pressure, scalars)?;
            return Ok(());
        }
    }
}

fn apply_diagonal_fallback(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    residual: &ConservedResidual,
    gamma: Real,
    min_pressure: Real,
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for cell in 0..fields.num_cells() {
        let scale = implicit_scale(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.volumes[cell],
            scalars.omega,
        );
        let increment = residual_cell_vector(residual, cell);
        let base = u0.cell_state(cell)?;
        let effective = max_physical_increment_scale(&base, increment, scale, gamma, min_pressure);
        if effective > 0.0 {
            write_cell_state(
                fields,
                cell,
                &state_after_increment(&base, increment, effective),
            );
        } else {
            write_cell_state(fields, cell, &base);
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ConservedResidual;
    use crate::physics::{ConservedState, FreestreamParams};

    #[test]
    fn sweep_keeps_uniform_freestream_physical() {
        let mesh = StructuredMesh3d::uniform_box("box", 3, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let mut fields =
            ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        let min_pressure = 1.0e-6;
        primitives
            .fill_from_conserved(&fields, &eos, min_pressure)
            .expect("prim");
        let residual = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let n = mesh.num_cells();
        let dt = vec![0.01; n];
        let sigma = vec![10.0; n];
        let volumes = mesh.cell_volumes();
        let mut params = LuSgsSweep3dParams {
            mesh: &mesh,
            eos: &eos,
            primitives: &mut primitives,
            min_pressure,
            backward_damping: 0.5,
        };
        lu_sgs_sweep_3d(
            &mut fields,
            &residual,
            &mut params,
            &dt,
            &sigma,
            &volumes,
            1.0,
            eos.gamma,
        )
        .expect("sweep");
        assert!(fields_are_physical(&fields, eos.gamma, params.min_pressure).expect("check"));
    }

    #[test]
    fn line_search_recovers_from_nonphysical_sweep_candidate() {
        let mesh = StructuredMesh3d::uniform_box("box", 2, 1, 1, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let base_state = ConservedState {
            density: 1.0,
            momentum: [1.0, 0.0, 0.0],
            total_energy: 2.5,
        };
        let u0 = ConservedFields::uniform(mesh.num_cells(), base_state).expect("u0");
        let mut bad = u0.clone();
        bad.momentum_x.values_mut()[0] = 100.0;
        let mut fields = u0.clone();
        let residual = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let scalars = LuSgsSweepScalars {
            dt: &[0.01; 2],
            sigma: &[1.0; 2],
            volumes: &[1.0; 2],
            omega: 1.0,
            gamma: eos.gamma,
        };
        stabilize_sweep_update(&mut fields, &u0, &bad, &residual, 0.0, eos.gamma, &scalars)
            .expect("stabilize");
        assert!(fields_are_physical(&fields, eos.gamma, 0.0).expect("check"));
    }
}
