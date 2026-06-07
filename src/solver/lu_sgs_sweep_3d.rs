//! 3D LU-SGS 双扫：前向 (+i,+j,+k) 与后向 (−i,−j,−k) 单元耦合扫掠。
//!
//! 标量谱半径近似邻居耦合；逐单元正性限制、后扫阻尼与全场线搜索用于强激波稳定化。
//! **不含**面通量增量步——残差 R 已包含全部面通量贡献；face sweep 会重复计入导致发散。

#![allow(clippy::too_many_arguments)]

use tracing::info_span;

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::StructuredMesh3d;
use crate::physics::IdealGasEoS;

use crate::solver::lu_sgs_common::{
    LuSgsSweepScalars, apply_limited_cell_increment, conserved_vector, implicit_scale,
    refresh_primitive_at_cell, residual_cell_vector, scale_source, stabilize_sweep_update,
};
use crate::solver::spectral_radius::face_spectral_radius;

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
                let scale = implicit_scale(scalars.dt[idx], scalars.sigma[idx], scalars.omega);
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
                let scale = implicit_scale(scalars.dt[idx], scalars.sigma[idx], scalars.omega);
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
    refresh_primitive_at_cell(
        fields,
        cell,
        params.eos,
        params.min_pressure,
        params.primitives,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ConservedResidual;
    use crate::physics::{ConservedState, FreestreamParams};
    use crate::solver::lu_sgs_common::{
        LuSgsSweepScalars, fields_are_physical, stabilize_sweep_update,
    };

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
