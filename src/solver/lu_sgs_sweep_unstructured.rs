//! 非结构 3D LU-SGS 双扫：按 `CellId` 顺序做前向/后向单元耦合扫掠。
//!
//! 残差已包含完整面通量；扫掠仅使用标量谱半径近似邻居耦合，避免重复计算面通量增量。

use tracing::info_span;

use crate::core::{FaceId, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::field::{
    ConservedFields, ConservedResidual, PrimitiveFields, is_physical_conserved,
    max_physical_increment_scale, state_after_increment,
};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use crate::solver::spectral_radius::face_spectral_radius;

struct LuSgsSweepScalars<'a> {
    dt: &'a [Real],
    sigma: &'a [Real],
    volumes: &'a [Real],
    omega: Real,
    gamma: Real,
}

/// 非结构 LU-SGS 扫掠参数。
pub struct LuSgsSweepUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub primitives: &'a mut PrimitiveFields,
    pub min_pressure: Real,
    pub backward_damping: Real,
}

/// 非结构 LU-SGS sweep 的逐单元时间步与标量参数。
pub struct LuSgsSweepUnstructuredInput<'a> {
    pub dt: &'a [Real],
    pub sigma: &'a [Real],
    pub volumes: &'a [Real],
    pub couplings: &'a LuSgsUnstructuredCouplings,
    pub omega: Real,
    pub gamma: Real,
}

#[derive(Clone, Copy)]
struct CellCoupling {
    neighbor: usize,
    area: Real,
    normal: Vector3,
}

/// 非结构 LU-SGS 拓扑邻接缓存，仅依赖网格 face owner/neighbor。
pub struct LuSgsUnstructuredCouplings {
    cells: Vec<Vec<CellCoupling>>,
}

impl LuSgsUnstructuredCouplings {
    pub fn from_mesh(mesh: &UnstructuredMesh3d) -> Result<Self> {
        build_cell_couplings(mesh).map(|cells| Self { cells })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// 非结构 LU-SGS 双扫，按 CellId 单调顺序定义下/上三角邻接。
pub fn lu_sgs_sweep_unstructured(
    fields: &mut ConservedFields,
    residual: &ConservedResidual,
    params: &mut LuSgsSweepUnstructuredParams<'_>,
    input: LuSgsSweepUnstructuredInput<'_>,
) -> Result<()> {
    let n = fields.num_cells();
    if residual.num_cells() != n
        || input.dt.len() != n
        || input.sigma.len() != n
        || input.volumes.len() != n
        || input.couplings.len() != n
    {
        return Err(AsimuError::Solver(
            "lu_sgs_sweep_unstructured: 场/残差/dt/sigma/volume 长度不一致".to_string(),
        ));
    }
    let u0 = fields.clone();
    let scalars = LuSgsSweepScalars {
        dt: input.dt,
        sigma: input.sigma,
        volumes: input.volumes,
        omega: input.omega,
        gamma: input.gamma,
    };
    {
        let _span = info_span!("lu_sgs_unstructured_forward").entered();
        forward_sweep(
            fields,
            &u0,
            residual,
            params,
            &input.couplings.cells,
            &scalars,
        )?;
    }
    {
        let _span = info_span!("lu_sgs_unstructured_backward").entered();
        backward_sweep(fields, &u0, params, &input.couplings.cells, &scalars)?;
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
    )
}

fn build_cell_couplings(mesh: &UnstructuredMesh3d) -> Result<Vec<Vec<CellCoupling>>> {
    let mut couplings = vec![Vec::new(); mesh.num_cells()];
    for face in 0..mesh.num_faces() {
        let face_id = FaceId(face as u32);
        let Some(neighbor_id) = mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner_id = mesh.face_owner(face_id)?;
        let owner = owner_id.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        let metric = mesh.face_metric(face_id);
        couplings[owner].push(CellCoupling {
            neighbor,
            area: metric.area,
            normal: metric.normal,
        });
        couplings[neighbor].push(CellCoupling {
            neighbor: owner,
            area: metric.area,
            normal: metric.normal,
        });
    }
    Ok(couplings)
}

fn forward_sweep(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    residual: &ConservedResidual,
    params: &mut LuSgsSweepUnstructuredParams<'_>,
    couplings: &[Vec<CellCoupling>],
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()) {
        let scale = implicit_scale(scalars.dt[cell], scalars.sigma[cell], scalars.omega);
        let mut source = residual_cell_vector(residual, cell);
        for coupling in cell_couplings.iter().filter(|c| c.neighbor < cell) {
            add_coupling_delta(
                &mut source,
                cell,
                *coupling,
                scalars.volumes[cell],
                fields,
                u0,
                params,
            );
        }
        apply_limited_cell_increment(
            fields,
            cell,
            scale,
            source,
            scalars.gamma,
            params.min_pressure,
        )?;
        refresh_primitive(params, fields, cell)?;
    }
    Ok(())
}

fn backward_sweep(
    fields: &mut ConservedFields,
    u0: &ConservedFields,
    params: &mut LuSgsSweepUnstructuredParams<'_>,
    couplings: &[Vec<CellCoupling>],
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()).rev() {
        let scale = implicit_scale(scalars.dt[cell], scalars.sigma[cell], scalars.omega);
        let mut source = [0.0; 5];
        for coupling in cell_couplings.iter().filter(|c| c.neighbor > cell) {
            add_coupling_delta(
                &mut source,
                cell,
                *coupling,
                scalars.volumes[cell],
                fields,
                u0,
                params,
            );
        }
        if source.iter().any(|c| c.abs() > Real::EPSILON) {
            let damped = scale_source(source, params.backward_damping);
            apply_limited_cell_increment(
                fields,
                cell,
                scale,
                damped,
                scalars.gamma,
                params.min_pressure,
            )?;
            refresh_primitive(params, fields, cell)?;
        }
    }
    Ok(())
}

fn implicit_scale(dt: Real, sigma: Real, omega: Real) -> Real {
    let denom = 1.0 + dt * sigma;
    if !(dt > 0.0 && omega > 0.0 && denom > 0.0) {
        return 0.0;
    }
    omega * dt / denom
}

fn add_coupling_delta(
    source: &mut [Real; 5],
    cell: usize,
    coupling: CellCoupling,
    volume: Real,
    fields: &ConservedFields,
    u0: &ConservedFields,
    params: &LuSgsSweepUnstructuredParams<'_>,
) {
    let lambda = face_spectral_radius(
        &params.primitives.cell_primitive(cell),
        &params.primitives.cell_primitive(coupling.neighbor),
        coupling.normal,
        params.eos.gamma,
    );
    let coef = coupling.area * lambda / volume.max(1.0e-30);
    let cur = conserved_vector(fields, coupling.neighbor);
    let old = conserved_vector(u0, coupling.neighbor);
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

fn fields_are_physical(fields: &ConservedFields, gamma: Real, min_pressure: Real) -> Result<bool> {
    for cell in 0..fields.num_cells() {
        let state = fields.cell_state(cell)?;
        if !is_physical_conserved(&state, gamma, min_pressure) {
            return Ok(false);
        }
    }
    Ok(true)
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
        let scale = implicit_scale(scalars.dt[cell], scalars.sigma[cell], scalars.omega);
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

fn refresh_primitive(
    params: &mut LuSgsSweepUnstructuredParams<'_>,
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
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::FreestreamParams;

    #[test]
    fn sweep_keeps_uniform_unstructured_freestream_physical() {
        let mesh = UnstructuredMesh3d::new(
            "two_tets",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![
                UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("cell"),
                UnstructuredCell::new(CellKind::Tet, vec![1, 2, 3, 4]).expect("cell"),
            ],
        )
        .expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let mut fields = ConservedFields::from_freestream(
            mesh.num_cells(),
            &eos,
            &FreestreamParams {
                mach: 0.2,
                ..FreestreamParams::default()
            },
        )
        .expect("fields");
        let min_pressure = 1.0e-8;
        let mut primitives = PrimitiveFields::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, min_pressure)
            .expect("fill");
        let residual = ConservedResidual::zeros(mesh.num_cells()).expect("rhs");
        let volumes = mesh.cell_volumes();
        let couplings = LuSgsUnstructuredCouplings::from_mesh(&mesh).expect("couplings");
        let dt = vec![0.01; mesh.num_cells()];
        let sigma = vec![10.0; mesh.num_cells()];
        let mut params = LuSgsSweepUnstructuredParams {
            mesh: &mesh,
            eos: &eos,
            primitives: &mut primitives,
            min_pressure,
            backward_damping: 0.5,
        };
        lu_sgs_sweep_unstructured(
            &mut fields,
            &residual,
            &mut params,
            LuSgsSweepUnstructuredInput {
                dt: &dt,
                sigma: &sigma,
                volumes: &volumes,
                couplings: &couplings,
                omega: 1.0,
                gamma: eos.gamma,
            },
        )
        .expect("sweep");
        assert!(fields_are_physical(&fields, eos.gamma, min_pressure).expect("physical"));
    }
}
