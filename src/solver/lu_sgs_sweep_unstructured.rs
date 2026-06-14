//! 非结构 3D LU-SGS 双扫：按 `CellId` 顺序做前向/后向单元耦合扫掠。
//!
//! 残差已包含完整面通量；扫掠仅使用标量谱半径近似邻居耦合，避免重复计算面通量增量。

use tracing::info_span;

use crate::core::{FaceId, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedResidual, PrimitiveFields};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use crate::solver::lu_sgs_common::{
    LuSgsSweepScalars, apply_limited_cell_increment, conserved_vector, implicit_scale,
    refresh_primitive_at_cell, residual_cell_vector, scale_source, stabilize_sweep_update,
};
use crate::solver::spectral_radius::face_spectral_radius;

/// 非结构 LU-SGS 扫掠参数。
pub struct LuSgsSweepUnstructuredParams<'a> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub primitives: &'a mut PrimitiveFields,
    pub min_pressure: Real,
    pub backward_damping: Real,
}

use crate::discretization::unstructured_face_cache_f32::LuSgsUnstructuredCouplingsF32;

/// 非结构 LU-SGS 拓扑邻接（f64 或 f32 预打包）。
pub enum LuSgsUnstructuredCouplingsRef<'a> {
    F64(&'a LuSgsUnstructuredCouplings),
    F32(&'a LuSgsUnstructuredCouplingsF32),
}

impl<'a> LuSgsUnstructuredCouplingsRef<'a> {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F64(c) => c.len(),
            Self::F32(c) => c.len(),
        }
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 非结构 LU-SGS sweep 的逐单元时间步与标量参数（f32 热路径）。
pub struct LuSgsSweepUnstructuredF32Input<'a> {
    pub dt: &'a [f32],
    pub sigma: &'a [f32],
    pub volumes: &'a [f32],
    pub couplings: LuSgsUnstructuredCouplingsRef<'a>,
    pub omega: f32,
    pub gamma: f32,
}

/// 非结构 LU-SGS sweep 的逐单元时间步与标量参数。
pub struct LuSgsSweepUnstructuredInput<'a> {
    pub dt: &'a [Real],
    pub sigma: &'a [Real],
    pub volumes: &'a [Real],
    pub couplings: LuSgsUnstructuredCouplingsRef<'a>,
    pub omega: Real,
    pub gamma: Real,
}

#[derive(Clone, Copy)]
pub(crate) struct LuSgsCellCoupling {
    pub(crate) neighbor: usize,
    pub(crate) area: Real,
    pub(crate) normal: Vector3,
}

/// 非结构 LU-SGS 拓扑邻接缓存，仅依赖网格 face owner/neighbor。
pub struct LuSgsUnstructuredCouplings {
    cells: Vec<Vec<LuSgsCellCoupling>>,
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

    pub(crate) fn cells(&self) -> &[Vec<LuSgsCellCoupling>] {
        &self.cells
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
    let LuSgsUnstructuredCouplingsRef::F64(couplings) = input.couplings else {
        return Err(AsimuError::Solver(
            "lu_sgs_sweep_unstructured: 仅支持 f64 拓扑耦合".to_string(),
        ));
    };
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
        forward_sweep(fields, &u0, residual, params, couplings.cells(), &scalars)?;
    }
    {
        let _span = info_span!("lu_sgs_unstructured_backward").entered();
        backward_sweep(fields, &u0, params, couplings.cells(), &scalars)?;
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

fn build_cell_couplings(mesh: &UnstructuredMesh3d) -> Result<Vec<Vec<LuSgsCellCoupling>>> {
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
        couplings[owner].push(LuSgsCellCoupling {
            neighbor,
            area: metric.area,
            normal: metric.normal,
        });
        couplings[neighbor].push(LuSgsCellCoupling {
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
    couplings: &[Vec<LuSgsCellCoupling>],
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
    couplings: &[Vec<LuSgsCellCoupling>],
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

fn add_coupling_delta(
    source: &mut [Real; 5],
    cell: usize,
    coupling: LuSgsCellCoupling,
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

fn refresh_primitive(
    params: &mut LuSgsSweepUnstructuredParams<'_>,
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
    use crate::field::ConservedFields;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::FreestreamParams;
    use crate::solver::lu_sgs_common::fields_are_physical;

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
                couplings: LuSgsUnstructuredCouplingsRef::F64(&couplings),
                omega: 1.0,
                gamma: eos.gamma,
            },
        )
        .expect("sweep");
        assert!(fields_are_physical(&fields, eos.gamma, min_pressure).expect("physical"));
    }
}
