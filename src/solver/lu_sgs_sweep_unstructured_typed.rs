//! 非结构 LU-SGS 双扫 typed 路径（f32 面耦合谱半径原生计算）。

use tracing::info_span;

use crate::core::{ComputeFloat, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use crate::solver::lu_sgs_common::{
    LuSgsSweepScalars, apply_limited_cell_increment_typed, conserved_vector_typed, implicit_scale,
    refresh_primitive_at_cell_typed, residual_cell_vector_typed, scale_source,
    stabilize_sweep_update_typed,
};
use crate::solver::lu_sgs_sweep_unstructured::{LuSgsCellCoupling, LuSgsSweepUnstructuredInput};
use crate::solver::spectral_radius::face_spectral_radius;
use crate::solver::spectral_radius_f32::{FacePrimitiveLaneF32, face_spectral_radius_f32};

/// typed 非结构 LU-SGS 扫掠参数。
pub struct LuSgsSweepUnstructuredTypedParams<'a, T: ComputeFloat> {
    pub mesh: &'a UnstructuredMesh3d,
    pub eos: &'a IdealGasEoS,
    pub primitives: &'a mut PrimitiveFieldsT<T>,
    pub min_pressure: Real,
    pub backward_damping: Real,
}

/// 面耦合谱半径：f32 走原生 `face_spectral_radius_f32`，f64 与既有路径一致。
pub trait LuSgsUnstructuredSweepTyped: ComputeFloat {
    fn face_coupling_lambda(
        primitives: &PrimitiveFieldsT<Self>,
        cell: usize,
        neighbor: usize,
        normal: Vector3,
        gamma: Real,
    ) -> Real;
}

impl LuSgsUnstructuredSweepTyped for f32 {
    fn face_coupling_lambda(
        primitives: &PrimitiveFieldsT<f32>,
        cell: usize,
        neighbor: usize,
        normal: Vector3,
        gamma: Real,
    ) -> Real {
        let gamma_f32 = gamma as f32;
        face_spectral_radius_f32(
            prim_lane_f32(primitives, cell),
            prim_lane_f32(primitives, neighbor),
            [normal.x as f32, normal.y as f32, normal.z as f32],
            gamma_f32,
        )
        .to_real()
    }
}

impl LuSgsUnstructuredSweepTyped for f64 {
    fn face_coupling_lambda(
        primitives: &PrimitiveFieldsT<f64>,
        cell: usize,
        neighbor: usize,
        normal: Vector3,
        gamma: Real,
    ) -> Real {
        face_spectral_radius(
            &primitives.cell_primitive(cell),
            &primitives.cell_primitive(neighbor),
            normal,
            gamma,
        )
    }
}

/// typed 非结构 LU-SGS 双扫。
pub fn lu_sgs_sweep_unstructured_typed<T: LuSgsUnstructuredSweepTyped>(
    fields: &mut ConservedFieldsT<T>,
    residual: &ConservedResidualT<T>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, T>,
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
            "lu_sgs_sweep_unstructured_typed: 场/残差/dt/sigma/volume 长度不一致".to_string(),
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
        let _span = info_span!(
            "lu_sgs_unstructured_forward_typed",
            precision = T::PRECISION.label()
        )
        .entered();
        forward_sweep_typed(
            fields,
            &u0,
            residual,
            params,
            input.couplings.cells(),
            &scalars,
        )?;
    }
    {
        let _span = info_span!(
            "lu_sgs_unstructured_backward_typed",
            precision = T::PRECISION.label()
        )
        .entered();
        backward_sweep_typed(fields, &u0, params, input.couplings.cells(), &scalars)?;
    }
    let u_sweep = fields.clone();
    stabilize_sweep_update_typed(
        fields,
        &u0,
        &u_sweep,
        residual,
        params.min_pressure,
        params.eos.gamma,
        &scalars,
    )
}

fn forward_sweep_typed<T: LuSgsUnstructuredSweepTyped>(
    fields: &mut ConservedFieldsT<T>,
    u0: &ConservedFieldsT<T>,
    residual: &ConservedResidualT<T>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, T>,
    couplings: &[Vec<LuSgsCellCoupling>],
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()) {
        let scale = implicit_scale(scalars.dt[cell], scalars.sigma[cell], scalars.omega);
        let mut source = residual_cell_vector_typed(residual, cell);
        for coupling in cell_couplings.iter().filter(|c| c.neighbor < cell) {
            add_coupling_delta_typed(
                &mut source,
                cell,
                *coupling,
                scalars.volumes[cell],
                fields,
                u0,
                params,
            );
        }
        apply_limited_cell_increment_typed(
            fields,
            cell,
            scale,
            source,
            scalars.gamma,
            params.min_pressure,
        )?;
        refresh_primitive_typed(params, fields, cell)?;
    }
    Ok(())
}

fn backward_sweep_typed<T: LuSgsUnstructuredSweepTyped>(
    fields: &mut ConservedFieldsT<T>,
    u0: &ConservedFieldsT<T>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, T>,
    couplings: &[Vec<LuSgsCellCoupling>],
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()).rev() {
        let scale = implicit_scale(scalars.dt[cell], scalars.sigma[cell], scalars.omega);
        let mut source = [0.0; 5];
        for coupling in cell_couplings.iter().filter(|c| c.neighbor > cell) {
            add_coupling_delta_typed(
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
            apply_limited_cell_increment_typed(
                fields,
                cell,
                scale,
                damped,
                scalars.gamma,
                params.min_pressure,
            )?;
            refresh_primitive_typed(params, fields, cell)?;
        }
    }
    Ok(())
}

fn add_coupling_delta_typed<T: LuSgsUnstructuredSweepTyped>(
    source: &mut [Real; 5],
    cell: usize,
    coupling: LuSgsCellCoupling,
    volume: Real,
    fields: &ConservedFieldsT<T>,
    u0: &ConservedFieldsT<T>,
    params: &LuSgsSweepUnstructuredTypedParams<'_, T>,
) {
    let lambda = T::face_coupling_lambda(
        params.primitives,
        cell,
        coupling.neighbor,
        coupling.normal,
        params.eos.gamma,
    );
    let coef = coupling.area * lambda / volume.max(1.0e-30);
    let cur = conserved_vector_typed(fields, coupling.neighbor);
    let old = conserved_vector_typed(u0, coupling.neighbor);
    for (s, (&c, &o)) in source.iter_mut().zip(cur.iter().zip(old.iter())) {
        *s -= coef * (c - o);
    }
}

fn refresh_primitive_typed<T: LuSgsUnstructuredSweepTyped>(
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, T>,
    fields: &ConservedFieldsT<T>,
    cell: usize,
) -> Result<()> {
    refresh_primitive_at_cell_typed(
        fields,
        cell,
        params.eos,
        params.min_pressure,
        params.primitives,
    )
}

fn prim_lane_f32(primitives: &PrimitiveFieldsT<f32>, cell: usize) -> FacePrimitiveLaneF32 {
    FacePrimitiveLaneF32 {
        rho: primitives.density.values()[cell],
        pressure: primitives.pressure.values()[cell],
        velocity: [
            primitives.velocity_x.values()[cell],
            primitives.velocity_y.values()[cell],
            primitives.velocity_z.values()[cell],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ConservedFieldsT;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::FreestreamParams;
    use crate::solver::lu_sgs_common::fields_are_physical_typed;
    use crate::solver::lu_sgs_sweep_unstructured::LuSgsUnstructuredCouplings;

    fn uniform_freestream_sweep_mesh() -> UnstructuredMesh3d {
        UnstructuredMesh3d::new(
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
        .expect("mesh")
    }

    #[test]
    fn f32_sweep_keeps_uniform_unstructured_freestream_physical() {
        let mesh = uniform_freestream_sweep_mesh();
        let eos = IdealGasEoS::AIR_STANDARD;
        let mut fields = ConservedFieldsT::<f32>::from_freestream(
            mesh.num_cells(),
            &eos,
            &FreestreamParams {
                mach: 0.2,
                ..FreestreamParams::default()
            },
        )
        .expect("fields");
        let min_pressure = 1.0e-8;
        let mut primitives = PrimitiveFieldsT::<f32>::zeros(mesh.num_cells()).expect("prim");
        primitives
            .fill_from_conserved(&fields, &eos, min_pressure)
            .expect("fill");
        let residual = ConservedResidualT::<f32>::zeros(mesh.num_cells()).expect("rhs");
        let volumes = mesh.cell_volumes();
        let couplings = LuSgsUnstructuredCouplings::from_mesh(&mesh).expect("couplings");
        let dt = vec![0.01; mesh.num_cells()];
        let sigma = vec![10.0; mesh.num_cells()];
        let mut params = LuSgsSweepUnstructuredTypedParams {
            mesh: &mesh,
            eos: &eos,
            primitives: &mut primitives,
            min_pressure,
            backward_damping: 0.5,
        };
        lu_sgs_sweep_unstructured_typed(
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
        assert!(fields_are_physical_typed(&fields, eos.gamma, min_pressure).expect("physical"));
    }
}
