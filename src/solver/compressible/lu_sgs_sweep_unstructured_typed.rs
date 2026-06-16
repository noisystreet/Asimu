//! 非结构 LU-SGS 双扫 typed 路径（f32 面耦合谱半径原生计算）。

use tracing::info_span;

use crate::core::{ComputeFloat, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT, PrimitiveFieldsT};
use crate::mesh::UnstructuredMesh3d;
use crate::physics::IdealGasEoS;

use crate::discretization::unstructured_face_cache_f32::LuSgsCellCouplingF32;
use crate::solver::compressible::lu_sgs_common::{
    LuSgsSweepScalars, LuSgsSweepScalarsF32, PrimitiveRefreshLane,
    apply_limited_cell_increment_f32, apply_limited_cell_increment_typed, conserved_vector_f32,
    conserved_vector_typed, implicit_scale, implicit_scale_f32, refresh_primitive_at_cell_typed,
    residual_cell_vector_f32, residual_cell_vector_typed, scale_source, scale_source_f32,
    stabilize_sweep_update_f32, stabilize_sweep_update_typed,
};
use crate::solver::compressible::lu_sgs_sweep_unstructured::{
    LuSgsCellCoupling, LuSgsSweepUnstructuredF32Input, LuSgsSweepUnstructuredInput,
    LuSgsUnstructuredCouplingsRef,
};
use crate::solver::compressible::spectral_radius::face_spectral_radius;
use crate::solver::compressible::spectral_radius_f32::{
    FacePrimitiveLaneF32, face_spectral_radius_f32,
};

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
pub fn lu_sgs_sweep_unstructured_typed<T: LuSgsUnstructuredSweepTyped + PrimitiveRefreshLane>(
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
        inv_dt_phys: input.inv_dt_phys,
    };
    match input.couplings {
        LuSgsUnstructuredCouplingsRef::F64(couplings) => {
            {
                let _span = info_span!(
                    "lu_sgs_unstructured_forward_typed",
                    precision = T::PRECISION.label()
                )
                .entered();
                forward_sweep_typed(fields, &u0, residual, params, couplings.cells(), &scalars)?;
            }
            {
                let _span = info_span!(
                    "lu_sgs_unstructured_backward_typed",
                    precision = T::PRECISION.label()
                )
                .entered();
                backward_sweep_typed(fields, &u0, params, couplings.cells(), &scalars)?;
            }
        }
        LuSgsUnstructuredCouplingsRef::F32(_) => {
            return Err(AsimuError::Solver(
                "lu_sgs_sweep_unstructured_typed: f32 耦合请调用 lu_sgs_sweep_unstructured_f32"
                    .to_string(),
            ));
        }
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

fn forward_sweep_typed<T: LuSgsUnstructuredSweepTyped + PrimitiveRefreshLane>(
    fields: &mut ConservedFieldsT<T>,
    u0: &ConservedFieldsT<T>,
    residual: &ConservedResidualT<T>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, T>,
    couplings: &[Vec<LuSgsCellCoupling>],
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()) {
        let scale = implicit_scale(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
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

fn backward_sweep_typed<T: LuSgsUnstructuredSweepTyped + PrimitiveRefreshLane>(
    fields: &mut ConservedFieldsT<T>,
    u0: &ConservedFieldsT<T>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, T>,
    couplings: &[Vec<LuSgsCellCoupling>],
    scalars: &LuSgsSweepScalars<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()).rev() {
        let scale = implicit_scale(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
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

fn forward_sweep_f32_couplings(
    fields: &mut ConservedFieldsT<f32>,
    u0: &ConservedFieldsT<f32>,
    residual: &ConservedResidualT<f32>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, f32>,
    couplings: &[Vec<LuSgsCellCouplingF32>],
    scalars: &LuSgsSweepScalarsF32<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()) {
        let scale = implicit_scale_f32(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
        let mut source = residual_cell_vector_f32(residual, cell);
        for coupling in cell_couplings.iter().filter(|c| c.neighbor < cell) {
            add_coupling_delta_f32(
                &mut source,
                cell,
                *coupling,
                scalars.volumes[cell],
                fields,
                u0,
                params,
            );
        }
        apply_limited_cell_increment_f32(
            fields,
            cell,
            scale,
            source,
            scalars.gamma,
            params.min_pressure as f32,
        )?;
        refresh_primitive_typed(params, fields, cell)?;
    }
    Ok(())
}

fn backward_sweep_f32_couplings(
    fields: &mut ConservedFieldsT<f32>,
    u0: &ConservedFieldsT<f32>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, f32>,
    couplings: &[Vec<LuSgsCellCouplingF32>],
    scalars: &LuSgsSweepScalarsF32<'_>,
) -> Result<()> {
    for (cell, cell_couplings) in couplings.iter().enumerate().take(fields.num_cells()).rev() {
        let scale = implicit_scale_f32(
            scalars.dt[cell],
            scalars.sigma[cell],
            scalars.omega,
            scalars.inv_dt_phys,
        );
        let mut source = [0.0_f32; 5];
        let damp = params.backward_damping as f32;
        for coupling in cell_couplings.iter().filter(|c| c.neighbor > cell) {
            add_coupling_delta_f32(
                &mut source,
                cell,
                *coupling,
                scalars.volumes[cell],
                fields,
                u0,
                params,
            );
        }
        if source.iter().any(|c| c.abs() > f32::EPSILON) {
            let damped = scale_source_f32(source, damp);
            apply_limited_cell_increment_f32(
                fields,
                cell,
                scale,
                damped,
                scalars.gamma,
                params.min_pressure as f32,
            )?;
            refresh_primitive_typed(params, fields, cell)?;
        }
    }
    Ok(())
}

fn add_coupling_delta_f32(
    source: &mut [f32; 5],
    cell: usize,
    coupling: LuSgsCellCouplingF32,
    volume: f32,
    fields: &ConservedFieldsT<f32>,
    u0: &ConservedFieldsT<f32>,
    params: &LuSgsSweepUnstructuredTypedParams<'_, f32>,
) {
    let lambda = face_spectral_radius_f32(
        prim_lane_f32(params.primitives, cell),
        prim_lane_f32(params.primitives, coupling.neighbor),
        coupling.normal,
        params.eos.gamma as f32,
    );
    let coef = coupling.area * lambda / volume.max(1.0e-30_f32);
    let cur = conserved_vector_f32(fields, coupling.neighbor);
    let old = conserved_vector_f32(u0, coupling.neighbor);
    for (s, (&c, &o)) in source.iter_mut().zip(cur.iter().zip(old.iter())) {
        *s -= coef * (c - o);
    }
}

/// f32 预打包耦合的非结构 LU-SGS 双扫。
pub fn lu_sgs_sweep_unstructured_f32(
    fields: &mut ConservedFieldsT<f32>,
    residual: &ConservedResidualT<f32>,
    params: &mut LuSgsSweepUnstructuredTypedParams<'_, f32>,
    input: LuSgsSweepUnstructuredF32Input<'_>,
) -> Result<()> {
    let n = fields.num_cells();
    if residual.num_cells() != n
        || input.dt.len() != n
        || input.sigma.len() != n
        || input.volumes.len() != n
        || input.couplings.len() != n
    {
        return Err(AsimuError::Solver(
            "lu_sgs_sweep_unstructured_f32: 场/残差/dt/sigma/volume 长度不一致".to_string(),
        ));
    }
    let LuSgsUnstructuredCouplingsRef::F32(couplings) = input.couplings else {
        return Err(AsimuError::Solver(
            "lu_sgs_sweep_unstructured_f32: 须使用 f32 拓扑耦合".to_string(),
        ));
    };
    let u0 = fields.clone();
    let scalars = LuSgsSweepScalarsF32 {
        dt: input.dt,
        sigma: input.sigma,
        volumes: input.volumes,
        omega: input.omega,
        gamma: input.gamma,
        inv_dt_phys: input.inv_dt_phys,
    };
    let min_p = params.min_pressure as f32;
    {
        let _span = info_span!("lu_sgs_unstructured_forward_f32").entered();
        forward_sweep_f32_couplings(fields, &u0, residual, params, couplings.cells(), &scalars)?;
    }
    {
        let _span = info_span!("lu_sgs_unstructured_backward_f32").entered();
        backward_sweep_f32_couplings(fields, &u0, params, couplings.cells(), &scalars)?;
    }
    let u_sweep = fields.clone();
    stabilize_sweep_update_f32(
        fields,
        &u0,
        &u_sweep,
        residual,
        min_p,
        scalars.gamma,
        &scalars,
    )
}

fn refresh_primitive_typed<T: LuSgsUnstructuredSweepTyped + PrimitiveRefreshLane>(
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
    use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet, WallHeat};
    use crate::discretization::UnstructuredSolverMeshCache;
    use crate::field::ConservedFieldsT;
    use crate::mesh::{CellKind, UnstructuredCell};
    use crate::physics::FreestreamParams;
    use crate::solver::compressible::lu_sgs_common::fields_are_physical_f32;
    use crate::solver::compressible::lu_sgs_sweep_unstructured::{
        LuSgsSweepUnstructuredF32Input, LuSgsUnstructuredCouplingsRef,
    };

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
        let volumes: Vec<f32> = mesh.cell_volumes().iter().map(|v| *v as f32).collect();
        let dt: Vec<f32> = vec![0.01_f32; mesh.num_cells()];
        let sigma: Vec<f32> = vec![10.0_f32; mesh.num_cells()];
        let patches = BoundarySet::new(vec![BoundaryPatch::new(
            "wall",
            (0..mesh.num_faces())
                .map(|f| crate::core::FaceId(f as u32))
                .collect(),
            BoundaryKind::Wall {
                no_slip: true,
                heat: WallHeat::Adiabatic,
            },
        )]);
        let mesh_cache = UnstructuredSolverMeshCache::from_mesh(&mesh, &patches).expect("cache");
        let mut params = LuSgsSweepUnstructuredTypedParams {
            mesh: &mesh,
            eos: &eos,
            primitives: &mut primitives,
            min_pressure,
            backward_damping: 0.5,
        };
        lu_sgs_sweep_unstructured_f32(
            &mut fields,
            &residual,
            &mut params,
            LuSgsSweepUnstructuredF32Input {
                dt: &dt,
                sigma: &sigma,
                volumes: &volumes,
                couplings: LuSgsUnstructuredCouplingsRef::F32(&mesh_cache.lusgs_couplings_f32),
                omega: 1.0,
                gamma: eos.gamma as f32,
                inv_dt_phys: 0.0,
            },
        )
        .expect("sweep");
        assert!(
            fields_are_physical_f32(&fields, eos.gamma as f32, min_pressure as f32)
                .expect("physical")
        );
    }
}
