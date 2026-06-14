//! 非结构 typed 显式时间推进精度分发（f32 当地时间步缓冲）。

use crate::core::Real;
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFieldsT, ConservedResidualT};
use crate::solver::time::{
    TimeIntegrationScheme, euler_step, euler_step_local, euler_step_local_f32, rk4_step,
    rk4_step_local, rk4_step_local_f32,
};

use super::{
    UnstructuredRhsDispatchImpl, UnstructuredRunEnvTyped, UnstructuredStepWorkTyped,
    UnstructuredTypedRhsWork, assemble_unstructured_typed_rhs,
};

pub(crate) fn advance_unstructured_explicit_typed<T: UnstructuredExplicitTimeAdvance>(
    env: &UnstructuredRunEnvTyped<'_>,
    fields: &mut ConservedFieldsT<T>,
    work: &mut UnstructuredStepWorkTyped<T>,
    dt: Real,
    p_floor: Real,
) -> Result<()> {
    T::advance_unstructured_explicit(env, fields, work, dt, p_floor)
}

/// 显式时间推进精度分发（f32 当地时间步走原生 f32 缓冲）。
pub(crate) trait UnstructuredExplicitTimeAdvance: UnstructuredRhsDispatchImpl {
    fn advance_unstructured_explicit(
        env: &UnstructuredRunEnvTyped<'_>,
        fields: &mut ConservedFieldsT<Self>,
        work: &mut UnstructuredStepWorkTyped<Self>,
        dt: Real,
        p_floor: Real,
    ) -> Result<()>;
}

impl UnstructuredExplicitTimeAdvance for f32 {
    fn advance_unstructured_explicit(
        env: &UnstructuredRunEnvTyped<'_>,
        fields: &mut ConservedFieldsT<f32>,
        work: &mut UnstructuredStepWorkTyped<f32>,
        dt: Real,
        p_floor: Real,
    ) -> Result<()> {
        let local = env.config.local_time_step;
        let scheme = env.config.time_scheme;
        let eos = env.config.eos;
        let mut reuse_current_state = true;
        let mut rhs_work = UnstructuredTypedRhsWork {
            ghosts: &mut work.ghosts,
            primitives: &mut work.primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
            viscous_grad_scratch_f32: &mut work.viscous_grad_scratch_f32,
            mesh_cache: &work.mesh_cache,
            exec: &mut work.exec,
        };
        let evaluate = |u: &ConservedFieldsT<f32>, r: &mut ConservedResidualT<f32>| {
            let refresh = !reuse_current_state;
            reuse_current_state = false;
            assemble_unstructured_typed_rhs(env, &mut rhs_work, u, r, refresh, p_floor)
        };
        match (scheme, local) {
            (TimeIntegrationScheme::Rk4, true) => rk4_step_local_f32(
                fields,
                &mut work.storage,
                &work.timestep.cell_dts_f32,
                evaluate,
                Some(eos),
                p_floor,
            ),
            (TimeIntegrationScheme::Rk4, false) => {
                rk4_step(fields, &mut work.storage, dt, evaluate)
            }
            (TimeIntegrationScheme::Euler, true) => euler_step_local_f32(
                fields,
                &mut work.storage,
                &work.timestep.cell_dts_f32,
                evaluate,
                Some(eos),
                p_floor,
            ),
            (TimeIntegrationScheme::Euler, false) => {
                euler_step(fields, &mut work.storage, dt, evaluate, Some(eos), p_floor)
            }
            _ => Err(AsimuError::Solver(
                "非结构 typed 显式推进收到不支持的时间格式".to_string(),
            )),
        }
    }
}

impl UnstructuredExplicitTimeAdvance for f64 {
    fn advance_unstructured_explicit(
        env: &UnstructuredRunEnvTyped<'_>,
        fields: &mut ConservedFieldsT<f64>,
        work: &mut UnstructuredStepWorkTyped<f64>,
        dt: Real,
        p_floor: Real,
    ) -> Result<()> {
        let local = env.config.local_time_step;
        let scheme = env.config.time_scheme;
        let eos = env.config.eos;
        let mut reuse_current_state = true;
        let mut rhs_work = UnstructuredTypedRhsWork {
            ghosts: &mut work.ghosts,
            primitives: &mut work.primitives,
            gradients: &mut work.gradients,
            viscous_scratch: &mut work.viscous_scratch,
            viscous_grad_scratch_f32: &mut work.viscous_grad_scratch_f32,
            mesh_cache: &work.mesh_cache,
            exec: &mut work.exec,
        };
        let evaluate = |u: &ConservedFieldsT<f64>, r: &mut ConservedResidualT<f64>| {
            let refresh = !reuse_current_state;
            reuse_current_state = false;
            assemble_unstructured_typed_rhs(env, &mut rhs_work, u, r, refresh, p_floor)
        };
        match (scheme, local) {
            (TimeIntegrationScheme::Rk4, true) => rk4_step_local(
                fields,
                &mut work.storage,
                &work.timestep.cell_dts,
                evaluate,
                Some(eos),
                p_floor,
            ),
            (TimeIntegrationScheme::Rk4, false) => {
                rk4_step(fields, &mut work.storage, dt, evaluate)
            }
            (TimeIntegrationScheme::Euler, true) => euler_step_local(
                fields,
                &mut work.storage,
                &work.timestep.cell_dts,
                evaluate,
                Some(eos),
                p_floor,
            ),
            (TimeIntegrationScheme::Euler, false) => {
                euler_step(fields, &mut work.storage, dt, evaluate, Some(eos), p_floor)
            }
            _ => Err(AsimuError::Solver(
                "非结构 typed 显式推进收到不支持的时间格式".to_string(),
            )),
        }
    }
}
