//! 非结构 3D 可压缩时间推进驱动（ADR 0018：f64 委托 typed 实现）。

use crate::error::Result;
use crate::field::ConservedFields;
use crate::solver::CompressibleStepInfo;
use crate::solver::compressible::unstructured_driver_typed::run_unstructured_typed_with_observer;

/// 非结构可压缩外层步只读视图（observer 回调参数）。
#[derive(Debug, Clone, Copy)]
pub struct CompressibleUnstructuredStepView<'a> {
    pub info: &'a CompressibleStepInfo,
    pub history: &'a [CompressibleStepInfo],
    pub fields: &'a ConservedFields,
}

/// 非结构推进配置（由 case 层从 `CaseSpec` 组装）。
pub struct UnstructuredDriverConfig<'a> {
    pub solver: &'a crate::solver::CompressibleEulerSolver,
    pub mesh: &'a crate::mesh::UnstructuredMesh3d,
    pub eos: &'a crate::physics::IdealGasEoS,
    pub freestream: &'a crate::physics::FreestreamParams,
    pub inviscid: &'a crate::discretization::InviscidFluxConfig,
    pub patches: &'a crate::boundary::BoundarySet,
    pub reference: Option<&'a crate::physics::ReferenceScales>,
    pub viscous: Option<&'a crate::physics::ViscousPhysicsConfig>,
    pub fixed_dt: Option<crate::core::Real>,
    pub local_time_step: bool,
    pub time_scheme: crate::solver::time::TimeIntegrationScheme,
    pub lu_sgs: crate::solver::time::LuSgsConfig,
    pub cfl_schedule: crate::solver::time::CflSchedule,
    pub max_steps: u64,
    pub residual_tolerance: Option<crate::core::Real>,
    pub exec_config: crate::exec::ExecConfig,
    /// 双时间步配置；`scheme = dual_time` 时必填。
    pub dual_time: Option<crate::solver::time::DualTimeConfig>,
    /// 间隔流场写出步长（`solution_every`）；observer 仅在这些步同步 host 守恒场。
    pub observer_field_sync_interval: Option<u64>,
}

/// observer 是否需 host 守恒场（间隔流场写出步）。
#[must_use]
pub(crate) fn observer_field_sync_due(interval: Option<u64>, step: u64) -> bool {
    interval.is_some_and(|every| every > 0 && step % every == 0)
}

/// 非结构 f64 推进（薄包装：单一路径 `run_unstructured_typed_with_observer::<f64>`）。
pub fn run_unstructured_with_observer(
    config: &UnstructuredDriverConfig<'_>,
    fields: &mut ConservedFields,
    mut observe_step: impl FnMut(CompressibleUnstructuredStepView<'_>) -> Result<()>,
) -> Result<Vec<CompressibleStepInfo>> {
    let (history, out) = run_unstructured_typed_with_observer::<f64>(config, fields, |step| {
        observe_step(step)?;
        Ok(())
    })?;
    *fields = out;
    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::observer_field_sync_due;

    #[test]
    fn observer_field_sync_due_matches_solution_every() {
        assert!(!observer_field_sync_due(None, 1));
        assert!(!observer_field_sync_due(Some(0), 200));
        assert!(observer_field_sync_due(Some(200), 200));
        assert!(!observer_field_sync_due(Some(200), 199));
    }
}
