//! 可压缩 Euler 求解器配置（与 `CompressibleEulerSolver` 解耦，便于控制 `mod.rs` 体量）。

use crate::physics::ViscousPhysicsConfig;
use crate::solver::time::{
    CflSchedule, LuSgsConfig, ResidualSmoothingConfig, RungeKutta4Config, TimeIntegrationScheme,
};

use super::{CompressibleTimeMode, gmres_implicit_3d::GmresImplicitConfig};

/// 显式可压缩 Euler 求解器配置。
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleEulerConfig {
    pub time: RungeKutta4Config,
    pub inviscid: crate::discretization::InviscidFluxConfig,
    /// `Some` 时叠加层流粘性通量（Navier-Stokes）。
    pub viscous: Option<ViscousPhysicsConfig>,
    pub cfl_schedule: CflSchedule,
    pub time_mode: CompressibleTimeMode,
    pub local_time_step: bool,
    /// 时间积分格式（`rk4` 默认；`euler` 排错；`lu_sgs`/`gmres` 隐式伪时间）。
    pub time_scheme: TimeIntegrationScheme,
    /// `lu_sgs` 松弛因子等（显式格式下忽略）。
    pub lu_sgs: LuSgsConfig,
    pub gmres: GmresImplicitConfig,
    pub residual_smoothing: ResidualSmoothingConfig,
    pub low_mach_preconditioning: Option<crate::solver::time::LowMachPreconditioningConfig>,
}

impl Default for CompressibleEulerConfig {
    fn default() -> Self {
        Self {
            time: RungeKutta4Config::default(),
            inviscid: crate::discretization::InviscidFluxConfig::default(),
            viscous: None,
            cfl_schedule: CflSchedule::constant(0.4),
            time_mode: CompressibleTimeMode::Transient,
            local_time_step: false,
            time_scheme: TimeIntegrationScheme::Rk4,
            lu_sgs: LuSgsConfig::default(),
            gmres: GmresImplicitConfig::default(),
            residual_smoothing: ResidualSmoothingConfig::default(),
            low_mach_preconditioning: None,
        }
    }
}
