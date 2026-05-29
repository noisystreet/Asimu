//! 求解器运行时状态（显式封装，禁止隐式全局）。

use crate::core::Real;

/// 时间推进与迭代共享的可变状态。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolverState {
    pub pseudo_step: u32,
    pub physical_time: Real,
    pub iteration: u32,
}
