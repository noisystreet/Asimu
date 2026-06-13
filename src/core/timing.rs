//! 轻量计时 helper，供 solver/case 诊断日志复用。

use std::time::Instant;

use super::Real;

/// 自 `start` 起经过的 wall time（毫秒）。
#[must_use]
pub fn elapsed_ms(start: Instant) -> Real {
    start.elapsed().as_secs_f64() * 1000.0
}
