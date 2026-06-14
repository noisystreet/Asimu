//! 不可压缩算例时间推进语义（稳态耦合 / 伪时间 / 物理瞬态）。

use crate::io::{CaseSpec, CaseTimeMode};
use crate::solver::TimeIntegrationScheme;

/// `[time]` 在不可压路径上的推进语义（与 `CaseTimeMode` × `scheme` 组合对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompressibleTimeAdvanceKind {
    /// `mode = steady`：SIMPLEC 耦合迭代至收敛。
    SteadyCoupling,
    /// `mode = transient` 且 `scheme` 缺省或为 `simplec`：伪时间步进。
    SteadyPseudoTime,
    /// `mode = transient` 且 `scheme = piso | bdf1`：物理时间推进。
    PhysicalTransient,
}

#[must_use]
pub fn incompressible_time_advance_kind(case: &CaseSpec) -> IncompressibleTimeAdvanceKind {
    incompressible_time_advance_from_config(&case.time)
}

#[must_use]
pub fn incompressible_time_advance_from_config(
    time: &crate::io::CaseTimeConfig,
) -> IncompressibleTimeAdvanceKind {
    match time.mode {
        CaseTimeMode::Steady => IncompressibleTimeAdvanceKind::SteadyCoupling,
        CaseTimeMode::Transient => match time.resolved_time_scheme() {
            TimeIntegrationScheme::Piso | TimeIntegrationScheme::Bdf1 => {
                IncompressibleTimeAdvanceKind::PhysicalTransient
            }
            _ => IncompressibleTimeAdvanceKind::SteadyPseudoTime,
        },
    }
}

#[must_use]
pub const fn incompressible_physical_transient(kind: IncompressibleTimeAdvanceKind) -> bool {
    matches!(kind, IncompressibleTimeAdvanceKind::PhysicalTransient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{CaseTimeConfig, CaseTimeMode};

    #[test]
    fn steady_mode_is_coupling() {
        assert_eq!(
            incompressible_time_advance_from_config(&CaseTimeConfig {
                mode: CaseTimeMode::Steady,
                ..CaseTimeConfig::default()
            }),
            IncompressibleTimeAdvanceKind::SteadyCoupling
        );
    }

    #[test]
    fn transient_without_scheme_is_pseudo_time() {
        assert_eq!(
            incompressible_time_advance_from_config(&CaseTimeConfig {
                mode: CaseTimeMode::Transient,
                dt: Some(0.01),
                ..CaseTimeConfig::default()
            }),
            IncompressibleTimeAdvanceKind::SteadyPseudoTime
        );
    }

    #[test]
    fn transient_bdf1_is_physical() {
        assert_eq!(
            incompressible_time_advance_from_config(&CaseTimeConfig {
                mode: CaseTimeMode::Transient,
                scheme: Some(TimeIntegrationScheme::Bdf1),
                dt: Some(0.01),
                ..CaseTimeConfig::default()
            }),
            IncompressibleTimeAdvanceKind::PhysicalTransient
        );
    }
}
