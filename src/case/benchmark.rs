//! 标准 V&V benchmark 注册表（集中 `benchmark_id` 语义，避免 case 编排层散落字符串匹配）。

use crate::error::Result;
use crate::field::IncompressibleFields;
use crate::io::{CaseSpec, IncompressibleCaseConfig};
use crate::mesh::StructuredMesh3d;

use super::taylor_green::{taylor_green_initial_fields, taylor_green_prepare_initial_fields};

/// 不可压 benchmark 目录中已知的 `benchmark_id`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownIncompressibleBenchmark {
    TaylorGreen3d,
    LidDrivenCavityRe100,
    ChannelPoiseuille,
}

impl KnownIncompressibleBenchmark {
    #[must_use]
    pub fn parse(id: &str) -> Option<Self> {
        match id {
            "taylor_green_3d" => Some(Self::TaylorGreen3d),
            "channel_poiseuille" => Some(Self::ChannelPoiseuille),
            id if id.starts_with("lid_driven_cavity_re100") => Some(Self::LidDrivenCavityRe100),
            _ => None,
        }
    }

    #[must_use]
    pub fn from_case(case: &CaseSpec) -> Option<Self> {
        case.benchmark_id.as_deref().and_then(Self::parse)
    }

    #[must_use]
    pub const fn tracks_kinetic_energy(self) -> bool {
        matches!(self, Self::TaylorGreen3d)
    }

    #[must_use]
    pub const fn emits_centerline_profiles(self) -> bool {
        matches!(self, Self::LidDrivenCavityRe100 | Self::ChannelPoiseuille)
    }

    /// 若 benchmark 定义专用初场则返回 `Some`；否则调用方回退 uniform 初值。
    pub fn initial_fields(
        self,
        mesh: &StructuredMesh3d,
        _config: &IncompressibleCaseConfig,
    ) -> Result<Option<IncompressibleFields>> {
        match self {
            Self::TaylorGreen3d => Ok(Some(taylor_green_initial_fields(mesh)?)),
            Self::LidDrivenCavityRe100 | Self::ChannelPoiseuille => Ok(None),
        }
    }

    /// 在 `initial_fields` 之后做 benchmark 专用预处理（如 Rhie-Chow 压力投影）。
    pub fn prepare_initial_fields(
        self,
        mesh: &StructuredMesh3d,
        config: &IncompressibleCaseConfig,
        boundary: &crate::boundary::BoundarySet,
        fields: IncompressibleFields,
    ) -> Result<IncompressibleFields> {
        match self {
            Self::TaylorGreen3d => {
                taylor_green_prepare_initial_fields(mesh, config, boundary, fields)
            }
            Self::LidDrivenCavityRe100 | Self::ChannelPoiseuille => Ok(fields),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_benchmark_ids() {
        assert_eq!(
            KnownIncompressibleBenchmark::parse("taylor_green_3d"),
            Some(KnownIncompressibleBenchmark::TaylorGreen3d)
        );
        assert_eq!(
            KnownIncompressibleBenchmark::parse("lid_driven_cavity_re100"),
            Some(KnownIncompressibleBenchmark::LidDrivenCavityRe100)
        );
        assert!(KnownIncompressibleBenchmark::parse("unknown").is_none());
    }

    #[test]
    fn taylor_green_tracks_kinetic_energy() {
        assert!(KnownIncompressibleBenchmark::TaylorGreen3d.tracks_kinetic_energy());
        assert!(!KnownIncompressibleBenchmark::LidDrivenCavityRe100.tracks_kinetic_energy());
    }
}
