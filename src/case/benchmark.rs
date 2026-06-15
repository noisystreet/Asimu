//! 标准 V&V benchmark 注册表（集中 `benchmark_id` 语义，避免 case 编排层散落字符串匹配）。

use crate::discretization::IncompressibleFaceFluxField;
use crate::error::Result;
use crate::field::IncompressibleFields;
use crate::io::{CaseSpec, IncompressibleCaseConfig};
use crate::mesh::StructuredMesh3d;

use super::taylor_green::{taylor_green_initial_fields, taylor_green_prepare_initial_fields};

/// 不可压 benchmark 初场与首步 coupling 状态。
#[derive(Debug, Clone, PartialEq)]
pub struct IncompressibleInitialState {
    pub fields: IncompressibleFields,
    pub initial_face_flux: Option<IncompressibleFaceFluxField>,
}

/// 不可压 benchmark 目录中已知的 `benchmark_id`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownIncompressibleBenchmark {
    TaylorGreen3d,
    LidDrivenCavityRe100,
    ChannelPoiseuille,
}

/// 构建不可压 benchmark / 通用 uniform 初场与可选 Rhie-Chow 面通量播种。
pub fn build_incompressible_initial_state(
    benchmark: Option<KnownIncompressibleBenchmark>,
    mesh: &StructuredMesh3d,
    config: &IncompressibleCaseConfig,
    boundary: &crate::boundary::BoundarySet,
    pseudo_time_step: crate::core::Real,
) -> Result<IncompressibleInitialState> {
    let state = if let Some(benchmark) = benchmark {
        if let Some(fields) = benchmark.initial_fields(mesh, config)? {
            benchmark.prepare_initial_state(mesh, config, boundary, pseudo_time_step, fields)?
        } else {
            IncompressibleInitialState {
                fields: IncompressibleFields::uniform(
                    mesh.num_cells(),
                    config.pressure,
                    config.velocity,
                )?,
                initial_face_flux: None,
            }
        }
    } else {
        IncompressibleInitialState {
            fields: IncompressibleFields::uniform(
                mesh.num_cells(),
                config.pressure,
                config.velocity,
            )?,
            initial_face_flux: None,
        }
    };
    state.fields.validate_len(mesh.num_cells())?;
    Ok(state)
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

    /// 在 `initial_fields` 之后做 benchmark 专用预处理（如 Rhie-Chow 压力投影与面通量播种）。
    pub fn prepare_initial_state(
        self,
        mesh: &StructuredMesh3d,
        config: &IncompressibleCaseConfig,
        boundary: &crate::boundary::BoundarySet,
        pseudo_time_step: crate::core::Real,
        fields: IncompressibleFields,
    ) -> Result<IncompressibleInitialState> {
        match self {
            Self::TaylorGreen3d => {
                let prepared = taylor_green_prepare_initial_fields(
                    mesh,
                    config,
                    boundary,
                    pseudo_time_step,
                    fields,
                )?;
                Ok(IncompressibleInitialState {
                    fields: prepared.fields,
                    initial_face_flux: Some(prepared.face_flux),
                })
            }
            Self::LidDrivenCavityRe100 | Self::ChannelPoiseuille => {
                Ok(IncompressibleInitialState {
                    fields,
                    initial_face_flux: None,
                })
            }
        }
    }

    /// 兼容旧调用：仅返回预处理后的 cell-centered 场。
    #[allow(dead_code)]
    pub fn prepare_initial_fields(
        self,
        mesh: &StructuredMesh3d,
        config: &IncompressibleCaseConfig,
        boundary: &crate::boundary::BoundarySet,
        pseudo_time_step: crate::core::Real,
        fields: IncompressibleFields,
    ) -> Result<IncompressibleFields> {
        Ok(self
            .prepare_initial_state(mesh, config, boundary, pseudo_time_step, fields)?
            .fields)
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
