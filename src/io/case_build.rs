//! 算例初场构建（Parse → Validate → **Build** 之 Build 阶段）。

use crate::error::Result;
use crate::field::{ConservedFields, Fields, ScalarField};
use crate::io::CaseSpec;
use crate::io::restart;
use crate::mesh::StructuredBlock3d;

/// 1D 扩散初场。
pub fn initial_fields(case: &CaseSpec) -> Result<Fields> {
    let mesh_1d = case.mesh.as_1d()?;
    Fields::from_initial_set(mesh_1d, &case.initial)
}

/// 1D 标量初场（缺省为零场）。
pub fn initial_scalar(case: &CaseSpec, name: &str) -> Result<ScalarField> {
    let mesh_1d = case.mesh.as_1d()?;
    case.initial.build_scalar_or_zero(name, mesh_1d)
}

/// 单域 3D 守恒初场；`[restart]` 优先。
pub fn conserved_fields(case: &CaseSpec) -> Result<ConservedFields> {
    if let Some(path) = &case.restart {
        return restart::load_conserved_fields(path);
    }
    restart::initial_freestream_conserved_fields(
        case.mesh.num_cells(),
        &case.physics.eos()?,
        case.reference.as_ref(),
        case.physics.viscous.as_ref(),
        case.freestream.or(case.fluid_initial.freestream),
    )
}

/// 多块 structured 守恒初场；`[restart]` 使用 version=2 多块 TOML。
pub fn multiblock_conserved_fields(
    case: &CaseSpec,
    blocks: &[StructuredBlock3d],
) -> Result<Vec<ConservedFields>> {
    restart::initial_multiblock_conserved_fields(
        case.restart.as_deref(),
        blocks,
        &case.physics.eos()?,
        case.reference.as_ref(),
        case.physics.viscous.as_ref(),
        case.freestream.or(case.fluid_initial.freestream),
    )
}
