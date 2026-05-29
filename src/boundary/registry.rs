//! BC 调度注册表（类比 CFL3D `bc.F` → `bcXXXX.F` 分派）。

use crate::error::Result;

use super::kind::BoundaryKind;

/// 边界条件数值处理器标识。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcHandler {
    DiffusionDirichlet,
    DiffusionNeumann,
    Wall,
    Farfield,
    Inlet,
    Outlet,
    Symmetry,
    Periodic,
    TurbulentInlet,
}

/// 名称 → 处理器映射。
#[derive(Debug, Clone, Default)]
pub struct BoundaryRegistry;

impl BoundaryRegistry {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// 根据 `BoundaryKind` 选择处理器。
    #[must_use]
    pub fn handler_for(kind: &BoundaryKind) -> BcHandler {
        match kind {
            BoundaryKind::Dirichlet { .. } => BcHandler::DiffusionDirichlet,
            BoundaryKind::Neumann { .. } => BcHandler::DiffusionNeumann,
            BoundaryKind::Wall { .. } => BcHandler::Wall,
            BoundaryKind::Farfield { .. } => BcHandler::Farfield,
            BoundaryKind::Inlet { .. } => BcHandler::Inlet,
            BoundaryKind::Outlet { .. } => BcHandler::Outlet,
            BoundaryKind::Symmetry => BcHandler::Symmetry,
            BoundaryKind::Periodic { .. } => BcHandler::Periodic,
            BoundaryKind::TurbulentInlet { .. } => BcHandler::TurbulentInlet,
        }
    }

    pub fn is_compressible(kind: &BoundaryKind) -> bool {
        !matches!(
            kind,
            BoundaryKind::Dirichlet { .. } | BoundaryKind::Neumann { .. }
        )
    }

    /// 校验 patch 列表。
    pub fn validate_patches(patches: &[super::patch::BoundaryPatch]) -> Result<()> {
        for patch in patches {
            if patch.name.trim().is_empty() {
                return Err(crate::error::AsimuError::Boundary(
                    "边界 patch 名称不能为空".to_string(),
                ));
            }
            if patch.face_ids.is_empty() {
                return Err(crate::error::AsimuError::Boundary(format!(
                    "边界 patch \"{}\" 未关联任何面",
                    patch.name
                )));
            }
        }
        Ok(())
    }
}
