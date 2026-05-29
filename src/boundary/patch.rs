//! 边界 patch 数据（CFL3D 每面分段 BC 的 asimu 等价物）。

use crate::core::FaceId;

use super::kind::BoundaryKind;

/// 单个逻辑边界 patch（如 `left`、`right`）。
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryPatch {
    pub name: String,
    pub face_ids: Vec<FaceId>,
    pub kind: BoundaryKind,
}

impl BoundaryPatch {
    pub fn new(name: impl Into<String>, face_ids: Vec<FaceId>, kind: BoundaryKind) -> Self {
        Self {
            name: name.into(),
            face_ids,
            kind,
        }
    }
}

/// 算例全部边界 patch（有序列表，施加顺序与 CFL3D `bc.F` 遍历段一致）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundarySet {
    patches: Vec<BoundaryPatch>,
}

impl BoundarySet {
    #[must_use]
    pub fn new(patches: Vec<BoundaryPatch>) -> Self {
        Self { patches }
    }

    #[must_use]
    pub fn patches(&self) -> &[BoundaryPatch] {
        &self.patches
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    pub fn push(&mut self, patch: BoundaryPatch) {
        self.patches.push(patch);
    }

    pub fn find(&self, name: &str) -> Option<&BoundaryPatch> {
        self.patches.iter().find(|p| p.name == name)
    }
}
