//! 边界 patch 数据（CFL3D 每面分段 BC 的 asimu 等价物）。

use tracing::info;

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

    pub fn patches_mut(&mut self) -> &mut [BoundaryPatch] {
        &mut self.patches
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

    #[must_use]
    pub fn has_periodic_pair(&self, a: &str, b: &str) -> bool {
        let mut a_to_b = false;
        let mut b_to_a = false;
        for patch in &self.patches {
            match (patch.name.as_str(), &patch.kind) {
                (name, BoundaryKind::Periodic { partner }) if name == a && partner == b => {
                    a_to_b = true;
                }
                (name, BoundaryKind::Periodic { partner }) if name == b && partner == a => {
                    b_to_a = true;
                }
                _ => {}
            }
        }
        a_to_b && b_to_a
    }

    /// 将各边界 patch 名称与边界条件写入日志（`info` 级别）。
    pub fn log_patches(&self) {
        let patches = self.patches();
        if patches.is_empty() {
            info!("边界条件：无 patch");
            return;
        }
        info!(
            count = patches.len(),
            "边界条件：{} 个 patch",
            patches.len()
        );
        for patch in patches {
            info!(
                patch = %patch.name,
                faces = patch.face_ids.len(),
                kind = %patch.kind.summary_label(),
                detail = %patch.kind.detail_label(),
                "边界 patch"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_reciprocal_periodic_pair() {
        let set = BoundarySet::new(vec![
            BoundaryPatch::new(
                "i_min",
                Vec::new(),
                BoundaryKind::Periodic {
                    partner: "i_max".to_string(),
                },
            ),
            BoundaryPatch::new(
                "i_max",
                Vec::new(),
                BoundaryKind::Periodic {
                    partner: "i_min".to_string(),
                },
            ),
        ]);

        assert!(set.has_periodic_pair("i_min", "i_max"));
        assert!(!set.has_periodic_pair("j_min", "j_max"));
    }
}
