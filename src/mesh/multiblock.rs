//! 多块结构化 3D 网格容器。

use std::collections::BTreeSet;

use crate::core::Real;
use crate::error::{AsimuError, Result};

use super::{MeshMetricMode, StructuredMesh3d};

/// 3D 结构化 block 的 1-based 顶点索引范围。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredIndexRange3d {
    pub imin: i32,
    pub imax: i32,
    pub jmin: i32,
    pub jmax: i32,
    pub kmin: i32,
    pub kmax: i32,
}

/// 多块结构化网格中的一条 block 间 1-to-1 接口。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredBlockInterface3d {
    pub owner_block: String,
    pub donor_block: String,
    pub owner_range: StructuredIndexRange3d,
    pub donor_range: StructuredIndexRange3d,
    pub transform: [i32; 3],
}

/// 单个结构化网格块及其在全局场数组中的单元偏移。
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredBlock3d {
    pub name: String,
    pub mesh: StructuredMesh3d,
    pub cell_offset: usize,
}

/// 多块结构化 3D 网格。
///
/// 首版只表达 block 集合与全局单元编号范围；跨块接口连通、守恒通量交换
/// 和多块求解器会在后续阶段接入。
#[derive(Debug, Clone, PartialEq)]
pub struct MultiBlockStructuredMesh3d {
    pub name: String,
    blocks: Vec<StructuredBlock3d>,
    interfaces: Vec<StructuredBlockInterface3d>,
    num_cells: usize,
    num_nodes: usize,
}

impl MultiBlockStructuredMesh3d {
    /// 将单块结构化网格包装为 1-block 多块容器（无接口）。
    pub fn from_single_mesh(mesh: StructuredMesh3d) -> Result<Self> {
        let name = mesh.name.clone();
        Self::new(name, vec![mesh])
    }

    pub fn new(name: impl Into<String>, meshes: Vec<StructuredMesh3d>) -> Result<Self> {
        Self::with_interfaces(name, meshes, Vec::new())
    }

    pub fn with_interfaces(
        name: impl Into<String>,
        meshes: Vec<StructuredMesh3d>,
        interfaces: Vec<StructuredBlockInterface3d>,
    ) -> Result<Self> {
        if meshes.is_empty() {
            return Err(AsimuError::Mesh(
                "多块结构化网格至少需要 1 个 block".to_string(),
            ));
        }

        let mut seen = BTreeSet::new();
        let mut blocks = Vec::with_capacity(meshes.len());
        let mut cell_offset = 0usize;
        let mut num_nodes = 0usize;

        for mesh in meshes {
            if mesh.name.trim().is_empty() {
                return Err(AsimuError::Mesh(
                    "多块结构化网格 block 名称不能为空".to_string(),
                ));
            }
            if !seen.insert(mesh.name.clone()) {
                return Err(AsimuError::Mesh(format!(
                    "多块结构化网格 block 名称重复：{}",
                    mesh.name
                )));
            }
            let block_cells = mesh.num_cells();
            num_nodes += mesh.num_nodes();
            blocks.push(StructuredBlock3d {
                name: mesh.name.clone(),
                mesh,
                cell_offset,
            });
            cell_offset += block_cells;
        }

        validate_interfaces(&seen, &interfaces)?;

        Ok(Self {
            name: name.into(),
            blocks,
            interfaces,
            num_cells: cell_offset,
            num_nodes,
        })
    }

    #[must_use]
    pub fn blocks(&self) -> &[StructuredBlock3d] {
        &self.blocks
    }

    #[must_use]
    pub fn interfaces(&self) -> &[StructuredBlockInterface3d] {
        &self.interfaces
    }

    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[must_use]
    pub fn block(&self, name: &str) -> Option<&StructuredBlock3d> {
        self.blocks.iter().find(|block| block.name == name)
    }

    /// 将所有 block 节点坐标乘以 `factor`。
    pub fn scale_coordinates(&mut self, factor: Real) {
        for block in &mut self.blocks {
            block.mesh.scale_coordinates(factor);
        }
    }

    pub fn set_metric_mode(&mut self, mode: MeshMetricMode) {
        for block in &mut self.blocks {
            block.mesh.set_metric_mode(mode);
        }
    }

    pub fn rebuild_metric_cache_if_needed(&mut self) -> Result<()> {
        for block in &mut self.blocks {
            block.mesh.rebuild_metric_cache_if_needed()?;
        }
        Ok(())
    }
}

fn validate_interfaces(
    block_names: &BTreeSet<String>,
    interfaces: &[StructuredBlockInterface3d],
) -> Result<()> {
    for interface in interfaces {
        if !block_names.contains(&interface.owner_block) {
            return Err(AsimuError::Mesh(format!(
                "多块接口引用未知 owner block：{}",
                interface.owner_block
            )));
        }
        if !block_names.contains(&interface.donor_block) {
            return Err(AsimuError::Mesh(format!(
                "多块接口引用未知 donor block：{}",
                interface.donor_block
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str, nx: usize) -> StructuredMesh3d {
        StructuredMesh3d::uniform_box(name, nx, 1, 1, nx as Real, 1.0, 1.0).expect("block")
    }

    #[test]
    fn wraps_single_structured_mesh() {
        let mesh = MultiBlockStructuredMesh3d::from_single_mesh(block("solo", 4)).expect("wrap");
        assert_eq!(mesh.num_blocks(), 1);
        assert_eq!(mesh.num_cells(), 4);
        assert!(mesh.interfaces().is_empty());
    }

    #[test]
    fn assigns_global_cell_offsets() {
        let mesh = MultiBlockStructuredMesh3d::new("multi", vec![block("a", 2), block("b", 3)])
            .expect("multi");

        assert_eq!(mesh.num_blocks(), 2);
        assert_eq!(mesh.num_cells(), 5);
        assert_eq!(mesh.blocks()[0].cell_offset, 0);
        assert_eq!(mesh.blocks()[1].cell_offset, 2);
        assert_eq!(mesh.block("b").expect("block").mesh.num_cells(), 3);
    }

    #[test]
    fn rejects_duplicate_block_names() {
        let err = MultiBlockStructuredMesh3d::new("multi", vec![block("a", 1), block("a", 1)])
            .expect_err("duplicate");

        assert!(matches!(err, AsimuError::Mesh(_)));
    }

    #[test]
    fn stores_block_interfaces() {
        let mesh = MultiBlockStructuredMesh3d::with_interfaces(
            "multi",
            vec![block("a", 1), block("b", 1)],
            vec![StructuredBlockInterface3d {
                owner_block: "a".to_string(),
                donor_block: "b".to_string(),
                owner_range: StructuredIndexRange3d {
                    imin: 2,
                    imax: 2,
                    jmin: 1,
                    jmax: 2,
                    kmin: 1,
                    kmax: 2,
                },
                donor_range: StructuredIndexRange3d {
                    imin: 1,
                    imax: 1,
                    jmin: 1,
                    jmax: 2,
                    kmin: 1,
                    kmax: 2,
                },
                transform: [1, 2, 3],
            }],
        )
        .expect("multi");

        assert_eq!(mesh.interfaces().len(), 1);
    }
}
