//! 非结构单元扫掠顺序（离线 reorder 工具与求解器共享）。

use std::collections::VecDeque;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AsimuError, Result};
use crate::mesh::UnstructuredMesh3d;

const ORDER_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellOrderFile {
    pub version: u32,
    pub strategy: String,
    pub num_cells: usize,
    pub order: Vec<usize>,
}

impl CellOrderFile {
    pub fn new(strategy: impl Into<String>, order: Vec<usize>) -> Result<Self> {
        let num_cells = order.len();
        validate_cell_order(&order, num_cells)?;
        Ok(Self {
            version: ORDER_VERSION,
            strategy: strategy.into(),
            num_cells,
            order,
        })
    }

    pub fn validate(&self, expected_cells: usize) -> Result<()> {
        if self.version != ORDER_VERSION {
            return Err(AsimuError::Config(format!(
                "cell_order version {} 不受支持（当前支持 {ORDER_VERSION}）",
                self.version
            )));
        }
        if self.num_cells != expected_cells {
            return Err(AsimuError::Config(format!(
                "cell_order 单元数 {} 与网格单元数 {expected_cells} 不一致",
                self.num_cells
            )));
        }
        validate_cell_order(&self.order, expected_cells)
    }
}

pub fn load_cell_order_file(path: &Path, expected_cells: usize) -> Result<CellOrderFile> {
    let text = std::fs::read_to_string(path)?;
    let order: CellOrderFile = toml::from_str(&text)?;
    order.validate(expected_cells)?;
    Ok(order)
}

pub fn write_cell_order_file(path: &Path, order: &CellOrderFile) -> Result<()> {
    let text = toml::to_string_pretty(order)
        .map_err(|err| AsimuError::Config(format!("cell_order 序列化失败: {err}")))?;
    std::fs::write(path, text)?;
    Ok(())
}

pub fn identity_order(num_cells: usize) -> Vec<usize> {
    (0..num_cells).collect()
}

pub fn bfs_order(mesh: &UnstructuredMesh3d) -> Result<Vec<usize>> {
    let adjacency = cell_adjacency(mesh)?;
    Ok(component_bfs_order(&adjacency))
}

pub fn rcm_order(mesh: &UnstructuredMesh3d) -> Result<Vec<usize>> {
    let adjacency = cell_adjacency(mesh)?;
    let mut order = component_bfs_order(&adjacency);
    order.reverse();
    Ok(order)
}

pub fn cell_order_rank(order: &[usize]) -> Result<Vec<usize>> {
    validate_cell_order(order, order.len())?;
    let mut rank = vec![0; order.len()];
    for (position, &cell) in order.iter().enumerate() {
        rank[cell] = position;
    }
    Ok(rank)
}

pub fn validate_cell_order(order: &[usize], expected_cells: usize) -> Result<()> {
    if order.len() != expected_cells {
        return Err(AsimuError::Config(format!(
            "cell_order 长度 {} 与网格单元数 {expected_cells} 不一致",
            order.len()
        )));
    }
    let mut seen = vec![false; expected_cells];
    for &cell in order {
        if cell >= expected_cells {
            return Err(AsimuError::Config(format!(
                "cell_order 包含越界单元 {cell}，网格单元数 {expected_cells}"
            )));
        }
        if seen[cell] {
            return Err(AsimuError::Config(format!(
                "cell_order 重复包含单元 {cell}"
            )));
        }
        seen[cell] = true;
    }
    Ok(())
}

fn cell_adjacency(mesh: &UnstructuredMesh3d) -> Result<Vec<Vec<usize>>> {
    let mut adjacency = vec![Vec::new(); mesh.num_cells()];
    for face in 0..mesh.num_faces() {
        let face_id = crate::core::FaceId(face as u32);
        let Some(neighbor_id) = mesh.face_neighbor(face_id)? else {
            continue;
        };
        let owner = mesh.face_owner(face_id)?.index() as usize;
        let neighbor = neighbor_id.index() as usize;
        adjacency[owner].push(neighbor);
        adjacency[neighbor].push(owner);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    Ok(adjacency)
}

fn component_bfs_order(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let mut visited = vec![false; adjacency.len()];
    let mut order = Vec::with_capacity(adjacency.len());
    while let Some(start) = next_unvisited_min_degree(adjacency, &visited) {
        push_bfs_component(adjacency, start, &mut visited, &mut order);
    }
    order
}

fn next_unvisited_min_degree(adjacency: &[Vec<usize>], visited: &[bool]) -> Option<usize> {
    adjacency
        .iter()
        .enumerate()
        .filter(|(cell, _)| !visited[*cell])
        .min_by_key(|(_, neighbors)| neighbors.len())
        .map(|(cell, _)| cell)
}

fn push_bfs_component(
    adjacency: &[Vec<usize>],
    start: usize,
    visited: &mut [bool],
    order: &mut Vec<usize>,
) {
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);
    while let Some(cell) = queue.pop_front() {
        order.push(cell);
        let mut neighbors = adjacency[cell].clone();
        neighbors.sort_by_key(|&neighbor| adjacency[neighbor].len());
        for neighbor in neighbors {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{CellKind, UnstructuredCell};

    #[test]
    fn rejects_duplicate_order() {
        let err = validate_cell_order(&[0, 1, 1], 3).expect_err("duplicate");
        assert!(err.to_string().contains("重复"));
    }

    #[test]
    fn rcm_order_is_valid_on_two_tets() {
        let mesh = UnstructuredMesh3d::new(
            "two_tets",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![
                UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("tet0"),
                UnstructuredCell::new(CellKind::Tet, vec![1, 2, 3, 4]).expect("tet1"),
            ],
        )
        .expect("mesh");
        let order = rcm_order(&mesh).expect("rcm");
        validate_cell_order(&order, mesh.num_cells()).expect("valid");
    }
}
