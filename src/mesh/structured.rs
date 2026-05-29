//! 结构化网格（2D / 3D），用于 VTS 读入与 FVM。

use crate::error::{AsimuError, Result};

/// 2D 或 3D 结构化网格。
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredMesh {
    D2(StructuredMesh2d),
    D3(StructuredMesh3d),
}

impl StructuredMesh {
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::D2(m) => &m.name,
            Self::D3(m) => &m.name,
        }
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::D2(_) => 2,
            Self::D3(_) => 3,
        }
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        match self {
            Self::D2(m) => m.num_cells(),
            Self::D3(m) => m.num_cells(),
        }
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        match self {
            Self::D2(m) => m.num_nodes(),
            Self::D3(m) => m.num_nodes(),
        }
    }
}

/// 2D 结构化网格（单元数 `nx × ny`）。
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredMesh2d {
    pub name: String,
    pub nx: usize,
    pub ny: usize,
    pub points_x: Vec<f64>,
    pub points_y: Vec<f64>,
}

impl StructuredMesh2d {
    pub fn new(
        name: impl Into<String>,
        nx: usize,
        ny: usize,
        points_x: Vec<f64>,
        points_y: Vec<f64>,
    ) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(AsimuError::Mesh("nx 与 ny 必须大于 0".to_string()));
        }
        let expected = (nx + 1) * (ny + 1);
        if points_x.len() != expected || points_y.len() != expected {
            return Err(AsimuError::Mesh(format!(
                "节点坐标长度应为 {expected}，实际 x={} y={}",
                points_x.len(),
                points_y.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            nx,
            ny,
            points_x,
            points_y,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.points_x.len()
    }

    #[must_use]
    pub fn node_x(&self, i: usize, j: usize) -> f64 {
        self.points_x[i + j * (self.nx + 1)]
    }

    #[must_use]
    pub fn node_y(&self, i: usize, j: usize) -> f64 {
        self.points_y[i + j * (self.nx + 1)]
    }
}

/// 3D 结构化网格（单元数 `nx × ny × nz`）。
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredMesh3d {
    pub name: String,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub points_x: Vec<f64>,
    pub points_y: Vec<f64>,
    pub points_z: Vec<f64>,
}

impl StructuredMesh3d {
    pub fn new(
        name: impl Into<String>,
        nx: usize,
        ny: usize,
        nz: usize,
        points_x: Vec<f64>,
        points_y: Vec<f64>,
        points_z: Vec<f64>,
    ) -> Result<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(AsimuError::Mesh("nx、ny、nz 必须大于 0".to_string()));
        }
        let expected = (nx + 1) * (ny + 1) * (nz + 1);
        if points_x.len() != expected || points_y.len() != expected || points_z.len() != expected {
            return Err(AsimuError::Mesh(format!(
                "节点坐标长度应为 {expected}，实际 x={} y={} z={}",
                points_x.len(),
                points_y.len(),
                points_z.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            nx,
            ny,
            nz,
            points_x,
            points_y,
            points_z,
        })
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.points_x.len()
    }

    #[must_use]
    pub fn node_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)
    }

    #[must_use]
    pub fn node_x(&self, i: usize, j: usize, k: usize) -> f64 {
        self.points_x[self.node_index(i, j, k)]
    }

    #[must_use]
    pub fn node_y(&self, i: usize, j: usize, k: usize) -> f64 {
        self.points_y[self.node_index(i, j, k)]
    }

    #[must_use]
    pub fn node_z(&self, i: usize, j: usize, k: usize) -> f64 {
        self.points_z[self.node_index(i, j, k)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square_2x2() -> StructuredMesh2d {
        let nx = 2;
        let ny = 2;
        let mut px = Vec::new();
        let mut py = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                px.push(i as f64);
                py.push(j as f64);
            }
        }
        StructuredMesh2d::new("unit", nx, ny, px, py).expect("mesh")
    }

    #[test]
    fn stores_node_coordinates() {
        let mesh = unit_square_2x2();
        assert_eq!(mesh.num_cells(), 4);
        assert_eq!(mesh.num_nodes(), 9);
        assert_eq!(mesh.node_x(2, 2), 2.0);
        assert_eq!(mesh.node_y(1, 0), 0.0);
    }
}
