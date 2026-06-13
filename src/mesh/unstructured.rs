//! 混合单元非结构 3D 网格（M1：拓扑构造 + 几何度量）。
//!
//! 节点顺序遵循 VTK 线性单元约定（与 VTU `types` 一致），见各 `CellKind` 文档。

#[path = "unstructured_geometry.rs"]
mod geometry;
#[path = "unstructured_templates.rs"]
mod templates;

#[cfg(test)]
#[path = "unstructured_tests.rs"]
mod tests;

use std::collections::HashMap;

use crate::core::{CellId, FaceId, NodeId, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::mesh::boundary::BoundaryMesh;

use super::metrics::{CellMetric, FaceMetric};
use super::{BoundaryMesh3d, FaceGeometry3d};
use geometry::{
    cell_center, orient_metric_outward_from, quad_face_metric, reverse_face_nodes, tri_face_metric,
    volume_from_outward_faces,
};
use templates::{LocalFaceSpec, local_faces};

/// 支持的 3D 线性单元类型（VTK：TET=10, HEX=12, WEDGE=13, PYRAMID=14）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellKind {
    Tet,
    Hex,
    Pyramid,
    Prism,
}

impl CellKind {
    #[must_use]
    pub const fn node_count(self) -> usize {
        match self {
            Self::Tet => 4,
            Self::Hex => 8,
            Self::Pyramid => 5,
            Self::Prism => 6,
        }
    }

    #[must_use]
    pub const fn vtk_type(self) -> u8 {
        match self {
            Self::Tet => 10,
            Self::Hex => 12,
            Self::Prism => 13,
            Self::Pyramid => 14,
        }
    }

    pub fn from_vtk_type(value: u8) -> Result<Self> {
        match value {
            10 => Ok(Self::Tet),
            12 => Ok(Self::Hex),
            13 => Ok(Self::Prism),
            14 => Ok(Self::Pyramid),
            other => Err(AsimuError::Mesh(format!(
                "不支持的 VTK 单元类型 {other}（M1 支持 10/12/13/14）"
            ))),
        }
    }

    /// CGNS `ElementType_t`（与 `io::cgns::ffi` 常量一致）。
    #[cfg(feature = "io-cgns")]
    #[must_use]
    pub const fn cgns_element_type(self) -> i32 {
        match self {
            Self::Tet => 10,
            Self::Pyramid => 12,
            Self::Prism => 14,
            Self::Hex => 17,
        }
    }
}

/// 单个非结构单元（全局节点索引）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnstructuredCell {
    pub kind: CellKind,
    pub nodes: Vec<NodeId>,
}

impl UnstructuredCell {
    pub fn new(kind: CellKind, nodes: Vec<usize>) -> Result<Self> {
        let expected = kind.node_count();
        if nodes.len() != expected {
            return Err(AsimuError::Mesh(format!(
                "{kind:?} 须 {expected} 个节点，实际 {} 个",
                nodes.len()
            )));
        }
        Ok(Self {
            kind,
            nodes: nodes
                .into_iter()
                .map(|index| {
                    u32::try_from(index)
                        .map(NodeId)
                        .map_err(|_| AsimuError::Mesh(format!("节点索引 {index} 超出 u32 范围")))
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

/// 混合单元非结构 3D 网格（构造期完成面拓扑与度量）。
#[derive(Debug, Clone, PartialEq)]
pub struct UnstructuredMesh3d {
    name: String,
    points: Vec<[Real; 3]>,
    cells: Vec<UnstructuredCell>,
    face_nodes: Vec<Vec<usize>>,
    face_owner: Vec<CellId>,
    face_neighbor: Vec<Option<CellId>>,
    cell_metrics: Vec<CellMetric>,
    face_metrics: Vec<FaceMetric>,
}

impl UnstructuredMesh3d {
    pub fn new(
        name: impl Into<String>,
        points: Vec<[Real; 3]>,
        cells: Vec<UnstructuredCell>,
    ) -> Result<Self> {
        if points.is_empty() {
            return Err(AsimuError::Mesh("非结构网格缺少节点".to_string()));
        }
        if cells.is_empty() {
            return Err(AsimuError::Mesh("非结构网格缺少单元".to_string()));
        }
        validate_points(&points)?;
        validate_cells(&points, &cells)?;
        let topology = build_face_topology(&points, &cells)?;
        Ok(Self {
            name: name.into(),
            points,
            cells,
            face_nodes: topology.face_nodes,
            face_owner: topology.face_owner,
            face_neighbor: topology.face_neighbor,
            cell_metrics: topology.cell_metrics,
            face_metrics: topology.face_metrics,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn points(&self) -> &[[Real; 3]] {
        &self.points
    }

    pub fn scale_coordinates(&mut self, factor: Real) -> Result<()> {
        if factor <= 0.0 || !factor.is_finite() {
            return Err(AsimuError::Mesh(format!(
                "非结构网格缩放因子必须为正且有限，实际 {factor}"
            )));
        }
        for point in &mut self.points {
            point[0] *= factor;
            point[1] *= factor;
            point[2] *= factor;
        }
        let area_scale = factor * factor;
        let volume_scale = area_scale * factor;
        for metric in &mut self.cell_metrics {
            metric.volume *= volume_scale;
            metric.center = scale_vector(metric.center, factor);
        }
        for metric in &mut self.face_metrics {
            metric.area_vector = scale_vector(metric.area_vector, area_scale);
            metric.area *= area_scale;
            metric.center = scale_vector(metric.center, factor);
        }
        Ok(())
    }

    #[must_use]
    pub fn cells(&self) -> &[UnstructuredCell] {
        &self.cells
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.points.len()
    }

    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.face_owner.len()
    }

    #[must_use]
    pub fn cell_kind(&self, cell: CellId) -> CellKind {
        self.cells[cell.index() as usize].kind
    }

    #[must_use]
    pub fn cell_metric(&self, cell: CellId) -> &CellMetric {
        &self.cell_metrics[cell.index() as usize]
    }

    #[must_use]
    pub fn face_metric(&self, face: FaceId) -> &FaceMetric {
        &self.face_metrics[face.index() as usize]
    }

    pub fn face_owner(&self, face: FaceId) -> Result<CellId> {
        self.face_owner
            .get(face.index() as usize)
            .copied()
            .ok_or_else(|| AsimuError::Mesh(format!("无效 FaceId({})", face.index())))
    }

    pub fn face_neighbor(&self, face: FaceId) -> Result<Option<CellId>> {
        self.face_neighbor
            .get(face.index() as usize)
            .copied()
            .ok_or_else(|| AsimuError::Mesh(format!("无效 FaceId({})", face.index())))
    }

    pub fn face_node_indices(&self, face: FaceId) -> Result<&[usize]> {
        self.face_nodes
            .get(face.index() as usize)
            .map(|nodes| nodes.as_slice())
            .ok_or_else(|| AsimuError::Mesh(format!("无效 FaceId({})", face.index())))
    }

    #[must_use]
    pub fn cell_volumes(&self) -> Vec<Real> {
        self.cell_metrics
            .iter()
            .map(|metric| metric.volume)
            .collect()
    }
}

impl BoundaryMesh for UnstructuredMesh3d {
    fn num_cells(&self) -> usize {
        self.num_cells()
    }

    fn face_owner(&self, face: FaceId) -> Result<CellId> {
        UnstructuredMesh3d::face_owner(self, face)
    }

    fn face_spacing(&self, face: FaceId) -> Result<Real> {
        let owner = self.face_owner(face)?;
        let owner_metric = self.cell_metric(owner);
        let face_metric = self.face_metric(face);
        if face_metric.area <= Real::EPSILON {
            Ok(Real::INFINITY)
        } else {
            Ok(Vector3::new(
                owner_metric.center.x - face_metric.center.x,
                owner_metric.center.y - face_metric.center.y,
                owner_metric.center.z - face_metric.center.z,
            )
            .magnitude())
        }
    }

    fn face_outward_normal(&self, face: FaceId) -> Result<Real> {
        Ok(self.face_metric(face).normal.x)
    }

    fn resolve_logical_boundary(&self, name: &str) -> Result<Vec<FaceId>> {
        Err(AsimuError::Mesh(format!(
            "非结构网格不支持逻辑边界名 \"{name}\"；请使用显式 BoundaryPatch"
        )))
    }
}

impl BoundaryMesh3d for UnstructuredMesh3d {
    fn face_geometry_3d(&self, face: FaceId) -> Result<FaceGeometry3d> {
        let metric = self.face_metric(face);
        Ok(FaceGeometry3d {
            normal: metric.normal,
            spacing: self.face_spacing(face)?,
            area: metric.area,
            center: metric.center,
        })
    }
}

struct TopologyBuild {
    face_nodes: Vec<Vec<usize>>,
    face_owner: Vec<CellId>,
    face_neighbor: Vec<Option<CellId>>,
    cell_metrics: Vec<CellMetric>,
    face_metrics: Vec<FaceMetric>,
}

struct FaceTopologyLists {
    face_nodes: Vec<Vec<usize>>,
    face_owner: Vec<CellId>,
    face_neighbor: Vec<Option<CellId>>,
    face_metrics: Vec<FaceMetric>,
}

#[derive(Debug, Clone)]
struct FaceHit {
    nodes: Vec<usize>,
    cell: usize,
}

fn validate_points(points: &[[Real; 3]]) -> Result<()> {
    for (index, point) in points.iter().enumerate() {
        if !point[0].is_finite() || !point[1].is_finite() || !point[2].is_finite() {
            return Err(AsimuError::Mesh(format!("节点 {index} 坐标非有限")));
        }
    }
    Ok(())
}

fn validate_cells(points: &[[Real; 3]], cells: &[UnstructuredCell]) -> Result<()> {
    let max_node = points.len();
    for (cell_index, cell) in cells.iter().enumerate() {
        if cell.nodes.len() != cell.kind.node_count() {
            return Err(AsimuError::Mesh(format!(
                "单元 {cell_index} 节点数与 {kind:?} 不符",
                kind = cell.kind
            )));
        }
        for node in &cell.nodes {
            if node.index() as usize >= max_node {
                return Err(AsimuError::Mesh(format!(
                    "单元 {cell_index} 节点索引 {} 越界（共 {max_node} 个节点）",
                    node.index()
                )));
            }
        }
    }
    Ok(())
}

fn scale_vector(vector: Vector3, factor: Real) -> Vector3 {
    Vector3::new(vector.x * factor, vector.y * factor, vector.z * factor)
}

fn build_face_topology(points: &[[Real; 3]], cells: &[UnstructuredCell]) -> Result<TopologyBuild> {
    let cell_metrics = build_cell_metrics(points, cells)?;
    let registry = build_face_registry(points, cells)?;
    let faces = finalize_faces_from_registry(points, &registry)?;
    Ok(TopologyBuild {
        face_nodes: faces.face_nodes,
        face_owner: faces.face_owner,
        face_neighbor: faces.face_neighbor,
        cell_metrics,
        face_metrics: faces.face_metrics,
    })
}

fn build_cell_metrics(points: &[[Real; 3]], cells: &[UnstructuredCell]) -> Result<Vec<CellMetric>> {
    let mut cell_metrics = Vec::with_capacity(cells.len());
    for (cell_index, cell) in cells.iter().enumerate() {
        let local_metrics = local_face_metrics(points, cell)?;
        let volume = volume_from_outward_faces(&local_metrics);
        if volume <= Real::EPSILON {
            return Err(AsimuError::Mesh(format!(
                "单元 {cell_index} 体积非正: {volume}"
            )));
        }
        let node_indices: Vec<usize> = cell.nodes.iter().map(|n| n.index() as usize).collect();
        cell_metrics.push(CellMetric {
            volume,
            center: cell_center(points, &node_indices),
        });
    }
    Ok(cell_metrics)
}

fn build_face_registry(
    points: &[[Real; 3]],
    cells: &[UnstructuredCell],
) -> Result<HashMap<Vec<usize>, Vec<FaceHit>>> {
    let mut registry: HashMap<Vec<usize>, Vec<FaceHit>> = HashMap::new();
    for (cell_index, cell) in cells.iter().enumerate() {
        let global_nodes: Vec<usize> = cell.nodes.iter().map(|n| n.index() as usize).collect();
        let center = cell_center(points, &global_nodes);
        for spec in local_faces(cell.kind) {
            let mut nodes = global_face_nodes(&global_nodes, *spec);
            let metric = face_metric_from_nodes(points, &nodes)?;
            if scalar_dot(metric.area_vector, vec_to_face(metric.center, center)) < 0.0 {
                nodes = reverse_face_nodes(&nodes);
            }
            let key = face_key(&nodes);
            registry.entry(key).or_default().push(FaceHit {
                nodes,
                cell: cell_index,
            });
        }
    }
    Ok(registry)
}

fn finalize_faces_from_registry(
    points: &[[Real; 3]],
    registry: &HashMap<Vec<usize>, Vec<FaceHit>>,
) -> Result<FaceTopologyLists> {
    let mut face_nodes = Vec::new();
    let mut face_owner = Vec::new();
    let mut face_neighbor = Vec::new();
    let mut face_metrics = Vec::new();

    for hits in registry.values() {
        match hits.len() {
            1 => push_boundary_face(
                points,
                &hits[0],
                &mut face_nodes,
                &mut face_owner,
                &mut face_neighbor,
                &mut face_metrics,
            )?,
            2 => push_interior_face(
                points,
                &hits[0],
                &hits[1],
                &mut face_nodes,
                &mut face_owner,
                &mut face_neighbor,
                &mut face_metrics,
            )?,
            count => {
                return Err(AsimuError::Mesh(format!(
                    "面 {count} 个单元共享同一节点集（非流形），节点键 {:?}",
                    face_key(&hits[0].nodes)
                )));
            }
        }
    }

    Ok(FaceTopologyLists {
        face_nodes,
        face_owner,
        face_neighbor,
        face_metrics,
    })
}

fn push_boundary_face(
    points: &[[Real; 3]],
    hit: &FaceHit,
    face_nodes: &mut Vec<Vec<usize>>,
    face_owner: &mut Vec<CellId>,
    face_neighbor: &mut Vec<Option<CellId>>,
    face_metrics: &mut Vec<FaceMetric>,
) -> Result<()> {
    let metric = face_metric_from_nodes(points, &hit.nodes)?;
    if metric.area <= Real::EPSILON {
        return Err(AsimuError::Mesh(format!(
            "边界面单元 {} 面积非正",
            hit.cell
        )));
    }
    face_nodes.push(hit.nodes.clone());
    face_owner.push(cell_id_from_index(hit.cell)?);
    face_neighbor.push(None);
    face_metrics.push(metric);
    Ok(())
}

fn push_interior_face(
    points: &[[Real; 3]],
    left_hit: &FaceHit,
    right_hit: &FaceHit,
    face_nodes: &mut Vec<Vec<usize>>,
    face_owner: &mut Vec<CellId>,
    face_neighbor: &mut Vec<Option<CellId>>,
    face_metrics: &mut Vec<FaceMetric>,
) -> Result<()> {
    let (left, right) = orient_interior_pair(left_hit, right_hit)?;
    let metric = face_metric_from_nodes(points, &left.nodes)?;
    if metric.area <= Real::EPSILON {
        return Err(AsimuError::Mesh(format!(
            "内部面单元 {}-{} 面积非正",
            left.cell, right.cell
        )));
    }
    face_nodes.push(left.nodes.clone());
    face_owner.push(cell_id_from_index(left.cell)?);
    face_neighbor.push(Some(cell_id_from_index(right.cell)?));
    face_metrics.push(metric);
    Ok(())
}

fn cell_id_from_index(cell_index: usize) -> Result<CellId> {
    u32::try_from(cell_index)
        .map(CellId)
        .map_err(|_| AsimuError::Mesh("单元索引超出 CellId 范围".to_string()))
}

fn local_face_metrics(points: &[[Real; 3]], cell: &UnstructuredCell) -> Result<Vec<FaceMetric>> {
    let global: Vec<usize> = cell.nodes.iter().map(|n| n.index() as usize).collect();
    let center = cell_center(points, &global);
    local_faces(cell.kind)
        .iter()
        .map(|spec| {
            let nodes = global_face_nodes(&global, *spec);
            let metric = face_metric_from_nodes(points, &nodes)?;
            Ok(orient_metric_outward_from(metric, center))
        })
        .collect()
}

fn global_face_nodes(global: &[usize], spec: LocalFaceSpec) -> Vec<usize> {
    match spec {
        LocalFaceSpec::Tri([a, b, c]) => vec![global[a], global[b], global[c]],
        LocalFaceSpec::Quad([a, b, c, d]) => vec![global[a], global[b], global[c], global[d]],
    }
}

fn face_key(nodes: &[usize]) -> Vec<usize> {
    let mut key: Vec<usize> = nodes.to_vec();
    key.sort_unstable();
    key
}

fn face_metric_from_nodes(points: &[[Real; 3]], nodes: &[usize]) -> Result<FaceMetric> {
    match nodes.len() {
        3 => Ok(tri_face_metric(points, [nodes[0], nodes[1], nodes[2]])),
        4 => Ok(quad_face_metric(
            points,
            [nodes[0], nodes[1], nodes[2], nodes[3]],
        )),
        n => Err(AsimuError::Mesh(format!("面节点数 {n} 无效"))),
    }
}

fn vec_to_face(face_center: Vector3, cell_center: Vector3) -> Vector3 {
    Vector3::new(
        face_center.x - cell_center.x,
        face_center.y - cell_center.y,
        face_center.z - cell_center.z,
    )
}

fn scalar_dot(a: Vector3, b: Vector3) -> Real {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn orient_interior_pair(left: &FaceHit, right: &FaceHit) -> Result<(FaceHit, FaceHit)> {
    if left.cell == right.cell {
        return Err(AsimuError::Mesh("内部面两侧单元相同".to_string()));
    }
    if face_key(&left.nodes) != face_key(&right.nodes) {
        return Err(AsimuError::Mesh("内部面节点集不一致".to_string()));
    }
    if left.cell <= right.cell {
        Ok((left.clone(), right.clone()))
    } else {
        Ok((right.clone(), left.clone()))
    }
}
