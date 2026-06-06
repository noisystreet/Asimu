//! CGNS Unstructured zone 读入。

use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

use crate::boundary::{BoundaryKind, BoundaryPatch, BoundarySet};
use crate::core::FaceId;
use crate::error::{AsimuError, Result};
use crate::io::limits::validate_cell_count;
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

use super::ffi::{
    BC_ELEMENT_LIST, BC_ELEMENT_RANGE, BC_POINT_LIST, BC_POINT_RANGE, CgSize, ELEM_HEXA_8,
    ELEM_MIXED, ELEM_PENTA_6, ELEM_PYRA_5, ELEM_QUAD_4, ELEM_TETRA_4, ELEM_TRI_3,
    GRID_LOCATION_FACE_CENTER, REAL_DOUBLE, ZONE_UNSTRUCTURED, cg_ElementDataSize,
    cg_boco_gridlocation_read, cg_boco_info, cg_boco_read, cg_coord_read, cg_elements_read,
    cg_nbocos, cg_nsections, cg_section_read, cg_zone_read,
};
use super::read::{CGNS_LOCK, CgnsFile, CgnsZoneInfo, c_str_to_string, cgns_lock_error, check_cg};

/// CGNS 非结构单 zone 读入结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CgnsUnstructuredLoadResult {
    pub zone: CgnsZoneInfo,
    pub mesh: UnstructuredMesh3d,
    pub boundary: BoundarySet,
}

struct CgnsSectionInfo {
    name: String,
    element_type: i32,
    start: CgSize,
    end: CgSize,
}

struct CgnsUnstructuredElements {
    cells: Vec<UnstructuredCell>,
    boundary_faces: Vec<CgnsBoundaryFaceElement>,
}

struct CgnsBoundaryFaceElement {
    element_id: CgSize,
    nodes: Vec<usize>,
}

/// 读取指定 CGNS Unstructured zone（1-based）为 `UnstructuredMesh3d`。
pub fn load_cgns_unstructured_zone(
    path: &Path,
    zone_index: usize,
) -> Result<CgnsUnstructuredLoadResult> {
    let _guard = CGNS_LOCK.lock().map_err(|_| cgns_lock_error())?;
    load_cgns_unstructured_zone_locked(path, zone_index)
}

fn load_cgns_unstructured_zone_locked(
    path: &Path,
    zone_index: usize,
) -> Result<CgnsUnstructuredLoadResult> {
    const BASE: i32 = 1;
    if zone_index == 0 {
        return Err(AsimuError::Mesh("zone_index 从 1 开始".to_string()));
    }
    let file = CgnsFile::open(path)?;
    let nzones = file.nzones(BASE)?;
    if zone_index > nzones {
        return Err(AsimuError::Mesh(format!(
            "zone_index={zone_index} 超出范围（共 {nzones} 个 zone）"
        )));
    }
    let zone = zone_index as i32;
    let info = unstructured_zone_info(&file, BASE, zone)?;
    let node_count = read_unstructured_node_count(&file, BASE, zone)?;
    let points = file.read_unstructured_points(BASE, zone, node_count)?;
    let elements = file.read_unstructured_elements(BASE, zone)?;
    let mesh_name = if info.name.is_empty() {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("cgns_unstructured")
            .to_string()
    } else {
        info.name.clone()
    };
    let mesh = UnstructuredMesh3d::new(mesh_name, points, elements.cells)?;
    let boundary =
        file.read_unstructured_zone_bocos(BASE, zone, &mesh, &elements.boundary_faces)?;
    Ok(CgnsUnstructuredLoadResult {
        zone: info,
        mesh,
        boundary,
    })
}

pub(super) fn unstructured_zone_info(
    file: &CgnsFile,
    base: i32,
    zone: i32,
) -> Result<CgnsZoneInfo> {
    let mut name = [0i8; 33];
    let mut sizes = [0 as CgSize; 3];
    check_cg(unsafe {
        cg_zone_read(
            file.index,
            base,
            zone,
            name.as_mut_ptr(),
            sizes.as_mut_ptr(),
        )
    })?;
    let zone_type = file.zone_type(base, zone)?;
    if zone_type != ZONE_UNSTRUCTURED {
        return Err(AsimuError::Mesh(format!(
            "zone {zone} 非 Unstructured 类型（type={zone_type}）"
        )));
    }
    let vertices = usize::try_from(sizes[0])
        .map_err(|_| AsimuError::Mesh(format!("zone {zone} 顶点数无效: {}", sizes[0])))?;
    let cells = usize::try_from(sizes[1])
        .map_err(|_| AsimuError::Mesh(format!("zone {zone} 单元数无效: {}", sizes[1])))?;
    if vertices == 0 {
        return Err(AsimuError::Mesh(format!("zone {zone} 顶点数为 0")));
    }
    validate_cell_count(cells as u64)?;
    Ok(CgnsZoneInfo {
        index: zone as usize,
        name: c_str_to_string(&name)?,
        nx: cells,
        ny: 1,
        nz: 1,
    })
}

impl CgnsFile {
    fn read_unstructured_points(
        &self,
        base: i32,
        zone: i32,
        node_count: usize,
    ) -> Result<Vec<[f64; 3]>> {
        let mut points_x = vec![0.0; node_count];
        let mut points_y = vec![0.0; node_count];
        let mut points_z = vec![0.0; node_count];
        let rmin = [1 as CgSize];
        let rmax = [node_count as CgSize];
        for (coord, buf) in [
            ("CoordinateX", &mut points_x),
            ("CoordinateY", &mut points_y),
            ("CoordinateZ", &mut points_z),
        ] {
            let cname = CString::new(coord).expect("static coord name");
            check_cg(unsafe {
                cg_coord_read(
                    self.index,
                    base,
                    zone,
                    cname.as_ptr(),
                    REAL_DOUBLE,
                    rmin.as_ptr(),
                    rmax.as_ptr(),
                    buf.as_mut_ptr().cast(),
                )
            })?;
        }
        Ok(points_x
            .into_iter()
            .zip(points_y)
            .zip(points_z)
            .map(|((x, y), z)| [x, y, z])
            .collect())
    }

    fn read_unstructured_elements(&self, base: i32, zone: i32) -> Result<CgnsUnstructuredElements> {
        let mut nsections = 0;
        check_cg(unsafe { cg_nsections(self.index, base, zone, &mut nsections) })?;
        let mut cells = Vec::new();
        let mut boundary_faces = Vec::new();
        for section in 1..=nsections {
            let info = self.section_info(base, zone, section)?;
            let mut data_size = 0 as CgSize;
            check_cg(unsafe {
                cg_ElementDataSize(self.index, base, zone, section, &mut data_size)
            })?;
            let mut elements = vec![
                0 as CgSize;
                usize::try_from(data_size).map_err(|_| {
                    AsimuError::Mesh(format!(
                        "section {} element data size 无效: {data_size}",
                        info.name
                    ))
                })?
            ];
            check_cg(unsafe {
                cg_elements_read(
                    self.index,
                    base,
                    zone,
                    section,
                    elements.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })?;
            append_section_elements(&info, &elements, &mut cells, &mut boundary_faces)?;
        }
        if cells.is_empty() {
            return Err(AsimuError::Mesh(
                "CGNS Unstructured zone 未读到支持的体单元 sections".to_string(),
            ));
        }
        Ok(CgnsUnstructuredElements {
            cells,
            boundary_faces,
        })
    }

    fn section_info(&self, base: i32, zone: i32, section: i32) -> Result<CgnsSectionInfo> {
        let mut name = [0i8; 33];
        let mut element_type = 0;
        let mut start = 0 as CgSize;
        let mut end = 0 as CgSize;
        let mut nbndry = 0;
        let mut parent_flag = 0;
        check_cg(unsafe {
            cg_section_read(
                self.index,
                base,
                zone,
                section,
                name.as_mut_ptr(),
                &mut element_type,
                &mut start,
                &mut end,
                &mut nbndry,
                &mut parent_flag,
            )
        })?;
        Ok(CgnsSectionInfo {
            name: c_str_to_string(&name)?,
            element_type,
            start,
            end,
        })
    }

    fn read_unstructured_zone_bocos(
        &self,
        base: i32,
        zone: i32,
        mesh: &UnstructuredMesh3d,
        boundary_faces: &[CgnsBoundaryFaceElement],
    ) -> Result<BoundarySet> {
        let element_to_face = build_element_to_face_map(mesh, boundary_faces)?;
        let families = self.read_family_bc_map(base)?;
        let mut nbocos = 0;
        check_cg(unsafe { cg_nbocos(self.index, base, zone, &mut nbocos) })?;
        let mut patches = Vec::with_capacity(nbocos as usize);
        for boco in 1..=nbocos {
            let info = self.boco_info(base, zone, boco)?;
            let location = self.boco_grid_location(base, zone, boco)?;
            if location != GRID_LOCATION_FACE_CENTER {
                continue;
            }
            let element_ids =
                self.read_boco_element_ids(base, zone, boco, info.ptset_type, info.npnts)?;
            let face_ids = map_boco_elements_to_faces(&info.name, &element_ids, &element_to_face)?;
            let resolved =
                self.resolve_boco_type(base, zone, boco, info.bocotype, &families, &info.name)?;
            patches.push(BoundaryPatch::new(
                info.name,
                face_ids,
                BoundaryKind::from_cgns_bctype(resolved.bctype, &resolved.label),
            ));
        }
        Ok(BoundarySet::new(patches))
    }

    fn boco_info(&self, base: i32, zone: i32, boco: i32) -> Result<CgnsBocoInfo> {
        let mut name = [0i8; 33];
        let mut bocotype = 0;
        let mut ptset_type = 0;
        let mut npnts = 0 as CgSize;
        let mut normalindex = 0;
        let mut normal_list_size = 0 as CgSize;
        let mut normaldatatype = 0;
        let mut ndataset = 0;
        check_cg(unsafe {
            cg_boco_info(
                self.index,
                base,
                zone,
                boco,
                name.as_mut_ptr(),
                &mut bocotype,
                &mut ptset_type,
                &mut npnts,
                &mut normalindex,
                &mut normal_list_size,
                &mut normaldatatype,
                &mut ndataset,
            )
        })?;
        Ok(CgnsBocoInfo {
            name: c_str_to_string(&name)?,
            bocotype,
            ptset_type,
            npnts,
        })
    }

    fn boco_grid_location(&self, base: i32, zone: i32, boco: i32) -> Result<i32> {
        let mut location = 0;
        check_cg(unsafe {
            cg_boco_gridlocation_read(self.index, base, zone, boco, &mut location)
        })?;
        Ok(location)
    }

    fn read_boco_element_ids(
        &self,
        base: i32,
        zone: i32,
        boco: i32,
        ptset_type: i32,
        npnts: CgSize,
    ) -> Result<Vec<CgSize>> {
        let len = boco_point_buffer_len(ptset_type, npnts)?;
        let mut pnts = vec![0 as CgSize; len];
        check_cg(unsafe {
            cg_boco_read(
                self.index,
                base,
                zone,
                boco,
                pnts.as_mut_ptr().cast(),
                std::ptr::null_mut(),
            )
        })?;
        match ptset_type {
            BC_POINT_RANGE | BC_ELEMENT_RANGE => expand_element_range(&pnts),
            BC_POINT_LIST | BC_ELEMENT_LIST => Ok(pnts),
            other => Err(AsimuError::Boundary(format!(
                "CGNS 非结构 ZoneBC PointSetType {other} 暂不支持"
            ))),
        }
    }
}

struct CgnsBocoInfo {
    name: String,
    bocotype: i32,
    ptset_type: i32,
    npnts: CgSize,
}

fn read_unstructured_node_count(file: &CgnsFile, base: i32, zone: i32) -> Result<usize> {
    let mut name = [0i8; 33];
    let mut sizes = [0 as CgSize; 3];
    check_cg(unsafe {
        cg_zone_read(
            file.index,
            base,
            zone,
            name.as_mut_ptr(),
            sizes.as_mut_ptr(),
        )
    })?;
    usize::try_from(sizes[0])
        .map_err(|_| AsimuError::Mesh(format!("Unstructured zone 顶点数无效: {}", sizes[0])))
}

fn append_section_elements(
    info: &CgnsSectionInfo,
    elements: &[CgSize],
    cells: &mut Vec<UnstructuredCell>,
    boundary_faces: &mut Vec<CgnsBoundaryFaceElement>,
) -> Result<()> {
    if info.element_type == ELEM_MIXED {
        append_mixed_section_elements(info, elements, cells, boundary_faces)
    } else if let Some(kind) = cell_kind_from_cgns_element(info.element_type) {
        append_fixed_section_cells(info, elements, kind, cells)
    } else if is_boundary_element_section(info.element_type) {
        append_fixed_boundary_faces(info, elements, boundary_faces)
    } else {
        Err(AsimuError::Mesh(format!(
            "section {} 含不支持的 CGNS ElementType {}",
            info.name, info.element_type
        )))
    }
}

fn append_fixed_section_cells(
    info: &CgnsSectionInfo,
    elements: &[CgSize],
    kind: CellKind,
    cells: &mut Vec<UnstructuredCell>,
) -> Result<()> {
    let count = section_element_count(info)?;
    let npe = kind.node_count();
    let expected = count
        .checked_mul(npe)
        .ok_or_else(|| AsimuError::Mesh(format!("section {} connectivity 长度溢出", info.name)))?;
    if elements.len() != expected {
        return Err(AsimuError::Mesh(format!(
            "section {} connectivity 长度应为 {expected}，实际 {}",
            info.name,
            elements.len()
        )));
    }
    for chunk in elements.chunks_exact(npe) {
        cells.push(UnstructuredCell::new(
            kind,
            cgns_nodes_to_zero_based(chunk)?,
        )?);
    }
    Ok(())
}

fn append_fixed_boundary_faces(
    info: &CgnsSectionInfo,
    elements: &[CgSize],
    boundary_faces: &mut Vec<CgnsBoundaryFaceElement>,
) -> Result<()> {
    let count = section_element_count(info)?;
    let npe = cgns_element_node_count(info.element_type).expect("boundary type checked");
    let expected = count
        .checked_mul(npe)
        .ok_or_else(|| AsimuError::Mesh(format!("section {} connectivity 长度溢出", info.name)))?;
    if elements.len() != expected {
        return Err(AsimuError::Mesh(format!(
            "section {} boundary connectivity 长度应为 {expected}，实际 {}",
            info.name,
            elements.len()
        )));
    }
    for (offset, chunk) in elements.chunks_exact(npe).enumerate() {
        boundary_faces.push(CgnsBoundaryFaceElement {
            element_id: info.start + offset as CgSize,
            nodes: cgns_nodes_to_zero_based(chunk)?,
        });
    }
    Ok(())
}

fn append_mixed_section_elements(
    info: &CgnsSectionInfo,
    elements: &[CgSize],
    cells: &mut Vec<UnstructuredCell>,
    boundary_faces: &mut Vec<CgnsBoundaryFaceElement>,
) -> Result<()> {
    let expected_cells = section_element_count(info)?;
    let mut parsed_cells = 0usize;
    let mut pos = 0usize;
    while pos < elements.len() {
        let elem_type = i32::try_from(elements[pos]).map_err(|_| {
            AsimuError::Mesh(format!(
                "section {} MIXED ElementType 无效: {}",
                info.name, elements[pos]
            ))
        })?;
        pos += 1;
        let Some(npe) = cgns_element_node_count(elem_type) else {
            return Err(AsimuError::Mesh(format!(
                "section {} MIXED 含不支持 ElementType {elem_type}",
                info.name
            )));
        };
        let end = pos.checked_add(npe).ok_or_else(|| {
            AsimuError::Mesh(format!("section {} MIXED connectivity 溢出", info.name))
        })?;
        if end > elements.len() {
            return Err(AsimuError::Mesh(format!(
                "section {} MIXED 单元 connectivity 超出数组",
                info.name
            )));
        }
        if let Some(kind) = cell_kind_from_cgns_element(elem_type) {
            cells.push(UnstructuredCell::new(
                kind,
                cgns_nodes_to_zero_based(&elements[pos..end])?,
            )?);
        } else if is_boundary_element_section(elem_type) {
            boundary_faces.push(CgnsBoundaryFaceElement {
                element_id: info.start + parsed_cells as CgSize,
                nodes: cgns_nodes_to_zero_based(&elements[pos..end])?,
            });
        }
        parsed_cells += 1;
        pos = end;
    }
    if parsed_cells != expected_cells {
        return Err(AsimuError::Mesh(format!(
            "section {} MIXED 元素数应为 {expected_cells}，实际 {parsed_cells}",
            info.name
        )));
    }
    Ok(())
}

fn build_element_to_face_map(
    mesh: &UnstructuredMesh3d,
    boundary_faces: &[CgnsBoundaryFaceElement],
) -> Result<HashMap<CgSize, FaceId>> {
    let mut key_to_face = HashMap::new();
    for face in 0..mesh.num_faces() {
        let face_id = FaceId(face as u32);
        if mesh.face_neighbor(face_id)?.is_some() {
            continue;
        }
        let mut key = mesh.face_node_indices(face_id)?.to_vec();
        key.sort_unstable();
        key_to_face.insert(key, face_id);
    }

    let mut element_to_face = HashMap::with_capacity(boundary_faces.len());
    for element in boundary_faces {
        let mut key = element.nodes.clone();
        key.sort_unstable();
        let Some(&face_id) = key_to_face.get(&key) else {
            return Err(AsimuError::Boundary(format!(
                "CGNS boundary element {} 未匹配到网格边界面",
                element.element_id
            )));
        };
        element_to_face.insert(element.element_id, face_id);
    }
    Ok(element_to_face)
}

fn map_boco_elements_to_faces(
    boco_name: &str,
    element_ids: &[CgSize],
    element_to_face: &HashMap<CgSize, FaceId>,
) -> Result<Vec<FaceId>> {
    let mut faces = Vec::with_capacity(element_ids.len());
    for &element_id in element_ids {
        let Some(&face) = element_to_face.get(&element_id) else {
            return Err(AsimuError::Boundary(format!(
                "CGNS BC {boco_name} 引用的 boundary element {element_id} 不存在"
            )));
        };
        faces.push(face);
    }
    Ok(faces)
}

fn boco_point_buffer_len(ptset_type: i32, npnts: CgSize) -> Result<usize> {
    if npnts <= 0 {
        return Err(AsimuError::Boundary(format!(
            "CGNS ZoneBC npnts 非法: {npnts}"
        )));
    }
    match ptset_type {
        BC_POINT_RANGE | BC_ELEMENT_RANGE => Ok(2),
        BC_POINT_LIST | BC_ELEMENT_LIST => usize::try_from(npnts).map_err(|_| {
            AsimuError::Boundary(format!("CGNS ZoneBC npnts 超出 usize 范围: {npnts}"))
        }),
        other => Err(AsimuError::Boundary(format!(
            "CGNS 非结构 ZoneBC PointSetType {other} 暂不支持"
        ))),
    }
}

fn expand_element_range(pnts: &[CgSize]) -> Result<Vec<CgSize>> {
    if pnts.len() != 2 || pnts[0] <= 0 || pnts[1] < pnts[0] {
        return Err(AsimuError::Boundary(format!(
            "CGNS ZoneBC ElementRange 非法: {pnts:?}"
        )));
    }
    Ok((pnts[0]..=pnts[1]).collect())
}

fn section_element_count(info: &CgnsSectionInfo) -> Result<usize> {
    if info.end < info.start {
        return Err(AsimuError::Mesh(format!(
            "section {} ElementRange 非法: {}..{}",
            info.name, info.start, info.end
        )));
    }
    let total = info.end - info.start + 1;
    usize::try_from(total).map_err(|_| {
        AsimuError::Mesh(format!(
            "section {} ElementRange 数量无效: {total}",
            info.name
        ))
    })
}

fn cgns_nodes_to_zero_based(nodes: &[CgSize]) -> Result<Vec<usize>> {
    nodes
        .iter()
        .map(|&node| {
            if node <= 0 {
                return Err(AsimuError::Mesh(format!("CGNS 节点索引 {node} 非正")));
            }
            usize::try_from(node - 1)
                .map_err(|_| AsimuError::Mesh(format!("CGNS 节点索引 {node} 超出 usize 范围")))
        })
        .collect()
}

fn cell_kind_from_cgns_element(element_type: i32) -> Option<CellKind> {
    match element_type {
        ELEM_TETRA_4 => Some(CellKind::Tet),
        ELEM_PYRA_5 => Some(CellKind::Pyramid),
        ELEM_PENTA_6 => Some(CellKind::Prism),
        ELEM_HEXA_8 => Some(CellKind::Hex),
        _ => None,
    }
}

fn cgns_element_node_count(element_type: i32) -> Option<usize> {
    match element_type {
        ELEM_TRI_3 => Some(3),
        ELEM_QUAD_4 => Some(4),
        ELEM_TETRA_4 => Some(4),
        ELEM_PYRA_5 => Some(5),
        ELEM_PENTA_6 => Some(6),
        ELEM_HEXA_8 => Some(8),
        _ => None,
    }
}

fn is_boundary_element_section(element_type: i32) -> bool {
    matches!(element_type, ELEM_TRI_3 | ELEM_QUAD_4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dualellipsoid_mix_cgns_path() -> Option<PathBuf> {
        std::env::var("ASIMU_MIX_CGNS_PATH")
            .map(PathBuf::from)
            .ok()
            .filter(|p| p.is_file())
            .or_else(|| {
                Some(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("output/case_dualellipsoid/mix.cgns"),
                )
                .filter(|p| p.is_file())
            })
    }

    #[test]
    fn loads_dualellipsoid_mixed_unstructured_when_present() {
        let Some(path) = dualellipsoid_mix_cgns_path() else {
            return;
        };
        let loaded = load_cgns_unstructured_zone(&path, 1).expect("load mixed cgns");
        assert!(loaded.mesh.num_nodes() > 0);
        assert!(loaded.mesh.num_cells() > 0);
        assert!(loaded.mesh.num_faces() > 0);
        assert_eq!(loaded.boundary.patches().len(), 3);
        assert!(
            loaded
                .boundary
                .patches()
                .iter()
                .all(|patch| !patch.face_ids.is_empty())
        );
        let report = crate::mesh::check_unstructured_mesh3d(
            &loaded.mesh,
            Some(&loaded.boundary),
            "mix.cgns",
        );
        assert!(report.passed(), "{report}");
    }
}
