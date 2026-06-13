//! 多块 3D 可压缩求解的接口映射。

use std::collections::BTreeSet;

use crate::boundary::{BoundaryKind, BoundaryPatch};
use crate::core::{FaceId, Real, Vector3};
use crate::error::{AsimuError, Result};
use crate::mesh::{
    BoundaryMesh3d, LogicalFace3d, MultiBlockStructuredMesh3d, StructuredBlock3d,
    StructuredIndexRange3d, StructuredMesh3d,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockInterfaceLink {
    pub(crate) face: FaceId,
    pub(crate) owner_cell: usize,
    pub(crate) donor_block_index: usize,
    pub(crate) donor_cell: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SharedInterfaceFace {
    pub(crate) owner_block_index: usize,
    pub(crate) owner_cell: usize,
    pub(crate) donor_block_index: usize,
    pub(crate) donor_cell: usize,
    pub(crate) face: FaceId,
    pub(crate) normal: Vector3,
    pub(crate) owner_scale: Real,
    pub(crate) donor_scale: Real,
}

pub(crate) struct MultiblockInterfaceMetadata {
    pub(crate) links: Vec<Vec<BlockInterfaceLink>>,
    pub(crate) patches: Vec<Vec<BoundaryPatch>>,
    pub(crate) shared_faces: Vec<SharedInterfaceFace>,
}

pub(crate) fn build_multiblock_interface_metadata(
    mesh: &MultiBlockStructuredMesh3d,
) -> Result<MultiblockInterfaceMetadata> {
    let mut metadata = MultiblockInterfaceMetadata {
        links: (0..mesh.num_blocks()).map(|_| Vec::new()).collect(),
        patches: (0..mesh.num_blocks()).map(|_| Vec::new()).collect(),
        shared_faces: Vec::new(),
    };
    let mut seen_faces = BTreeSet::new();
    let block_volumes: Vec<Vec<Real>> = mesh
        .blocks()
        .iter()
        .map(|block| block.mesh.cell_volumes())
        .collect();
    for (interface_index, interface) in mesh.interfaces().iter().enumerate() {
        let owner_index = block_index(mesh.blocks(), &interface.owner_block)?;
        let donor_index = block_index(mesh.blocks(), &interface.donor_block)?;
        let owner_block = &mesh.blocks()[owner_index];
        let donor_block = &mesh.blocks()[donor_index];
        let owner_entries = interface_owner_entries(&owner_block.mesh, &interface.owner_range)?;
        let donor_cells = interface_donor_cells(
            &owner_block.mesh,
            &donor_block.mesh,
            &interface.owner_range,
            &interface.donor_range,
            interface.transform,
        )?;
        let owner_faces: Vec<FaceId> = owner_entries.iter().map(|entry| entry.face).collect();
        if owner_faces.len() != donor_cells.len() {
            return Err(AsimuError::Mesh(format!(
                "多块接口 {} -> {} 面数不匹配：owner={} donor={}",
                interface.owner_block,
                interface.donor_block,
                owner_faces.len(),
                donor_cells.len()
            )));
        }
        metadata.patches[owner_index].push(BoundaryPatch::new(
            format!("__interface/{}/{}", interface.donor_block, interface_index),
            owner_faces.clone(),
            BoundaryKind::Periodic {
                partner: interface.donor_block.clone(),
            },
        ));
        for (entry, donor_cell) in owner_entries.into_iter().zip(donor_cells) {
            let owner_cell = owner_block.mesh.cell_index(
                (entry.cell_coord[0] - 1) as usize,
                (entry.cell_coord[1] - 1) as usize,
                (entry.cell_coord[2] - 1) as usize,
            );
            let link = BlockInterfaceLink {
                face: entry.face,
                owner_cell,
                donor_block_index: donor_index,
                donor_cell,
            };
            let key = canonical_interface_key(owner_index, &link);
            if seen_faces.insert(key) {
                let geom = owner_block.mesh.face_geometry_3d(entry.face)?;
                metadata.shared_faces.push(SharedInterfaceFace {
                    owner_block_index: owner_index,
                    owner_cell,
                    donor_block_index: donor_index,
                    donor_cell,
                    face: entry.face,
                    normal: geom.normal,
                    owner_scale: -geom.area / block_volumes[owner_index][owner_cell],
                    donor_scale: geom.area / block_volumes[donor_index][donor_cell],
                });
            }
            metadata.links[owner_index].push(link);
        }
    }
    Ok(metadata)
}

fn canonical_interface_key(
    owner_block: usize,
    link: &BlockInterfaceLink,
) -> (usize, usize, usize, usize) {
    let a = (owner_block, link.owner_cell);
    let b = (link.donor_block_index, link.donor_cell);
    if a <= b {
        (a.0, a.1, b.0, b.1)
    } else {
        (b.0, b.1, a.0, a.1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InterfaceOwnerEntry {
    face: FaceId,
    cell_coord: [i32; 3],
}

fn block_index(blocks: &[StructuredBlock3d], name: &str) -> Result<usize> {
    blocks
        .iter()
        .position(|block| block.name == name)
        .ok_or_else(|| AsimuError::Mesh(format!("多块接口引用未知 block：{name}")))
}

fn interface_owner_entries(
    mesh: &StructuredMesh3d,
    range: &StructuredIndexRange3d,
) -> Result<Vec<InterfaceOwnerEntry>> {
    let face = detect_interface_face(mesh, range)?;
    let mut entries = Vec::new();
    match face {
        LogicalFace3d::IMin | LogicalFace3d::IMax => {
            let i = boundary_cell_coord(range.imin, mesh.nx as i32);
            for k in range_axis_values(range.kmin, range.kmax, mesh.nz as i32) {
                for j in range_axis_values(range.jmin, range.jmax, mesh.ny as i32) {
                    let local = (j - 1) + (k - 1) * mesh.ny as i32;
                    entries.push(InterfaceOwnerEntry {
                        face: face.encode(local as u32),
                        cell_coord: [i, j, k],
                    });
                }
            }
        }
        LogicalFace3d::JMin | LogicalFace3d::JMax => {
            let j = boundary_cell_coord(range.jmin, mesh.ny as i32);
            for k in range_axis_values(range.kmin, range.kmax, mesh.nz as i32) {
                for i in range_axis_values(range.imin, range.imax, mesh.nx as i32) {
                    let local = (i - 1) + (k - 1) * mesh.nx as i32;
                    entries.push(InterfaceOwnerEntry {
                        face: face.encode(local as u32),
                        cell_coord: [i, j, k],
                    });
                }
            }
        }
        LogicalFace3d::KMin | LogicalFace3d::KMax => {
            let k = boundary_cell_coord(range.kmin, mesh.nz as i32);
            for j in range_axis_values(range.jmin, range.jmax, mesh.ny as i32) {
                for i in range_axis_values(range.imin, range.imax, mesh.nx as i32) {
                    let local = (i - 1) + (j - 1) * mesh.nx as i32;
                    entries.push(InterfaceOwnerEntry {
                        face: face.encode(local as u32),
                        cell_coord: [i, j, k],
                    });
                }
            }
        }
    }
    Ok(entries)
}

fn interface_donor_cells(
    owner_mesh: &StructuredMesh3d,
    donor_mesh: &StructuredMesh3d,
    owner_range: &StructuredIndexRange3d,
    donor_range: &StructuredIndexRange3d,
    transform: [i32; 3],
) -> Result<Vec<usize>> {
    validate_cgns_transform(transform)?;
    let owner_entries = interface_owner_entries(owner_mesh, owner_range)?;
    let mut cells = Vec::with_capacity(owner_entries.len());
    for entry in owner_entries {
        let donor = transform_owner_cell_to_donor(
            entry.cell_coord,
            owner_range,
            donor_range,
            transform,
            owner_mesh,
            donor_mesh,
        )?;
        cells.push(donor_mesh.cell_index(
            (donor[0] - 1) as usize,
            (donor[1] - 1) as usize,
            (donor[2] - 1) as usize,
        ));
    }
    Ok(cells)
}

fn transform_owner_cell_to_donor(
    owner_cell: [i32; 3],
    owner_range: &StructuredIndexRange3d,
    donor_range: &StructuredIndexRange3d,
    transform: [i32; 3],
    owner_mesh: &StructuredMesh3d,
    donor_mesh: &StructuredMesh3d,
) -> Result<[i32; 3]> {
    let owner_bounds = range_bounds(owner_range);
    let donor_bounds = range_bounds(donor_range);
    let owner_cells = [
        owner_mesh.nx as i32,
        owner_mesh.ny as i32,
        owner_mesh.nz as i32,
    ];
    let donor_cells = [
        donor_mesh.nx as i32,
        donor_mesh.ny as i32,
        donor_mesh.nz as i32,
    ];
    let mut donor = [0; 3];
    for owner_axis in 0..3 {
        let donor_axis = transform[owner_axis].unsigned_abs() as usize - 1;
        let owner_coord = owner_cell[owner_axis];
        let owner_start = cell_index_value(owner_bounds[owner_axis].0, owner_cells[owner_axis]);
        let donor_start = cell_index_value(donor_bounds[donor_axis].0, donor_cells[donor_axis]);
        donor[donor_axis] = if transform[owner_axis] > 0 {
            donor_start + (owner_coord - owner_start)
        } else {
            donor_start - (owner_coord - owner_start)
        };
    }
    validate_donor_cell_coord(donor, donor_mesh)?;
    Ok(donor)
}

fn validate_cgns_transform(transform: [i32; 3]) -> Result<()> {
    let mut seen = [false; 3];
    for component in transform {
        let axis = component.unsigned_abs() as usize;
        if !(1..=3).contains(&axis) || seen[axis - 1] {
            return Err(AsimuError::Mesh(format!(
                "CGNS 1-to-1 transform 无效：{:?}",
                transform
            )));
        }
        seen[axis - 1] = true;
    }
    Ok(())
}

fn range_bounds(range: &StructuredIndexRange3d) -> [(i32, i32); 3] {
    [
        (range.imin, range.imax),
        (range.jmin, range.jmax),
        (range.kmin, range.kmax),
    ]
}

fn validate_donor_cell_coord(coord: [i32; 3], mesh: &StructuredMesh3d) -> Result<()> {
    let max = [mesh.nx as i32, mesh.ny as i32, mesh.nz as i32];
    for axis in 0..3 {
        if coord[axis] < 1 || coord[axis] > max[axis] {
            return Err(AsimuError::Mesh(format!(
                "多块接口 donor 单元坐标越界：coord={coord:?}, dims={max:?}"
            )));
        }
    }
    Ok(())
}

fn detect_interface_face(
    mesh: &StructuredMesh3d,
    range: &StructuredIndexRange3d,
) -> Result<LogicalFace3d> {
    let nx = mesh.nx as i32;
    let ny = mesh.ny as i32;
    let nz = mesh.nz as i32;
    if range.imin == 1 && range.imax == 1 {
        return Ok(LogicalFace3d::IMin);
    }
    if range.imin == nx + 1 && range.imax == nx + 1 {
        return Ok(LogicalFace3d::IMax);
    }
    if range.jmin == 1 && range.jmax == 1 {
        return Ok(LogicalFace3d::JMin);
    }
    if range.jmin == ny + 1 && range.jmax == ny + 1 {
        return Ok(LogicalFace3d::JMax);
    }
    if range.kmin == 1 && range.kmax == 1 {
        return Ok(LogicalFace3d::KMin);
    }
    if range.kmin == nz + 1 && range.kmax == nz + 1 {
        return Ok(LogicalFace3d::KMax);
    }
    Err(AsimuError::Boundary(format!(
        "无法识别多块接口 PointRange 对应逻辑面：{:?}",
        range
    )))
}

fn range_axis_values(start: i32, end: i32, n_cells: i32) -> Vec<i32> {
    let start = cell_index_value(start, n_cells);
    let end = cell_index_value(end, n_cells);
    if start <= end {
        (start..=end).collect()
    } else {
        (end..=start).rev().collect()
    }
}

fn cell_index_value(index: i32, n_cells: i32) -> i32 {
    if index == n_cells + 1 {
        n_cells
    } else {
        index.clamp(1, n_cells)
    }
}

fn boundary_cell_index(index: i32, n_cells: i32) -> usize {
    if index <= 1 {
        0
    } else {
        (n_cells - 1) as usize
    }
}

fn boundary_cell_coord(index: i32, n_cells: i32) -> i32 {
    boundary_cell_index(index, n_cells) as i32 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(
        imin: i32,
        imax: i32,
        jmin: i32,
        jmax: i32,
        kmin: i32,
        kmax: i32,
    ) -> StructuredIndexRange3d {
        StructuredIndexRange3d {
            imin,
            imax,
            jmin,
            jmax,
            kmin,
            kmax,
        }
    }

    #[test]
    fn interface_transform_reverses_tangent_axis() {
        let owner = StructuredMesh3d::uniform_box("owner", 1, 2, 1, 1.0, 2.0, 1.0).expect("owner");
        let donor = StructuredMesh3d::uniform_box("donor", 1, 2, 1, 1.0, 2.0, 1.0).expect("donor");
        let cells = interface_donor_cells(
            &owner,
            &donor,
            &range(2, 2, 1, 3, 1, 2),
            &range(1, 1, 3, 1, 1, 2),
            [1, -2, 3],
        )
        .expect("cells");

        assert_eq!(
            cells,
            vec![donor.cell_index(0, 1, 0), donor.cell_index(0, 0, 0)]
        );
    }

    #[test]
    fn interface_transform_swaps_tangent_axes() {
        let owner = StructuredMesh3d::uniform_box("owner", 1, 2, 3, 1.0, 2.0, 3.0).expect("owner");
        let donor = StructuredMesh3d::uniform_box("donor", 1, 3, 2, 1.0, 3.0, 2.0).expect("donor");
        let cells = interface_donor_cells(
            &owner,
            &donor,
            &range(2, 2, 1, 3, 1, 4),
            &range(1, 1, 1, 4, 1, 3),
            [1, 3, 2],
        )
        .expect("cells");

        assert_eq!(cells[0], donor.cell_index(0, 0, 0));
        assert_eq!(cells[1], donor.cell_index(0, 0, 1));
        assert_eq!(cells[2], donor.cell_index(0, 1, 0));
        assert_eq!(cells[5], donor.cell_index(0, 2, 1));
    }
}
