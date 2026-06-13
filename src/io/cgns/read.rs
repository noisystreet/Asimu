//! CGNS 结构化 zone 读取。

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Mutex;

use crate::boundary::{BoundarySet, cgns_bc};
use crate::error::{AsimuError, Result};
use crate::io::limits::{io_error, validate_cell_count, validate_file_size, validate_input_path};
use crate::mesh::{StructuredIndexRange3d, StructuredMesh, StructuredMesh3d};

use super::ffi::{
    BC_POINT_RANGE, CG_MODE_READ, CG_OK, CgSize, REAL_DOUBLE, ZONE_STRUCTURED, ZONE_UNSTRUCTURED,
    asimu_cg_read_boco_family_name, cg_1to1_read, cg_boco_info, cg_boco_read, cg_close,
    cg_coord_read, cg_fambc_read, cg_family_read, cg_get_error, cg_n1to1, cg_nbocos, cg_nfamilies,
    cg_nzones, cg_open, cg_zone_read, cg_zone_type,
};
use super::unstructured;
use super::zonebc::{CgnsPointRange, boundary_set_from_cgns, patch_from_cgns};

pub(crate) static CGNS_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CgnsZoneInfo {
    pub index: usize,
    pub name: String,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CgnsLoadResult {
    pub zone: CgnsZoneInfo,
    pub mesh: StructuredMesh,
    pub boundary: BoundarySet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cgns1to1Connection {
    pub zone_name: String,
    pub connect_name: String,
    pub donor_name: String,
    pub range: StructuredIndexRange3d,
    pub donor_range: StructuredIndexRange3d,
    pub transform: [i32; 3],
}

pub(super) struct CgnsFile {
    pub(super) index: i32,
}

pub(super) struct ResolvedBocoType {
    pub(super) bctype: i32,
    pub(super) label: String,
}

impl CgnsFile {
    pub(super) fn open(path: &Path) -> Result<Self> {
        validate_input_path(path)?;
        let bytes = std::fs::metadata(path)?;
        validate_file_size(bytes.len(), "CGNS 文件")?;

        let cpath = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| io_error(std::io::ErrorKind::InvalidInput, "CGNS 路径含内嵌 NUL 字节"))?;
        let mut index = 0;
        let err = unsafe { cg_open(cpath.as_ptr(), CG_MODE_READ, &mut index) };
        check_cg(err)?;
        Ok(Self { index })
    }

    pub(super) fn nzones(&self, base: i32) -> Result<usize> {
        let mut nzones = 0;
        let err = unsafe { cg_nzones(self.index, base, &mut nzones) };
        check_cg(err)?;
        Ok(nzones as usize)
    }

    fn zone_info(&self, base: i32, zone: i32) -> Result<CgnsZoneInfo> {
        let mut name = [0i8; 33];
        let mut sizes = [0 as CgSize; 3];
        let err = unsafe {
            cg_zone_read(
                self.index,
                base,
                zone,
                name.as_mut_ptr(),
                sizes.as_mut_ptr(),
            )
        };
        check_cg(err)?;

        let mut zone_type = 0;
        let err = unsafe { cg_zone_type(self.index, base, zone, &mut zone_type) };
        check_cg(err)?;
        if zone_type != ZONE_STRUCTURED {
            return Err(AsimuError::Mesh(format!(
                "zone {zone} 非 Structured 类型（type={zone_type}），暂不支持"
            )));
        }

        let ni = sizes[0] as usize;
        let nj = sizes[1] as usize;
        let nk = sizes[2] as usize;
        if ni < 2 || nj < 2 || nk < 2 {
            return Err(AsimuError::Mesh(format!(
                "zone {zone} 顶点尺寸无效: {ni}×{nj}×{nk}"
            )));
        }
        let nx = ni - 1;
        let ny = nj - 1;
        let nz = nk - 1;
        validate_cell_count((nx * ny * nz) as u64)?;

        let zone_name = c_str_to_string(&name)?;
        Ok(CgnsZoneInfo {
            index: zone as usize,
            name: zone_name,
            nx,
            ny,
            nz,
        })
    }

    fn any_zone_info(&self, base: i32, zone: i32) -> Result<CgnsZoneInfo> {
        let zone_type = self.zone_type(base, zone)?;
        if zone_type == ZONE_STRUCTURED {
            self.zone_info(base, zone)
        } else if zone_type == ZONE_UNSTRUCTURED {
            unstructured::unstructured_zone_info(self, base, zone)
        } else {
            Err(AsimuError::Mesh(format!(
                "zone {zone} 类型 {zone_type} 不支持"
            )))
        }
    }

    pub(super) fn zone_type(&self, base: i32, zone: i32) -> Result<i32> {
        let mut zone_type = 0;
        let err = unsafe { cg_zone_type(self.index, base, zone, &mut zone_type) };
        check_cg(err)?;
        Ok(zone_type)
    }

    fn read_coords(
        &self,
        base: i32,
        zone: i32,
        info: &CgnsZoneInfo,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let ni = info.nx + 1;
        let nj = info.ny + 1;
        let nk = info.nz + 1;
        let npts = ni
            .checked_mul(nj)
            .and_then(|n| n.checked_mul(nk))
            .ok_or_else(|| AsimuError::Mesh("zone 节点数溢出".to_string()))?;

        let mut points_x = vec![0.0; npts];
        let mut points_y = vec![0.0; npts];
        let mut points_z = vec![0.0; npts];
        let rmin = [1, 1, 1];
        let rmax = [ni as CgSize, nj as CgSize, nk as CgSize];

        for (coord, buf) in [
            ("CoordinateX", &mut points_x),
            ("CoordinateY", &mut points_y),
            ("CoordinateZ", &mut points_z),
        ] {
            let cname = CString::new(coord).expect("static coord name");
            let err = unsafe {
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
            };
            check_cg(err)?;
        }
        Ok((points_x, points_y, points_z))
    }

    pub(super) fn read_family_bc_map(&self, base: i32) -> Result<HashMap<String, i32>> {
        let mut nfamilies = 0;
        check_cg(unsafe { cg_nfamilies(self.index, base, &mut nfamilies) })?;
        let mut map = HashMap::with_capacity(nfamilies as usize);
        for family in 1..=nfamilies {
            let mut name = [0i8; 33];
            let mut nbocos = 0;
            let mut ngeos = 0;
            check_cg(unsafe {
                cg_family_read(
                    self.index,
                    base,
                    family,
                    name.as_mut_ptr(),
                    &mut nbocos,
                    &mut ngeos,
                )
            })?;
            if nbocos < 1 {
                continue;
            }
            let mut fambc_name = [0i8; 33];
            let mut bctype = 0;
            check_cg(unsafe {
                cg_fambc_read(
                    self.index,
                    base,
                    family,
                    1,
                    fambc_name.as_mut_ptr(),
                    &mut bctype,
                )
            })?;
            map.insert(c_str_to_string(&name)?, bctype);
        }
        Ok(map)
    }

    pub(super) fn resolve_boco_type(
        &self,
        base: i32,
        zone: i32,
        boco: i32,
        bocotype: i32,
        _families: &HashMap<String, i32>,
        boco_name: &str,
    ) -> Result<ResolvedBocoType> {
        if bocotype != cgns_bc::BC_FAMILY_SPECIFIED {
            return Ok(ResolvedBocoType {
                bctype: bocotype,
                label: boco_name.to_string(),
            });
        }
        let mut family_name = [0i8; 33];
        check_cg(unsafe {
            asimu_cg_read_boco_family_name(self.index, base, zone, boco, family_name.as_mut_ptr())
        })?;
        let family = c_str_to_string(&family_name)?;
        Ok(ResolvedBocoType {
            bctype: cgns_bc::BC_FAMILY_SPECIFIED,
            label: family,
        })
    }

    fn read_boco_point_range(&self, base: i32, zone: i32, boco: i32) -> Result<CgnsPointRange> {
        let mut pnts = [0i32; 6];
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
        Ok(CgnsPointRange {
            imin: pnts[0],
            imax: pnts[3],
            jmin: pnts[1],
            jmax: pnts[4],
            kmin: pnts[2],
            kmax: pnts[5],
        })
    }

    fn read_zone_1to1_connections(
        &self,
        base: i32,
        zone: i32,
        zone_name: &str,
    ) -> Result<Vec<Cgns1to1Connection>> {
        let mut count = 0;
        check_cg(unsafe { cg_n1to1(self.index, base, zone, &mut count) })?;
        let mut connections = Vec::with_capacity(count as usize);
        for connect in 1..=count {
            let mut connect_name = [0i8; 33];
            let mut donor_name = [0i8; 33];
            let mut range = [0 as CgSize; 6];
            let mut donor_range = [0 as CgSize; 6];
            let mut transform = [0i32; 3];
            check_cg(unsafe {
                cg_1to1_read(
                    self.index,
                    base,
                    zone,
                    connect,
                    connect_name.as_mut_ptr(),
                    donor_name.as_mut_ptr(),
                    range.as_mut_ptr(),
                    donor_range.as_mut_ptr(),
                    transform.as_mut_ptr(),
                )
            })?;
            connections.push(Cgns1to1Connection {
                zone_name: zone_name.to_string(),
                connect_name: c_str_to_string(&connect_name)?,
                donor_name: c_str_to_string(&donor_name)?,
                range: structured_range_from_cgns(range),
                donor_range: structured_range_from_cgns(donor_range),
                transform,
            });
        }
        Ok(connections)
    }

    fn read_zone_bocos(
        &self,
        base: i32,
        zone: i32,
        mesh: &StructuredMesh3d,
    ) -> Result<BoundarySet> {
        let families = self.read_family_bc_map(base)?;
        let mut nbocos = 0;
        check_cg(unsafe { cg_nbocos(self.index, base, zone, &mut nbocos) })?;
        let mut patches = Vec::with_capacity(nbocos as usize);
        for boco in 1..=nbocos {
            let mut name = [0i8; 33];
            let mut bocotype = 0;
            let mut ptset_type = 0;
            let mut npnts = 0;
            let mut normalindex = 0;
            let mut normal_list_size = 0;
            let mut normaldatatype = 0;
            let mut ndataset = 0;
            let err = unsafe {
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
            };
            check_cg(err)?;
            if ptset_type != BC_POINT_RANGE {
                continue;
            }
            let range = self.read_boco_point_range(base, zone, boco)?;
            let boco_name = c_str_to_string(&name)?;
            let resolved =
                self.resolve_boco_type(base, zone, boco, bocotype, &families, &boco_name)?;
            patches.push(patch_from_cgns(
                boco_name,
                resolved.bctype,
                &resolved.label,
                range,
                mesh,
            )?);
        }
        Ok(boundary_set_from_cgns(patches))
    }
}

impl Drop for CgnsFile {
    fn drop(&mut self) {
        let _ = unsafe { cg_close(self.index) };
    }
}

pub fn list_cgns_zones(path: &Path) -> Result<Vec<CgnsZoneInfo>> {
    let _guard = CGNS_LOCK.lock().map_err(|_| cgns_lock_error())?;
    list_cgns_zones_locked(path)
}

fn list_cgns_zones_locked(path: &Path) -> Result<Vec<CgnsZoneInfo>> {
    const BASE: i32 = 1;
    let file = CgnsFile::open(path)?;
    let nzones = file.nzones(BASE)?;
    let mut zones = Vec::with_capacity(nzones);
    for zone in 1..=nzones as i32 {
        zones.push(file.any_zone_info(BASE, zone)?);
    }
    Ok(zones)
}

pub fn load_cgns_zone(path: &Path, zone_index: usize) -> Result<CgnsLoadResult> {
    let _guard = CGNS_LOCK.lock().map_err(|_| cgns_lock_error())?;
    load_cgns_zone_locked(path, zone_index)
}

fn load_cgns_zone_locked(path: &Path, zone_index: usize) -> Result<CgnsLoadResult> {
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
    let info = file.zone_info(BASE, zone)?;
    let (points_x, points_y, points_z) = file.read_coords(BASE, zone, &info)?;
    let mesh_name = if info.name.is_empty() {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("cgns")
            .to_string()
    } else {
        info.name.clone()
    };
    let mesh3d = StructuredMesh3d::new(
        mesh_name.clone(),
        info.nx,
        info.ny,
        info.nz,
        points_x,
        points_y,
        points_z,
    )?;
    let boundary = file.read_zone_bocos(BASE, zone, &mesh3d)?;
    let mesh = StructuredMesh::D3(mesh3d);
    Ok(CgnsLoadResult {
        zone: info,
        mesh,
        boundary,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct CgnsMultiLoadResult {
    pub zones: Vec<CgnsLoadResult>,
    pub interfaces: Vec<Cgns1to1Connection>,
    /// 多 block 导出时生成的 `.vtm` 路径（仅 `export_cgns_to_vtm` 设置）。
    pub vtm_path: Option<std::path::PathBuf>,
}

pub fn load_cgns_all_zones(path: &Path) -> Result<CgnsMultiLoadResult> {
    let _guard = CGNS_LOCK.lock().map_err(|_| cgns_lock_error())?;
    load_cgns_all_zones_locked(path)
}

fn load_cgns_all_zones_locked(path: &Path) -> Result<CgnsMultiLoadResult> {
    const BASE: i32 = 1;
    let file = CgnsFile::open(path)?;
    let nzones = file.nzones(BASE)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cgns")
        .to_string();
    let mut zones = Vec::with_capacity(nzones);
    let mut interfaces = Vec::new();
    for zone in 1..=nzones as i32 {
        let info = file.zone_info(BASE, zone)?;
        let (points_x, points_y, points_z) = file.read_coords(BASE, zone, &info)?;
        let mesh_name = if info.name.is_empty() {
            format!("{stem}_zone{zone:02}")
        } else {
            info.name.clone()
        };
        let mesh3d = StructuredMesh3d::new(
            mesh_name.clone(),
            info.nx,
            info.ny,
            info.nz,
            points_x,
            points_y,
            points_z,
        )?;
        let boundary = file.read_zone_bocos(BASE, zone, &mesh3d)?;
        interfaces.extend(file.read_zone_1to1_connections(BASE, zone, &mesh_name)?);
        let mesh = StructuredMesh::D3(mesh3d);
        zones.push(CgnsLoadResult {
            zone: info,
            mesh,
            boundary,
        });
    }
    Ok(CgnsMultiLoadResult {
        zones,
        interfaces,
        vtm_path: None,
    })
}

/// 将 CGNS 全部 zone 导出为 `.vtm` + 子 VTS（ParaView 请打开 `.vtm`）。
#[cfg(feature = "io-vtk")]
pub fn export_cgns_to_vtm(input: &Path, output: &Path) -> Result<CgnsMultiLoadResult> {
    let loaded = load_cgns_all_zones(input)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh");
    let file_names: Vec<String> = loaded
        .zones
        .iter()
        .map(|z| format!("{stem}_zone{:02}.vts", z.zone.index))
        .collect();
    let block_names: Vec<String> = loaded
        .zones
        .iter()
        .map(|z| {
            if z.zone.name.trim().is_empty() {
                format!("Zone_{:02}", z.zone.index)
            } else {
                z.zone.name.trim().to_string()
            }
        })
        .collect();
    for (z, file_name) in loaded.zones.iter().zip(file_names.iter()) {
        crate::io::vtk::write_vts(&z.mesh, &parent.join(file_name))?;
    }
    let block_refs: Vec<crate::io::vtk::VtmBlock<'_>> = block_names
        .iter()
        .zip(file_names.iter())
        .map(|(name, file)| crate::io::vtk::VtmBlock { name, file })
        .collect();
    let vtm_path = parent.join(format!("{stem}.vtm"));
    crate::io::vtk::write_vtm(&block_refs, &vtm_path)?;
    Ok(CgnsMultiLoadResult {
        zones: loaded.zones,
        interfaces: loaded.interfaces,
        vtm_path: Some(vtm_path),
    })
}

/// 兼容旧名：等价于 [`export_cgns_to_vtm`]。
#[cfg(feature = "io-vtk")]
pub fn export_cgns_to_vts(input: &Path, output: &Path) -> Result<CgnsMultiLoadResult> {
    export_cgns_to_vtm(input, output)
}

#[cfg(not(feature = "io-vtk"))]
pub fn export_cgns_to_vts(_input: &Path, _output: &Path) -> Result<CgnsMultiLoadResult> {
    Err(AsimuError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "导出 VTS 需要启用 feature io-vtk",
    )))
}

/// 将 CGNS zone 导出为 VTS（需 features `io-cgns` + `io-vtk`）。
#[cfg(feature = "io-vtk")]
pub fn export_cgns_zone_to_vts(
    input: &Path,
    zone_index: usize,
    output: &Path,
) -> Result<CgnsLoadResult> {
    let loaded = load_cgns_zone(input, zone_index)?;
    crate::io::vtk::write_vts(&loaded.mesh, output)?;
    Ok(loaded)
}

#[cfg(not(feature = "io-vtk"))]
pub fn export_cgns_zone_to_vts(
    _input: &Path,
    _zone_index: usize,
    _output: &Path,
) -> Result<CgnsLoadResult> {
    Err(AsimuError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "导出 VTS 需要启用 feature io-vtk",
    )))
}

pub(super) fn c_str_to_string(buf: &[i8]) -> Result<String> {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let bytes: Vec<u8> = buf[..end].iter().map(|&b| b as u8).collect();
    String::from_utf8(bytes).map_err(|e| AsimuError::Mesh(format!("zone 名称非 UTF-8: {e}")))
}

fn structured_range_from_cgns(range: [CgSize; 6]) -> StructuredIndexRange3d {
    StructuredIndexRange3d {
        imin: range[0],
        imax: range[3],
        jmin: range[1],
        jmax: range[4],
        kmin: range[2],
        kmax: range[5],
    }
}

pub(super) fn check_cg(err: i32) -> Result<()> {
    if err == CG_OK {
        return Ok(());
    }
    let msg = unsafe {
        CStr::from_ptr(cg_get_error())
            .to_string_lossy()
            .into_owned()
    };
    Err(AsimuError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("CGNS 错误 ({err}): {msg}"),
    )))
}

pub(super) fn cgns_lock_error() -> AsimuError {
    AsimuError::Io(std::io::Error::other("CGNS 全局锁已损坏"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dlr_f6_path() -> Option<PathBuf> {
        std::env::var("ASIMU_CGNS_PATH")
            .map(PathBuf::from)
            .ok()
            .filter(|p| p.is_file())
            .or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .map(|dir| dir.join("dlr-f6.coar.cgns"))
                    .filter(|p| p.is_file())
            })
    }

    fn cylinder_cgns_path() -> Option<PathBuf> {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|dir| dir.join("cylinder.cgns"))
            .filter(|p| p.is_file())
    }

    #[test]
    fn lists_dlr_f6_zones_when_present() {
        let Some(path) = dlr_f6_path() else {
            return;
        };
        let zones = list_cgns_zones(&path).expect("list zones");
        assert_eq!(zones.len(), 26);
        assert_eq!(zones[0].nx, 216);
        assert_eq!(zones[0].ny, 56);
        assert_eq!(zones[0].nz, 8);
    }

    #[test]
    fn loads_smallest_dlr_f6_zone_when_present() {
        let Some(path) = dlr_f6_path() else {
            return;
        };
        let loaded = load_cgns_zone(&path, 26).expect("load zone 26");
        let mesh = match loaded.mesh {
            StructuredMesh::D3(m) => m,
            StructuredMesh::D2(_) => panic!("expected 3d"),
        };
        assert_eq!(mesh.nx, 16);
        assert_eq!(mesh.ny, 24);
        assert_eq!(mesh.nz, 48);
        assert_eq!(mesh.num_nodes(), 17 * 25 * 49);
    }

    #[test]
    fn loads_dlr_f6_zone1_boundaries_when_present() {
        use crate::boundary::BoundaryKind;

        let Some(path) = dlr_f6_path() else {
            return;
        };
        let loaded = load_cgns_zone(&path, 1).expect("load zone 1");
        assert_eq!(loaded.boundary.patches().len(), 4);
        assert!(
            loaded
                .boundary
                .patches()
                .iter()
                .all(|patch| !patch.face_ids.is_empty())
        );
        assert!(
            loaded
                .boundary
                .patches()
                .iter()
                .any(|patch| { matches!(patch.kind, BoundaryKind::Farfield { .. }) })
        );
    }

    #[test]
    fn loads_cylinder_cgns_boundaries_when_present() {
        use crate::boundary::BoundaryKind;

        let Some(path) = cylinder_cgns_path() else {
            return;
        };
        let loaded = load_cgns_zone(&path, 1).expect("load cylinder");
        assert_eq!(loaded.zone.name, "blk-1");
        assert_eq!(loaded.boundary.patches().len(), 9);
        assert!(
            loaded
                .boundary
                .patches()
                .iter()
                .all(|patch| !patch.face_ids.is_empty())
        );
        assert_eq!(
            loaded
                .boundary
                .patches()
                .iter()
                .map(|patch| patch.face_ids.len())
                .sum::<usize>(),
            19_899
        );
        assert_eq!(
            loaded
                .boundary
                .patches()
                .iter()
                .filter(|patch| matches!(patch.kind, BoundaryKind::Symmetry))
                .count(),
            3
        );
        assert_eq!(
            loaded
                .boundary
                .patches()
                .iter()
                .filter(|patch| matches!(patch.kind, BoundaryKind::Inlet { .. }))
                .count(),
            2
        );
        assert_eq!(
            loaded
                .boundary
                .patches()
                .iter()
                .filter(|patch| matches!(patch.kind, BoundaryKind::Outlet { .. }))
                .count(),
            2
        );
        assert_eq!(
            loaded
                .boundary
                .patches()
                .iter()
                .filter(|patch| matches!(patch.kind, BoundaryKind::Wall { .. }))
                .count(),
            2
        );
    }

    #[cfg(all(feature = "io-vtk", feature = "slow-tests"))]
    #[test]
    fn exports_dlr_f6_all_zones_to_vtm_when_present() {
        use std::env;

        let Some(path) = dlr_f6_path() else {
            return;
        };
        let out = env::temp_dir().join("asimu_dlr_f6_all.vts");
        let loaded = export_cgns_to_vtm(&path, &out).expect("export all");
        assert_eq!(loaded.zones.len(), 26);
        let vtm = loaded.vtm_path.expect("vtm path");
        assert!(vtm.is_file());
        let vtm_text = std::fs::read_to_string(&vtm).expect("read vtm");
        assert!(vtm_text.contains("vtkMultiBlockDataSet"));
        assert_eq!(vtm_text.matches("<DataSet ").count(), 26);
        let zone_vts = out.parent().unwrap().join("asimu_dlr_f6_all_zone01.vts");
        assert!(zone_vts.is_file());
        let _ = std::fs::remove_file(vtm);
        for i in 1..=26 {
            let _ = std::fs::remove_file(
                out.parent()
                    .unwrap()
                    .join(format!("asimu_dlr_f6_all_zone{i:02}.vts")),
            );
        }
    }

    #[cfg(all(feature = "io-vtk", feature = "slow-tests"))]
    #[test]
    fn exports_dlr_f6_zone_to_vts_when_present() {
        use std::env;

        let Some(path) = dlr_f6_path() else {
            return;
        };
        let out = env::temp_dir().join("asimu_dlr_f6_zone26.vts");
        let loaded = export_cgns_zone_to_vts(&path, 26, &out).expect("export");
        assert!(out.is_file());
        let roundtrip = crate::io::load_vts(&out).expect("reload vts");
        assert_eq!(roundtrip.mesh.num_nodes(), loaded.mesh.num_nodes());
        let _ = std::fs::remove_file(out);
    }
}
