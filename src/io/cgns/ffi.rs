//! CGNS MLL 最小 FFI（链接系统 `libcgns`）。

#![allow(unsafe_code)]

use std::os::raw::{c_char, c_int, c_void};

pub const CG_MODE_READ: c_int = 0;
pub const CG_OK: c_int = 0;

pub const ZONE_STRUCTURED: ZoneType = 2;
pub const REAL_DOUBLE: DataType = 4;
/// CGNS 3.4 `PointSetType_t::PointRange`（旧版枚举值为 2）。
pub const BC_POINT_RANGE: PointSetType = 4;

pub type ZoneType = c_int;
pub type DataType = c_int;
pub type PointSetType = c_int;
pub type CgSize = c_int;

unsafe extern "C" {
    pub fn cg_open(filename: *const c_char, mode: c_int, fn_: *mut c_int) -> c_int;
    pub fn cg_close(fn_: c_int) -> c_int;
    pub fn cg_nzones(fn_: c_int, base: c_int, nzones: *mut c_int) -> c_int;
    pub fn cg_zone_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        zonename: *mut c_char,
        size: *mut CgSize,
    ) -> c_int;
    pub fn cg_zone_type(fn_: c_int, base: c_int, zone: c_int, type_: *mut ZoneType) -> c_int;
    pub fn cg_coord_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        coordname: *const c_char,
        datatype: DataType,
        rmin: *const CgSize,
        rmax: *const CgSize,
        data: *mut c_void,
    ) -> c_int;
    pub fn cg_nfamilies(fn_: c_int, base: c_int, nfamilies: *mut c_int) -> c_int;
    pub fn cg_family_read(
        fn_: c_int,
        base: c_int,
        family: c_int,
        family_name: *mut c_char,
        nbocos: *mut c_int,
        ngeos: *mut c_int,
    ) -> c_int;
    pub fn cg_fambc_read(
        fn_: c_int,
        base: c_int,
        family: c_int,
        bc: c_int,
        fambc_name: *mut c_char,
        bocotype: *mut c_int,
    ) -> c_int;
    pub fn cg_nbocos(fn_: c_int, base: c_int, zone: c_int, nbocos: *mut c_int) -> c_int;
    pub fn cg_boco_info(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        boco: c_int,
        boconame: *mut c_char,
        bocotype: *mut c_int,
        ptset_type: *mut PointSetType,
        npnts: *mut c_int,
        normalindex: *mut c_int,
        normal_list_size: *mut c_int,
        normaldatatype: *mut DataType,
        ndataset: *mut c_int,
    ) -> c_int;
    pub fn cg_boco_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        boco: c_int,
        pnts: *mut c_void,
        normal_list: *mut c_void,
    ) -> c_int;
    pub fn cg_get_error() -> *const c_char;
    pub fn asimu_cg_read_boco_family_name(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        boco: c_int,
        family_name: *mut c_char,
    ) -> c_int;
    #[allow(dead_code)]
    pub fn asimu_cg_write_structured_flow(
        filename: *const c_char,
        basename: *const c_char,
        zonename: *const c_char,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        points_x: *const f64,
        points_y: *const f64,
        points_z: *const f64,
        rho: *const f64,
        u: *const f64,
        v: *const f64,
        w: *const f64,
        p: *const f64,
        mach: *const f64,
        temperature: *const f64,
        physical_time: f64,
    ) -> c_int;
}
