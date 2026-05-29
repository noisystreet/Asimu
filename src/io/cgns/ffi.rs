//! CGNS MLL 最小 FFI（链接系统 `libcgns`）。

#![allow(unsafe_code)]

use std::os::raw::{c_char, c_int, c_void};

pub const CG_MODE_READ: c_int = 0;
pub const CG_OK: c_int = 0;

pub const ZONE_STRUCTURED: ZoneType = 2;
pub const REAL_DOUBLE: DataType = 4;
pub const BC_POINT_RANGE: PointSetType = 2;

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
}
