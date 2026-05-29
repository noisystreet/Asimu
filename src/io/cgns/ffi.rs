//! CGNS MLL 最小 FFI（链接系统 `libcgns`）。

#![allow(unsafe_code)]

use std::os::raw::{c_char, c_int, c_void};

pub const CG_MODE_READ: c_int = 0;
pub const CG_OK: c_int = 0;

pub const ZONE_STRUCTURED: ZoneType = 2;
pub const REAL_DOUBLE: DataType = 4;

pub type ZoneType = c_int;
pub type DataType = c_int;
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
    pub fn cg_get_error() -> *const c_char;
}
