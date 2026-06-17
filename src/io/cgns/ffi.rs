//! CGNS MLL 最小 FFI（链接系统 `libcgns`）。

#![allow(unsafe_code)]

use std::os::raw::{c_char, c_int, c_void};

pub const CG_MODE_READ: c_int = 0;
pub const CG_OK: c_int = 0;

pub const ZONE_STRUCTURED: ZoneType = 2;
pub const ZONE_UNSTRUCTURED: ZoneType = 3;
pub const REAL_DOUBLE: DataType = 4;
/// CGNS 3.4 `PointSetType_t::PointRange`（旧版枚举值为 2）。
pub const BC_POINT_RANGE: PointSetType = 4;
pub const BC_POINT_LIST: PointSetType = 2;
pub const BC_ELEMENT_RANGE: PointSetType = 6;
pub const BC_ELEMENT_LIST: PointSetType = 7;
pub const GRID_LOCATION_FACE_CENTER: c_int = 4;

pub type ZoneType = c_int;
pub type DataType = c_int;
pub type PointSetType = c_int;
pub type CgSize = c_int;

pub const ELEM_TRI_3: c_int = 5;
pub const ELEM_QUAD_4: c_int = 7;
pub const ELEM_TETRA_4: c_int = 10;
pub const ELEM_PYRA_5: c_int = 12;
pub const ELEM_PENTA_6: c_int = 14;
pub const ELEM_HEXA_8: c_int = 17;
pub const ELEM_MIXED: c_int = 20;

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
    pub fn cg_boco_gridlocation_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        boco: c_int,
        location: *mut c_int,
    ) -> c_int;
    pub fn cg_get_error() -> *const c_char;
    pub fn cg_n1to1(fn_: c_int, base: c_int, zone: c_int, n1to1: *mut c_int) -> c_int;
    pub fn cg_1to1_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        connect: c_int,
        connectname: *mut c_char,
        donorname: *mut c_char,
        range: *mut CgSize,
        donor_range: *mut CgSize,
        transform: *mut c_int,
    ) -> c_int;
    pub fn cg_nsections(fn_: c_int, base: c_int, zone: c_int, nsections: *mut c_int) -> c_int;
    pub fn cg_nsols(fn_: c_int, base: c_int, zone: c_int, nsols: *mut c_int) -> c_int;
    pub fn cg_sol_info(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        sol: c_int,
        solname: *mut c_char,
        location: *mut c_int,
    ) -> c_int;
    pub fn cg_nfields(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        sol: c_int,
        nfields: *mut c_int,
    ) -> c_int;
    pub fn cg_field_info(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        sol: c_int,
        field: c_int,
        datatype: *mut DataType,
        fieldname: *mut c_char,
    ) -> c_int;
    pub fn cg_field_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        sol: c_int,
        fieldname: *const c_char,
        datatype: DataType,
        rmin: *const CgSize,
        rmax: *const CgSize,
        data: *mut c_void,
    ) -> c_int;
    pub fn cg_section_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        section: c_int,
        section_name: *mut c_char,
        element_type: *mut c_int,
        start: *mut CgSize,
        end: *mut CgSize,
        nbndry: *mut c_int,
        parent_flag: *mut c_int,
    ) -> c_int;
    pub fn cg_ElementDataSize(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        section: c_int,
        element_data_size: *mut CgSize,
    ) -> c_int;
    pub fn cg_elements_read(
        fn_: c_int,
        base: c_int,
        zone: c_int,
        section: c_int,
        elements: *mut CgSize,
        parent_data: *mut CgSize,
    ) -> c_int;
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
    pub fn asimu_cg_write_multiblock_structured_flow(
        filename: *const c_char,
        basename: *const c_char,
        zone_count: c_int,
        zonenames: *const *const c_char,
        nx: *const c_int,
        ny: *const c_int,
        nz: *const c_int,
        points_x: *const *const f64,
        points_y: *const *const f64,
        points_z: *const *const f64,
        rho: *const *const f64,
        u: *const *const f64,
        v: *const *const f64,
        w: *const *const f64,
        p: *const *const f64,
        mach: *const *const f64,
        temperature: *const *const f64,
        physical_time: f64,
    ) -> c_int;
    pub fn asimu_cg_write_structured_solution_fields(
        filename: *const c_char,
        basename: *const c_char,
        zonename: *const c_char,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        points_x: *const f64,
        points_y: *const f64,
        points_z: *const f64,
        field_count: c_int,
        field_names: *const *const c_char,
        field_values: *const *const f64,
        physical_time: f64,
    ) -> c_int;
    pub fn asimu_cg_write_unstructured_flow(
        filename: *const c_char,
        basename: *const c_char,
        zonename: *const c_char,
        num_nodes: c_int,
        num_cells: c_int,
        points_x: *const f64,
        points_y: *const f64,
        points_z: *const f64,
        section_count: c_int,
        section_names: *const *const c_char,
        element_types: *const c_int,
        section_starts: *const c_int,
        section_ends: *const c_int,
        section_connectivity: *const *const CgSize,
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
