//! CGNS 结构化网格读入（系统 `libcgns`）。

#![allow(unsafe_code)]

mod ffi;
mod read;
mod write;
mod zonebc;

#[cfg(feature = "io-vtk")]
pub use read::export_cgns_to_vtm;
pub use read::{
    CgnsLoadResult, CgnsMultiLoadResult, CgnsZoneInfo, export_cgns_to_vts, export_cgns_zone_to_vts,
    list_cgns_zones, load_cgns_all_zones, load_cgns_zone,
};
pub use write::write_flow_cgns;
pub use zonebc::{CgnsPointRange, patch_from_cgns};
