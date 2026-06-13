//! CGNS 结构化网格读入（系统 `libcgns`）。

#![allow(unsafe_code)]

mod ffi;
mod read;
mod unstructured;
mod write;
mod zonebc;

#[cfg(feature = "io-vtk")]
pub use read::export_cgns_to_vtm;
pub use read::{
    Cgns1to1Connection, CgnsLoadResult, CgnsMultiLoadResult, CgnsZoneInfo, export_cgns_to_vts,
    export_cgns_zone_to_vts, list_cgns_zones, load_cgns_all_zones, load_cgns_zone,
};
pub use unstructured::{CgnsUnstructuredLoadResult, load_cgns_unstructured_zone};
pub use write::{
    StructuredVertexSolution, VertexScalarFieldView, write_flow_cgns, write_flow_cgns_unstructured,
    write_multiblock_flow_cgns, write_structured_vertex_solution_cgns,
};
pub use zonebc::{CgnsPointRange, patch_from_cgns};
