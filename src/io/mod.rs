//! 输入/输出适配层。
//!
//! - 遗留单行 case：`load_mesh_from_case`
//! - VTK VTS（二进制 appended）：feature `io-vtk` → [`vtk::load_vts`]

mod case;
mod limits;
mod mesh_report;
mod nondimensional;
mod residual;
mod restart;
mod vertex_field;

pub use case::{
    CaseMesh, CaseObservabilityConfig, CaseOutputConfig, CaseSpec, CaseTimeConfig, CaseTimeMode,
    SodCaseConfig, load_case, parse_case_str, resolve_case_output_path,
};
pub use residual::write_residual_csv;
pub use restart::{load_conserved_fields, write_conserved_fields};

#[cfg(feature = "io-cgns")]
pub mod cgns;

#[cfg(feature = "io-vtk")]
pub mod vtk;

use std::path::Path;

use crate::error::{AsimuError, Result};
use crate::mesh::Mesh;

pub use limits::{validate_cell_count, validate_file_size, validate_input_path};
pub use mesh_report::{
    BoundaryPatchSummary, MeshReport, report_case_mesh, report_mesh1d, report_mesh3d,
    report_structured_mesh,
};

#[cfg(feature = "io-cgns")]
pub use mesh_report::report_cgns_zone;

#[cfg(feature = "io-vtk")]
pub use mesh_report::report_vts;

#[cfg(feature = "io-vtk")]
pub use vtk::{
    VtmBlock, VtsLoadResult, load_vts, write_flow_vts, write_flow_vtu, write_vtm, write_vts,
};

#[cfg(feature = "io-cgns-vts")]
pub use cgns::export_cgns_to_vtm;
#[cfg(feature = "io-cgns")]
pub use cgns::{
    Cgns1to1Connection, CgnsLoadResult, CgnsMultiLoadResult, CgnsZoneInfo, export_cgns_to_vts,
    export_cgns_zone_to_vts, list_cgns_zones, load_cgns_all_zones, load_cgns_zone, write_flow_cgns,
    write_multiblock_flow_cgns,
};

/// 从占位 case 文件加载网格元数据。
///
/// 格式约定：`name=<mesh_name>;cells=<count>`
pub fn load_mesh_from_case(path: &Path) -> Result<Mesh> {
    validate_input_path(path)?;
    let content = std::fs::read_to_string(path)?;
    parse_case_content(&content)
}

fn parse_case_content(content: &str) -> Result<Mesh> {
    let mut name = String::from("unnamed");
    let mut cells: Option<usize> = None;

    for part in content.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(value) = part.strip_prefix("name=") {
            name = value.to_string();
        } else if let Some(value) = part.strip_prefix("cells=") {
            cells = Some(value.parse().map_err(|_| {
                AsimuError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "cells 必须为整数",
                ))
            })?);
        }
    }

    let cell_count = cells.ok_or_else(|| {
        AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "缺少 cells 字段",
        ))
    })?;

    validate_cell_count(cell_count as u64)?;
    Mesh::new(name, cell_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_case_content() {
        let mesh = parse_case_content("name=channel;cells=128").expect("parse");
        assert_eq!(mesh.name, "channel");
        assert_eq!(mesh.cell_count, 128);
    }
}
