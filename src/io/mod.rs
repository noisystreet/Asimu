//! 输入/输出适配层。
//!
//! - 遗留单行 case：`load_mesh_from_case`
//! - VTK VTS（二进制 appended）：feature `io-vtk` → [`vtk::load_vts`]

mod limits;

#[cfg(feature = "io-vtk")]
pub mod vtk;

use std::path::Path;

use crate::error::{AsimuError, Result};
use crate::mesh::Mesh;

pub use limits::{validate_cell_count, validate_file_size, validate_input_path};

#[cfg(feature = "io-vtk")]
pub use vtk::{VtsLoadResult, load_vts};

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
