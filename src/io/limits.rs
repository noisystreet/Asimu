//! I/O 资源上限与路径校验（Parse 阶段）。

use std::path::{Component, Path};

use crate::error::{AsimuError, Result};

/// 单文件大小上限（与 SECURITY.md 一致）。
pub const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// 最大单元数（与 SECURITY.md 一致）。
pub const MAX_CELLS: u64 = 100_000_000;

pub fn io_error(kind: std::io::ErrorKind, message: impl Into<String>) -> AsimuError {
    AsimuError::Io(std::io::Error::new(kind, message.into()))
}

pub fn validate_input_path(path: &Path) -> Result<()> {
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(io_error(
            std::io::ErrorKind::PermissionDenied,
            "路径不得包含 '..'",
        ));
    }
    Ok(())
}

pub fn validate_file_size(bytes: u64, label: &str) -> Result<()> {
    if bytes > MAX_FILE_BYTES {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("{label} 超过大小上限 {MAX_FILE_BYTES} 字节"),
        ));
    }
    Ok(())
}

pub fn validate_cell_count(cells: u64) -> Result<()> {
    if cells == 0 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "单元数必须大于 0",
        ));
    }
    if cells > MAX_CELLS {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("单元数 {cells} 超过上限 {MAX_CELLS}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn rejects_parent_dir() {
        let path = PathBuf::from("../secret/mesh.vts");
        assert!(validate_input_path(&path).is_err());
    }
}
