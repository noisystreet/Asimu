//! Checkpoint / restart 场数据 I/O（TOML 格式）。

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AsimuError, Result};
use crate::field::ConservedFields;
use crate::field::ScalarField;

const RESTART_VERSION: u32 = 1;

/// Restart 文件内容。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RestartToml {
    version: u32,
    num_cells: usize,
    density: Vec<f64>,
    momentum_x: Vec<f64>,
    momentum_y: Vec<f64>,
    momentum_z: Vec<f64>,
    total_energy: Vec<f64>,
}

/// 从 restart 文件加载守恒场。
pub fn load_conserved_fields(path: &Path) -> Result<ConservedFields> {
    let content = std::fs::read_to_string(path).map_err(|err| {
        AsimuError::Io(std::io::Error::new(
            err.kind(),
            format!("无法读取 restart {}: {err}", path.display()),
        ))
    })?;
    let raw: RestartToml = toml::from_str(&content)?;
    if raw.version != RESTART_VERSION {
        return Err(AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("不支持的 restart 版本 {}", raw.version),
        )));
    }
    let n = raw.num_cells;
    validate_len(n, &raw.density, "density")?;
    validate_len(n, &raw.momentum_x, "momentum_x")?;
    validate_len(n, &raw.momentum_y, "momentum_y")?;
    validate_len(n, &raw.momentum_z, "momentum_z")?;
    validate_len(n, &raw.total_energy, "total_energy")?;
    Ok(ConservedFields {
        density: ScalarField::from_values(raw.density)?,
        momentum_x: ScalarField::from_values(raw.momentum_x)?,
        momentum_y: ScalarField::from_values(raw.momentum_y)?,
        momentum_z: ScalarField::from_values(raw.momentum_z)?,
        total_energy: ScalarField::from_values(raw.total_energy)?,
    })
}

/// 写出 restart 文件。
pub fn write_conserved_fields(path: &Path, fields: &ConservedFields) -> Result<()> {
    let snapshot = RestartToml {
        version: RESTART_VERSION,
        num_cells: fields.num_cells(),
        density: fields.density.values().to_vec(),
        momentum_x: fields.momentum_x.values().to_vec(),
        momentum_y: fields.momentum_y.values().to_vec(),
        momentum_z: fields.momentum_z.values().to_vec(),
        total_energy: fields.total_energy.values().to_vec(),
    };
    let content = toml::to_string_pretty(&snapshot).map_err(|err| {
        AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("序列化 restart 失败: {err}"),
        ))
    })?;
    std::fs::write(path, content)?;
    Ok(())
}

fn validate_len(n: usize, data: &[f64], name: &str) -> Result<()> {
    if data.len() != n {
        return Err(AsimuError::Field(format!(
            "restart {name} 长度 {} 与 num_cells {n} 不一致",
            data.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{FreestreamParams, IdealGasEoS};

    #[test]
    fn restart_roundtrip() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields =
            ConservedFields::from_freestream(4, &eos, &FreestreamParams::default()).expect("fields");
        let path = std::env::temp_dir().join("asimu_restart_test.toml");
        write_conserved_fields(&path, &fields).expect("write");
        let loaded = load_conserved_fields(&path).expect("load");
        assert_eq!(loaded.density.values(), fields.density.values());
        let _ = std::fs::remove_file(path);
    }
}
