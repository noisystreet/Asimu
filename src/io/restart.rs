//! Checkpoint / restart 场数据 I/O（TOML 格式）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AsimuError, Result};
use crate::field::ConservedFields;
use crate::field::ScalarField;
use crate::mesh::StructuredBlock3d;
use crate::physics::{FreestreamContext, FreestreamParams, IdealGasEoS, ReferenceScales};

const RESTART_VERSION_SINGLE: u32 = 1;
const RESTART_VERSION_MULTIBLOCK: u32 = 2;

/// Restart 文件内容（单 block，version = 1）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SingleRestartToml {
    version: u32,
    num_cells: usize,
    density: Vec<f64>,
    momentum_x: Vec<f64>,
    momentum_y: Vec<f64>,
    momentum_z: Vec<f64>,
    total_energy: Vec<f64>,
}

/// Restart 文件内容（单 block 条目，version = 2）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BlockRestartToml {
    name: String,
    num_cells: usize,
    density: Vec<f64>,
    momentum_x: Vec<f64>,
    momentum_y: Vec<f64>,
    momentum_z: Vec<f64>,
    total_energy: Vec<f64>,
}

/// Restart 文件内容（多块，version = 2）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct MultiblockRestartToml {
    version: u32,
    blocks: Vec<BlockRestartToml>,
}

enum RestartPayload {
    Single(ConservedFields),
    Multiblock(Vec<(String, ConservedFields)>),
}

/// 解析 case TOML 中的 restart 相对路径。
pub fn resolve_restart_path(path: PathBuf, case_dir: Option<&Path>) -> PathBuf {
    if path.is_absolute() {
        path
    } else if let Some(dir) = case_dir {
        dir.join(path)
    } else {
        path
    }
}

/// 按 block 顺序构建多块守恒初场（restart 或均匀来流）。
pub fn initial_multiblock_conserved_fields(
    restart: Option<&Path>,
    blocks: &[StructuredBlock3d],
    eos: &IdealGasEoS,
    reference: Option<&ReferenceScales>,
    viscous: Option<&crate::physics::ViscousPhysicsConfig>,
    freestream: Option<FreestreamParams>,
) -> Result<Vec<ConservedFields>> {
    if let Some(path) = restart {
        let names: Vec<&str> = blocks.iter().map(|block| block.name.as_str()).collect();
        return load_multiblock_conserved_fields(path, &names);
    }
    let fs = freestream
        .ok_or_else(|| AsimuError::Field("须指定 [freestream] 或 [restart]".to_string()))?;
    let ctx = FreestreamContext::new(eos, reference, viscous);
    blocks
        .iter()
        .map(|block| ConservedFields::from_freestream_context(block.mesh.num_cells(), &ctx, &fs))
        .collect()
}

/// 从均匀来流构建单块守恒初场。
pub fn initial_freestream_conserved_fields(
    num_cells: usize,
    eos: &IdealGasEoS,
    reference: Option<&ReferenceScales>,
    viscous: Option<&crate::physics::ViscousPhysicsConfig>,
    freestream: Option<FreestreamParams>,
) -> Result<ConservedFields> {
    let fs = freestream
        .ok_or_else(|| AsimuError::Field("须指定 [freestream] 或 [restart]".to_string()))?;
    let ctx = FreestreamContext::new(eos, reference, viscous);
    ConservedFields::from_freestream_context(num_cells, &ctx, &fs)
}

/// 从 restart 文件加载守恒场（单 block）。
pub fn load_conserved_fields(path: &Path) -> Result<ConservedFields> {
    match read_restart_file(path)? {
        RestartPayload::Single(fields) => Ok(fields),
        RestartPayload::Multiblock(_) => Err(AsimuError::Field(
            "restart version=2 含多个 block，请使用 load_multiblock_conserved_fields".to_string(),
        )),
    }
}

/// 按 mesh block 顺序从 restart 文件加载多块守恒场。
pub fn load_multiblock_conserved_fields(
    path: &Path,
    block_names: &[&str],
) -> Result<Vec<ConservedFields>> {
    match read_restart_file(path)? {
        RestartPayload::Single(fields) => {
            if block_names.len() != 1 {
                return Err(AsimuError::Field(format!(
                    "restart version=1 仅适用于单 block 网格，当前 mesh 含 {} 个 block",
                    block_names.len()
                )));
            }
            Ok(vec![fields])
        }
        RestartPayload::Multiblock(blocks) => assemble_multiblock_fields(blocks, block_names),
    }
}

/// 写出 restart 文件（单 block）。
pub fn write_conserved_fields(path: &Path, fields: &ConservedFields) -> Result<()> {
    let snapshot = SingleRestartToml {
        version: RESTART_VERSION_SINGLE,
        num_cells: fields.num_cells(),
        density: fields.density.values().to_vec(),
        momentum_x: fields.momentum_x.values().to_vec(),
        momentum_y: fields.momentum_y.values().to_vec(),
        momentum_z: fields.momentum_z.values().to_vec(),
        total_energy: fields.total_energy.values().to_vec(),
    };
    write_restart_toml(path, &snapshot)
}

/// 写出多块 restart 文件（version = 2）。
pub fn write_multiblock_conserved_fields(
    path: &Path,
    blocks: &[(&str, &ConservedFields)],
) -> Result<()> {
    let snapshot = MultiblockRestartToml {
        version: RESTART_VERSION_MULTIBLOCK,
        blocks: blocks
            .iter()
            .map(|(name, fields)| BlockRestartToml {
                name: (*name).to_string(),
                num_cells: fields.num_cells(),
                density: fields.density.values().to_vec(),
                momentum_x: fields.momentum_x.values().to_vec(),
                momentum_y: fields.momentum_y.values().to_vec(),
                momentum_z: fields.momentum_z.values().to_vec(),
                total_energy: fields.total_energy.values().to_vec(),
            })
            .collect(),
    };
    write_restart_toml(path, &snapshot)
}

fn read_restart_file(path: &Path) -> Result<RestartPayload> {
    let content = read_restart_text(path)?;
    let version = parse_restart_version(&content)?;
    match version {
        RESTART_VERSION_SINGLE => {
            let raw: SingleRestartToml = toml::from_str(&content)?;
            Ok(RestartPayload::Single(fields_from_single(raw)?))
        }
        RESTART_VERSION_MULTIBLOCK => {
            let raw: MultiblockRestartToml = toml::from_str(&content)?;
            if raw.version != RESTART_VERSION_MULTIBLOCK {
                return Err(restart_version_error(raw.version));
            }
            let mut blocks = Vec::with_capacity(raw.blocks.len());
            for block in raw.blocks {
                let fields = fields_from_block(block)?;
                blocks.push(fields);
            }
            Ok(RestartPayload::Multiblock(blocks))
        }
        other => Err(restart_version_error(other)),
    }
}

fn parse_restart_version(content: &str) -> Result<u32> {
    let table: toml::Table = toml::from_str(content)?;
    table
        .get("version")
        .and_then(toml::Value::as_integer)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            AsimuError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "restart 缺少 version 字段",
            ))
        })
}

fn fields_from_single(raw: SingleRestartToml) -> Result<ConservedFields> {
    if raw.version != RESTART_VERSION_SINGLE {
        return Err(restart_version_error(raw.version));
    }
    fields_from_arrays(
        raw.num_cells,
        &raw.density,
        &raw.momentum_x,
        &raw.momentum_y,
        &raw.momentum_z,
        &raw.total_energy,
    )
}

fn fields_from_block(raw: BlockRestartToml) -> Result<(String, ConservedFields)> {
    let fields = fields_from_arrays(
        raw.num_cells,
        &raw.density,
        &raw.momentum_x,
        &raw.momentum_y,
        &raw.momentum_z,
        &raw.total_energy,
    )?;
    Ok((raw.name, fields))
}

fn fields_from_arrays(
    num_cells: usize,
    density: &[f64],
    momentum_x: &[f64],
    momentum_y: &[f64],
    momentum_z: &[f64],
    total_energy: &[f64],
) -> Result<ConservedFields> {
    validate_len(num_cells, density, "density")?;
    validate_len(num_cells, momentum_x, "momentum_x")?;
    validate_len(num_cells, momentum_y, "momentum_y")?;
    validate_len(num_cells, momentum_z, "momentum_z")?;
    validate_len(num_cells, total_energy, "total_energy")?;
    Ok(ConservedFields {
        density: ScalarField::from_values(density.to_vec())?,
        momentum_x: ScalarField::from_values(momentum_x.to_vec())?,
        momentum_y: ScalarField::from_values(momentum_y.to_vec())?,
        momentum_z: ScalarField::from_values(momentum_z.to_vec())?,
        total_energy: ScalarField::from_values(total_energy.to_vec())?,
    })
}

fn assemble_multiblock_fields(
    blocks: Vec<(String, ConservedFields)>,
    block_names: &[&str],
) -> Result<Vec<ConservedFields>> {
    let by_name: HashMap<String, ConservedFields> = blocks.into_iter().collect();
    let mut out = Vec::with_capacity(block_names.len());
    for name in block_names {
        let fields = by_name
            .get(*name)
            .ok_or_else(|| AsimuError::Field(format!("restart 缺少 block \"{name}\"")))?;
        out.push(fields.clone());
    }
    Ok(out)
}

fn read_restart_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|err| {
        AsimuError::Io(std::io::Error::new(
            err.kind(),
            format!("无法读取 restart {}: {err}", path.display()),
        ))
    })
}

fn write_restart_toml<T: Serialize>(path: &Path, snapshot: &T) -> Result<()> {
    let content = toml::to_string_pretty(snapshot).map_err(|err| {
        AsimuError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("序列化 restart 失败: {err}"),
        ))
    })?;
    std::fs::write(path, content)?;
    Ok(())
}

fn restart_version_error(version: u32) -> AsimuError {
    AsimuError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("不支持的 restart 版本 {version}"),
    ))
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
        let fields = ConservedFields::from_freestream(4, &eos, &FreestreamParams::default())
            .expect("fields");
        let path = std::env::temp_dir().join("asimu_restart_test.toml");
        write_conserved_fields(&path, &fields).expect("write");
        let loaded = load_conserved_fields(&path).expect("load");
        assert_eq!(loaded.density.values(), fields.density.values());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn multiblock_restart_roundtrip() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let block_a = ConservedFields::from_freestream(2, &eos, &FreestreamParams::default())
            .expect("block a");
        let block_b = ConservedFields::from_freestream(3, &eos, &FreestreamParams::default())
            .expect("block b");
        let path = std::env::temp_dir().join("asimu_multiblock_restart_test.toml");
        write_multiblock_conserved_fields(&path, &[("a", &block_a), ("b", &block_b)])
            .expect("write");
        let loaded = load_multiblock_conserved_fields(&path, &["a", "b"]).expect("load ordered");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].num_cells(), 2);
        assert_eq!(loaded[1].num_cells(), 3);
        assert_eq!(loaded[0].density.values(), block_a.density.values());
        assert_eq!(loaded[1].density.values(), block_b.density.values());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn multiblock_restart_rejects_missing_block() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let block_a = ConservedFields::from_freestream(1, &eos, &FreestreamParams::default())
            .expect("block a");
        let path = std::env::temp_dir().join("asimu_multiblock_restart_missing_test.toml");
        write_multiblock_conserved_fields(&path, &[("a", &block_a)]).expect("write");
        let err = load_multiblock_conserved_fields(&path, &["a", "b"]).expect_err("missing");
        assert!(err.to_string().contains("缺少 block \"b\""));
        let _ = std::fs::remove_file(path);
    }
}
