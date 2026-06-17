//! Checkpoint / restart 场数据 I/O（TOML 格式；ADR 0016 §6）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::{ComputeFloat, ComputePrecision, parse_compute_precision};
use crate::error::{AsimuError, Result};
use crate::field::{ConservedFields, ConservedFieldsT, ScalarField};
use crate::mesh::StructuredBlock3d;
use crate::physics::{
    ConservedState, FreestreamContext, FreestreamParams, IdealGasEoS, PrimitiveState,
    ReferenceScales,
};

const RESTART_VERSION_SINGLE: u32 = 1;
const RESTART_VERSION_MULTIBLOCK: u32 = 2;

/// Restart 文件内标注的核心计算精度；缺省视为 `f64`（兼容旧文件）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartPrecision(pub ComputePrecision);

impl RestartPrecision {
    #[must_use]
    pub const fn compute_precision(self) -> ComputePrecision {
        self.0
    }
}

/// Restart 文件内容（单 block，version = 1）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SingleRestartToml {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compute_precision: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compute_precision: Option<String>,
    blocks: Vec<BlockRestartToml>,
}

enum RestartPayload {
    Single {
        precision: RestartPrecision,
        fields: ConservedFields,
    },
    Multiblock {
        precision: RestartPrecision,
        blocks: Vec<(String, ConservedFields)>,
    },
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
    expected_precision: ComputePrecision,
) -> Result<Vec<ConservedFields>> {
    if let Some(path) = restart {
        let names: Vec<&str> = blocks.iter().map(|block| block.name.as_str()).collect();
        return load_multiblock_conserved_fields_checked(path, &names, expected_precision);
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

/// 从 restart 文件加载守恒场（单 block）；不校验 case 精度（兼容旧调用）。
pub fn load_conserved_fields(path: &Path) -> Result<ConservedFields> {
    match read_restart_file(path)? {
        RestartPayload::Single { fields, .. } => Ok(fields),
        RestartPayload::Multiblock { .. } => Err(AsimuError::Field(
            "restart version=2 含多个 block，请使用 load_multiblock_conserved_fields".to_string(),
        )),
    }
}

/// 读取 restart 文件标注的 `compute_precision`（缺省 `f64`）。
pub fn read_restart_precision(path: &Path) -> Result<RestartPrecision> {
    Ok(match read_restart_file(path)? {
        RestartPayload::Single { precision, .. } => precision,
        RestartPayload::Multiblock { precision, .. } => precision,
    })
}

/// 从 restart 文件加载单 block 守恒场，并校验与 case 精度一致。
pub fn load_conserved_fields_checked(
    path: &Path,
    expected: ComputePrecision,
) -> Result<ConservedFields> {
    match read_restart_file(path)? {
        RestartPayload::Single { precision, fields } => {
            ensure_restart_precision_matches(precision, expected)?;
            Ok(fields)
        }
        RestartPayload::Multiblock { .. } => Err(AsimuError::Field(
            "restart version=2 含多个 block，请使用 load_multiblock_conserved_fields_checked"
                .to_string(),
        )),
    }
}

/// 从非结构 `flow.cgns`（CellCenter 原始量）加载并转换为无量纲守恒场。
///
/// 约束：
/// - 当前仅读取 zone=1；
/// - 输入 `flow.cgns` 为有量纲 SI，转换时按 `reference` 缩放至 \(*\) 变量；
/// - 守恒量通过 `ConservedState::from_primitive` 与 `eos` 一致重建。
#[cfg(feature = "io-cgns")]
pub fn load_conserved_fields_from_flow_cgns(
    path: &Path,
    expected_num_cells: usize,
    eos: &IdealGasEoS,
    reference: &ReferenceScales,
) -> Result<ConservedFields> {
    let flow = crate::io::cgns::load_cgns_unstructured_flow_solution(path, 1)?;
    if flow.zone.nx != expected_num_cells {
        return Err(AsimuError::Field(format!(
            "flow.cgns 单元数 {} 与网格 {} 不一致",
            flow.zone.nx, expected_num_cells
        )));
    }
    let mut density = Vec::with_capacity(expected_num_cells);
    let mut momentum_x = Vec::with_capacity(expected_num_cells);
    let mut momentum_y = Vec::with_capacity(expected_num_cells);
    let mut momentum_z = Vec::with_capacity(expected_num_cells);
    let mut total_energy = Vec::with_capacity(expected_num_cells);
    for i in 0..expected_num_cells {
        let rho_si = flow.density[i];
        let ux_si = flow.velocity_x[i];
        let uy_si = flow.velocity_y[i];
        let uz_si = flow.velocity_z[i];
        let p_si = flow.pressure[i];
        if !rho_si.is_finite()
            || !ux_si.is_finite()
            || !uy_si.is_finite()
            || !uz_si.is_finite()
            || !p_si.is_finite()
        {
            return Err(AsimuError::Field(format!(
                "flow.cgns 第 {i} 个单元含非有限值"
            )));
        }
        if rho_si <= 0.0 || p_si <= 0.0 {
            return Err(AsimuError::Field(format!(
                "flow.cgns 第 {i} 个单元密度/压力非正：rho={rho_si}, p={p_si}"
            )));
        }
        let rho = rho_si / reference.density;
        let pressure = p_si / reference.pressure;
        let velocity = [
            ux_si / reference.velocity,
            uy_si / reference.velocity,
            uz_si / reference.velocity,
        ];
        let temperature = pressure / (rho * eos.gas_constant);
        let cons = ConservedState::from_primitive(
            eos,
            &PrimitiveState {
                density: rho,
                velocity,
                pressure,
                temperature,
            },
        )?;
        density.push(cons.density);
        momentum_x.push(cons.momentum[0]);
        momentum_y.push(cons.momentum[1]);
        momentum_z.push(cons.momentum[2]);
        total_energy.push(cons.total_energy);
    }
    fields_from_arrays(
        expected_num_cells,
        &density,
        &momentum_x,
        &momentum_y,
        &momentum_z,
        &total_energy,
    )
}

/// `io-cgns` 关闭时占位报错，避免静默回退到错误格式。
#[cfg(not(feature = "io-cgns"))]
pub fn load_conserved_fields_from_flow_cgns(
    _path: &Path,
    _expected_num_cells: usize,
    _eos: &IdealGasEoS,
    _reference: &ReferenceScales,
) -> Result<ConservedFields> {
    Err(AsimuError::Config(
        "从 flow.cgns 读取初场须启用 feature io-cgns".to_string(),
    ))
}

/// 从 restart 加载 typed 守恒场（校验文件精度与 `T` 一致）。
pub fn load_conserved_fields_typed<T: ComputeFloat>(path: &Path) -> Result<ConservedFieldsT<T>> {
    let fields = load_conserved_fields_checked(path, T::PRECISION)?;
    ConservedFieldsT::from_real_fields(&fields)
}

/// 按 mesh block 顺序从 restart 文件加载多块守恒场（不校验 case 精度）。
pub fn load_multiblock_conserved_fields(
    path: &Path,
    block_names: &[&str],
) -> Result<Vec<ConservedFields>> {
    match read_restart_file(path)? {
        RestartPayload::Single { fields, .. } => {
            if block_names.len() != 1 {
                return Err(AsimuError::Field(format!(
                    "restart version=1 仅适用于单 block 网格，当前 mesh 含 {} 个 block",
                    block_names.len()
                )));
            }
            Ok(vec![fields])
        }
        RestartPayload::Multiblock { blocks, .. } => {
            assemble_multiblock_fields(blocks, block_names)
        }
    }
}

/// 按 mesh block 顺序加载多块 restart，并校验与 case 精度一致。
pub fn load_multiblock_conserved_fields_checked(
    path: &Path,
    block_names: &[&str],
    expected: ComputePrecision,
) -> Result<Vec<ConservedFields>> {
    match read_restart_file(path)? {
        RestartPayload::Single { .. } => {
            if block_names.len() != 1 {
                return Err(AsimuError::Field(format!(
                    "restart version=1 仅适用于单 block 网格，当前 mesh 含 {} 个 block",
                    block_names.len()
                )));
            }
            load_conserved_fields_checked(path, expected).map(|fields| vec![fields])
        }
        RestartPayload::Multiblock { precision, blocks } => {
            ensure_restart_precision_matches(precision, expected)?;
            assemble_multiblock_fields(blocks, block_names)
        }
    }
}

/// 写出 restart 文件（单 block，`f64`）。
pub fn write_conserved_fields(path: &Path, fields: &ConservedFields) -> Result<()> {
    write_conserved_fields_with_precision(path, fields, ComputePrecision::F64)
}

/// 写出带精度标注的单 block restart。
pub fn write_conserved_fields_with_precision(
    path: &Path,
    fields: &ConservedFields,
    precision: ComputePrecision,
) -> Result<()> {
    let snapshot = SingleRestartToml {
        version: RESTART_VERSION_SINGLE,
        compute_precision: serialize_restart_precision(precision),
        num_cells: fields.num_cells(),
        density: fields.density.values().to_vec(),
        momentum_x: fields.momentum_x.values().to_vec(),
        momentum_y: fields.momentum_y.values().to_vec(),
        momentum_z: fields.momentum_z.values().to_vec(),
        total_energy: fields.total_energy.values().to_vec(),
    };
    write_restart_toml(path, &snapshot)
}

/// 写出 typed 单 block restart（精度由 `T` 决定）。
pub fn write_conserved_fields_typed<T: ComputeFloat>(
    path: &Path,
    fields: &ConservedFieldsT<T>,
) -> Result<()> {
    write_conserved_fields_with_precision(path, &fields.cast_real()?, T::PRECISION)
}

/// 写出多块 restart 文件（version = 2，`f64`）。
pub fn write_multiblock_conserved_fields(
    path: &Path,
    blocks: &[(&str, &ConservedFields)],
) -> Result<()> {
    write_multiblock_conserved_fields_with_precision(path, blocks, ComputePrecision::F64)
}

/// 写出带精度标注的多块 restart。
pub fn write_multiblock_conserved_fields_with_precision(
    path: &Path,
    blocks: &[(&str, &ConservedFields)],
    precision: ComputePrecision,
) -> Result<()> {
    let snapshot = MultiblockRestartToml {
        version: RESTART_VERSION_MULTIBLOCK,
        compute_precision: serialize_restart_precision(precision),
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
            let (precision, fields) = fields_from_single(raw)?;
            Ok(RestartPayload::Single { precision, fields })
        }
        RESTART_VERSION_MULTIBLOCK => {
            let raw: MultiblockRestartToml = toml::from_str(&content)?;
            if raw.version != RESTART_VERSION_MULTIBLOCK {
                return Err(restart_version_error(raw.version));
            }
            let precision = parse_restart_precision_field(raw.compute_precision.as_deref())?;
            let mut blocks = Vec::with_capacity(raw.blocks.len());
            for block in raw.blocks {
                blocks.push(fields_from_block(block)?);
            }
            Ok(RestartPayload::Multiblock { precision, blocks })
        }
        other => Err(restart_version_error(other)),
    }
}

fn parse_restart_precision_field(raw: Option<&str>) -> Result<RestartPrecision> {
    Ok(RestartPrecision(match raw {
        None => ComputePrecision::F64,
        Some(value) => parse_compute_precision(value)?,
    }))
}

fn serialize_restart_precision(precision: ComputePrecision) -> Option<String> {
    match precision {
        ComputePrecision::F64 => None,
        ComputePrecision::F32 => Some(ComputePrecision::F32.label().to_string()),
    }
}

fn ensure_restart_precision_matches(
    file: RestartPrecision,
    expected: ComputePrecision,
) -> Result<()> {
    if file.0 != expected {
        return Err(AsimuError::Field(format!(
            "restart compute_precision = \"{}\" 与 case [numerics] \"{}\" 不一致；跨精度 restart 暂不支持",
            file.0.label(),
            expected.label()
        )));
    }
    Ok(())
}

fn fields_from_single(raw: SingleRestartToml) -> Result<(RestartPrecision, ConservedFields)> {
    if raw.version != RESTART_VERSION_SINGLE {
        return Err(restart_version_error(raw.version));
    }
    let precision = parse_restart_precision_field(raw.compute_precision.as_deref())?;
    let fields = fields_from_arrays(
        raw.num_cells,
        &raw.density,
        &raw.momentum_x,
        &raw.momentum_y,
        &raw.momentum_z,
        &raw.total_energy,
    )?;
    Ok((precision, fields))
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

    #[test]
    fn f32_restart_roundtrip_and_precision_tag() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let real =
            ConservedFields::from_freestream(3, &eos, &FreestreamParams::default()).expect("f64");
        let typed = ConservedFieldsT::<f32>::from_real_fields(&real).expect("f32");
        let path = std::env::temp_dir().join("asimu_restart_f32_test.toml");
        write_conserved_fields_typed(&path, &typed).expect("write");
        assert_eq!(
            read_restart_precision(&path)
                .expect("precision")
                .compute_precision(),
            ComputePrecision::F32
        );
        let loaded = load_conserved_fields_typed::<f32>(&path).expect("load f32");
        assert_eq!(
            loaded.density.to_real_values(),
            typed.density.to_real_values()
        );
        let err = load_conserved_fields_checked(&path, ComputePrecision::F64).expect_err("cross");
        assert!(err.to_string().contains("不一致"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn legacy_restart_without_tag_is_f64_only() {
        let eos = IdealGasEoS::AIR_STANDARD;
        let fields = ConservedFields::from_freestream(2, &eos, &FreestreamParams::default())
            .expect("fields");
        let path = std::env::temp_dir().join("asimu_restart_legacy_test.toml");
        write_conserved_fields(&path, &fields).expect("write");
        assert_eq!(
            read_restart_precision(&path)
                .expect("precision")
                .compute_precision(),
            ComputePrecision::F64
        );
        load_conserved_fields_checked(&path, ComputePrecision::F64).expect("f64 case");
        let err =
            load_conserved_fields_checked(&path, ComputePrecision::F32).expect_err("f32 case");
        assert!(err.to_string().contains("f64"));
        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "io-cgns")]
    #[test]
    fn loads_unstructured_flow_cgns_and_converts_to_nondimensional_conserved() {
        use crate::io::write_flow_cgns_unstructured;
        use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};
        use crate::physics::ReferenceScales;

        let mesh = UnstructuredMesh3d::new(
            "tet",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![UnstructuredCell::new(CellKind::Tet, vec![0, 1, 2, 3]).expect("tet")],
        )
        .expect("mesh");
        let eos_dim = IdealGasEoS::AIR_STANDARD;
        let fs_dim = FreestreamParams {
            mach: 0.25,
            pressure: 101_325.0,
            temperature: 300.0,
            ..FreestreamParams::default()
        };
        let reference = ReferenceScales::from_freestream(&eos_dim, &fs_dim, None).expect("ref");
        let mut eos_nd = eos_dim;
        eos_nd.gas_constant = reference.nondimensional_gas_constant();
        let fields_nd = ConservedFields::from_freestream(mesh.num_cells(), &eos_dim, &fs_dim)
            .expect("nondimensional fields");
        let fields_dim = fields_nd
            .to_dimensional(&reference)
            .expect("dimensional fields");

        let path = std::env::temp_dir().join("asimu_restart_from_flow_cgns_test.cgns");
        write_flow_cgns_unstructured(&path, &mesh, &fields_dim, &eos_dim, 0.0, 1.0e-6)
            .expect("write flow cgns");
        let loaded =
            load_conserved_fields_from_flow_cgns(&path, mesh.num_cells(), &eos_nd, &reference)
                .expect("load from flow cgns");

        assert!((loaded.density.values()[0] - fields_nd.density.values()[0]).abs() < 1.0e-10);
        assert!(
            (loaded.total_energy.values()[0] - fields_nd.total_energy.values()[0]).abs() < 1.0e-10
        );
        let _ = std::fs::remove_file(path);
    }
}
