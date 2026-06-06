//! VTK XML UnstructuredGrid（`.vtu`）读入。
//!
//! 支持 ASCII 与 inline `format="binary"` DataArray；首版只读取 Points 与 Cells。

use std::path::Path;

use base64::Engine;
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

use crate::error::Result;
use crate::io::limits::{io_error, validate_cell_count, validate_file_size, validate_input_path};
use crate::mesh::{CellKind, UnstructuredCell, UnstructuredMesh3d};

#[derive(Debug, Clone, PartialEq)]
pub struct VtuLoadResult {
    pub mesh: UnstructuredMesh3d,
}

pub fn load_vtu(path: &Path) -> Result<VtuLoadResult> {
    validate_input_path(path)?;
    let bytes = std::fs::read(path)?;
    validate_file_size(bytes.len() as u64, "VTU 文件")?;
    let content = std::str::from_utf8(&bytes).map_err(|err| {
        io_error(
            std::io::ErrorKind::InvalidData,
            format!("VTU 非 UTF-8 XML: {err}"),
        )
    })?;
    let mesh_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("vtu");
    load_vtu_from_str(mesh_name, content)
}

fn load_vtu_from_str(mesh_name: &str, content: &str) -> Result<VtuLoadResult> {
    let parsed = parse_vtu_xml(content)?;
    validate_cell_count(parsed.num_cells as u64)?;
    let points = decode_points(&parsed.points, parsed.num_points)?;
    let connectivity = decode_usize_array(&parsed.connectivity)?;
    let offsets = decode_usize_array(&parsed.offsets)?;
    let types = decode_u8_array(&parsed.types)?;
    let cells = build_cells(&connectivity, &offsets, &types)?;
    Ok(VtuLoadResult {
        mesh: UnstructuredMesh3d::new(mesh_name, points, cells)?,
    })
}

#[derive(Debug, Clone)]
struct VtuDataArray {
    vtk_type: String,
    format: String,
    text: String,
    components: usize,
}

#[derive(Debug)]
struct ParsedVtu {
    num_points: usize,
    num_cells: usize,
    points: VtuDataArray,
    connectivity: VtuDataArray,
    offsets: VtuDataArray,
    types: VtuDataArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum XmlContext {
    #[default]
    None,
    Points,
    Cells,
    DataArray(DataArrayTarget),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataArrayTarget {
    Points,
    Connectivity,
    Offsets,
    Types,
}

#[derive(Default)]
struct VtuParseState {
    saw_unstructured_grid: bool,
    num_points: Option<usize>,
    num_cells: Option<usize>,
    byte_order: Option<String>,
    context: XmlContext,
    current_array: Option<VtuDataArray>,
    points: Option<VtuDataArray>,
    connectivity: Option<VtuDataArray>,
    offsets: Option<VtuDataArray>,
    types: Option<VtuDataArray>,
}

fn parse_vtu_xml(content: &str) -> Result<ParsedVtu> {
    let mut reader = Reader::from_str(content);
    reader.config_mut().trim_text(true);
    let mut state = VtuParseState::default();

    loop {
        match reader
            .read_event()
            .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?
        {
            Event::Start(e) => apply_start(&e, &mut state)?,
            Event::Empty(e) => apply_empty(&e, &mut state)?,
            Event::Text(e) => append_text(&e, &mut state)?,
            Event::End(e) => apply_end(e.name().as_ref(), &mut state)?,
            Event::Eof => break,
            _ => {}
        }
    }
    build_parsed_vtu(state)
}

fn apply_start(e: &BytesStart<'_>, state: &mut VtuParseState) -> Result<()> {
    match e.name().as_ref() {
        b"VTKFile" => apply_vtk_file_start(e, state),
        b"UnstructuredGrid" => {
            state.saw_unstructured_grid = true;
            Ok(())
        }
        b"Piece" => apply_piece_start(e, state),
        b"Points" => {
            state.context = XmlContext::Points;
            Ok(())
        }
        b"Cells" => {
            state.context = XmlContext::Cells;
            Ok(())
        }
        b"DataArray" => apply_data_array_start(e, state),
        _ => Ok(()),
    }
}

fn apply_empty(e: &BytesStart<'_>, state: &mut VtuParseState) -> Result<()> {
    apply_start(e, state)?;
    if matches!(state.context, XmlContext::DataArray(_)) {
        finish_data_array(state)?;
    }
    Ok(())
}

fn apply_vtk_file_start(e: &BytesStart<'_>, state: &mut VtuParseState) -> Result<()> {
    for attr in e.attributes().flatten() {
        match attr.key.as_ref() {
            b"type" => {
                let t = String::from_utf8_lossy(&attr.value);
                if t != "UnstructuredGrid" {
                    return Err(io_error(
                        std::io::ErrorKind::InvalidData,
                        format!("需要 UnstructuredGrid，实际为 {t}"),
                    ));
                }
            }
            b"byte_order" => {
                state.byte_order = Some(String::from_utf8_lossy(&attr.value).into_owned());
            }
            b"compressor" => {
                return Err(io_error(
                    std::io::ErrorKind::InvalidData,
                    "VTU 读入暂不支持 compressor",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn apply_piece_start(e: &BytesStart<'_>, state: &mut VtuParseState) -> Result<()> {
    state.num_points = parse_usize_attr(e, b"NumberOfPoints")?;
    state.num_cells = parse_usize_attr(e, b"NumberOfCells")?;
    Ok(())
}

fn apply_data_array_start(e: &BytesStart<'_>, state: &mut VtuParseState) -> Result<()> {
    let target = match state.context {
        XmlContext::Points => Some(DataArrayTarget::Points),
        XmlContext::Cells => cell_data_array_target(e)?,
        _ => None,
    };
    let Some(target) = target else {
        return Ok(());
    };
    let format = attribute_value(e, b"format")?.unwrap_or_else(|| "ascii".to_string());
    if format == "appended" {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "VTU 读入暂不支持 appended DataArray",
        ));
    }
    state.current_array = Some(VtuDataArray {
        vtk_type: attribute_value(e, b"type")?.unwrap_or_else(|| "Float64".to_string()),
        format,
        text: String::new(),
        components: parse_usize_attr(e, b"NumberOfComponents")?.unwrap_or(1),
    });
    state.context = XmlContext::DataArray(target);
    Ok(())
}

fn cell_data_array_target(e: &BytesStart<'_>) -> Result<Option<DataArrayTarget>> {
    Ok(match attribute_value(e, b"Name")?.as_deref() {
        Some("connectivity") => Some(DataArrayTarget::Connectivity),
        Some("offsets") => Some(DataArrayTarget::Offsets),
        Some("types") => Some(DataArrayTarget::Types),
        _ => None,
    })
}

fn append_text(e: &quick_xml::events::BytesText<'_>, state: &mut VtuParseState) -> Result<()> {
    if !matches!(state.context, XmlContext::DataArray(_)) {
        return Ok(());
    }
    let text = e
        .unescape()
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?;
    if let Some(array) = state.current_array.as_mut() {
        array.text.push_str(&text);
    }
    Ok(())
}

fn apply_end(name: &[u8], state: &mut VtuParseState) -> Result<()> {
    match name {
        b"DataArray" => finish_data_array(state),
        b"Points" | b"Cells" => {
            state.context = XmlContext::None;
            Ok(())
        }
        _ => Ok(()),
    }
}

fn finish_data_array(state: &mut VtuParseState) -> Result<()> {
    let XmlContext::DataArray(target) = state.context else {
        return Ok(());
    };
    let array = state.current_array.take().ok_or_else(|| {
        io_error(
            std::io::ErrorKind::InvalidData,
            "DataArray 结束但缺少当前数组",
        )
    })?;
    match target {
        DataArrayTarget::Points => state.points = Some(array),
        DataArrayTarget::Connectivity => state.connectivity = Some(array),
        DataArrayTarget::Offsets => state.offsets = Some(array),
        DataArrayTarget::Types => state.types = Some(array),
    }
    state.context = match target {
        DataArrayTarget::Points => XmlContext::Points,
        _ => XmlContext::Cells,
    };
    Ok(())
}

fn build_parsed_vtu(state: VtuParseState) -> Result<ParsedVtu> {
    if !state.saw_unstructured_grid {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "缺少 UnstructuredGrid 元素",
        ));
    }
    ensure_little_endian(state.byte_order.as_deref())?;
    let arrays = require_vtu_arrays(state)?;
    Ok(ParsedVtu {
        num_points: arrays.num_points,
        num_cells: arrays.num_cells,
        points: arrays.points,
        connectivity: arrays.connectivity,
        offsets: arrays.offsets,
        types: arrays.types,
    })
}

struct RequiredVtuArrays {
    num_points: usize,
    num_cells: usize,
    points: VtuDataArray,
    connectivity: VtuDataArray,
    offsets: VtuDataArray,
    types: VtuDataArray,
}

fn ensure_little_endian(byte_order: Option<&str>) -> Result<()> {
    let order = byte_order.unwrap_or("LittleEndian");
    if order == "LittleEndian" {
        return Ok(());
    }
    Err(io_error(
        std::io::ErrorKind::InvalidData,
        format!("暂不支持 byte_order={order}，仅 LittleEndian"),
    ))
}

fn require_vtu_arrays(state: VtuParseState) -> Result<RequiredVtuArrays> {
    Ok(RequiredVtuArrays {
        num_points: state.num_points.ok_or_else(|| {
            io_error(std::io::ErrorKind::InvalidData, "Piece 缺少 NumberOfPoints")
        })?,
        num_cells: state
            .num_cells
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "Piece 缺少 NumberOfCells"))?,
        points: state
            .points
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 Points DataArray"))?,
        connectivity: state
            .connectivity
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 Cells/connectivity"))?,
        offsets: state
            .offsets
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 Cells/offsets"))?,
        types: state
            .types
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 Cells/types"))?,
    })
}

fn decode_points(array: &VtuDataArray, num_points: usize) -> Result<Vec<[f64; 3]>> {
    if array.components != 3 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!(
                "Points NumberOfComponents 必须为 3，实际为 {}",
                array.components
            ),
        ));
    }
    let values = decode_f64_array(array)?;
    if values.len() != num_points * 3 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!(
                "Points 数量不匹配：期望 {} 个标量，实际 {}",
                num_points * 3,
                values.len()
            ),
        ));
    }
    Ok(values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

fn decode_f64_array(array: &VtuDataArray) -> Result<Vec<f64>> {
    match array.format.as_str() {
        "ascii" => array
            .text
            .split_whitespace()
            .map(|part| {
                part.parse::<f64>().map_err(|_| {
                    io_error(
                        std::io::ErrorKind::InvalidData,
                        format!("Float 数值无效: {part}"),
                    )
                })
            })
            .collect(),
        "binary" => decode_binary_f64_array(array),
        other => Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("不支持的 DataArray format={other}"),
        )),
    }
}

fn decode_usize_array(array: &VtuDataArray) -> Result<Vec<usize>> {
    match array.format.as_str() {
        "ascii" => array
            .text
            .split_whitespace()
            .map(parse_nonnegative_index)
            .collect(),
        "binary" => decode_binary_usize_array(array),
        other => Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("不支持的 DataArray format={other}"),
        )),
    }
}

fn decode_u8_array(array: &VtuDataArray) -> Result<Vec<u8>> {
    match array.format.as_str() {
        "ascii" => array
            .text
            .split_whitespace()
            .map(|part| {
                part.parse::<u8>().map_err(|_| {
                    io_error(
                        std::io::ErrorKind::InvalidData,
                        format!("UInt8 无效: {part}"),
                    )
                })
            })
            .collect(),
        "binary" => decode_binary_u8_array(array),
        other => Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("不支持的 DataArray format={other}"),
        )),
    }
}

fn decode_binary_payload(array: &VtuDataArray) -> Result<Vec<u8>> {
    let encoded: String = array.text.chars().filter(|c| !c.is_whitespace()).collect();
    let block = base64::engine::general_purpose::STANDARD
        .decode(encoded.as_bytes())
        .map_err(|err| {
            io_error(
                std::io::ErrorKind::InvalidData,
                format!("base64 解码失败: {err}"),
            )
        })?;
    if block.len() < 4 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "binary DataArray 缺少 UInt32 长度头",
        ));
    }
    let len = u32::from_le_bytes(block[0..4].try_into().expect("4 bytes")) as usize;
    if block.len() != 4 + len {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!(
                "binary DataArray 长度头为 {len}，实际 payload {}",
                block.len() - 4
            ),
        ));
    }
    Ok(block[4..].to_vec())
}

fn decode_binary_f64_array(array: &VtuDataArray) -> Result<Vec<f64>> {
    let payload = decode_binary_payload(array)?;
    match array.vtk_type.as_str() {
        "Float64" => decode_chunks(&payload, 8, |bytes| {
            f64::from_le_bytes(bytes.try_into().expect("8 bytes"))
        }),
        "Float32" => decode_chunks(&payload, 4, |bytes| {
            f64::from(f32::from_le_bytes(bytes.try_into().expect("4 bytes")))
        }),
        other => Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("Float DataArray 不支持 type={other}"),
        )),
    }
}

fn decode_binary_usize_array(array: &VtuDataArray) -> Result<Vec<usize>> {
    let payload = decode_binary_payload(array)?;
    match array.vtk_type.as_str() {
        "Int64" => decode_chunks(&payload, 8, |bytes| {
            i64::from_le_bytes(bytes.try_into().expect("8 bytes"))
        })?
        .into_iter()
        .map(i64_to_usize)
        .collect(),
        "Int32" => decode_chunks(&payload, 4, |bytes| {
            i32::from_le_bytes(bytes.try_into().expect("4 bytes")) as i64
        })?
        .into_iter()
        .map(i64_to_usize)
        .collect(),
        other => Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("Index DataArray 不支持 type={other}"),
        )),
    }
}

fn decode_binary_u8_array(array: &VtuDataArray) -> Result<Vec<u8>> {
    let payload = decode_binary_payload(array)?;
    if array.vtk_type != "UInt8" {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("types DataArray 需要 UInt8，实际 {}", array.vtk_type),
        ));
    }
    Ok(payload)
}

fn decode_chunks<T>(payload: &[u8], width: usize, decode: impl Fn(&[u8]) -> T) -> Result<Vec<T>> {
    if payload.len() % width != 0 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("binary payload 长度 {} 不是 {width} 的倍数", payload.len()),
        ));
    }
    Ok(payload.chunks_exact(width).map(decode).collect())
}

fn build_cells(
    connectivity: &[usize],
    offsets: &[usize],
    types: &[u8],
) -> Result<Vec<UnstructuredCell>> {
    if offsets.len() != types.len() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!(
                "offsets 数量 {} 与 types 数量 {} 不一致",
                offsets.len(),
                types.len()
            ),
        ));
    }
    let mut start = 0usize;
    let mut cells = Vec::with_capacity(types.len());
    for (cell_index, (&end, &vtk_type)) in offsets.iter().zip(types.iter()).enumerate() {
        if end < start || end > connectivity.len() {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                format!(
                    "单元 {cell_index} offset={end} 非法（start={start}, conn={}）",
                    connectivity.len()
                ),
            ));
        }
        let kind = CellKind::from_vtk_type(vtk_type)?;
        cells.push(UnstructuredCell::new(
            kind,
            connectivity[start..end].to_vec(),
        )?);
        start = end;
    }
    if start != connectivity.len() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!(
                "connectivity 尾部未使用：offset 最终 {start}，长度 {}",
                connectivity.len()
            ),
        ));
    }
    Ok(cells)
}

fn parse_nonnegative_index(part: &str) -> Result<usize> {
    let value = part.parse::<i64>().map_err(|_| {
        io_error(
            std::io::ErrorKind::InvalidData,
            format!("整数索引无效: {part}"),
        )
    })?;
    i64_to_usize(value)
}

fn i64_to_usize(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        io_error(
            std::io::ErrorKind::InvalidData,
            format!("索引 {value} 为负或超出 usize"),
        )
    })
}

fn attribute_value(e: &BytesStart<'_>, key: &[u8]) -> Result<Option<String>> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == key {
            return Ok(Some(String::from_utf8_lossy(&attr.value).into_owned()));
        }
    }
    Ok(None)
}

fn parse_usize_attr(e: &BytesStart<'_>, key: &[u8]) -> Result<Option<usize>> {
    attribute_value(e, key)?
        .map(|raw| {
            raw.parse::<usize>().map_err(|_| {
                io_error(
                    std::io::ErrorKind::InvalidData,
                    format!("属性 {}={raw} 不是 usize", String::from_utf8_lossy(key)),
                )
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CellId, approx_eq};

    #[test]
    fn loads_ascii_mixed_vtu() {
        let xml = r#"
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="8" NumberOfCells="2">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0  1 0 0  0 1 0  0 0 1
          1 1 0  1 0 1  0 1 1  1 1 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">
          0 1 2 3  1 4 2 7 5
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">4 9</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10 14</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#;
        let loaded = load_vtu_from_str("mixed", xml).expect("load");
        assert_eq!(loaded.mesh.num_cells(), 2);
        assert_eq!(loaded.mesh.cell_kind(CellId(0)), CellKind::Tet);
        assert_eq!(loaded.mesh.cell_kind(CellId(1)), CellKind::Pyramid);
        assert!(loaded.mesh.cell_metric(CellId(0)).volume > 0.0);
        assert!(approx_eq(loaded.mesh.points()[7][2], 1.0, 1.0e-12));
    }
}
