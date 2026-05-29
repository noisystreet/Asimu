//! VTK XML StructuredGrid（`.vts`）**二进制 appended** 读入。
//!
//! 支持：LittleEndian appended、Float32/Float64 Points、可选 zlib（`vtkZLibDataCompressor`）、2D/3D。
//! 不支持 ASCII、inline binary、多 Piece。见 ADR 0007。

use std::io::Read;
use std::path::Path;

use base64::Engine;
use flate2::read::ZlibDecoder;
use quick_xml::Reader;
use quick_xml::events::Event;

use crate::error::Result;
use crate::io::limits::{io_error, validate_cell_count, validate_file_size, validate_input_path};
use crate::mesh::{StructuredMesh, StructuredMesh2d, StructuredMesh3d};

/// VTS 读入结果（网格几何；PointData/CellData 后续 PR）。
#[derive(Debug, Clone, PartialEq)]
pub struct VtsLoadResult {
    pub mesh: StructuredMesh,
}

/// 从 `.vts` 文件加载结构化网格（`Points` + appended base64）。
pub fn load_vts(path: &Path) -> Result<VtsLoadResult> {
    validate_input_path(path)?;
    let bytes = std::fs::read(path)?;
    validate_file_size(bytes.len() as u64, "VTS 文件")?;
    let content = std::str::from_utf8(&bytes).map_err(|e| {
        io_error(
            std::io::ErrorKind::InvalidData,
            format!("VTS 非 UTF-8 XML: {e}"),
        )
    })?;
    load_vts_from_str(path, content)
}

fn load_vts_from_str(path: &Path, content: &str) -> Result<VtsLoadResult> {
    let parsed = parse_vts_xml(content)?;
    let mesh_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("vts")
        .to_string();

    let nx = extent_cells(parsed.extent[1] - parsed.extent[0]);
    let ny = extent_cells(parsed.extent[3] - parsed.extent[2]);
    let nz = extent_cells(parsed.extent[5] - parsed.extent[4]);
    let is_3d = parsed.extent[5] > parsed.extent[4];

    let num_points = if is_3d {
        (parsed.extent[1] - parsed.extent[0] + 1) as usize
            * (parsed.extent[3] - parsed.extent[2] + 1) as usize
            * (parsed.extent[5] - parsed.extent[4] + 1) as usize
    } else {
        (parsed.extent[1] - parsed.extent[0] + 1) as usize
            * (parsed.extent[3] - parsed.extent[2] + 1) as usize
    };

    validate_cell_count(if is_3d {
        (nx * ny * nz) as u64
    } else {
        (nx * ny) as u64
    })?;

    let appended = decode_appended_base64(&parsed.appended_base64)?;
    let unified = if parsed.compressed {
        build_uncompressed_appended(&appended, &parsed.appended_array_offsets)?
    } else {
        appended
    };
    let block = read_appended_block(&unified, parsed.points_offset as usize)?;
    let (points_x, points_y, points_z) = decode_points_xyz(
        &block,
        parsed.points_scalar,
        num_points,
        parsed.points_components,
    )?;

    let mesh = if is_3d {
        StructuredMesh::D3(StructuredMesh3d::new(
            mesh_name, nx, ny, nz, points_x, points_y, points_z,
        )?)
    } else {
        StructuredMesh::D2(StructuredMesh2d::new(
            mesh_name, nx, ny, points_x, points_y,
        )?)
    };

    Ok(VtsLoadResult { mesh })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    Float32,
    Float64,
}

#[derive(Debug)]
struct ParsedVts {
    extent: [i32; 6],
    points_offset: u32,
    points_scalar: ScalarKind,
    points_components: u32,
    appended_base64: String,
    appended_array_offsets: Vec<u32>,
    compressed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum XmlContext {
    #[default]
    None,
    Points,
    AppendedData,
}

#[derive(Debug, Default)]
struct VtsParseState {
    extent: Option<[i32; 6]>,
    points_format: Option<String>,
    points_offset: Option<u32>,
    points_type: Option<String>,
    points_components: Option<u32>,
    appended_base64: Option<String>,
    appended_array_offsets: Vec<u32>,
    byte_order: Option<String>,
    compressed: bool,
    context: XmlContext,
    saw_structured_grid: bool,
}

fn parse_vts_xml(content: &str) -> Result<ParsedVts> {
    let mut reader = Reader::from_str(content);
    reader.config_mut().trim_text(true);

    let mut state = VtsParseState {
        context: XmlContext::None,
        ..Default::default()
    };

    loop {
        match reader
            .read_event()
            .map_err(|e| io_error(std::io::ErrorKind::InvalidData, e.to_string()))?
        {
            Event::Start(e) => apply_element_start(&e, &mut state)?,
            Event::Empty(e) => apply_element_start(&e, &mut state)?,
            Event::Text(e) if state.context == XmlContext::AppendedData => {
                let text = e
                    .unescape()
                    .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?
                    .into_owned();
                state.appended_base64 = Some(strip_appended_prefix(&text));
            }
            Event::End(e) => match e.name().as_ref() {
                b"Points" => state.context = XmlContext::None,
                b"AppendedData" => state.context = XmlContext::None,
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }

    if !state.saw_structured_grid {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "缺少 StructuredGrid 元素",
        ));
    }

    let extent = state
        .extent
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 WholeExtent / Extent"))?;

    let format = state
        .points_format
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 Points DataArray"))?;

    match format.as_str() {
        "appended" => {}
        "ascii" => {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                "不支持 ASCII VTS，请使用 appended 二进制格式",
            ));
        }
        other => {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                format!("不支持的 DataArray format=\"{other}\"，仅支持 appended"),
            ));
        }
    }

    let points_offset = state.points_offset.ok_or_else(|| {
        io_error(
            std::io::ErrorKind::InvalidData,
            "Points DataArray 缺少 offset",
        )
    })?;

    let points_type = state.points_type.unwrap_or_else(|| "Float64".to_string());
    let points_scalar = match points_type.as_str() {
        "Float64" => ScalarKind::Float64,
        "Float32" => ScalarKind::Float32,
        other => {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                format!("Points 不支持 type=\"{other}\""),
            ));
        }
    };

    let points_components = state.points_components.unwrap_or(3);
    if points_components != 3 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("Points NumberOfComponents 必须为 3，实际为 {points_components}"),
        ));
    }

    let order = state
        .byte_order
        .unwrap_or_else(|| "LittleEndian".to_string());
    if order != "LittleEndian" {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("暂不支持 byte_order={order}，仅 LittleEndian"),
        ));
    }

    let appended_base64 = state
        .appended_base64
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "缺少 AppendedData 段"))?;

    Ok(ParsedVts {
        extent,
        points_offset,
        points_scalar,
        points_components,
        appended_base64,
        appended_array_offsets: state.appended_array_offsets,
        compressed: state.compressed,
    })
}

fn apply_element_start(
    e: &quick_xml::events::BytesStart<'_>,
    state: &mut VtsParseState,
) -> Result<()> {
    match e.name().as_ref() {
        b"VTKFile" => {
            for attr in e.attributes().flatten() {
                match attr.key.as_ref() {
                    b"byte_order" => {
                        state.byte_order = Some(String::from_utf8_lossy(&attr.value).into_owned());
                    }
                    b"compressor" => {
                        let name = String::from_utf8_lossy(&attr.value);
                        if name == "vtkZLibDataCompressor" {
                            state.compressed = true;
                        } else {
                            return Err(io_error(
                                std::io::ErrorKind::InvalidData,
                                format!("不支持的 compressor=\"{name}\""),
                            ));
                        }
                    }
                    b"type" => {
                        let t = String::from_utf8_lossy(&attr.value);
                        if t != "StructuredGrid" {
                            return Err(io_error(
                                std::io::ErrorKind::InvalidData,
                                format!("需要 StructuredGrid，实际为 {t}"),
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
        b"StructuredGrid" => {
            state.saw_structured_grid = true;
            if let Some(value) = attribute_value(e, b"WholeExtent")? {
                state.extent = Some(parse_extent(&value)?);
            }
        }
        b"Piece" => {
            if let Some(value) = attribute_value(e, b"Extent")? {
                state.extent = Some(parse_extent(&value)?);
            }
        }
        b"Points" => state.context = XmlContext::Points,
        b"DataArray" => {
            let format = attribute_value(e, b"format")?;
            if format.as_deref() == Some("appended") {
                if let Some(offset) = attribute_value(e, b"offset")? {
                    let offset = offset.parse().map_err(|_| {
                        io_error(std::io::ErrorKind::InvalidData, "DataArray offset 无效")
                    })?;
                    state.appended_array_offsets.push(offset);
                }
            }
            if state.context == XmlContext::Points {
                if state.points_format.is_some() {
                    return Err(io_error(
                        std::io::ErrorKind::InvalidData,
                        "VTS 仅支持单个 Points DataArray",
                    ));
                }
                state.points_format = format;
                state.points_type = attribute_value(e, b"type")?;
                state.points_components =
                    attribute_value(e, b"NumberOfComponents")?.and_then(|s| s.parse().ok());
                if let Some(offset) = attribute_value(e, b"offset")? {
                    state.points_offset = Some(offset.parse().map_err(|_| {
                        io_error(
                            std::io::ErrorKind::InvalidData,
                            "Points DataArray offset 无效",
                        )
                    })?);
                }
            }
        }
        b"AppendedData" => state.context = XmlContext::AppendedData,
        _ => {}
    }
    Ok(())
}

fn attribute_value(e: &quick_xml::events::BytesStart<'_>, key: &[u8]) -> Result<Option<String>> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == key {
            return Ok(Some(String::from_utf8_lossy(&attr.value).into_owned()));
        }
    }
    Ok(None)
}

fn parse_extent(raw: &str) -> Result<[i32; 6]> {
    let parts: Vec<&str> = raw.split_whitespace().collect();
    if parts.len() != 6 {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("extent 需要 6 个整数，实际为 \"{raw}\""),
        ));
    }
    let mut extent = [0i32; 6];
    for (idx, part) in parts.iter().enumerate() {
        extent[idx] = part.parse().map_err(|_| {
            io_error(
                std::io::ErrorKind::InvalidData,
                format!("extent 整数无效: {part}"),
            )
        })?;
    }
    Ok(extent)
}

fn extent_cells(delta: i32) -> usize {
    delta.max(0) as usize
}

fn strip_appended_prefix(text: &str) -> String {
    text.trim_start_matches('_').trim().to_string()
}

/// VTK 将 appended 二进制按块独立 base64 编码后拼接；块边界以 padding（`=`）结束。
fn decode_appended_base64(base64_text: &str) -> Result<Vec<u8>> {
    let trimmed: String = base64_text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < trimmed.len() {
        let mut end = pos + 4;
        let mut decoded_block = None;
        while end <= trimmed.len() {
            let part = &trimmed[pos..end];
            if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(part.as_bytes()) {
                decoded_block = Some(decoded);
                if part.ends_with('=') {
                    break;
                }
            }
            end += 4;
        }
        let Some(decoded) = decoded_block else {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                format!("base64 解码失败: 偏移 {pos} 处无法解析块"),
            ));
        };
        out.extend_from_slice(&decoded);
        pos = end;
    }
    Ok(out)
}

/// 将 zlib 压缩的 appended 流还原为与未压缩 VTS 相同的统一缓冲区布局。
fn build_uncompressed_appended(compressed: &[u8], offsets: &[u32]) -> Result<Vec<u8>> {
    let chunks = parse_vtk_compressed_chunks(compressed)?;
    if chunks.len() != offsets.len() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!(
                "压缩 appended 块数 ({}) 与 DataArray offset 数 ({}) 不一致",
                chunks.len(),
                offsets.len()
            ),
        ));
    }
    let total_len = offsets
        .iter()
        .zip(chunks.iter())
        .map(|(offset, chunk)| *offset as usize + 4 + chunk.len())
        .max()
        .unwrap_or(0);
    let mut unified = vec![0u8; total_len];
    for (offset, chunk) in offsets.iter().zip(chunks) {
        let start = *offset as usize;
        unified[start..start + 4].copy_from_slice(&(chunk.len() as u32).to_le_bytes());
        unified[start + 4..start + 4 + chunk.len()].copy_from_slice(&chunk);
    }
    Ok(unified)
}

fn parse_vtk_compressed_chunks(compressed: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut pos = 0;
    let mut chunks = Vec::new();
    while pos + 16 <= compressed.len() {
        let num_blocks = read_u32_le(compressed, pos)?;
        if num_blocks != 1 {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                format!("暂不支持 num_blocks={num_blocks} 的压缩块"),
            ));
        }
        pos += 4;
        pos += 4; // block_size（通常 32768），读取后跳过
        let uncompressed_size = read_u32_le(compressed, pos)? as usize;
        pos += 4;
        let compressed_size = read_u32_le(compressed, pos)? as usize;
        pos += 4;
        let end = pos
            .checked_add(compressed_size)
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "压缩块长度溢出"))?;
        if end > compressed.len() {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                "压缩块超出 appended 数据范围",
            ));
        }
        let payload = &compressed[pos..end];
        pos = end;
        let decoded = decompress_zlib(payload)?;
        if decoded.len() != uncompressed_size {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                format!(
                    "压缩块解压大小应为 {uncompressed_size}，实际为 {}",
                    decoded.len()
                ),
            ));
        }
        chunks.push(decoded);
    }
    if pos != compressed.len() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("appended 压缩数据尾部残留 {} 字节", compressed.len() - pos),
        ));
    }
    Ok(chunks)
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32> {
    let bytes: [u8; 4] = data[offset..offset + 4]
        .try_into()
        .map_err(|_| io_error(std::io::ErrorKind::InvalidData, "读取 u32 越界"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_appended_block(data: &[u8], offset: usize) -> Result<Vec<u8>> {
    if offset + 4 > data.len() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "appended offset 超出数据范围",
        ));
    }
    let block_len =
        u32::from_le_bytes(data[offset..offset + 4].try_into().expect("4 bytes")) as usize;
    let start = offset + 4;
    let end = start
        .checked_add(block_len)
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "appended block 长度溢出"))?;
    if end > data.len() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "appended block 超出数据范围",
        ));
    }
    Ok(data[start..end].to_vec())
}

fn decompress_zlib(compressed: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(compressed);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).map_err(|e| {
        io_error(
            std::io::ErrorKind::InvalidData,
            format!("zlib 解压失败: {e}"),
        )
    })?;
    Ok(out)
}

fn decode_points_xyz(
    raw: &[u8],
    scalar: ScalarKind,
    num_points: usize,
    components: u32,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let component_count = components as usize;
    let bytes_per_value = match scalar {
        ScalarKind::Float32 => 4,
        ScalarKind::Float64 => 8,
    };
    let expected = num_points
        .checked_mul(component_count)
        .and_then(|n| n.checked_mul(bytes_per_value))
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "Points 尺寸溢出"))?;
    if raw.len() != expected {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            format!("Points 字节数应为 {expected}，实际为 {}", raw.len()),
        ));
    }

    let mut points_x = Vec::with_capacity(num_points);
    let mut points_y = Vec::with_capacity(num_points);
    let mut points_z = Vec::with_capacity(num_points);
    let stride = component_count * bytes_per_value;

    for point_idx in 0..num_points {
        let base = point_idx * stride;
        points_x.push(read_scalar(raw, base, scalar)?);
        points_y.push(read_scalar(raw, base + bytes_per_value, scalar)?);
        points_z.push(read_scalar(raw, base + 2 * bytes_per_value, scalar)?);
    }
    Ok((points_x, points_y, points_z))
}

fn read_scalar(raw: &[u8], offset: usize, scalar: ScalarKind) -> Result<f64> {
    match scalar {
        ScalarKind::Float64 => {
            let bytes: [u8; 8] = raw[offset..offset + 8]
                .try_into()
                .map_err(|_| io_error(std::io::ErrorKind::InvalidData, "Float64 越界"))?;
            Ok(f64::from_le_bytes(bytes))
        }
        ScalarKind::Float32 => {
            let bytes: [u8; 4] = raw[offset..offset + 4]
                .try_into()
                .map_err(|_| io_error(std::io::ErrorKind::InvalidData, "Float32 越界"))?;
            Ok(f64::from(f32::from_le_bytes(bytes)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mesh")
            .join(name)
    }

    #[test]
    fn rejects_ascii_vts() {
        let path = fixture_path("ascii_reject.vts");
        let err = load_vts(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ASCII") || msg.contains("ascii"));
    }

    #[test]
    fn loads_binary_vts_2x2() {
        let path = fixture_path("unit_square_2x2_binary.vts");
        let result = load_vts(&path).expect("load vts");
        let mesh = match result.mesh {
            StructuredMesh::D2(m) => m,
            StructuredMesh::D3(_) => panic!("expected 2d mesh"),
        };
        assert_eq!(mesh.nx, 2);
        assert_eq!(mesh.ny, 2);
        assert_eq!(mesh.num_cells(), 4);
        assert_eq!(mesh.node_x(2, 2), 2.0);
        assert_eq!(mesh.node_y(0, 2), 2.0);
    }

    #[test]
    fn loads_binary_vts_float32() {
        let path = fixture_path("unit_square_2x2_binary_f32.vts");
        let result = load_vts(&path).expect("load vts f32");
        let mesh = match result.mesh {
            StructuredMesh::D2(m) => m,
            StructuredMesh::D3(_) => panic!("expected 2d mesh"),
        };
        assert_eq!(mesh.nx, 2);
        assert_eq!(mesh.node_y(1, 1), 1.0);
    }

    #[test]
    fn loads_pyvista_structured_grid_when_present() {
        let path = std::env::var("ASIMU_VTS_PATH")
            .map(PathBuf::from)
            .ok()
            .filter(|p| p.is_file())
            .or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .map(|dir| dir.join("StructuredGrid.vts"))
                    .filter(|p| p.is_file())
            });
        let Some(path) = path else {
            return;
        };
        let result = load_vts(&path).expect("pyvista StructuredGrid.vts");
        match result.mesh {
            StructuredMesh::D3(m) => {
                assert_eq!(m.nx, 5);
                assert_eq!(m.ny, 11);
                assert_eq!(m.nz, 8);
                assert_eq!(m.num_nodes(), 6 * 12 * 9);
            }
            StructuredMesh::D2(_) => panic!("expected 3d mesh"),
        }
    }
}
