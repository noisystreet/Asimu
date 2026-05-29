//! VTK XML MultiBlock（`.vtm`）写出 — 引用同目录下的子 VTS 文件。

use std::path::Path;

use crate::error::Result;
use crate::io::limits::validate_input_path;

/// 子块描述：`(显示名称, 相对 .vtm 的路径)`。
pub struct VtmBlock<'a> {
    pub name: &'a str,
    pub file: &'a str,
}

/// 写出 `.vtm` 清单，供 ParaView 打开多 block 结构化网格。
pub fn write_vtm(blocks: &[VtmBlock<'_>], path: &Path) -> Result<()> {
    validate_input_path(path)?;
    if blocks.is_empty() {
        return Err(crate::error::AsimuError::Mesh(
            "write_vtm 需要至少一个子块".to_string(),
        ));
    }
    let mut datasets = String::new();
    for (index, block) in blocks.iter().enumerate() {
        datasets.push_str(&format!(
            r#"    <DataSet index="{index}" name="{}" file="{}"/>
"#,
            xml_escape(block.name),
            xml_escape(block.file),
        ));
    }
    let xml = format!(
        r#"<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <vtkMultiBlockDataSet>
{datasets}  </vtkMultiBlockDataSet>
</VTKFile>
"#
    );
    std::fs::write(path, xml).map_err(crate::error::AsimuError::from)
}

fn xml_escape(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
