//! VTK XML StructuredGrid（`.vts`）**二进制 appended** 写出（未压缩）。

use std::path::Path;

use base64::Engine;

use crate::error::Result;
use crate::io::limits::validate_input_path;
use crate::mesh::{StructuredMesh, StructuredMesh2d, StructuredMesh3d};

/// 将结构化网格写出为 appended 二进制 VTS（Float64，LittleEndian，无压缩）。
pub fn write_vts(mesh: &StructuredMesh, path: &Path) -> Result<()> {
    validate_input_path(path)?;
    let xml = match mesh {
        StructuredMesh::D2(m) => build_vts_xml_2d(m),
        StructuredMesh::D3(m) => build_vts_xml_3d(m),
    };
    std::fs::write(path, xml).map_err(crate::error::AsimuError::from)
}

fn build_vts_xml_2d(mesh: &StructuredMesh2d) -> String {
    let block = encode_points_block(|out| {
        for j in 0..=mesh.ny {
            for i in 0..=mesh.nx {
                write_f64(out, mesh.node_x(i, j));
                write_f64(out, mesh.node_y(i, j));
                write_f64(out, 0.0);
            }
        }
    });
    format_vts(mesh.nx, mesh.ny, 0, &block)
}

fn build_vts_xml_3d(mesh: &StructuredMesh3d) -> String {
    let block = encode_mesh3d_points(mesh);
    format_vts(mesh.nx, mesh.ny, mesh.nz, &block)
}

fn encode_mesh3d_points(mesh: &StructuredMesh3d) -> String {
    let mut binary = Vec::new();
    append_points_block(&mut binary, mesh);
    encode_appended_binary(&binary)
}

fn append_points_block(binary: &mut Vec<u8>, mesh: &StructuredMesh3d) {
    let mut payload = Vec::new();
    write_mesh3d_points(&mut payload, mesh);
    binary.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    binary.extend(payload);
}

fn write_mesh3d_points(out: &mut Vec<u8>, mesh: &StructuredMesh3d) {
    for k in 0..=mesh.nz {
        for j in 0..=mesh.ny {
            for i in 0..=mesh.nx {
                write_f64(out, mesh.node_x(i, j, k));
                write_f64(out, mesh.node_y(i, j, k));
                write_f64(out, mesh.node_z(i, j, k));
            }
        }
    }
}

fn encode_appended_binary(binary: &[u8]) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(binary);
    format!("_{encoded}")
}

fn encode_points_block(fill: impl FnOnce(&mut Vec<u8>)) -> String {
    let mut payload = Vec::new();
    fill(&mut payload);
    let mut block = Vec::with_capacity(4 + payload.len());
    block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    block.extend(payload);
    let encoded = base64::engine::general_purpose::STANDARD.encode(&block);
    format!("_{encoded}")
}

fn format_vts(nx: usize, ny: usize, nz: usize, appended: &str) -> String {
    format!(
        r#"<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <StructuredGrid WholeExtent="0 {nx} 0 {ny} 0 {nz}">
    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="appended" offset="0"/>
      </Points>
    </Piece>
  </StructuredGrid>
  <AppendedData encoding="base64">
{appended}</AppendedData>
</VTKFile>
"#
    )
}

fn write_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn roundtrip_unit_square_fixture() {
        let path = env::temp_dir().join("asimu_write_vts_test.vts");
        let mesh = StructuredMesh::D2(
            StructuredMesh2d::new(
                "unit",
                2,
                2,
                vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            )
            .expect("mesh"),
        );
        write_vts(&mesh, &path).expect("write");
        let loaded = crate::io::load_vts(&path).expect("read");
        match loaded.mesh {
            StructuredMesh::D2(m) => {
                assert_eq!(m.nx, 2);
                assert_eq!(m.node_x(2, 2), 2.0);
            }
            StructuredMesh::D3(_) => panic!("expected 2d"),
        }
        let _ = std::fs::remove_file(path);
    }
}
