//! VTK XML StructuredGrid（`.vts`）写出。
//!
//! 纯网格仍用 appended；流场 `write_flow_vts` 用 inline `format="binary"`（ParaView 兼容）。

use std::path::Path;

use base64::Engine;

use crate::error::Result;
use crate::field::ConservedFields;
use crate::io::limits::validate_input_path;
use crate::io::vertex_field::gather_cell_primitives;
use crate::mesh::{StructuredMesh, StructuredMesh2d, StructuredMesh3d};
use crate::physics::IdealGasEoS;

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

/// 3D 流场 VTS（`CellData` 单元中心 ρ/p/速度分量，ParaView 着色用 Cell Data）。
pub fn write_flow_vts(
    path: &Path,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<()> {
    validate_input_path(path)?;
    let (rho, u, v, w, p) = gather_cell_primitives(mesh, fields, eos, min_pressure)?;
    let scalars = FlowScalarsB64 {
        rho: encode_f64_block(&rho),
        p: encode_f64_block(&p),
        u: encode_f64_block(&u),
        v: encode_f64_block(&v),
        w: encode_f64_block(&w),
    };
    let xml = format_flow_vts_xml(
        mesh.nx,
        mesh.ny,
        mesh.nz,
        &scalars,
        &encode_mesh3d_points_inline(mesh),
    );
    std::fs::write(path, xml).map_err(crate::error::AsimuError::from)
}

fn encode_f64_block(values: &[f64]) -> String {
    let mut payload = Vec::new();
    for &value in values {
        write_f64(&mut payload, value);
    }
    encode_binary_array(&payload)
}

fn encode_mesh3d_points_inline(mesh: &StructuredMesh3d) -> String {
    let mut payload = Vec::new();
    write_mesh3d_points(&mut payload, mesh);
    encode_binary_array(&payload)
}

fn encode_binary_array(payload: &[u8]) -> String {
    let mut block = Vec::with_capacity(4 + payload.len());
    block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    block.extend_from_slice(payload);
    base64::engine::general_purpose::STANDARD.encode(block)
}

fn encode_mesh3d_points(mesh: &StructuredMesh3d) -> String {
    encode_appended_binary(&encode_mesh3d_points_block(mesh))
}

fn encode_mesh3d_points_block(mesh: &StructuredMesh3d) -> Vec<u8> {
    let mut payload = Vec::new();
    write_mesh3d_points(&mut payload, mesh);
    wrap_payload_block(&payload)
}

fn wrap_payload_block(payload: &[u8]) -> Vec<u8> {
    let mut block = Vec::with_capacity(4 + payload.len());
    block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    block.extend_from_slice(payload);
    block
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

struct FlowScalarsB64 {
    rho: String,
    p: String,
    u: String,
    v: String,
    w: String,
}

fn format_flow_vts_xml(
    nx: usize,
    ny: usize,
    nz: usize,
    scalars: &FlowScalarsB64,
    pts: &str,
) -> String {
    format!(
        r#"<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <StructuredGrid WholeExtent="0 {nx} 0 {ny} 0 {nz}">
    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">
      <PointData>
      </PointData>
      <CellData Scalars="Density">
        <DataArray type="Float64" Name="Density" NumberOfComponents="1" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="Pressure" NumberOfComponents="1" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="VelocityX" NumberOfComponents="1" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="VelocityY" NumberOfComponents="1" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="VelocityZ" NumberOfComponents="1" format="binary">{}</DataArray>
      </CellData>
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="binary">{pts}</DataArray>
      </Points>
    </Piece>
  </StructuredGrid>
</VTKFile>
"#,
        scalars.rho, scalars.p, scalars.u, scalars.v, scalars.w,
    )
}

fn write_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ConservedFields;
    use crate::physics::{FreestreamParams, IdealGasEoS};
    use std::env;

    #[test]
    fn flow_vts_writes_cell_data_at_cell_count() {
        let path = env::temp_dir().join("asimu_flow_vts_test.vts");
        let mesh = StructuredMesh3d::uniform_box("t", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        write_flow_vts(&path, &mesh, &fields, &eos, 1.0e-6).expect("write");
        let xml = std::fs::read_to_string(&path).expect("read");
        assert!(xml.contains("<CellData Scalars=\"Density\">"));
        assert!(xml.contains(r#"Name="Density" NumberOfComponents="1" format="binary""#));
        let status = std::process::Command::new("python3")
            .arg("-c")
            .arg(format!(
                r#"
import vtk
r = vtk.vtkXMLStructuredGridReader()
r.SetFileName({path:?})
r.Update()
g = r.GetOutput()
assert g.GetNumberOfCells() == 8, g.GetNumberOfCells()
d = g.GetCellData().GetArray("Density")
lo, hi = d.GetRange()
assert 0 < lo <= hi < 1e6, (lo, hi)
print("ok")
"#,
                path = path.display().to_string()
            ))
            .status()
            .expect("python");
        assert!(status.success());
        let _ = std::fs::remove_file(path);
    }

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
