//! VTK XML UnstructuredGrid（`.vtu`）写出 — ParaView 兼容性最佳。
//!
//! 使用各 `DataArray` 的 inline `format="binary"`（非 `AppendedData`），避免部分 VTK/ParaView
//! 版本对未压缩 appended 块的读取缺陷。

use std::path::Path;

use base64::Engine;

use crate::error::Result;
use crate::field::ConservedFields;
use crate::io::limits::validate_input_path;
use crate::io::vertex_field::gather_cell_primitives;
use crate::mesh::StructuredMesh3d;
use crate::physics::IdealGasEoS;

/// VTK_HEXAHEDRON
const VTK_HEXAHEDRON: u8 = 12;

/// 3D 流场 VTU（六面体非结构网格 + 单元中心标量，ParaView 推荐格式）。
pub fn write_flow_vtu(
    path: &Path,
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<()> {
    validate_input_path(path)?;
    let (rho, u, v, w, p, mach, temperature) =
        gather_cell_primitives(mesh, fields, eos, min_pressure)?;
    let npts = mesh.num_nodes();
    let ncells = mesh.num_cells();

    let mut connectivity = Vec::with_capacity(ncells * 8);
    let mut offsets = Vec::with_capacity(ncells);
    let mut types = Vec::with_capacity(ncells);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let nodes = [
                    mesh.node_index(i, j, k),
                    mesh.node_index(i + 1, j, k),
                    mesh.node_index(i + 1, j + 1, k),
                    mesh.node_index(i, j + 1, k),
                    mesh.node_index(i, j, k + 1),
                    mesh.node_index(i + 1, j, k + 1),
                    mesh.node_index(i + 1, j + 1, k + 1),
                    mesh.node_index(i, j + 1, k + 1),
                ];
                connectivity.extend(nodes);
                offsets.push(connectivity.len());
                types.push(VTK_HEXAHEDRON);
            }
        }
    }

    let scalars = FlowScalarsB64 {
        rho: encode_f64_block(&rho),
        p: encode_f64_block(&p),
        u: encode_f64_block(&u),
        v: encode_f64_block(&v),
        w: encode_f64_block(&w),
        mach: encode_f64_block(&mach),
        temperature: encode_f64_block(&temperature),
    };
    let topo = FlowVtuTopologyB64 {
        pts: encode_points_block(mesh),
        conn: encode_i64_block(&connectivity),
        off: encode_i64_block_usize(&offsets),
        types: encode_u8_block(&types),
    };
    let xml = format_flow_vtu_xml(npts, ncells, &scalars, &topo);
    std::fs::write(path, xml).map_err(crate::error::AsimuError::from)
}

fn encode_points_block(mesh: &StructuredMesh3d) -> String {
    let mut payload = Vec::new();
    for k in 0..=mesh.nz {
        for j in 0..=mesh.ny {
            for i in 0..=mesh.nx {
                write_f64(&mut payload, mesh.node_x(i, j, k));
                write_f64(&mut payload, mesh.node_y(i, j, k));
                write_f64(&mut payload, mesh.node_z(i, j, k));
            }
        }
    }
    encode_binary_array(&payload)
}

fn encode_f64_block(values: &[f64]) -> String {
    let mut payload = Vec::new();
    for &value in values {
        write_f64(&mut payload, value);
    }
    encode_binary_array(&payload)
}

fn encode_i64_block(values: &[usize]) -> String {
    let mut payload = Vec::new();
    for &value in values {
        payload.extend_from_slice(&(value as i64).to_le_bytes());
    }
    encode_binary_array(&payload)
}

fn encode_i64_block_usize(values: &[usize]) -> String {
    encode_i64_block(values)
}

fn encode_u8_block(values: &[u8]) -> String {
    encode_binary_array(values)
}

fn encode_binary_array(payload: &[u8]) -> String {
    let mut block = Vec::with_capacity(4 + payload.len());
    block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    block.extend_from_slice(payload);
    base64::engine::general_purpose::STANDARD.encode(block)
}

struct FlowScalarsB64 {
    rho: String,
    p: String,
    u: String,
    v: String,
    w: String,
    mach: String,
    temperature: String,
}

struct FlowVtuTopologyB64 {
    pts: String,
    conn: String,
    off: String,
    types: String,
}

fn format_flow_vtu_xml(
    npts: usize,
    ncells: usize,
    scalars: &FlowScalarsB64,
    topo: &FlowVtuTopologyB64,
) -> String {
    format!(
        r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">
      <PointData>
      </PointData>
      <CellData Scalars="Density">
        <DataArray type="Float64" Name="Density" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="Pressure" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="VelocityX" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="VelocityY" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="VelocityZ" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="MachNumber" format="binary">{}</DataArray>
        <DataArray type="Float64" Name="Temperature" format="binary">{}</DataArray>
      </CellData>
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="binary">{}</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="binary">{}</DataArray>
        <DataArray type="Int64" Name="offsets" format="binary">{}</DataArray>
        <DataArray type="UInt8" Name="types" format="binary">{}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#,
        scalars.rho,
        scalars.p,
        scalars.u,
        scalars.v,
        scalars.w,
        scalars.mach,
        scalars.temperature,
        topo.pts,
        topo.conn,
        topo.off,
        topo.types,
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
    fn flow_vtu_readable_by_vtk() {
        let path = env::temp_dir().join("asimu_flow_vtu_vtk_test.vtu");
        let mesh = StructuredMesh3d::uniform_box("t", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams::default();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        write_flow_vtu(&path, &mesh, &fields, &eos, 1.0e-6).expect("write");
        let status = std::process::Command::new("python3")
            .arg("-c")
            .arg(format!(
                r#"
import vtk
r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName({path:?})
r.Update()
g = r.GetOutput()
assert g.GetNumberOfCells() == 8, g.GetNumberOfCells()
assert all(g.GetCell(i).GetNumberOfPoints() == 8 for i in range(g.GetNumberOfCells()))
d = g.GetCellData().GetArray("Density")
lo, hi = d.GetRange()
assert 0 < lo <= hi < 1e6, (lo, hi)
print("ok")
"#,
                path = path.display().to_string()
            ))
            .status()
            .expect("python");
        assert!(status.success(), "vtk python read failed");
        let _ = std::fs::remove_file(path);
    }
}
