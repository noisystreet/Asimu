//! 单元中心场 → 顶点场（ParaView / CGNS Vertex 写出共用）。

#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
use crate::error::Result;
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
use crate::field::ConservedFields;
#[cfg(any(feature = "io-vtk", feature = "io-cgns", test))]
use crate::mesh::StructuredMesh3d;
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
use crate::physics::{IdealGasEoS, PrimitiveState};

/// 单元中心原始变量（ρ, u, v, w, p），长度 = `mesh.num_cells()`，VTK 顺序 k–j–i。
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
pub type CellPrimitiveArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// 顶点原始变量数组（ρ, u, v, w, p），长度 = `mesh.num_nodes()`。
#[cfg(feature = "io-cgns")]
pub type VertexPrimitiveArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
pub fn gather_cell_primitives(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<CellPrimitiveArrays> {
    let n = mesh.num_cells();
    let mut rho_c = Vec::with_capacity(n);
    let mut u_c = Vec::with_capacity(n);
    let mut v_c = Vec::with_capacity(n);
    let mut w_c = Vec::with_capacity(n);
    let mut p_c = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let prim = fields.primitive_at(mesh.cell_index(i, j, k), eos, min_pressure)?;
                push_primitive(&mut rho_c, &mut u_c, &mut v_c, &mut w_c, &mut p_c, &prim);
            }
        }
    }
    Ok((rho_c, u_c, v_c, w_c, p_c))
}

#[cfg(feature = "io-cgns")]
pub fn gather_vertex_primitives(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<VertexPrimitiveArrays> {
    let (rho_c, u_c, v_c, w_c, p_c) = gather_cell_primitives(mesh, fields, eos, min_pressure)?;
    Ok((
        scatter_cell_scalar_to_vertices(mesh, &rho_c),
        scatter_cell_scalar_to_vertices(mesh, &u_c),
        scatter_cell_scalar_to_vertices(mesh, &v_c),
        scatter_cell_scalar_to_vertices(mesh, &w_c),
        scatter_cell_scalar_to_vertices(mesh, &p_c),
    ))
}

/// 单元标量场按结构化网格节点平均到顶点。
#[cfg(any(feature = "io-cgns", test))]
pub fn scatter_cell_scalar_to_vertices(mesh: &StructuredMesh3d, cell: &[f64]) -> Vec<f64> {
    let npts = mesh.num_nodes();
    let mut node = vec![0.0; npts];
    let mut count = vec![0u32; npts];
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let c = cell[mesh.cell_index(i, j, k)];
                for dk in 0..2 {
                    for dj in 0..2 {
                        for di in 0..2 {
                            let idx = mesh.node_index(i + di, j + dj, k + dk);
                            node[idx] += c;
                            count[idx] += 1;
                        }
                    }
                }
            }
        }
    }
    for (value, n) in node.iter_mut().zip(count.iter()) {
        if *n > 0 {
            *value /= f64::from(*n);
        }
    }
    node
}

#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
fn push_primitive(
    rho: &mut Vec<f64>,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    w: &mut Vec<f64>,
    p: &mut Vec<f64>,
    prim: &PrimitiveState,
) {
    rho.push(prim.density);
    u.push(prim.velocity[0]);
    v.push(prim.velocity[1]);
    w.push(prim.velocity[2]);
    p.push(prim.pressure);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::StructuredMesh3d;

    #[test]
    fn scatter_uniform_cell_field_to_vertices() {
        let mesh = StructuredMesh3d::new(
            "box",
            2,
            2,
            1,
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 2.0,
            ],
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 1.0,
            ],
            vec![0.0; 18],
        )
        .expect("mesh");
        let cell = vec![3.0; mesh.num_cells()];
        let node = scatter_cell_scalar_to_vertices(&mesh, &cell);
        assert_eq!(node.len(), mesh.num_nodes());
        assert!(node.iter().all(|&v| (v - 3.0).abs() < 1.0e-12));
    }
}
