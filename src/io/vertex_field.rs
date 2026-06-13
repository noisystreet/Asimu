//! 单元中心场 → 顶点场（ParaView / CGNS Vertex 写出共用）。

#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
use crate::error::Result;
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
use crate::field::ConservedFields;
#[cfg(any(feature = "io-vtk", feature = "io-cgns", test))]
use crate::mesh::StructuredMesh3d;
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
use crate::physics::{IdealGasEoS, PrimitiveState};

/// 单元中心流场（ρ, u, v, w, p, Mach, T），长度 = `mesh.num_cells()`，VTK 顺序 k–j–i。
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
pub type CellPrimitiveArrays = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
);

/// 顶点流场（ρ, u, v, w, p, Mach, T），长度 = `mesh.num_nodes()`。
#[cfg(feature = "io-cgns")]
pub type VertexPrimitiveArrays = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
);

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
    let mut mach_c = Vec::with_capacity(n);
    let mut t_c = Vec::with_capacity(n);
    for k in 0..mesh.nz {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let prim = fields.primitive_at(mesh.cell_index(i, j, k), eos, min_pressure)?;
                rho_c.push(prim.density);
                u_c.push(prim.velocity[0]);
                v_c.push(prim.velocity[1]);
                w_c.push(prim.velocity[2]);
                p_c.push(prim.pressure);
                mach_c.push(mach_from_primitive(&prim, eos));
                t_c.push(temperature_from_primitive(&prim, eos));
            }
        }
    }
    Ok((rho_c, u_c, v_c, w_c, p_c, mach_c, t_c))
}

/// 非结构单元中心流场（ρ, u, v, w, p, Mach, T），长度 = `fields.num_cells()`。
#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
pub fn gather_unstructured_cell_primitives(
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<CellPrimitiveArrays> {
    let n = fields.num_cells();
    let mut rho_c = Vec::with_capacity(n);
    let mut u_c = Vec::with_capacity(n);
    let mut v_c = Vec::with_capacity(n);
    let mut w_c = Vec::with_capacity(n);
    let mut p_c = Vec::with_capacity(n);
    let mut mach_c = Vec::with_capacity(n);
    let mut t_c = Vec::with_capacity(n);
    for cell in 0..n {
        let prim = fields.primitive_at(cell, eos, min_pressure)?;
        rho_c.push(prim.density);
        u_c.push(prim.velocity[0]);
        v_c.push(prim.velocity[1]);
        w_c.push(prim.velocity[2]);
        p_c.push(prim.pressure);
        mach_c.push(mach_from_primitive(&prim, eos));
        t_c.push(temperature_from_primitive(&prim, eos));
    }
    Ok((rho_c, u_c, v_c, w_c, p_c, mach_c, t_c))
}

#[cfg(feature = "io-cgns")]
pub fn gather_vertex_primitives(
    mesh: &StructuredMesh3d,
    fields: &ConservedFields,
    eos: &IdealGasEoS,
    min_pressure: f64,
) -> Result<VertexPrimitiveArrays> {
    let (rho_c, u_c, v_c, w_c, p_c, mach_c, t_c) =
        gather_cell_primitives(mesh, fields, eos, min_pressure)?;
    Ok((
        scatter_cell_scalar_to_vertices(mesh, &rho_c),
        scatter_cell_scalar_to_vertices(mesh, &u_c),
        scatter_cell_scalar_to_vertices(mesh, &v_c),
        scatter_cell_scalar_to_vertices(mesh, &w_c),
        scatter_cell_scalar_to_vertices(mesh, &p_c),
        scatter_cell_scalar_to_vertices(mesh, &mach_c),
        scatter_cell_scalar_to_vertices(mesh, &t_c),
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
#[must_use]
fn mach_from_primitive(prim: &PrimitiveState, eos: &IdealGasEoS) -> f64 {
    let rho = prim.density.max(1.0e-30);
    let p = prim.pressure.max(1.0e-30);
    let speed = (prim.velocity[0] * prim.velocity[0]
        + prim.velocity[1] * prim.velocity[1]
        + prim.velocity[2] * prim.velocity[2])
        .sqrt();
    let a = (eos.gamma * p / rho).sqrt().max(1.0e-30);
    speed / a
}

#[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
#[must_use]
fn temperature_from_primitive(prim: &PrimitiveState, eos: &IdealGasEoS) -> f64 {
    let rho = prim.density.max(1.0e-30);
    let p = prim.pressure.max(1.0e-30);
    p / (rho * eos.gas_constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
    use crate::field::ConservedFields;
    use crate::mesh::StructuredMesh3d;
    #[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
    use crate::physics::{FreestreamParams, IdealGasEoS};

    #[cfg(any(feature = "io-vtk", feature = "io-cgns"))]
    #[test]
    fn freestream_mach_matches_case() {
        let mesh = StructuredMesh3d::uniform_box("b", 2, 2, 2, 1.0, 1.0, 1.0).expect("mesh");
        let eos = IdealGasEoS::AIR_STANDARD;
        let fs = FreestreamParams {
            mach: 0.5,
            ..FreestreamParams::default()
        };
        let reference =
            crate::physics::ReferenceScales::from_freestream(&eos, &fs, None).expect("ref");
        let mut nd_eos = eos;
        nd_eos.gas_constant = reference.nondimensional_gas_constant();
        let fields = ConservedFields::from_freestream(mesh.num_cells(), &eos, &fs).expect("fields");
        let (_, _, _, _, _, mach, t) =
            gather_cell_primitives(&mesh, &fields, &nd_eos, 1.0e-6).expect("gather");
        let ref_t = fields
            .primitive_at(0, &nd_eos, 1.0e-6)
            .expect("prim")
            .temperature;
        assert!(mach.iter().all(|&m| (m - 0.5).abs() < 1.0e-4));
        assert!(t.iter().all(|&x| (x - ref_t).abs() < 1.0e-4));
    }

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
