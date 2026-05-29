//! 探测外部 `.vts` 是否可被 asimu 读取。
//!
//! ```bash
//! cargo run --example probe_vts --features io-vtk -- /path/to/mesh.vts
//! ```

use std::{env, path::PathBuf, process};

use asimu::io::load_vts;
use asimu::mesh::StructuredMesh;

fn main() {
    let path = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("用法: cargo run --example probe_vts --features io-vtk -- <file.vts>");
        process::exit(2);
    });

    match load_vts(&path) {
        Ok(loaded) => {
            let mesh = &loaded.mesh;
            println!("OK  {}", path.display());
            println!(
                "    dim={} cells={} nodes={}",
                mesh.dimension(),
                mesh.num_cells(),
                mesh.num_nodes()
            );
            match mesh {
                StructuredMesh::D2(m) => {
                    println!("    nx={} ny={}", m.nx, m.ny);
                    println!(
                        "    角点 (0,0)=({:.6}, {:.6})",
                        m.node_x(0, 0),
                        m.node_y(0, 0)
                    );
                }
                StructuredMesh::D3(m) => {
                    println!("    nx={} ny={} nz={}", m.nx, m.ny, m.nz);
                    println!(
                        "    角点 (0,0,0)=({:.6}, {:.6}, {:.6})",
                        m.node_x(0, 0, 0),
                        m.node_y(0, 0, 0),
                        m.node_z(0, 0, 0)
                    );
                }
            }
        }
        Err(err) => {
            eprintln!("FAIL  {}", path.display());
            eprintln!("    {err}");
            process::exit(1);
        }
    }
}
