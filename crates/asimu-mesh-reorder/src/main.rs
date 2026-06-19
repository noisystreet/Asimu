use std::path::PathBuf;

use asimu::error::{AsimuError, Result};
use asimu::io::{CaseMesh, load_case};
use asimu::mesh_order::{
    CellOrderFile, bfs_order, identity_order, rcm_order, write_cell_order_file,
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    /// 输入 case.toml；工具复用 case 的 mesh 解析入口读取 CGNS/VTU。
    #[arg(long)]
    case: PathBuf,
    /// 输出 order.toml。
    #[arg(long)]
    output: PathBuf,
    /// 单元排序策略。
    #[arg(long, value_enum, default_value_t = Strategy::Rcm)]
    strategy: Strategy,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Strategy {
    Identity,
    Bfs,
    Rcm,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let case = load_case(&cli.case)?;
    let CaseMesh::Unstructured3d(mesh) = &case.mesh else {
        return Err(AsimuError::Config(
            "asimu-mesh-reorder 第一版仅支持非结构 3D case".to_string(),
        ));
    };
    let order = match cli.strategy {
        Strategy::Identity => identity_order(mesh.num_cells()),
        Strategy::Bfs => bfs_order(mesh)?,
        Strategy::Rcm => rcm_order(mesh)?,
    };
    let file = CellOrderFile::new(cli.strategy.label(), order)?;
    write_cell_order_file(&cli.output, &file)?;
    Ok(())
}

impl Strategy {
    fn label(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Bfs => "bfs",
            Self::Rcm => "rcm",
        }
    }
}
