//! CLI 应用编排层。
//!
//! 负责配置加载、日志初始化与算例驱动；**不属于**稳定数值 library API。
//! v0.3+ 可演进为 `case` 模块（见 `docs/ARCHITECTURE.md`）。

use tracing::info;

use crate::config::{AppConfig, Cli};
use crate::error::Result;
use crate::mesh::Mesh;
use crate::solver::Solver;

/// CLI 应用主流程：加载配置 → 构建占位网格 → 运行求解器。
pub fn run(cli: Cli) -> Result<()> {
    let config = cli.load_config()?;
    crate::config::init_tracing(&config.logging.level)?;

    info!(
        version = env!("CARGO_PKG_VERSION"),
        max_iterations = config.solver.max_iterations,
        "asimu 启动"
    );

    let mesh = Mesh::new("demo", 64)?;
    let solver = Solver::new(config.solver);
    let result = solver.run(&mesh)?;

    info!(
        iterations = result.iterations,
        residual = result.residual,
        converged = result.converged,
        "求解完成"
    );

    Ok(())
}

/// 供集成测试使用的默认配置。
#[must_use]
pub fn demo_config() -> AppConfig {
    AppConfig::default()
}
