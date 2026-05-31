//! CLI 应用编排层。
//!
//! 负责配置加载、日志初始化与算例驱动；**不属于**稳定数值 library API。
//! v0.3+ 算例编排见 [`crate::case`]（见 `docs/ARCHITECTURE.md`）。

use tracing::info;

use crate::case;
use crate::config::{AppConfig, Cli};
use crate::error::{AsimuError, Result};

/// CLI 应用主流程：加载配置 → 运行算例或提示用法。
pub fn run(cli: Cli) -> Result<()> {
    let case_path = cli.case.clone();
    let chrome_trace = cli.chrome_trace.clone();
    let config = cli.load_config()?;

    let Some(case_path) = case_path else {
        return Err(AsimuError::Config(
            "请指定算例：asimu --case path/to/case.toml".to_string(),
        ));
    };

    let result =
        case::run_case_path_logged(&case_path, &config.logging.level, chrome_trace.as_deref())?;
    info!(version = env!("CARGO_PKG_VERSION"), "asimu 算例完成");
    info!(
        name = %result.name,
        benchmark_id = ?result.benchmark_id,
        summary = %result.summary,
        "算例完成"
    );
    Ok(())
}

/// 供集成测试使用的默认配置。
#[must_use]
pub fn demo_config() -> AppConfig {
    AppConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn run_diffusion_case_via_cli_path() {
        let cli = Cli {
            config: None,
            max_steps: None,
            log_level: Some("warn".to_string()),
            case: Some(Path::new("tests/benchmarks/1d_diffusion_analytical/case.toml").into()),
            chrome_trace: None,
        };
        run(cli).expect("run");
    }
}
