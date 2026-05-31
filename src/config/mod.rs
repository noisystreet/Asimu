//! 配置加载：命令行 > 环境变量 > 配置文件 > 默认值。
//!
//! 详见 `docs/ARCHITECTURE.md` 中的配置管理约定。

mod tracing_init;

use std::path::{Path, PathBuf};

use clap::Parser;
use serde::{Deserialize, Serialize};

pub use tracing_init::{TracingGuard, init_tracing};

use crate::error::{AsimuError, Result};

/// 全局运行配置。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    pub solver: SolverConfig,
    pub logging: LoggingConfig,
}

/// 求解器相关配置（全局 default.toml / CLI 占位；算例见 `case.toml` `[time]`）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SolverConfig {
    pub max_steps: u64,
}

/// 日志相关配置。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingConfig {
    pub level: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig { max_steps: 100 },
            logging: LoggingConfig {
                level: "info".to_string(),
            },
        }
    }
}

/// 命令行参数（优先级最高）。
#[derive(Debug, Parser)]
#[command(name = "asimu", about = "Rust CFD 求解器")]
pub struct Cli {
    /// 配置文件路径（TOML）
    #[arg(long, env = "ASIMU_CONFIG")]
    pub config: Option<PathBuf>,

    /// 最大时间步数（稳态伪时间 / 瞬态物理时间共用）
    #[arg(long, env = "ASIMU_MAX_STEPS")]
    pub max_steps: Option<u64>,

    /// 日志级别: error | warn | info | debug | trace
    #[arg(long, env = "ASIMU_LOG_LEVEL")]
    pub log_level: Option<String>,

    /// 算例文件（TOML）
    #[arg(long, env = "ASIMU_CASE", value_name = "CASE_TOML")]
    pub case: Option<PathBuf>,

    /// Chrome trace JSON（[ui.perfetto.dev](https://ui.perfetto.dev)），覆盖 case.toml。
    /// 无路径：`output/profiling/trace.json`；有路径：相对当前工作目录或绝对路径。
    #[arg(
        long,
        env = "ASIMU_CHROME_TRACE",
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub chrome_trace: Option<String>,
}

impl Cli {
    /// 解析命令行并合并配置来源。
    pub fn load_config(self) -> Result<AppConfig> {
        let mut config = if let Some(path) = self.config {
            load_config_file(&path)?
        } else {
            let default_path = Path::new("config/default.toml");
            if default_path.exists() {
                load_config_file(default_path)?
            } else {
                AppConfig::default()
            }
        };

        if let Some(max_steps) = self.max_steps {
            config.solver.max_steps = max_steps;
        }
        if let Some(level) = self.log_level {
            config.logging.level = level;
        }

        Ok(config)
    }
}

fn load_config_file(path: &Path) -> Result<AppConfig> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| AsimuError::Config(format!("无法读取配置文件 {}: {err}", path.display())))?;
    toml::from_str(&content)
        .map_err(|err| AsimuError::Config(format!("无法解析配置文件 {}: {err}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = AppConfig::default();
        assert!(config.solver.max_steps > 0);
    }

    #[test]
    fn cli_overrides_config_values() {
        let cli = Cli {
            config: None,
            max_steps: Some(42),
            log_level: Some("debug".to_string()),
            case: None,
            chrome_trace: None,
        };
        let config = cli.load_config().expect("load config");
        assert_eq!(config.solver.max_steps, 42);
        assert_eq!(config.logging.level, "debug");
    }
}
