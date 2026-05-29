//! 统一错误类型，业务逻辑优先返回 `Result<T, AsimuError>`。

use thiserror::Error;

/// asimu 库级错误。
#[derive(Debug, Error)]
pub enum AsimuError {
    #[error("配置错误: {0}")]
    Config(String),

    #[error("网格错误: {0}")]
    Mesh(String),

    #[error("场错误: {0}")]
    Field(String),

    #[error("线性代数错误: {0}")]
    Linalg(String),

    #[error("求解器错误: {0}")]
    Solver(String),

    #[error("执行后端错误: {0}")]
    Exec(String),

    #[error("I/O 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML 解析错误: {0}")]
    Toml(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, AsimuError>;
