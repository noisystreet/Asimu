//! tracing 订阅器：stderr 日志 + 可选 Chrome trace。

use std::path::{Path, PathBuf};

use time::{UtcOffset, format_description::parse};
use tracing::{info, warn};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{
    EnvFilter, Layer, fmt::time::OffsetTime, layer::SubscriberExt, util::SubscriberInitExt,
};

use crate::error::{AsimuError, Result};

/// 与原先 UTC 日志一致，但不带 `Z` 或 `+08:00` 等时区后缀。
const LOCAL_LOG_TIME_FORMAT: &str =
    "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:6]";

/// 保持 Chrome trace 写线程直至算例结束（drop 时 flush）。
pub struct TracingGuard {
    _chrome: Option<tracing_chrome::FlushGuard>,
    trace_path: Option<PathBuf>,
}

impl TracingGuard {
    fn noop() -> Self {
        Self {
            _chrome: None,
            trace_path: None,
        }
    }
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        let Some(path) = self.trace_path.as_ref() else {
            return;
        };
        // FlushGuard 先 drop，再检查文件。
        self._chrome.take();
        match std::fs::metadata(path) {
            Ok(meta) if meta.len() > 0 => {
                info!(
                    path = %path.display(),
                    bytes = meta.len(),
                    "Chrome trace 已写入"
                );
            }
            Ok(_) => {
                warn!(
                    path = %path.display(),
                    "Chrome trace 文件为空（release 构建请确认 tracing 启用 release_max_level_info）"
                );
            }
            Err(err) => {
                warn!(
                    path = %path.display(),
                    error = %err,
                    "Chrome trace 文件未生成"
                );
            }
        }
    }
}

/// 初始化 tracing：stderr 文本日志；`chrome_trace` 非空时额外写出 Chrome trace JSON。
pub fn init_tracing(level: &str, chrome_trace: Option<&Path>) -> Result<TracingGuard> {
    if tracing::dispatcher::has_been_set() {
        return Ok(TracingGuard::noop());
    }

    let filter = EnvFilter::try_new(level)
        .map_err(|err| AsimuError::Config(format!("无效的日志级别 `{level}`: {err}")))?;
    // 在单线程启动阶段固定本地时区偏移，避免多线程下 `now_local` 不可靠。
    let format = parse(LOCAL_LOG_TIME_FORMAT)
        .map_err(|err| AsimuError::Config(format!("无效的日志时间格式: {err}")))?;
    let offset = UtcOffset::current_local_offset()
        .map_err(|err| AsimuError::Config(format!("无法获取本地时区偏移: {err}")))?;
    let timer = OffsetTime::new(offset, format);

    if let Some(path) = chrome_trace {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        if let Some(dir) = parent {
            std::fs::create_dir_all(dir).map_err(|err| {
                AsimuError::Config(format!(
                    "无法创建 Chrome trace 目录 {}: {err}",
                    dir.display()
                ))
            })?;
        }
        let file = std::fs::File::create(path).map_err(|err| {
            AsimuError::Config(format!(
                "无法创建 Chrome trace 文件 {}: {err}",
                path.display()
            ))
        })?;
        // Chrome 层单独 filter：stderr 仍用 `level`；桶级 scatter 为 trace 级（`--log-level trace` 或显式 target）。
        let chrome_filter = EnvFilter::try_new(format!("{level},asimu::exec::scatter=trace"))
            .map_err(|err| AsimuError::Config(format!("无效的 Chrome trace filter: {err}")))?;
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .writer(file)
            .include_args(true)
            .build();
        let init_result = tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_writer(std::io::stderr)
                    .with_timer(timer.clone()),
            )
            .with(chrome_layer.with_filter(chrome_filter))
            .try_init();
        if let Err(err) = init_result {
            if tracing::dispatcher::has_been_set() {
                return Ok(TracingGuard::noop());
            }
            return Err(AsimuError::Config(format!("初始化日志失败: {err}")));
        }
        info!(path = %path.display(), "Chrome trace 已启用");
        return Ok(TracingGuard {
            _chrome: Some(guard),
            trace_path: Some(path.to_path_buf()),
        });
    }

    let init_result = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_timer(timer)
        .try_init();
    if let Err(err) = init_result {
        if tracing::dispatcher::has_been_set() {
            return Ok(TracingGuard::noop());
        }
        return Err(AsimuError::Config(format!("初始化日志失败: {err}")));
    }

    info!(level, "日志已初始化");
    Ok(TracingGuard::noop())
}
