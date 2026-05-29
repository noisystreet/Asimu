//! VTK XML 结构化网格（`.vts`）读入 — 仅二进制 appended 格式。

mod vts;

pub use vts::{VtsLoadResult, load_vts};
