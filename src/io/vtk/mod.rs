//! VTK XML 结构化网格（`.vts`）读入 — 仅二进制 appended 格式。

#[cfg(feature = "io-vtk")]
mod vtm_write;
mod vts;
#[cfg(feature = "io-vtk")]
mod vts_write;
#[cfg(feature = "io-vtk")]
mod vtu;
#[cfg(feature = "io-vtk")]
mod vtu_write;

#[cfg(feature = "io-vtk")]
pub use vtm_write::{VtmBlock, write_vtm};
pub use vts::{VtsLoadResult, load_vts};
#[cfg(feature = "io-vtk")]
pub use vts_write::{write_flow_vts, write_vts};
pub use vtu::{VtuLoadResult, load_vtu};
pub use vtu_write::write_flow_vtu;
