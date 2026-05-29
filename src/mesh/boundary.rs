//! 边界面拓扑抽象（CFL3D 面分段 BC 的网格侧接口）。

use crate::core::{CellId, FaceId, Real};
use crate::error::Result;

/// 提供边界面 → 单元映射的网格接口。
pub trait BoundaryMesh {
    fn num_cells(&self) -> usize;

    /// 面所属单元（边界面上取内侧单元）。
    fn face_owner(&self, face: FaceId) -> Result<CellId>;

    /// 面中心到 owner 单元中心的距离（1D 均匀网格为 `dx/2`）。
    fn face_spacing(&self, face: FaceId) -> Result<Real>;

    /// 1D 外法向：左边界为 `-1`，右边界为 `+1`。
    fn face_outward_normal(&self, face: FaceId) -> Result<Real>;

    /// 逻辑边界名 → 面列表（1D：`left` / `right`）。
    fn resolve_logical_boundary(&self, name: &str) -> Result<Vec<FaceId>>;
}
