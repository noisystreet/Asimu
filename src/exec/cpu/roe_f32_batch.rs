//! Roe 通量四路批处理 f32（`simd-fvm` 桶内 batch4；逐 lane 调用 `roe_f32`）。

use crate::discretization::inviscid_f32::{FaceNormalF32, InviscidFluxF32};
use crate::discretization::roe::RoeFluxConfig;
use crate::discretization::roe_f32::roe_flux_with_primitives_f32;
use crate::discretization::viscous_boundary_f32::PrimitiveStateF32;
use crate::physics::IdealGasEoS;

/// 四路一阶 Roe 面通量（f32 原变量）。
pub fn face_inviscid_flux_first_order_roe_batch4_f32(
    left: [&PrimitiveStateF32; 4],
    right: [&PrimitiveStateF32; 4],
    normals: [FaceNormalF32; 4],
    eos: &IdealGasEoS,
    config: &RoeFluxConfig,
) -> [Option<InviscidFluxF32>; 4] {
    let mut out = [None; 4];
    for i in 0..4 {
        out[i] = roe_flux_with_primitives_f32(left[i], right[i], normals[i], eos, config).ok();
    }
    out
}
