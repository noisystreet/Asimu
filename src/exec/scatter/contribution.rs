//! scatter 贡献与残差切片（exec 自有类型，不依赖 discretization）。

use std::ops::Range;

use crate::core::Real;
use crate::exec::context::ExecutionContext;

/// 单面粘性 scatter 贡献。
#[derive(Debug, Clone, Copy)]
pub struct ViscousScatterOp {
    pub owner: usize,
    pub neighbor: usize,
    pub owner_scale: Real,
    pub neighbor_scale: Real,
    pub flux_mx: Real,
    pub flux_my: Real,
    pub flux_mz: Real,
    pub flux_energy: Real,
}

/// 单面无粘 scatter 贡献。
#[derive(Debug, Clone, Copy)]
pub struct InviscidScatterOp {
    pub owner: usize,
    pub neighbor: usize,
    pub owner_scale: Real,
    pub neighbor_scale: Real,
    pub mass: Real,
    pub momentum: [Real; 3],
    pub energy: Real,
}

/// 粘性残差可变切片。
pub struct ViscousResidualMut<'a> {
    pub mx: &'a mut [Real],
    pub my: &'a mut [Real],
    pub mz: &'a mut [Real],
    pub energy: &'a mut [Real],
}

/// 无粘残差可变切片。
pub struct InviscidResidualMut<'a> {
    pub density: &'a mut [Real],
    pub mx: &'a mut [Real],
    pub my: &'a mut [Real],
    pub mz: &'a mut [Real],
    pub energy: &'a mut [Real],
}

/// 按 `valid` 掩码 scatter 粘性桶。
pub struct ViscousValidSlotScatter<'a, G, F> {
    pub ctx: &'a ExecutionContext,
    pub bucket_len: usize,
    pub geoms: &'a [G],
    pub fluxes: &'a [F],
    pub valid: &'a [bool],
    pub residual: ViscousResidualMut<'a>,
}

/// 按索引范围 scatter 粘性桶。
pub struct ViscousRangeScatter<'a, G, F> {
    pub ctx: &'a ExecutionContext,
    pub bucket_len: usize,
    pub geoms: &'a [G],
    pub fluxes: &'a [F],
    pub range: Range<usize>,
    pub residual: ViscousResidualMut<'a>,
}

/// 无粘 `(geom, flux)` 对 scatter。
pub struct InviscidPairScatter<'a, G, F> {
    pub ctx: &'a ExecutionContext,
    pub bucket_len: usize,
    pub pairs: &'a [(G, F)],
    pub residual: InviscidResidualMut<'a>,
}
