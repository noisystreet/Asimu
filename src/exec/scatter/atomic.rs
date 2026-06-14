//! f64 原子累加（ADR 0013 E1；`AtomicU64` + CAS）。

#![allow(unsafe_op_in_unsafe_fn)]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::core::Real;

use super::contribution::{
    InviscidScatterOp, InviscidScatterOpF32, ViscousScatterOp, ViscousScatterOpF32,
};
use super::ptr::SendMutPtr;

#[inline]
pub(super) fn fetch_add_f32(target: *mut f32, value: f32) {
    if value == 0.0 {
        return;
    }
    // SAFETY: `target` 来自已验证长度的 `&mut [f32]`；与 `AtomicU32` 同尺寸同对齐。
    unsafe {
        let atom = &*(target.cast::<AtomicU32>());
        let mut current = atom.load(Ordering::Relaxed);
        loop {
            let new = f32::from_bits(current) + value;
            match atom.compare_exchange_weak(
                current,
                new.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

#[inline]
pub(super) fn fetch_add_f64(target: *mut Real, value: Real) {
    if value == 0.0 {
        return;
    }
    // SAFETY: `target` 来自已验证长度的 `&mut [Real]`；与 `AtomicU64` 同尺寸同对齐。
    unsafe {
        let atom = &*(target.cast::<AtomicU64>());
        let mut current = atom.load(Ordering::Relaxed);
        loop {
            let new = Real::from_bits(current) + value;
            match atom.compare_exchange_weak(
                current,
                new.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ViscousResidualPtrs {
    mx: SendMutPtr,
    my: SendMutPtr,
    mz: SendMutPtr,
    energy: SendMutPtr,
}

impl ViscousResidualPtrs {
    pub(super) fn from_slices(
        mx: &mut [Real],
        my: &mut [Real],
        mz: &mut [Real],
        energy: &mut [Real],
    ) -> Self {
        Self {
            mx: SendMutPtr::new(mx.as_mut_ptr()),
            my: SendMutPtr::new(my.as_mut_ptr()),
            mz: SendMutPtr::new(mz.as_mut_ptr()),
            energy: SendMutPtr::new(energy.as_mut_ptr()),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct InviscidResidualPtrsF32 {
    density: super::ptr::SendMutPtrF32,
    mx: super::ptr::SendMutPtrF32,
    my: super::ptr::SendMutPtrF32,
    mz: super::ptr::SendMutPtrF32,
    energy: super::ptr::SendMutPtrF32,
}

impl InviscidResidualPtrsF32 {
    pub(super) fn from_slices(
        density: &mut [f32],
        mx: &mut [f32],
        my: &mut [f32],
        mz: &mut [f32],
        energy: &mut [f32],
    ) -> Self {
        use super::ptr::SendMutPtrF32;
        Self {
            density: SendMutPtrF32::new(density.as_mut_ptr()),
            mx: SendMutPtrF32::new(mx.as_mut_ptr()),
            my: SendMutPtrF32::new(my.as_mut_ptr()),
            mz: SendMutPtrF32::new(mz.as_mut_ptr()),
            energy: SendMutPtrF32::new(energy.as_mut_ptr()),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct ViscousResidualPtrsF32 {
    mx: super::ptr::SendMutPtrF32,
    my: super::ptr::SendMutPtrF32,
    mz: super::ptr::SendMutPtrF32,
    energy: super::ptr::SendMutPtrF32,
}

impl ViscousResidualPtrsF32 {
    pub(super) fn from_slices(
        mx: &mut [f32],
        my: &mut [f32],
        mz: &mut [f32],
        energy: &mut [f32],
    ) -> Self {
        use super::ptr::SendMutPtrF32;
        Self {
            mx: SendMutPtrF32::new(mx.as_mut_ptr()),
            my: SendMutPtrF32::new(my.as_mut_ptr()),
            mz: SendMutPtrF32::new(mz.as_mut_ptr()),
            energy: SendMutPtrF32::new(energy.as_mut_ptr()),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct InviscidResidualPtrs {
    density: SendMutPtr,
    mx: SendMutPtr,
    my: SendMutPtr,
    mz: SendMutPtr,
    energy: SendMutPtr,
}

impl InviscidResidualPtrs {
    pub(super) fn from_slices(
        density: &mut [Real],
        mx: &mut [Real],
        my: &mut [Real],
        mz: &mut [Real],
        energy: &mut [Real],
    ) -> Self {
        Self {
            density: SendMutPtr::new(density.as_mut_ptr()),
            mx: SendMutPtr::new(mx.as_mut_ptr()),
            my: SendMutPtr::new(my.as_mut_ptr()),
            mz: SendMutPtr::new(mz.as_mut_ptr()),
            energy: SendMutPtr::new(energy.as_mut_ptr()),
        }
    }
}

#[inline]
pub(super) unsafe fn scatter_viscous_op_atomic(op: ViscousScatterOp, ptrs: ViscousResidualPtrs) {
    fetch_add_f64(ptrs.mx.as_ptr().add(op.owner), op.owner_scale * op.flux_mx);
    fetch_add_f64(ptrs.my.as_ptr().add(op.owner), op.owner_scale * op.flux_my);
    fetch_add_f64(ptrs.mz.as_ptr().add(op.owner), op.owner_scale * op.flux_mz);
    fetch_add_f64(
        ptrs.energy.as_ptr().add(op.owner),
        op.owner_scale * op.flux_energy,
    );
    fetch_add_f64(
        ptrs.mx.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_mx,
    );
    fetch_add_f64(
        ptrs.my.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_my,
    );
    fetch_add_f64(
        ptrs.mz.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_mz,
    );
    fetch_add_f64(
        ptrs.energy.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_energy,
    );
}

#[inline]
pub(super) unsafe fn scatter_viscous_op_atomic_f32(
    op: ViscousScatterOpF32,
    ptrs: ViscousResidualPtrsF32,
) {
    fetch_add_f32(ptrs.mx.as_ptr().add(op.owner), op.owner_scale * op.flux_mx);
    fetch_add_f32(ptrs.my.as_ptr().add(op.owner), op.owner_scale * op.flux_my);
    fetch_add_f32(ptrs.mz.as_ptr().add(op.owner), op.owner_scale * op.flux_mz);
    fetch_add_f32(
        ptrs.energy.as_ptr().add(op.owner),
        op.owner_scale * op.flux_energy,
    );
    fetch_add_f32(
        ptrs.mx.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_mx,
    );
    fetch_add_f32(
        ptrs.my.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_my,
    );
    fetch_add_f32(
        ptrs.mz.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_mz,
    );
    fetch_add_f32(
        ptrs.energy.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.flux_energy,
    );
}

#[inline]
pub(super) unsafe fn scatter_inviscid_op_atomic_f32(
    op: InviscidScatterOpF32,
    ptrs: InviscidResidualPtrsF32,
) {
    fetch_add_f32(
        ptrs.density.as_ptr().add(op.owner),
        op.owner_scale * op.mass,
    );
    fetch_add_f32(
        ptrs.mx.as_ptr().add(op.owner),
        op.owner_scale * op.momentum[0],
    );
    fetch_add_f32(
        ptrs.my.as_ptr().add(op.owner),
        op.owner_scale * op.momentum[1],
    );
    fetch_add_f32(
        ptrs.mz.as_ptr().add(op.owner),
        op.owner_scale * op.momentum[2],
    );
    fetch_add_f32(
        ptrs.energy.as_ptr().add(op.owner),
        op.owner_scale * op.energy,
    );
    fetch_add_f32(
        ptrs.density.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.mass,
    );
    fetch_add_f32(
        ptrs.mx.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.momentum[0],
    );
    fetch_add_f32(
        ptrs.my.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.momentum[1],
    );
    fetch_add_f32(
        ptrs.mz.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.momentum[2],
    );
    fetch_add_f32(
        ptrs.energy.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.energy,
    );
}

#[inline]
pub(super) unsafe fn scatter_inviscid_op_atomic(op: InviscidScatterOp, ptrs: InviscidResidualPtrs) {
    fetch_add_f64(
        ptrs.density.as_ptr().add(op.owner),
        op.owner_scale * op.mass,
    );
    fetch_add_f64(
        ptrs.mx.as_ptr().add(op.owner),
        op.owner_scale * op.momentum[0],
    );
    fetch_add_f64(
        ptrs.my.as_ptr().add(op.owner),
        op.owner_scale * op.momentum[1],
    );
    fetch_add_f64(
        ptrs.mz.as_ptr().add(op.owner),
        op.owner_scale * op.momentum[2],
    );
    fetch_add_f64(
        ptrs.energy.as_ptr().add(op.owner),
        op.owner_scale * op.energy,
    );
    fetch_add_f64(
        ptrs.density.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.mass,
    );
    fetch_add_f64(
        ptrs.mx.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.momentum[0],
    );
    fetch_add_f64(
        ptrs.my.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.momentum[1],
    );
    fetch_add_f64(
        ptrs.mz.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.momentum[2],
    );
    fetch_add_f64(
        ptrs.energy.as_ptr().add(op.neighbor),
        op.neighbor_scale * op.energy,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::approx_eq;

    #[test]
    fn fetch_add_f32_matches_serial_add() {
        let mut values = [1.0_f32, 2.0, 3.0];
        fetch_add_f32(values.as_mut_ptr(), 0.5);
        fetch_add_f32(values.as_mut_ptr().wrapping_add(1), -0.25);
        assert!((values[0] - 1.5).abs() < 1.0e-6);
        assert!((values[1] - 1.75).abs() < 1.0e-6);
        assert!((values[2] - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn fetch_add_f64_matches_serial_add() {
        let mut values = [1.0, 2.0, 3.0];
        fetch_add_f64(values.as_mut_ptr(), 0.5);
        fetch_add_f64(values.as_mut_ptr().wrapping_add(1), -0.25);
        assert!(approx_eq(values[0], 1.5, 1.0e-15));
        assert!(approx_eq(values[1], 1.75, 1.0e-15));
        assert!(approx_eq(values[2], 3.0, 1.0e-15));
    }
}
