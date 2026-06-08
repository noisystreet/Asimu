//! f64 原子累加（ADR 0013 E1；`AtomicU64` + CAS）。

#![allow(unsafe_op_in_unsafe_fn)]

use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::Real;

use super::contribution::{InviscidScatterOp, ViscousScatterOp};
use super::ptr::SendMutPtr;

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
    fn fetch_add_f64_matches_serial_add() {
        let mut values = [1.0, 2.0, 3.0];
        fetch_add_f64(values.as_mut_ptr(), 0.5);
        fetch_add_f64(values.as_mut_ptr().wrapping_add(1), -0.25);
        assert!(approx_eq(values[0], 1.5, 1.0e-15));
        assert!(approx_eq(values[1], 1.75, 1.0e-15));
        assert!(approx_eq(values[2], 3.0, 1.0e-15));
    }
}
