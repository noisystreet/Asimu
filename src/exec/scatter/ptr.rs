//! 并行 atomic scatter 用 raw slice 指针（着色桶内无别名写）。

use crate::core::Real;

#[derive(Clone, Copy)]
pub(super) struct SendMutPtr(*mut Real);

// SAFETY: 仅在同着色桶 scatter 内使用；桶内面无共享单元。
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

impl SendMutPtr {
    pub(super) fn new(ptr: *mut Real) -> Self {
        Self(ptr)
    }

    pub(super) fn as_ptr(self) -> *mut Real {
        self.0
    }
}

#[derive(Clone, Copy)]
pub(super) struct SendMutPtrF32(*mut f32);

// SAFETY: 仅在同着色桶 scatter 内使用；桶内面无共享单元。
unsafe impl Send for SendMutPtrF32 {}
unsafe impl Sync for SendMutPtrF32 {}

impl SendMutPtrF32 {
    pub(super) fn new(ptr: *mut f32) -> Self {
        Self(ptr)
    }

    pub(super) fn as_ptr(self) -> *mut f32 {
        self.0
    }
}
