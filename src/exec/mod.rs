//! CPU/GPU 执行后端（v1.2+ 规划）。
//!
//! v0.x：`cpu` 子模块提供可选 SIMD 热算子（feature `simd-fvm`），
//! 标量回退路径始终可用；见 [ADR 0003](../docs/adr/0003-multi-precision-and-gpu.md)。

pub mod cpu;
