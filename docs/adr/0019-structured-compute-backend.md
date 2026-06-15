# ADR 0019: 结构化可压缩单一路径与 `StructuredComputeBackend` 聚合

- **状态**: 提议中（实现分阶段 S0–S5）
- **日期**: 2026-06-16
- **关联**: [ADR 0016](0016-runtime-compute-precision.md)、[ADR 0018](0018-unstructured-compute-backend.md)、[任务卡](../tasks/structured-f32-s0-s1.md)

## 背景

ADR 0016 P2 已引入结构化 typed 无粘装配与 `run_multiblock_structured_typed_with_observer`，但驱动逻辑集中在 `typed.rs` / `multiblock_driver_typed.rs`，f32 谱半径、粘性、LU-SGS 扫掠等仍与非结构能力矩阵不齐。参照 ADR 0018，在结构化路径建立聚合 trait 与分阶段里程碑，避免 `ComputeFloat` 子 trait bound 沿调用链扩散。

本 ADR **不**追求字面全 fp32；几何、CFL 编排、GMRES Krylov 等可保留 f64（ADR 0016 §4）。

## 决策

### 1. `StructuredComputeBackend` 聚合 trait

首版（S0）合并：

- `ComputeFloat`
- `LusgsDiagonalUpdateBackend`
- `InviscidFaceFluxTyped`
- `PrimitiveFillFromConserved`

后续阶段扩展：

| 阶段 | 追加 trait / 能力 |
|------|------------------|
| S1 | `StructuredSpectralRadiusTyped`、f32 Δt 缓冲 |
| S2 | 多块接口 f32 通量 |
| S3 | 结构化粘性 typed 装配 |
| S4 | `StructuredLusgsSweepTyped` |
| S5 | exec SIMD、f32 benchmark |

定义位置：`solver::compressible::structured_compute_backend`。

### 2. 驱动子模块（对标非结构）

```
structured_driver_typed.rs
structured_prepare_timestep_typed.rs
structured_explicit_typed.rs
structured_lusgs_typed.rs
```

### 3. 与非结构差异

- 梯度：中心差分 / MUSCL stencil，无 IDWLS。
- LU-SGS 耦合：i/j/k 方向预打包，非 CSR。
- CUDA：排在 CPU 路径对齐之后（远期）。

## 实现里程碑

| 阶段 | 交付 | 验证 |
|------|------|------|
| S0 | trait + 驱动拆分 | `make check`；无数值变更 |
| S1 | f32 谱半径 + 显式 dt | freestream f32≈f64 |
| S2 | 多块接口 f32 | 1-to-1 freestream |
| S3 | 粘性 typed | 均匀场零 RHS |
| S4 | LU-SGS 扫掠 f32 | 小盒 smoke |
| S5 | SIMD + V&V | benchmark 文档 |

## 兼容性

- 默认 `compute_precision = "f64"` 不变。
- 未实现组合在 Validate 阶段报错，不静默回退 f64。
