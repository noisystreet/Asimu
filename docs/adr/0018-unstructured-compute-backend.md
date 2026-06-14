# ADR 0018: 非结构可压缩单一路径与 `UnstructuredComputeBackend` 聚合

- **状态**: 已接受（实现分阶段 U0–U3）
- **日期**: 2026-06-13
- **关联**: [ADR 0016](0016-runtime-compute-precision.md)、[ADR 0017](0017-gpu-cuda-cudarc-multi-backend.md)、[ARCHITECTURE.md](../ARCHITECTURE.md) §8.4

## 背景

ADR 0016 P3 已引入非结构 typed 驱动（`run_unstructured_typed_with_observer::<T>`），但生产 **f64** 仍走 legacy 驱动 → `EvaluateRhsUnstructured` → `assemble_inviscid_residual_unstructured`，与 typed 路径并行维护。P1/P2 无粘通量 typed 化后，双轨成本高于收益：

| 问题 | 影响 |
|------|------|
| 两套 driver + RHS 上下文 | 修 bug / 加特性需改两处 |
| 7+ 个 `ComputeFloat` 子 trait bound 沿调用链传播 | 编译错误定位难、API 噪音大 |
| f64 typed 未挂接 `simd-fvm` 一阶内面 | 直接切换 case 可能性能回退 |
| `UnstructuredTypedRhsDispatch` 空标记 trait | 与 `DispatchImpl` 重复 |

本 ADR **不**扩展 `ComputeFloat` 至第三标量类型；bf16/SIMD 向量精度仍由 `exec`（ADR 0017）承载。

## 决策

### 1. 非结构生产路径统一为 typed 岛

| 精度 | case 入口 | 驱动 |
|------|-----------|------|
| f64 | `run_compressible_unstructured_3d_typed::<f64>` | `run_unstructured_typed_with_observer::<f64>` |
| f32 | `run_compressible_unstructured_3d_typed::<f32>` | `run_unstructured_typed_with_observer::<f32>` |

`run_unstructured_with_observer` 保留为 **薄包装**（委托 `::<f64>` typed 驱动），供既有测试与外部调用兼容；不再维护独立推进循环。

### 2. `UnstructuredComputeBackend` 聚合 trait

在 `solver::unstructured_compute_backend` 定义单一 supertrait，合并：

- `ComputeFloat`
- `LusgsDiagonalUpdateBackend`
- `InviscidFaceFluxTyped`
- `InviscidTypedScatterBackend`
- `ViscousTypedScatterBackend`
- `UnstructuredSpectralRadiusTyped`
- `LuSgsUnstructuredSweepTyped`
- `UnstructuredRhsDispatchImpl`（原 `rhs_dispatch::DispatchImpl`）

求解器 / case 边界仅写 `T: UnstructuredComputeBackend`；子 trait 仍保留在 `discretization` / `field` 供模块内分发。

### 3. f64 一阶内面 SIMD 挂载

`assemble_first_order_typed` 在 `T=f64` + `simd-fvm` 时，优先调用 `try_assemble_first_order_interior_simd_f64`（复用 `assembly_unstructured_inviscid_simd`），避免 typed 切换损失 Roe/HVL batch4 路径。

### 4. 暂保留、分阶段删除

| 项 | 阶段 |
|----|------|
| `assemble_inviscid_residual_unstructured` | U2 前保留（单元测试 / `compute_interior_inviscid_face_contribution`） |
| `EvaluateRhsUnstructured` | U1 后仅测试/文档引用，生产不经由 |
| `compressible_rhs_unstructured_typed.rs` | U1 删除（与 `DispatchImpl` 重复） |
| MUSCL f64 桥接 `muscl_f64_params` | **U3 已完成**：`assembly_unstructured_inviscid_f64` 直连 `GradientFieldsT<f64>` |

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 继续双 driver 至 P5 完成 | 无粘通量已全 typed，双轨无新增价值 |
| 全库 `CompressibleComputeBackend` 一次聚合 | 结构化与非结构 dispatch 不同，YAGNI |
| 删除 `run_unstructured_with_observer` | 破坏测试与 `docs/API.md` 过渡兼容 |

## 实现里程碑

| 阶段 | 交付 | 验证 |
|------|------|------|
| U0 | ADR + `UnstructuredComputeBackend` + case f64 → typed | `make check` |
| U1 | legacy driver 薄包装；删 `EvaluateRhsUnstructuredTyped` | 非结构 smoke / LU-SGS 测试 |
| U2 | f64 SIMD 挂接 typed 一阶内面 | `simd-fvm` 下 assembly 测试 |
| U3 | MUSCL f64 typed 原生装配（删 `muscl_f64_params`）；粘性 f64 桥接收缩（后续 PR） | 单 tet MUSCL freestream；dual_ellipsoid MUSCL golden |

## 兼容性

- 默认 `compute_precision = "f64"` 行为不变（数值路径改为 typed 实现，容差内等价）。
- `run_unstructured_with_observer` 签名不变。
- 未实现 typed 的路径在 Validate 阶段报错（ADR 0016 既有规则）。
