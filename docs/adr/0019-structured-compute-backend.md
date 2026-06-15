# ADR 0019: 结构化可压缩单一路径与 `StructuredComputeBackend` 聚合

- **状态**: 已接受（**S0 完成**；S1–S5 分阶段）
- **日期**: 2026-06-16
- **关联**: [ADR 0016](0016-runtime-compute-precision.md)、[ADR 0018](0018-unstructured-compute-backend.md)、[任务卡](../tasks/structured-f32-s0-s1.md)、[ARCHITECTURE.md](../ARCHITECTURE.md) §8.4

## 背景

ADR 0016 P2 已引入结构化 typed 无粘装配与 `run_multiblock_structured_typed_with_observer`，但驱动逻辑集中在 `typed.rs` / `multiblock_driver_typed.rs`，f32 谱半径、粘性、LU-SGS 扫掠等仍与非结构能力矩阵不齐。参照 ADR 0018，在结构化路径建立聚合 trait 与分阶段里程碑，避免 `ComputeFloat` 子 trait bound 沿调用链扩散。

本 ADR **不**追求字面全 fp32；几何、CFL 编排、GMRES Krylov 等可保留 f64（ADR 0016 §4）。

## 决策

### 1. 结构化生产路径（单块 / 多块）

| 精度 | case 入口 | 驱动 |
|------|-----------|------|
| f64 | `run_multiblock_structured_typed_with_observer::<f64>` | `CompressibleEulerSolver::advance_step_3d_typed::<f64>` |
| f32 | `run_multiblock_structured_typed_with_observer::<f32>` | `CompressibleEulerSolver::advance_step_3d_typed::<f32>` |

legacy f64 非 typed 多块路径在 S0 **保留**（数值等价）；新特性优先 typed 子模块扩展。

### 2. `StructuredComputeBackend` 聚合 trait

在 `solver::compressible::structured_compute_backend` 定义单一 supertrait（密封于 `f32` / `f64`），S0 合并：

- `ComputeFloat`
- `LusgsDiagonalUpdateBackend`
- `InviscidFaceFluxTyped`
- `PrimitiveFillFromConserved`

求解器 / RHS typed 边界写 `T: StructuredComputeBackend`；子 trait 仍保留在 `discretization` / `field` 供模块内分发。

后续阶段扩展：

| 阶段 | 追加 trait / 能力 |
|------|------------------|
| **S0** ✅ | 上表四项 + 驱动子模块拆分 |
| S1 | `StructuredSpectralRadiusTyped`、f32 \(\Delta t_i\) 缓冲 |
| S2 | 多块 1-to-1 接口 f32 通量 |
| S3 | 结构化粘性 typed 装配 |
| S4 | `StructuredLusgsSweepTyped`（`lusgs_sweep = true`） |
| S5 | exec SIMD、f32 benchmark 文档 |

### 3. 驱动子模块（S0-b ✅）

`typed.rs` 保留 GMRES typed 与 `advance_step_3d_typed` 编排；时间推进实现拆分至：

| 模块 | 职责 |
|------|------|
| `structured_prepare_timestep_typed.rs` | `prepare_spectral_timestep_3d_typed`、`prepare_lusgs_timestep_3d_typed` |
| `structured_explicit_typed.rs` | `advance_explicit_step_3d_typed`、`advance_explicit_step_typed` |
| `structured_lusgs_typed.rs` | `advance_lusgs_step_3d_typed` |
| `structured_typed_tests.rs` | f32/f64 freestream 对照单测 |

### 4. 与非结构差异

| 项 | 结构化 | 非结构（ADR 0018） |
|----|--------|-------------------|
| 梯度 | 中心差分 / MUSCL stencil | IDWLS + 限制器 |
| LU-SGS 耦合 | i/j/k 方向预打包 | CSR + `lusgs_couplings_f32` |
| 聚合 trait | `StructuredComputeBackend` | `UnstructuredComputeBackend` |
| CUDA | S5 之后评估 | ADR 0017 已部分落地 |

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 与非结构共用 `CompressibleComputeBackend` | 谱半径 / LU-SGS / 梯度 dispatch 不同，YAGNI |
| S0 同时改 f32 谱半径 | 违反「S0 无数值变更」；留 S1 |
| 删除 legacy 多块 f64 驱动 | 破坏过渡兼容与 golden |

## 实现里程碑

| 阶段 | 交付 | 验证 | 状态 |
|------|------|------|------|
| **S0-a** | `StructuredComputeBackend` + `impl f32/f64` | `make check` | ✅ |
| **S0-b** | 驱动拆分为 prepare / explicit / lusgs 子模块 | f64/f32 freestream 单测；无数值变更 | ✅ |
| **S0-c** | ADR 定稿 + API/CHANGELOG | 文档审查 | ✅ |
| S1 | f32 谱半径 + 显式 \(\Delta t_i\) 缓冲 | freestream f32≈f64 | 待办 |
| S2 | 多块接口 f32 通量 | 1-to-1 freestream | 待办 |
| S3 | 粘性 typed 装配 | 均匀场零 RHS | 待办 |
| S4 | LU-SGS 扫掠 f32 | 小盒 smoke | 待办 |
| S5 | SIMD + V&V benchmark | `docs/BENCHMARKS.md` | 待办 |

## S0 完成后能力矩阵（`compute_precision = "f32"`）

| 配置 | 支持 |
|------|------|
| 单块 / 多块 Euler 一阶 / MUSCL | ✅ |
| RK4 / Euler 显式 + LTS | ✅（\(\Delta t\) / \(\sigma\) 仍主要 f64，S1 改 f32） |
| LU-SGS 对角（`lusgs_sweep = false`） | ✅ |
| LU-SGS 扫掠 f32 | ❌（S4；Validate 报错） |
| 结构化粘性 f32 | ❌（S3） |
| 多块 1-to-1 接口通量 f32 | ❌（S2） |

## 兼容性

- 默认 `compute_precision = "f64"` 不变。
- S0 **无数值语义变更**；f32 未实现组合在 Validate 阶段报错，不静默回退 f64（ADR 0016 既有规则）。
- `run_multiblock_structured_with_observer` 保留；typed 路径为 f32 与后续 S1+ 扩展的主线。

## 后果

### 正面

- 结构化 f32 改造有与非结构对齐的 trait + 文件边界，便于 S1 起按里程碑 PR。
- `typed.rs` 圈复杂度下降，prepare / explicit / LU-SGS 可独立测试。

### 负面

- S0 仍维持 legacy + typed 双轨（至 S1 能力对齐前）；文档须标明矩阵缺口。
