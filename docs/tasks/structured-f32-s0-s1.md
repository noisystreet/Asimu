# 任务卡：结构化可压缩 f32 改造（S0–S5）

> 参照非结构 f32 路径（ADR 0016/0018）。目标：能力矩阵与非结构对齐的热路径形态，**非**字面全 fp32（几何/GMRES 线代等可保留 f64，见 ADR 0016 §4）。

## 原则

- S0：编排重构，数值不变（f64/f32 结果容差内一致）。
- S1：谱半径、Δt、显式推进改 f32 热路径；须 f32≈f64 对照测试。
- S2：多块 1-to-1 **共享接口**无粘通量改 f32 热路径；几何/scale 仍可 f64；须 2-block freestream f32≈f64 对照。
- S3–S5：见文末路线图；每阶段独立 PR，`make check` + 单独 commit。
- 每 PR：`make check`；未实现组合仍在 validate 报错，禁止静默回退 f64。

## PR 清单与状态

| ID | 阶段 | 标题 | 状态 |
|----|------|------|------|
| PR-1 | S0-a | `StructuredComputeBackend` + trait 骨架 | 已完成 |
| PR-2 | S0-b | 驱动子模块拆分（数值不变） | 已完成 |
| PR-3 | S0-c | ADR 0019 定稿 + API/CHANGELOG | 已完成 |
| PR-4 | S1-a | `StructuredFaceCacheF32` 面几何缓存 | 已完成 |
| PR-5 | S1-b | `cell_spectral_radius_3d_f32` + typed trait | 已完成 |
| PR-6 | S1-c | f32 时间步缓冲 + 显式推进闭环 | 已完成 |
| PR-7 | S2-a | 多块接口 f32 通量装配 + typed compute | 已完成 |
| PR-8 | S2-b | 驱动 wired + 2-block freestream 测试 + 文档 | 已完成 |

---

## PR-1（S0-a）：StructuredComputeBackend + trait 骨架

**交付**

- `src/solver/compressible/structured_compute_backend.rs`
- `impl StructuredComputeBackend for f32/f64`
- `docs/adr/0019-structured-compute-backend.md`（提议中）
- `mod.rs` 导出

**验收**：`make check`；无行为变更。

---

## PR-2（S0-b）：驱动子模块拆分

**新建**

- `structured_driver_typed.rs`
- `structured_explicit_typed.rs`
- `structured_prepare_timestep_typed.rs`
- `structured_lusgs_typed.rs`

**搬迁**（自 `typed.rs`）

| 原函数 | 新文件 |
|--------|--------|
| `advance_explicit_step_3d_typed` 等 | `structured_explicit_typed.rs` |
| `prepare_spectral/lusgs_timestep_3d_typed` | `structured_prepare_timestep_typed.rs` |
| `advance_lusgs_step_3d_typed` | `structured_lusgs_typed.rs` |

**验收**：f64 结果与 PR 前一致；`make check`。

---

## PR-3（S0-c）：ADR 0019 定稿

- ADR 状态 → 已接受
- `docs/API.md`、`CHANGELOG.md`
- 里程碑 S0–S5 表

---

## PR-4（S1-a）：StructuredFaceCacheF32

- `src/discretization/structured_face_cache_f32.rs`
- i/j/k 法向、面积、体积 f32 预打包
- `assembly_3d_typed` 内面读 cache
- 测试：uniform box f32 vs f64 几何

---

## PR-5（S1-b）：谱半径 f32

- `spectral_radius_3d_f32.rs`
- `StructuredSpectralRadiusTyped`
- 去掉 f32 路径 `cast_real` → `cell_spectral_radius_3d`
- 测试：freestream box σ_f32 ≈ σ_f64

---

## PR-6（S1-c）：时间步 + 显式闭环

- `StructuredTimestepBuffers`
- `euler/rk4_step_local_f32`（结构化 LTS）
- LU-SGS 对角用 f32 σ/dt
- 测试：RK4 freestream f32≈f64；CHANGELOG S1

---

## PR-7（S2-a）：多块接口 f32 通量装配

### 背景 / 缺口

- `multiblock_driver_typed` 已 wired `interface_residual`，但 `compute_shared_interface_residuals` **始终**用 f64 `PrimitiveFields` + `inviscid_boundary_face_flux_with_normal`。
- `apply_interface_residuals_typed` 仅把 f64 通量写入 `ConservedResidualT<f32>`（`add_flux_to_cell` 走 Real 分量）。
- S2 目标：接口通量计算与 scatter **原生 f32**，与块内 `assembly_3d_interior_f32` 对齐。

### 交付

| 项 | 说明 |
|----|------|
| `StructuredMultiblockInterfaceTyped` | 密封 trait（`f32`/`f64`）；挂入 `StructuredComputeBackend` supertrait |
| `compute_shared_interface_residuals_typed` | 精度分发；`f32` 填 `PrimitiveFieldsT<f32>`，不再经 f64 原始变量缓冲 |
| f32 通量求值 | 复用 `face_inviscid_flux_first_order_boundary_soa_f32`（或等价 helper）：owner 原始变量 f32 + donor `ConservedState` 外态 |
| typed contribution | `InterfaceResidualContribution` 扩展或并列 `InterfaceResidualContributionF32`（存 `InviscidFluxF32`）；避免热路径 Real 通量中间态 |
| `apply_interface_residuals_typed` | `f32` 走 `InviscidFluxF32` scatter（`accumulate_boundary_face_f32` 或专用 `add_flux_f32_to_cell`）；`f64` 保持现状 |
| 几何 | `SharedInterfaceFace.normal` / `owner_scale` / `donor_scale` 仍 f64（ADR 0016 §4）；scatter 时 cast `scale as f32` |

### 不在本 PR

- 接口 ghost `fill_interface_ghosts` 仍用 Real `ConservedFields` 快照（与 S1 GMRES 镜像策略一致）。
- 多块 GMRES / `lusgs_sweep = true` validate 拒绝规则不变。
- MUSCL 接口独立 stencil（接口语义与现 f64 边界面 Riemann 一致即可；均匀 freestream 验收不依赖 MUSCL）。

### 验收

- `make check`
- 单元测试：`compute_shared_interface_residuals_typed::<f32>` 在 2-face 玩具网格上 flux_f32 ≈ flux_f64（相对误差 < 1e-3）

---

## PR-8（S2-b）：驱动集成 + 2-block freestream 回归

### 交付

- `multiblock_driver_typed`：`T=f32` 时调用 `compute_shared_interface_residuals_typed`（替换 f64-only compute）
- 集成测试：2-block 1-to-1 I-face（参考 `src/mesh/multiblock.rs` `with_interfaces` fixture）+ 均匀 freestream
  - 单步 typed f32 vs f64：聚合 `residual_rms` 与关键 cell ρ 相对误差
  - 可选：仅接口 contribution 与块内 RHS 解耦断言（便于定位）
- `CHANGELOG.md` S2、`docs/API.md` trait/函数表、ADR 0019 里程碑 S2 → 已完成

### 验收

- `make check`
- 能力矩阵「多块 1-to-1 接口通量 f32」→ ✅

---

## 能力矩阵

| 配置 | S0 后 | S1 后 | S2 后 |
|------|--------|--------|--------|
| Euler 一阶/MUSCL f32 | ✅ | ✅ | ✅ |
| f32 LTS + RK4/Euler | ✅（dt f64） | ✅（dt f32） | ✅ |
| LU-SGS 对角 f32 | ✅ | ✅（σ/dt f32） | ✅ |
| LU-SGS 扫掠 f32 | ❌ | ❌ | ❌（S4） |
| 粘性 f32 | ❌ | ❌ | ❌（S3） |
| 多块 1-to-1 接口通量 f32 | ❌ | ❌ | ✅ |

---

## 后续路线图（概要）

| 阶段 | 交付 | 验证 |
|------|------|------|
| **S3** | 结构化粘性 typed 装配（`assembly_3d_viscous_f32` 或等价） | 均匀场零 RHS |
| **S4** | `StructuredLusgsSweepTyped`（`lusgs_sweep = true`） | 小盒 smoke |
| **S5** | exec SIMD + f32 benchmark 文档 | `docs/BENCHMARKS.md` |

---

## PR Checklist

- [ ] `make check`
- [ ] 无生产路径 `unwrap`
- [ ] f32 未实现能力 validate 报错
- [ ] 数值变更：f32 vs f64 测试 + CHANGELOG
- [ ] API/ADR 同步
