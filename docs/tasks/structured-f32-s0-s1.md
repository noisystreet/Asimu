# 任务卡：结构化可压缩 f32 改造 S0–S1

> 参照非结构 f32 路径（ADR 0016/0018）。目标：能力矩阵与非结构对齐的热路径形态，**非**字面全 fp32（几何/GMRES 线代等可保留 f64，见 ADR 0016 §4）。

## 原则

- S0：编排重构，数值不变（f64/f32 结果容差内一致）。
- S1：谱半径、Δt、显式推进改 f32 热路径；须 f32≈f64 对照测试。
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

## S0/S1 完成后能力矩阵

| 配置 | S0 后 | S1 后 |
|------|--------|--------|
| Euler 一阶/MUSCL f32 | ✅ | ✅ |
| f32 LTS + RK4/Euler | ✅（dt f64） | ✅（dt f32） |
| LU-SGS 对角 f32 | ✅ | ✅（σ/dt f32） |
| LU-SGS 扫掠 f32 | ❌ | ❌（S4） |
| 粘性 f32 | ❌ | ❌（S3） |
| 多块接口通量 f32 | ❌ | ❌（S2） |

## PR Checklist

- [ ] `make check`
- [ ] 无生产路径 `unwrap`
- [ ] f32 未实现能力 validate 报错
- [ ] 数值变更：f32 vs f64 测试 + CHANGELOG
- [ ] API/ADR 同步
