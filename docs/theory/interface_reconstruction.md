# 界面重构（Interface Reconstruction）

> 模块：`src/discretization/reconstruction.rs` · 版本：v1.x · 状态：**已实现（一阶分段常数）**

## 1. 动机

有限体积法在单元 \(i\) 上存储**分段常数**（或更高阶）的守恒量 \(\mathbf{U}_i\)。面 \(f\) 两侧的 Riemann 问题需要**界面左/右态** \(\mathbf{U}_L\)、\(\mathbf{U}_R\)，不能简单取相邻单元中心值而不区分 owner/neighbor 与 ghost。

**界面重构**从单元值（及边界 ghost）构造 \(\mathbf{U}_L\)、\(\mathbf{U}_R\)，再交给 Riemann 求解器（见 [inviscid_flux.md](inviscid_flux.md)）。

---

## 2. 一阶分段常数（Godunov / 最低阶）

对内部面 \(f\) 分隔 owner 单元 \(i\) 与 neighbor 单元 \(i+1\)：

$$
\mathbf{U}_L = \mathbf{U}_i, \qquad \mathbf{U}_R = \mathbf{U}_{i+1} \tag{1}
$$

对边界面 \(f\)：owner 为边界单元，neighbor 侧由 **ghost 单元** 给出 \(\mathbf{U}_{\mathrm{ghost}}\)：

$$
\mathbf{U}_L = \mathbf{U}_{\mathrm{owner}}, \qquad \mathbf{U}_R = \mathbf{U}_{\mathrm{ghost}} \tag{2}
$$

（法向指向域外时，ghost 在 owner 外侧；通量求解器用法向 \(\mathbf{n}\) 统一投影。）

**精度**：一阶 Roe 100 单元 L1(ρ) ≈ 0.02；MUSCL+van Albada+Roe ≈ 0.012（见 `tests/benchmarks/sod_1d/`）。

---

## 3. 法向与左右态约定

| 量 | asimu 约定 |
|----|------------|
| 面法向 \(\mathbf{n}\) | **owner → neighbor**，由网格给出 |
| `InterfaceStates::left` | 面 **owner 侧**（内侧）守恒态 |
| `InterfaceStates::right` | 面 **neighbor / ghost 侧**（外侧）守恒态 |
| 重构 | **与 \(\mathbf{n}\) 无关**；Roe 求解器负责法向投影 |

1D 内部面：`normal = (1,0,0)`，owner 在左；左边界 `normal = (-1,0,0)`，owner 仍为域内单元。

3D 结构化内部面：MUSCL 宽模板**沿面法向**取四点，切向 \((j,k)\) 等索引固定。例如 \(i\)-面 \((i,j,k)\mid(i+1,j,k)\) 使用 \((i-1,j,k), (i,j,k), (i+1,j,k), (i+2,j,k)\)；域内缺邻点时自动退化为单侧/一阶。

3D 边界面：owner 为边界单元，neighbor 为 ghost；宽模板沿法向向域内延伸，ghost 侧用 `right_of_neighbor` 取第二内层单元（如 `i_min` 面取 \((i,j,k)\) 的 \((1,j,k)\) 与 \((2,j,k)\)）。仅一层单元时退化为 owner 侧 MUSCL / 一阶。

---

## 4. 边界 ghost 与重构衔接

| 场景 | ghost 来源 | 代码 |
|------|------------|------|
| 1D 零梯度 | \(\mathbf{U}_{\mathrm{ghost}} = \mathbf{U}_{\mathrm{owner}}\) | `zero_gradient_ghosts_1d` |
| 1D 固定 ghost | 用户给定 | `InviscidBoundary1d::Fixed` |
| 3D 可压缩 BC | Farfield / Wall / Inlet / … | `apply_compressible_boundary_conditions` |

装配流程（每个 RK 阶段）：

```text
ghost ← BC(owner 态, patch 类型)
(U_L, U_R) ← reconstruct_first_order(owner, ghost_or_neighbor)
F̂ ← roe_flux(U_L, U_R, n)
```

---

## 5. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1) 一阶重构 | `reconstruct_first_order` | **已实现** |
| MUSCL + minmod / van Leer / van Albada | `reconstruct_muscl_1d` | **已实现**（1D/3D 内部面 + 3D 边界面，沿法向宽模板） |
| 面通量入口 | `face_inviscid_flux`（重构 + Roe/HLLC dispatch） | **已实现** |
| 1D 内部面 | `assemble_interior_faces_1d`（四点 MUSCL 模板） | **已实现** |
| 3D 内部面 | `muscl_stencil_3d`（i/j/k 法向四点模板） | **已实现** |
| 3D 边界面 | `flux_at_boundary_face`（ghost + 域内宽模板） | **已实现** |
| 1D 边界面 | 一阶 owner/ghost | **已实现** |

---

## 6. 高阶扩展

| 格式 | 界面值 | 状态 |
|------|--------|------|
| 一阶 PC | 式 (1) | **已实现** |
| MUSCL + minmod / van Leer / van Albada | 线性外推 + 限制 | **已实现**（1D/3D 内部面 + 3D 边界面） |
| WENO | 高阶多项式 | 远期 |

配置：`InviscidFluxConfig { reconstruction, limiter, scheme }`；预设 `muscl_hllc()`。

**一阶**：`reconstruction = first_order` 时界面取 owner/neighbor 单元值，不调用 MUSCL 限制器（`limiter` 字段忽略，元数据记为 `none`）；单调性由 Godunov 型 Riemann 通量保证。

新增格式时：扩展 `reconstruction.rs`，保持 `InterfaceStates` 接口不变，由 `face_inviscid_flux` 或策略枚举 dispatch。

---

## 7. 参考文献

1. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge. Ch. 4（重构与 Riemann 问题）、Ch. 6（高阶扩展）。
2. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer. Ch. 5–6（Godunov / 重构与通量）。
3. asimu ADR [0009](../adr/0009-compressible-navier-stokes.md) — MUSCL / 限制器路线。

---

## 8. 相关算例

- `tests/benchmarks/sod_1d/` — 一阶 Roe + RK4 vs 精确解
- `discretization::reconstruction::tests::first_order_passes_cell_values_unchanged`
