# 一维稳态扩散（FVM）

> 模块：`src/discretization/` · 版本：v0.2 · 状态：骨架（装配待实现）

## 1. 控制方程

一维稳态标量扩散：

$$
-\frac{d}{dx}\left(D \frac{d\phi}{dx}\right) = 0 \tag{1}
$$

其中 \(D\) 为分子扩散系数，\(\phi\) 为标量场。

## 2. 离散化

- **网格**：1D 结构化均匀网格，单元数 \(N\)，域长 \(L\)，\(\Delta x = L/N\)
- **方法**：有限体积法（FVM），界面通量用中心差分
- **守恒性**：界面通量连续，整体守恒

对单元 \(i\)，积分后：

$$
F_{i+1/2} - F_{i-1/2} = 0 \tag{2}
$$

扩散通量（式 2 中 \(F = -D \partial\phi/\partial x\)）：

$$
F_{i+1/2} \approx -D \frac{\phi_{i+1} - \phi_i}{\Delta x} \tag{3}
$$

## 3. 边界条件

| BC 类型 | 数学条件 | 离散处理 | 代码入口（规划） |
|---------|----------|----------|------------------|
| Dirichlet | \(\phi = \phi_b\) | 边界单元中心值固定 | `apply_dirichlet` |
| Neumann | \(-D \partial\phi/\partial n = q\) | 幽灵单元或通量修正 | `apply_neumann` |

**顺序**：先装配内部面通量，再按 patch 应用 BC（见 ARCHITECTURE §8.5.3）。

## 4. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1)–(3) 装配 | `discretization::assemble_diffusion` | 规划 |
| 占位校验 | `discretization::assemble_diffusion_placeholder` | **已实现** |
| BC 应用 | `discretization::apply_boundary_*` | 规划 |
| 线性求解 | `linalg::solve_cg` | 规划 |

## 5. 参考文献

1. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere. ISBN 978-0891165224. Ch. 5.
2. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. DOI [10.1007/978-3-319-55774-2](https://doi.org/10.1007/978-3-319-55774-2). Ch. 8.

## 6. 相关算例

- `tests/benchmarks/1d_diffusion_analytical/` — 均匀网格 L2 误差 vs 解析解 \(\phi(x) = x/L\)

Case 输入：[CASE_FORMAT.md](../CASE_FORMAT.md)

**非结构 3D 热传导（无量纲 FVM）** 见 [heat_conduction_fvm.md](heat_conduction_fvm.md)；本页保留 1D 有量纲骨架说明。
