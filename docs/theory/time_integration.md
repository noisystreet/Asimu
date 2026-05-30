# 时间积分（显式 RK4）

> 模块：`src/solver/time/`、`src/solver/compressible.rs` · 版本：v1.x · 状态：**已实现（RK4 + CFL）**

## 1. 半离散形式

可压缩 Euler 方程经空间 FVM 离散后，每个控制体 \(i\) 的守恒量 \(\mathbf{U}_i\) 满足常微分方程：

$$
\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t}
= -\frac{1}{V_i}\sum_{f\in\partial\Omega_i} \hat{\mathbf{F}}_f \cdot \mathbf{S}_f
\equiv \mathbf{R}_i(\mathbf{U}) \tag{1}
$$

其中 \(V_i\) 为单元体积，\(\hat{\mathbf{F}}_f\) 为面数值通量（见 [inviscid_flux.md](inviscid_flux.md)），\(\mathbf{S}_f = A_f \mathbf{n}_f\) 为面积向量（owner → neighbor 方向）。

**asimu 约定**：`ConservedResidual` 存的是式 (1) 右端 \(\mathrm{d}\mathbf{U}/\mathrm{d}t\)，RK4 对其积分。

---

## 2. 架构分层（ADR 0005）

| 组件 | 职责 | 模块 |
|------|------|------|
| `TimeIntegrator` | 时间步计数、物理时间 \(t\)、稳态/瞬态模式 | `solver/time` |
| `discretization` | 空间残差 \(\mathbf{R}(\mathbf{U})\) | `assemble_inviscid_residual_*` |
| `CompressibleEulerSolver` | 编排：CFL → RK4 阶段 → BC ghost 刷新 | `solver/compressible` |

```text
for each time step:
  dt ← suggest_dt_cfl(Δx, max|u|+a, CFL)
  U^{n+1} ← rk4_step(U^n, dt, evaluate_rhs)
  evaluate_rhs: BC ghost → assemble_inviscid_residual → R
  t ← t + dt
```

---

## 3. 经典四阶 Runge-Kutta（RK4）

Butcher 表（显式）：

| 阶段 | \(c\) | \(a\) |
|------|-------|-------|
| \(k_1\) | 0 | — |
| \(k_2\) | 1/2 | 1/2 |
| \(k_3\) | 1/2 | 1/2 |
| \(k_4\) | 1 | 1 |

$$
\begin{aligned}
\mathbf{U}^{(1)} &= \mathbf{U}^n \\
k_1 &= \mathbf{R}(\mathbf{U}^{(1)}) \\
\mathbf{U}^{(2)} &= \mathbf{U}^n + \tfrac{\Delta t}{2}\, k_1 \\
k_2 &= \mathbf{R}(\mathbf{U}^{(2)}) \\
\mathbf{U}^{(3)} &= \mathbf{U}^n + \tfrac{\Delta t}{2}\, k_2 \\
k_3 &= \mathbf{R}(\mathbf{U}^{(3)}) \\
\mathbf{U}^{(4)} &= \mathbf{U}^n + \Delta t\, k_3 \\
k_4 &= \mathbf{R}(\mathbf{U}^{(4)}) \\
\mathbf{U}^{n+1} &= \mathbf{U}^n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
\tag{2}
$$

**稳定性**：RK4 稳定域有限；可压缩无粘问题需 CFL 限制（§4）。每阶段重新评估 BC ghost 与残差，保证边界随阶段态更新。

---

## 4. CFL 时间步

一维/结构化网格上，建议时间步：

$$
\Delta t = \mathrm{CFL}\,\frac{\Delta x_{\min}}{(\max_i |u_i| + a_i)} \tag{3}
$$

| 符号 | 含义 | 代码 |
|------|------|------|
| \(\Delta x_{\min}\) | 最小单元尺度 | `mesh.dx()`（1D）或 `min(dx,dy,dz)`（3D） |
| \(|u|+a\) | 最大特征波速 | `max_wave_speed` |
| CFL | Courant 数 | `CompressibleEulerConfig::cfl`（默认 0.4） |

若 `RungeKutta4Config::dt > 0`，则使用固定 \(\Delta t\)（用于单测与 benchmark 复现）。

---

## 5. 稳态 vs 瞬态

| 实现 | 模式 | 说明 |
|------|------|------|
| `SteadyStateIntegrator` | `TimeMode::Steady` | 递增 `pseudo_step`，\(t=0\)，用于稳态扩散占位 |
| `RungeKutta4Integrator` | `TimeMode::Transient` | 递增 `time_step` 与 `physical_time` |

瞬态求解器状态见 `SolverState`：`physical_time`、`time_step`、`dt`。

---

## 6. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1) 残差装配 | `assemble_inviscid_residual_1d` / `_3d` | **已实现** |
| (2) RK4 阶段 | `rk4_step` | **已实现** |
| (2) 阶段场更新 | `ConservedFields::assign_axpy` | **已实现** |
| (2) 斜率组合 | `ConservedResidual::assign_rk4_increment` | **已实现** |
| (3) CFL | `suggested_dt_cfl`、`CompressibleEulerSolver::suggest_dt_*` | **已实现** |
| 多步编排 | `CompressibleEulerSolver::advance_step_1d` | **已实现** |
| Sod 至 \(t=t_f\) | `run_sod_benchmark` | **已实现** |

**RK 阶段与 BC**：`advance_step_1d` 在 `rk4_step` 的闭包内每阶段调用 `InviscidBoundary1d::resolve`（零梯度 ghost）再装配残差。

---

## 7. 参考文献

1. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. DOI [10.1007/978-3-319-55774-2](https://doi.org/10.1007/978-3-319-55774-2). Ch. 6（时间推进）、Ch. 11（可压缩流 CFL）。
2. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge. ISBN 978-0521009249. Ch. 8（显式 RK）。
3. asimu ADR [0005](../adr/0005-time-integration.md) — 时间推进抽象与配置。

---

## 8. 相关算例

- `tests/benchmarks/sod_1d/` — RK4 + Roe，\(t=0.2\) 密度 L1/L2 vs 精确 Riemann 解
- `solver/time/rk4::tests::rk4_integrates_linear_decay` — 标量线性衰减解析验证
- `solver/compressible::tests::uniform_1d_field_remains_stationary_over_steps` — 均匀场不变性
