# 时间积分（显式 RK4 / LU-SGS）

> 模块：`src/solver/time/`、`src/solver/compressible.rs`、`src/solver/spectral_radius.rs` · 版本：v1.x · 状态：**已实现**

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
| `CompressibleEulerSolver` | 编排：CFL → 时间推进 → BC ghost 刷新 | `solver/compressible` |

```text
for each time step:
  dt_i ← Blazek local Δt (§4)
  U^{n+1} ← rk4 / lu_sgs (evaluate_rhs)
  t ← t + dt_min
```

---

## 3. 经典四阶 Runge-Kutta（RK4）

Butcher 表（显式）见原实现；每阶段重新评估 BC ghost 与残差。

---

## 4. 局部时间步（Blazek Ch. 9.1，RK4 / LU-SGS 统一）

Blazek, *Computational Fluid Dynamics: Principles and Applications*（3rd ed.），**§9.1 Local Time-Stepping** 与隐式 LU-SGS 对角近似共用同一局部时间步：

$$
\sigma_i = \frac{1}{V_i}\sum_{f\in\partial\Omega_i} \lambda_f\,A_f,
\qquad
\lambda_f \approx \tfrac{1}{2}\bigl(|u_n|+a\bigr)_L + \tfrac{1}{2}\bigl(|u_n|+a\bigr)_R,
\tag{2}
$$

$$
\Delta t_i = \mathrm{CFL}\,\frac{V_i}{\sigma_i}.
\tag{3}
$$

| 符号 | 含义 | 代码 |
|------|------|------|
| \(V_i\) | 单元体积 | `StructuredMesh3d::cell_volumes` |
| \(\lambda_f\) | 面法向谱半径 | `face_spectral_radius` |
| \(\sigma_i\) | 单元谱半径和 | `cell_spectral_radius_3d` |
| \(\Delta t_i\) | 局部时间步 | `cell_local_dt_cfl_3d` / `compute_cell_dts_3d` |
| CFL | Courant 数 | `[time].cfl` / `cfl_schedule` |

**说明**：

- 不再使用 \(\Delta t=\mathrm{CFL}\,h_{\min}/(|u|+a)\) 与 \(\sigma\) 两套公式；\(h_{\min}\) 仅作几何诊断（见 [curvilinear_metrics.md](curvilinear_metrics.md) §3.3），**不**用于 3D 可压缩时间步。
- 计算 \(\sigma\) 前须 `apply_compressible_boundary_conditions`（边界面 ghost 与 RHS 一致）。
- `local_time_step = false` 时，全场取 \(\Delta t=\min_i \Delta t_i\)（Blazek 全局下限）。
- 若 `[time].dt > 0`，固定时间步覆盖上式。

### 4.1 LU-SGS 对角 / 双扫

隐式伪时间更新（阶段 C）：

$$
\Delta\mathbf{U}_i
= \omega\,\frac{\Delta t_i\,\mathbf{R}_i}{1 + \Delta t_i\,\sigma_i^{\mathrm{sp}}},
\qquad
\sigma_i^{\mathrm{sp}}=\frac{|u|+a}{h_i}.
\tag{4}
$$

**LU-SGS 与 RK4 的时间步约定**（asimu 实现）：

| 格式 | \(\Delta t_i\) | \(\sigma_i\)（隐式分母） |
|------|----------------|--------------------------|
| RK4 | \(\mathrm{CFL}\,V_i/\sigma_i^{\mathrm{vol}}\) | 仅用于 \(\Delta t\)；式 (2) 体积谱半径 |
| LU-SGS | \(\mathrm{CFL}\,h_i/(|u|+a)_i\) | \(\sigma_i^{\mathrm{sp}}=(|u|+a)_i/h_i\) |

其中 \(h_i\) 为 `cell_cfl_lengths`（相邻面间距最小值），\(\sigma_i^{\mathrm{vol}}=\frac{1}{V_i}\sum_f\lambda_f A_f\)。
曲网格贴体单元上 \(\sigma^{\mathrm{vol}}\) 常使 \(V/\sigma\) 过小，LU-SGS 若仍用体积谱半径会导致残差几乎不下降；故 **仅 LU-SGS** 采用面间距 CFL（Blazek 经典 CFL 与 LU 对角尺度一致）。

**阶段 D（`lusgs_sweep = true`，默认）**：在式 (3)(4) 基础上增加 i/j/k 前扫与后扫；扫掠内界面一阶通量。`lu_sgs` 须 `local_time_step = true`。

---

## 5. 稳态 vs 瞬态

| 实现 | 模式 | 说明 |
|------|------|------|
| `SteadyStateIntegrator` | `TimeMode::Steady` | 伪时间步计数 |
| `RungeKutta4Integrator` | `TimeMode::Transient` | 物理时间 \(t\) |

---

## 6. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1) 残差装配 | `assemble_inviscid_residual_3d` | **已实现** |
| (2)(3) 局部 \(\Delta t\) | `cell_spectral_radius_3d`, `cell_local_dt_cfl_3d`, `compute_cell_dts_3d` | **已实现** |
| RK4 | `rk4_step` / `rk4_step_local` | **已实现** |
| (4) LU-SGS | `lu_sgs_sweep_3d`, `lu_sgs_step_local` | **已实现** |

---

## 7. 参考文献

1. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Elsevier. **§9.1** 局部时间步；**§6** 隐式 LU-SGS 谱半径近似。
2. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. Ch. 6、Ch. 11.
3. asimu ADR [0005](../adr/0005-time-integration.md)、[0009](../adr/0009-compressible-navier-stokes.md)（CFL 与 \(\sigma_i\) 约定）。

---

## 8. 相关算例

- `tests/benchmarks/sod_1d/` — RK4 + Roe
- `[time] scheme = "lu_sgs"` + `local_time_step = true` — 稳态圆柱等
