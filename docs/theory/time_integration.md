# 时间积分（显式 RK4 / LU-SGS / GMRES）

> 模块：`src/solver/time/`、`src/solver/compressible.rs`、`src/solver/spectral_radius.rs` · 版本：v1.x · 状态：**已实现**

## 1. 半离散形式

可压缩 Euler / Navier-Stokes 方程经空间 FVM 离散后，每个控制体 \(i\) 的守恒量 \(\mathbf{U}_i\) 满足常微分方程：

$$
\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t}
:= -\frac{1}{V_i}\sum_{f\in\partial\Omega_i} \hat{\mathbf{F}}_f \cdot \mathbf{S}_f
\equiv \mathbf{R}_i(\mathbf{U}) \tag{1}
$$

其中 \(V_i\) 为单元体积，\(\hat{\mathbf{F}}_f\) 为面数值通量（见 [inviscid_flux.md](inviscid_flux.md)），\(\mathbf{S}_f = A_f \mathbf{n}_f\) 为面积向量（owner → neighbor 方向）。

**asimu 约定**：`ConservedResidual` 存的是式 (1) 右端 \(\mathrm{d}\mathbf{U}/\mathrm{d}t\)，RK4 / LU-SGS 对其推进。

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
  U^{n+1} ← rk4 / lu_sgs / gmres (evaluate_rhs)
  t ← t + dt_min
```

---

## 3. 经典四阶 Runge-Kutta（RK4）

Butcher 表（显式）见原实现；每阶段重新评估 BC ghost 与残差。

---

## 4. 局部时间步（Blazek §6.1.4 / §9.1，RK4 / LU-SGS 统一）

Blazek, *Computational Fluid Dynamics: Principles and Applications*（3rd ed.）给出结构网格有限体积局部时间步：

$$
\Delta t_i
= \mathrm{CFL}\,\frac{V_i}{\Lambda_i^c + C_v\Lambda_i^v}.
\tag{2}
$$

无粘谱半径按控制体所有面求和：

$$
\Lambda_i^c = \sum_{f\in\partial\Omega_i} \lambda_f A_f,
\qquad
\lambda_f \approx \tfrac{1}{2}\bigl(|u_n|+a\bigr)_L + \tfrac{1}{2}\bigl(|u_n|+a\bigr)_R.
\tag{3}
$$

Navier-Stokes 计算还叠加粘性/热扩散的抛物型上界：

$$
\Lambda_i^v = \sum_{f\in\partial\Omega_i} d_i\,\frac{A_f^2}{V_i},
\qquad
d_i=\max(\nu_i,\alpha_i),\quad
\nu_i=\frac{\mu_i}{\rho_i},\quad
\alpha_i=\frac{\mu_i}{\rho_i Pr}.
\tag{4}
$$

代码内部使用归一化谱半径：

$$
\sigma_i = \frac{\Lambda_i^c + C_v\Lambda_i^v}{V_i},
\qquad
\Delta t_i = \frac{\mathrm{CFL}}{\sigma_i}.
\tag{5}
$$

| 符号 | 含义 | 代码 |
|------|------|------|
| \(V_i\) | 单元体积 | `StructuredMesh3d::cell_volumes` |
| \(\lambda_f\) | 面法向谱半径 | `face_spectral_radius` |
| \(d_i\) | 最大动量/热扩散率 | `cell_viscous_diffusivity_max` |
| \(C_v\) | 粘性谱半径系数（当前 3D 中心差分上界取 6） | `PARABOLIC_SPECTRAL_FACTOR_3D` |
| \(\sigma_i\) | 单元归一化谱半径 | `cell_spectral_radius_3d` |
| \(\Delta t_i\) | 局部时间步 | `cell_local_dt_cfl_3d` / `compute_cell_dts_3d` |
| CFL | Courant 数 | `[time].cfl` / `cfl_schedule` |

**说明**：

- 不再使用 \(\Delta t=\mathrm{CFL}\,h_{\min}/(|u|+a)\) 与 face-sum \(\sigma\) 两套公式；\(h_{\min}\) 仅作几何诊断（见 [curvilinear_metrics.md](curvilinear_metrics.md) §3.3），**不**用于 3D 可压缩时间步。
- 计算 \(\sigma\) 前须 `apply_compressible_boundary_conditions`（边界面 ghost 与 RHS 一致）。
- `local_time_step = false` 时，全场取 \(\Delta t=\min_i \Delta t_i\)（Blazek 全局下限）。
- 若 `[time].dt > 0`，固定时间步覆盖上式。

### 4.1 LU-SGS 对角 / 双扫

隐式伪时间更新（阶段 C）：

$$
\Delta\mathbf{U}_i
:= \omega\,\frac{\Delta t_i\,\mathbf{R}_i}{1 + \Delta t_i\,\sigma_i}.
\tag{6}
$$

**LU-SGS 与 RK4 的时间步约定**（asimu 实现）：

| 格式 | \(\Delta t_i\) | \(\sigma_i\)（隐式分母） |
|------|----------------|--------------------------|
| RK4 | \(\mathrm{CFL}/\sigma_i\) | 仅用于 \(\Delta t\)；式 (5) |
| LU-SGS | \(\mathrm{CFL}/\sigma_i\) | 式 (6) 的对角/扫掠分母 |

`σ_i` 由 `cell_spectral_radius_3d` 统一给出；RK4 与 LU-SGS 不再分别维护 `h_i/(|u|+a)` 与 face-sum 两套局部 CFL。`cell_cfl_lengths` 仅保留作网格尺度诊断。

**阶段 D（`lusgs_sweep = true`）**：在式 (5)(6) 基础上增加 i/j/k 前扫与后扫，用标量谱半径近似邻居耦合项。实现含逐单元正性限制、`lusgs_sweep_backward_damping` 后扫阻尼，以及相对 \(U_0\) 的全场线搜索（失败时回退对角隐式更新）。默认 `lusgs_sweep = false` 仅执行阶段 C；`lu_sgs` 须 `local_time_step = true`。

---

## 5. Matrix-Free GMRES 隐式伪时间

`time.scheme = "gmres"` 求解线性化伪时间系统：

$$
\left(D_{\Delta t}-J_R\right)\Delta U = R(U),
\qquad
D_{\Delta t,i}=\frac{1}{\Delta t_i}I.
\tag{7}
$$

其中 \(J_R v\) 不显式装配，而用有限差分 \(J_R v \approx [R(U+\epsilon v)-R(U)]/\epsilon\)。GMRES 左预条件器使用式 (6) 的 LU-SGS 对角近似。有限差分扰动会先按单元缩放到正密度、正压力可行范围；求得 \(\Delta U\) 后，`CompressibleEulerSolver` 对更新系数 \(\alpha\) 做 \(1,1/2,\ldots\) 回退线搜索，并在写回时逐单元限制增量，确保更新场可恢复正密度与正压力后再接受。

当前 GMRES 路径仅用于 3D 可压缩稳态伪时间，须设置 `local_time_step = true`。

---

## 6. 方向分裂隐式残差光顺

稳态伪时间推进可在每次更新前对残差做方向分裂隐式光顺：

$$
\left(I-\epsilon\delta_{\xi\xi}\right)
\left(I-\epsilon\delta_{\eta\eta}\right)
\left(I-\epsilon\delta_{\zeta\zeta}\right)
\bar{\mathbf{R}} = \mathbf{R}.
\tag{8}
$$

每个方向沿结构网格线解常系数三对角系统。以 \(i\) 方向为例，内部点满足：

$$
-\epsilon \bar{R}_{i-1}
+(1+2\epsilon)\bar{R}_i
-\epsilon \bar{R}_{i+1}
=R_i.
\tag{9}
$$

线端采用零梯度退化：

$$
(1+\epsilon)\bar{R}_0-\epsilon\bar{R}_1=R_0,
\qquad
-\epsilon\bar{R}_{N-2}+(1+\epsilon)\bar{R}_{N-1}=R_{N-1}.
\tag{10}
$$

实现按 i→j→k 顺序作用于 \(\rho,\rho u,\rho v,\rho w,\rho E\) 五个残差分量。由于分量分别光顺可能破坏动能与总能增量的一致性，`smooth_residual_3d_limited` 会按单元检查
\(\mathbf{U}+\alpha\Delta t\,\bar{\mathbf{R}}\) 的密度与内能正性；若光顺残差不可接受，则将该单元残差回退到未光顺残差，必要时继续向零更新回退。

该操作只改变伪时间收敛路径，不改变稳态控制方程；因此仅在 `mode = "steady"` 的 3D 可压缩推进中启用，真实瞬态计算忽略该配置。

TOML 配置：

```toml
[time]
residual_smoothing = true
residual_smoothing_epsilon = 0.5
residual_smoothing_sweeps = 1
```

---

## 7. 稳态 vs 瞬态

| 实现 | 模式 | 说明 |
|------|------|------|
| `SteadyStateIntegrator` | `TimeMode::Steady` | 伪时间步计数 |
| `RungeKutta4Integrator` | `TimeMode::Transient` | 物理时间 \(t\) |

---

## 8. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1) 残差装配 | `assemble_inviscid_residual_3d` | **已实现** |
| (2)–(5) 局部 \(\Delta t\) | `cell_spectral_radius_3d`, `cell_local_dt_cfl_3d`, `compute_cell_dts_3d` | **已实现** |
| RK4 | `rk4_step` / `rk4_step_local` | **已实现** |
| (6) LU-SGS | `lu_sgs_sweep_3d`, `lu_sgs_step_local` | **已实现** |
| (7) GMRES | `solve_gmres_implicit_delta_3d`, `advance_gmres_step_3d` | **已实现** |
| (8)–(10) 残差光顺 | `smooth_residual_3d` | **已实现** |

---

## 9. 参考文献

1. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Elsevier. **§6.1.4** 最大时间步与粘性谱半径；**§9.1** 局部时间步；**§6.2** 隐式 LU-SGS 谱半径近似。
2. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. Ch. 6、Ch. 11.
3. asimu ADR [0005](../adr/0005-time-integration.md)、[0009](../adr/0009-compressible-navier-stokes.md)（CFL 与 \(\sigma_i\) 约定）。

---

## 10. 相关算例

- `tests/benchmarks/sod_1d/` — RK4 + Roe
- `[time] scheme = "lu_sgs"` + `local_time_step = true` — 稳态圆柱等
