# 三维不可压缩 NS：SIMPLEC 与 PISO

> 模块：`src/discretization/incompressible.rs` · `src/case/incompressible_3d.rs` · 版本：v0.3+ · 状态：I0/I1 部分实现（ADR 0015）
> ADR：[adr/0015-incompressible-navier-stokes-simplec-piso.md](../adr/0015-incompressible-navier-stokes-simplec-piso.md)

## 1. 控制方程

常密度、等温、层流：

\[
\nabla\cdot\mathbf{u} = 0 \tag{1}
\]

\[
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\mathbf{u}) = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} \tag{2}
\]

### 1.1 不可压缩无量纲化

不可压缩求解内部使用显式参考量，而不是像可压缩流那样由声速和来流热力学状态自动构造：

\[
L_{\mathrm{ref}} = \text{用户给定特征长度}, \qquad
U_{\mathrm{ref}} = \text{用户给定特征速度}, \qquad
\rho_{\mathrm{ref}} = \rho
\tag{2a}
\]

\[
t_{\mathrm{ref}}=\frac{L_{\mathrm{ref}}}{U_{\mathrm{ref}}}, \qquad
p_{\mathrm{ref}}=\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2, \qquad
Re=\frac{U_{\mathrm{ref}}L_{\mathrm{ref}}}{\nu}
\tag{2b}
\]

变量缩放：

\[
\mathbf{x}^*=\frac{\mathbf{x}}{L_{\mathrm{ref}}},
\quad
\mathbf{u}^*=\frac{\mathbf{u}}{U_{\mathrm{ref}}},
\quad
t^*=\frac{t}{t_{\mathrm{ref}}},
\quad
p^*=\frac{p}{p_{\mathrm{ref}}},
\quad
\nu^*=\frac{1}{Re}
\tag{2c}
\]

代入 (1)(2) 得：

\[
\nabla^*\cdot\mathbf{u}^* = 0 \tag{2d}
\]

\[
\frac{\partial \mathbf{u}^*}{\partial t^*}
+\nabla^*\cdot(\mathbf{u}^*\mathbf{u}^*)
=-\nabla^*p^*+\frac{1}{Re}\nabla^{*2}\mathbf{u}^*
\tag{2e}
\]

因此不可压缩核心算子不再携带有量纲密度；密度仅用于 \(p_{\mathrm{ref}}\)、输出还原与有量纲诊断。Case 输入仍用 SI，解析后 `CaseSpec` 内部切换为星号量，CGNS 输出再还原 SI。

## 2. 离散布局

- **FVM**，结构化六面体，**collocated**：\(p,\mathbf{u}\) 存于单元中心。
- 面质量通量 \(\dot{m}_f = \rho\,\mathbf{u}_f\cdot\mathbf{S}_f\) 经 **Rhie-Chow** 计算，避免压力棋盘格。
- I1 基础算子限定为 Cartesian 均匀结构化网格；边界缺失邻居暂按零法向梯度 ghost 处理，后续 SIMPLEC/PISO 装配将改由显式边界通量控制。

### 2.1 连续性残差（I1）

I1 用 cell-centered 有限差分近似连续性残差：

\[
R_c(P)=
\frac{u_E-u_W}{2\Delta x}
+\frac{v_N-v_S}{2\Delta y}
+\frac{w_T-w_B}{2\Delta z}
\tag{1a}
\]

边界单元的缺失邻居取 \(\phi_g=\phi_P\)，等价于当前 skeleton 的零法向梯度 ghost。该残差仅用于建立 pressure-velocity coupling 前的数据流与诊断，不替代后续 Rhie-Chow 面质量通量。

## 3. 通量格式

### 3.1 Rhie-Chow 面速度（`rhie_chow.rs`）

面 \(f\) 介于 owner \(O\) 与 neighbor \(N\)（边界仅 owner）：

\[
\mathbf{u}_f = \overline{\mathbf{u}}_f - \overline{\mathbf{D}}_f\left(\overline{\nabla p}_f - \frac{p_N - p_O}{|\mathbf{x}_N-\mathbf{x}_O|}\,\mathbf{e}_{ON}\right) \tag{3}
\]

\[
\dot{m}_f = \rho\,\mathbf{u}_f\cdot\mathbf{S}_f \tag{4}
\]

Rhie-Chow **仅**用于 \(\dot{m}_f\) 与压力修正源项 \(\nabla\cdot(\rho\mathbf{u}^*)\)。动量方程中的压力梯度用 cell-centered 面差分，不经 (3) 修正。

### 3.2 对流通量（`convection.rs`）

动量分量 \(\phi\in\{u,v,w\}\)：

\[
\Phi_f^{\mathrm{conv}} = \dot{m}_f \phi_f, \qquad
\phi_f =
\begin{cases}
\phi_O, & \dot{m}_f \ge 0 \\
\phi_N, & \dot{m}_f < 0
\end{cases}
\tag{5}
\]

| `ConvectionScheme` | 面值 \(\phi_f\) | 默认 |
|--------------------|-----------------|------|
| `upwind` | (5) | **是** |
| `central` | \(\frac{1}{2}(\phi_O+\phi_N)\) | 低 Re 调试 |
| `minmod` | upwind + \(\frac{1}{2}\psi(\nabla\phi)\cdot\Delta\mathbf{x}\) | I6 |
| `quick` | 三阶 QUICK stencil | I6 |

### 3.3 扩散通量（`diffusion.rs`）

\[
\Phi_f^{\mathrm{visc}} = -\rho\nu (\nabla \phi)_f \cdot \mathbf{S}_f \tag{6}
\]

内面 \((\nabla\phi)_f\) 为中心差分；壁面用 ghost \(\phi_g\)（§6）。

I1 先提供速度分量 Laplacian skeleton：

\[
\nabla^2 \phi_P \approx
\frac{\phi_E-2\phi_P+\phi_W}{\Delta x^2}
+\frac{\phi_N-2\phi_P+\phi_S}{\Delta y^2}
+\frac{\phi_T-2\phi_P+\phi_B}{\Delta z^2},
\qquad \phi\in\{u,v,w\}
\tag{6a}
\]

边界缺失邻居同 §2.1 使用 \(\phi_g=\phi_P\)。实际动量方程扩散通量仍以 (6) 为准，后续会在边界面显式注入 wall/inlet/outlet 条件。

### 3.4 压力梯度（`momentum.rs`）

\[
(\partial p/\partial x)_P \approx (p_E - p_W)/\Delta x_P \tag{7}
\]

## 4. 动量方程离散

对速度分量 \(\phi \in \{u,v,w\}\)：

\[
a_P \phi_P = \sum a_{nb}\phi_{nb} + H(\phi) - (p_E - p_W)_\phi \tag{8}
\]

\(H\) 含对流、扩散显式部分（不含 \(a_P\phi_P\) 与压力梯度）。

**欠松弛（SIMPLEC）**：\(a_P \leftarrow a_P/\alpha_u\)，\(H \leftarrow H + (1-\alpha_u)a_P\phi_P/\alpha_u\)。

## 5. SIMPLEC 压力修正（`simplec.rs` + `pressure_correction.rs`）

### 5.1 预测速度

由 (8) 得 \(\mathbf{u}^*\)（压力梯度用 \(p^n\)）。

### 5.2 一致系数

\[
a_P^c = a_P - \sum a_{nb} \tag{9}
\]

\[
d_P = \frac{V_P}{a_P^c} \tag{10}
\]

### 5.3 压力修正方程

\[
\nabla\cdot(\rho\, d\,\nabla p') = \nabla\cdot(\rho\,\mathbf{u}^*) \tag{11}
\]

I1 skeleton 尚未引入动量方程一致系数 \(d_P\)，先取 \(d=1\) 并装配 SPD 符号形式：

\[
-\rho\nabla^2 p' = \rho R_c \tag{11a}
\]

其中 \(R_c\) 来自 (1a)。Cartesian 7 点 stencil 的内点系数为：

\[
a_P = 2\rho\left(\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}+\frac{1}{\Delta z^2}\right),
\quad
a_{nb}=-\frac{\rho}{\Delta n^2}
\tag{11b}
\]

纯 Neumann 压力校正矩阵奇异；I1 通过 `pressure_reference_cell` 将一行替换为 \(p'=p'_{\mathrm{ref}}\)，后续真实边界条件与参考压力策略会随 SIMPLEC/PISO 装配完善。

### 5.4 修正

\[
p \leftarrow p + \alpha_p p' \tag{12}
\]

\[
\mathbf{u} \leftarrow \mathbf{u}^* - d\,\nabla p' \tag{13}
\]

## 6. 边界条件

Ghost 单元距 owner 中心法向距离 \(d_f\)。

| BC | \(\mathbf{u}\) ghost | \(p\) | \(p'\) | \(\dot{m}_f\) |
|----|----------------------|-------|--------|---------------|
| 无滑移壁 | \(\mathbf{u}_g = -\mathbf{u}_o\) | \(\partial p/\partial n=0\) | Neumann | 0 |
| 动壁 \(U_w\) | \(\mathbf{u}_g = 2U_w - \mathbf{u}_o\) | Neumann | Neumann | \(\rho U_w\cdot\mathbf{S}\) |
| 速度入口 \(u_b\) | \(\mathbf{u}_g = 2u_b - u_o\) | Neumann | Neumann | upwind |
| 压力出口 \(p_b\) | \(\partial u/\partial n=0\) | \(p=p_b\) | \(p'=0\) | upwind owner |
| 对称 | \(u_n=0\), \(\partial u_t/\partial n=0\) | Neumann | Neumann | \(u_n=0\) |

**Dirichlet ghost**（(B1)）：\(\phi_g = 2\phi_b - \phi_o\)。

**Neumann ghost**（(B2)）：\(\phi_g = \phi_o + d_f (\partial\phi/\partial n)_b\)；零梯度时 \(\phi_g=\phi_o\)。

详细分工见 [boundary_conditions.md](boundary_conditions.md) §9。

## 7. PISO 与时间积分

### 7.1 瞬态 BDF1 + PISO

\[
\frac{\rho V_P}{\Delta t}(\mathbf{u}_P - \mathbf{u}_P^n) + \sum_f \dot{m}_f \mathbf{u}_f = -\nabla p + \mu\nabla^2\mathbf{u} \tag{14}
\]

单步：

1. 解 (14) 得 \(\mathbf{u}^*\)；
2. 重复 \(k=1,\ldots,N\)：解 (11) → (12)(13)，**无** \(\alpha_p\)；
3. \(t \leftarrow t + \Delta t\)。

### 7.2 时间步长

\[
\Delta t \le \mathrm{CFL}\,\frac{V_P^{1/3}}{|\mathbf{u}|_P}, \qquad
\Delta t \le \mathrm{CFL}_\nu\,\frac{(\Delta x_P)^2}{\nu} \tag{15}
\]

### 7.3 模式对照

| `time.mode` | 耦合 | 说明 |
|-------------|------|------|
| `steady` | SIMPLEC | 外层迭代至残差收敛 |
| `pseudo_transient` | SIMPLEC + 局部 \(\Delta t_P\) | 加速稳态；不推进物理时间 |
| `transient` | PISO + BDF1 | 默认 `n_piso_correctors = 2` |

### 7.4 残差监控

- 连续性：\(\|\nabla\cdot(\rho\mathbf{u}^*)\|_\infty / (\rho U_{\mathrm{ref}})\)
- 动量：\(\|\mathbf{R}_u\|_\infty / (\rho U_{\mathrm{ref}}^2)\)

## 8. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1a) 连续性残差 | `discretization::compute_incompressible_divergence_3d` | **I1 已实现** |
| (6a) 速度 Laplacian skeleton | `discretization::compute_incompressible_velocity_laplacian_3d` | **I1 已实现** |
| (11a)(11b) 压力校正 Poisson skeleton | `discretization::assemble_incompressible_pressure_poisson_3d` | **I1 已实现** |
| (2a)–(2e) 不可压缩无量纲化 | `io::nondimensional::apply_nondimensionalization_for_incompressible` | **I1 已实现** |
| I1 runner 诊断闭环 | `case/incompressible_3d.rs` | **已实现：初始化、\(max|div(u)|\)、压力校正 CSR 装配与 GMRES 求解、CGNS 输出；尚未修正 \(p,\mathbf{u}\)** |
| (3)(4) Rhie-Chow | `discretization/incompressible/rhie_chow.rs` | 规划 |
| (5)(6) 对流/扩散 | `convection.rs`, `diffusion.rs` | 规划 |
| (8) 动量装配 | `momentum.rs` | 规划 |
| (9)(10) SIMPLEC 系数 | `momentum.rs` | 规划 |
| (11) 压力 Poisson | `pressure_correction.rs` | 规划 |
| BC ghost | `discretization/incompressible/bc.rs` | 规划 |
| SIMPLEC 循环 | `solver/incompressible/simplec.rs` | 规划 |
| PISO + BDF1 | `solver/incompressible/piso.rs` | 规划 |
| CFL / pseudo-transient | `solver/time/pseudo_transient.rs` | 规划 |
| CG 求解 \(p'\) | `linalg` + `solver/incompressible/linear.rs` | 规划 |

## 9. 参考文献

1. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere. ISBN 978-0891165224. Ch. 6–7.
2. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. DOI [10.1007/978-3-319-55774-2](https://doi.org/10.1007/978-3-319-55774-2). Ch. 8–9.
3. Versteeg, H. K., & Malalasekera, W. (2007). *An Introduction to Computational Fluid Dynamics* (2nd ed.). Pearson. Ch. 6–8.
4. Issa, R. I. (1986). Solution of the implicitly discretised fluid flow equations by operator-splitting. *Journal of Computational Physics*, 62(1), 40–65. DOI [10.1016/0021-9991(86)90099-9](https://doi.org/10.1016/0021-9991(86)90099-9).
5. Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.

## 10. 相关算例

- `tests/benchmarks/poiseuille_3d/` — I1
- `tests/benchmarks/lid_cavity_re100/` — I2
- `tests/benchmarks/taylor_green_3d/` — I3

Case 输入：[CASE_FORMAT.md](../CASE_FORMAT.md)（实现期扩展 `[solver.incompressible]`）
