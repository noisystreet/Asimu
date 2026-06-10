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
a_P \phi_P = \sum a_{nb}\phi_{nb} + H(\phi) - (p_E - p_W)_\phi + V_P f_\phi \tag{8}
\]

I1 结构化实现中，扩散项和一阶迎风对流项先进入左端矩阵；压力梯度、每单位质量体力 \(\mathbf{f}\) 与边界源项进入 RHS；后续会把更多格式与完整 \(H(\phi)\) 拆分补齐。

**欠松弛（SIMPLEC）**：\(a_P \leftarrow a_P/\alpha_u\)，\(H \leftarrow H + (1-\alpha_u)a_P\phi_P/\alpha_u\)。

## 5. SIMPLEC 压力修正（`simplec.rs` + `pressure_correction.rs`）

### 5.1 预测速度

由 (8) 得 \(\mathbf{u}^*\)（压力梯度用 \(p^n\)）。

动量预测使用伪瞬态格式，用于打通矩阵/RHS/一致系数的数据通路，包含内部扩散、一阶迎风对流、动量边界面贡献、压力梯度与速度欠松弛：

\[
\frac{V_P}{\Delta \tau}\phi_P^* + \sum_f F_f\phi_f^{up} - \nu \nabla^2 \phi_P^*
= \frac{V_P}{\Delta \tau}\phi_P^n - V_P(\nabla p^n)_\phi + V_P f_\phi,
\qquad \phi\in\{u,v,w\}
\tag{8a}
\]

Cartesian 结构网格上，内点扩散系数为：

\[
a_E=a_W=\nu\frac{\Delta y\Delta z}{\Delta x},\quad
a_N=a_S=\nu\frac{\Delta x\Delta z}{\Delta y},\quad
a_T=a_B=\nu\frac{\Delta x\Delta y}{\Delta z},
\tag{8b}
\]

面通量使用 cell-centered 速度线性插值得到 \(F_f=(\mathbf{u}_f\cdot\mathbf{n}_f)A_f\)，\(\phi_f^{up}\) 取一阶迎风。

\[
a_P=\frac{V_P}{\Delta\tau}+\sum a_{nb}^{diff}+\sum_f \max(F_f,0),\qquad
rhs_\phi=\frac{V_P}{\Delta\tau}\phi_P^n - V_P(\nabla p^n)_\phi + V_P f_\phi.
\tag{8c}
\]

`[incompressible].body_force` 输入 SI 加速度，解析后使用 \(\mathbf{f}^*=\mathbf{f}L_{\mathrm{ref}}/U_{\mathrm{ref}}^2\) 进入 (8c)。欠松弛按 (8) 后的规则修改对角与 RHS。动量边界面贡献来自 `BoundarySet`：速度入口、无滑移/动壁按 Dirichlet 速度加入扩散源项与入流迎风源项；压力出口按零梯度速度使用 owner 外推；对称/滑移壁去除法向通量；结构化 `i_min/i_max` 成对 `periodic` 作为内部 wrap 邻接进入动量矩阵与压力梯度。

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

I5 起，压力校正 RHS 使用 `compute_incompressible_rhie_chow_divergence_3d`
从面通量计算连续性残差。内部面质量通量为
\(\dot{m}_f=\rho A_f(\mathbf{u}_f\cdot\mathbf{n}_f-d_f(p_N-p_P)/\Delta n_f)\)，
其中 \(d_f=(d_P+d_N)/2\)。边界面通量由不可压缩边界条件给定：壁面/对称面法向通量为零，速度入口使用给定速度，压力出口使用 owner 速度零梯度外推；结构化 `i_min/i_max` 成对周期边界通过 wrap 面通量进入 Rhie-Chow 连续性残差。

I3 压力校正矩阵使用动量预测矩阵提供的 cell-centered \(d_P\)，内部面取
\(d_f=(d_P+d_N)/2\)。压力出口 owner 行施加 \(p'=0\)；当前 owner-cell
边界 skeleton 还会把无滑移壁、动壁和速度入口 owner 行作为 \(p'=0\) 约束，
避免压力校正与下一轮边界速度重施加互相抵消。若没有上述约束，则用
`pressure_reference_cell` 固定参考压力：

\[
-\nabla\cdot(\rho d_f\nabla p') = \rho R_c \tag{11a}
\]

其中 \(R_c\) 来自 (1a)。Cartesian 7 点 stencil 的内点系数为：

\[
a_P = \sum_f \rho\frac{d_f}{\Delta n_f^2},
\quad
a_{nb}=-\rho\frac{d_f}{\Delta n_f^2}
\tag{11b}
\]

纯 Neumann 压力校正矩阵奇异；无压力出口或速度约束 owner 行时，通过
`pressure_reference_cell` 将一行替换为 \(p'=p'_{\mathrm{ref}}\)，并在闭域 RHS
上移除非参考行均值以满足兼容性条件。

### 5.4 修正

\[
p \leftarrow p + \alpha_p p' \tag{12}
\]

`[incompressible].pressure_under_relaxation` 给出 \(\alpha_p\in(0,1]\)，默认 1。

\[
\mathbf{u} \leftarrow \mathbf{u}^* - \alpha_p d\,\nabla p' \tag{13}
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

当前实现分两层：`apply_incompressible_boundary_conditions_3d` 先把 `wall`、`velocity_inlet`、`pressure_outlet`、`symmetry` 施加到结构化边界 owner 单元并输出统计；`moving_wall` 在 owner-cell 层只施加无穿透约束，避免把壁面切向速度误当作 cell-centered 出流。SIMPLEC 每次动量预测与 \(p,\mathbf{u}\) 修正后会再次施加这些 owner-cell 约束，确保壁面/动壁法向速度不随压力校正漂移。`assemble_incompressible_momentum_predictor_with_boundary_3d` 再把速度 Dirichlet、动壁切向驱动、压力出口零梯度与对称/滑移法向约束转化为动量预测矩阵/RHS 的边界面贡献。`i_min/i_max` 成对 `periodic` 不改 owner 单元值，而是在动量、Rhie-Chow、压力校正和速度修正压力梯度中使用周期 wrap 邻接。

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

`solver::run_incompressible_simplec` 已提供 SIMPLEC 外层循环：`time.max_steps` 作为最大外层迭代数，
`time.tolerance` 为可选收敛阈值；每轮执行动量预测、压力校正、\(p,\mathbf{u}\)
修正，并把按 \(\alpha_p\) 缩放后的压力校正连续性残差
\(\max|b_p-\alpha_p A_p p'|\) 与 \(\max|A_u u^*-rhs_u|\) 写入残差历史。
预测残差仍来自 Rhie-Chow 面通量；`max_abs_corrected_divergence` 保留全量压力
校正方程线性残差 \(\max|b_p-A_p p'|\)，用于判断线性系统是否解好。设置
`time.tolerance` 时，欠松弛后的连续性残差、动量残差与
\(\max|\Delta\mathbf{u}|\) 速度更新量须同时满足阈值才标记收敛；未设置时仅执行固定
`max_steps`，`simplec_converged=false` 表示没有收敛判据。若残差或速度更新量出现非有限值，或任一监控量超过发散保护上限，runner 立即返回求解器错误；输出字段使用最后一次重施加边界后的修正场。

为排查封闭腔体收敛，runner 额外记录多类诊断：`max_abs_corrected_divergence`
表示全量压力校正方程自身的质量残差；`max_abs_underrelaxed_corrected_divergence`
表示实际欠松弛速度修正后仍剩余的压力校正连续性残差；`max_abs_corrected_field_divergence_before_boundary`
与 `max_abs_corrected_field_divergence_after_boundary` 则分别重新计算 \(p,\mathbf{u}\)
修正后、边界重施加前后的 cell-centered \(\nabla\cdot\mathbf{u}\)。这些指标不应混用：
全量压力方程残差用于判断线性系统是否解好，欠松弛残差用于 SIMPLEC 收敛，cell-centered 散度用于判断边界重施加和速度修正是否仍破坏真实速度场连续性。
`pressure_correction_rhs_active_sum` 记录跳过 \(p'=0\) identity 约束行后的 RHS 总和，用于检查闭域兼容性。
`max_abs_corrected_velocity_delta_interior` 与 `max_abs_corrected_velocity_delta_boundary`
把总速度更新量拆成非速度约束 owner 和速度约束边界 owner 两类，用于判断收敛受内部场演化还是边界 owner 重施加主导；SIMPLEC 收敛判据仍使用总速度更新量。

`[incompressible.linear.momentum]` 与 `[incompressible.linear.pressure]` 分别控制动量预测和压力校正线性求解的 GMRES `restart`、`max_iters` 与 `tolerance`。压力校正默认使用 `restart=64`、`max_iters=500`、`tolerance=1.0e-10`，避免小型 Poisson-like 系统被过早截断；当前首版仍使用 Identity 预条件器，后续会切换到更适合 Poisson 系统的 CG/ILU(0) 或 AMG 路径。

## 8. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (1a) 连续性残差 | `discretization::compute_incompressible_divergence_3d` | **I1 已实现** |
| (6a) 速度 Laplacian skeleton | `discretization::compute_incompressible_velocity_laplacian_3d` | **I1 已实现** |
| (8a)–(8c) 动量预测 | `discretization::assemble_incompressible_momentum_predictor_with_boundary_3d` | **已实现：内部扩散/迎风对流、边界面贡献、周期 x wrap、三分量 RHS** |
| (9)(10) SIMPLEC 系数 | `discretization::assemble_incompressible_momentum_predictor_with_boundary_3d` | **已实现：由动量矩阵一致系数计算 \(d_P\)** |
| (11a)(11b) 压力校正 Poisson skeleton | `discretization::assemble_incompressible_pressure_poisson_3d` | **I1 已实现：RHS 来自预测速度 \(u^*\) 的散度** |
| (2a)–(2e) 不可压缩无量纲化 | `io::nondimensional::apply_nondimensionalization_for_incompressible` | **I1 已实现** |
| I1 runner 诊断闭环 | `case/incompressible_3d.rs`, `solver::run_incompressible_simplec` | **已实现：case 负责输入/输出，solver 负责编排 SIMPLEC 迭代与收敛历史** |
| (3)(4) Rhie-Chow | `discretization::compute_incompressible_rhie_chow_divergence_3d` | **已实现：内部面压力-速度耦合通量、周期 x wrap 与边界面通量** |
| (5)(6) 对流/扩散 | `discretization::assemble_incompressible_momentum_predictor_with_boundary_3d` | **已实现：一阶迎风对流、中心扩散与边界贡献** |
| (8) 完整动量装配 | `discretization::assemble_incompressible_momentum_predictor_with_boundary_3d` | **部分实现：结构化 Cartesian 首版** |
| (9)(10) 完整 SIMPLEC 系数 | `discretization::assemble_incompressible_momentum_predictor_with_boundary_3d` | **部分实现：\(d_P\) 已导出，\(a_P/a_P^c/H(u)\) 仍待显式 API 化** |
| (11) 压力 Poisson | `discretization::assemble_incompressible_pressure_correction_3d` | **已实现：面插值 \(d_P\)、压力出口 \(p'=0\)、参考压力策略** |
| BC ghost | `discretization::apply_incompressible_boundary_conditions_3d` | **部分实现：cell-centered owner 应用与面通量贡献；ghost refresh 待补** |
| SIMPLEC 循环 | `solver::run_incompressible_simplec` | **已实现：外层迭代、连续性/动量收敛判据与最终修正场** |
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
