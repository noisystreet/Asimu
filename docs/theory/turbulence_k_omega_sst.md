# Menter k-ω SST 湍流闭包（可压 RANS）

> 模块：`src/physics/turbulence/` · `src/discretization/turbulence/` · 版本：v1.x · 状态：**规划（ADR 0014）**
> Case：`[physics.turbulence]` · BC：`turbulent_inlet` · 架构：[ADR 0014](../adr/0014-turbulence-k-omega-sst-rans.md)

## 1. 控制方程

可压 RANS 沿用 [ADR 0009](../adr/0009-compressible-navier-stokes.md) 守恒变量 \(\mathbf U=[\rho,\rho\mathbf u,\rho E]^T\)。湍流通过 **Boussinesq 涡粘** 进入 Newtonian 粘性应力（与 [nondimensional.md](nondimensional.md) 中 \(\mu^*=1/\mathrm{Re}\) 缩放兼容）：

\[
\mu_{\mathrm{eff}} = \mu_{\mathrm{lam}}(T) + \mu_t
\tag{1}
\]

**Menter k-ω SST**（2003 可压 RANS 常用形式）两个输运量 \(k\)（湍动能）、\(\omega\)（比耗散率）：

\[
\frac{\partial (\rho k)}{\partial t} + \nabla\cdot(\rho k \mathbf u)
= \nabla\cdot\left[\left(\mu + \sigma_k \mu_t\right)\nabla k\right] + P_k - \beta^* \rho k \omega
\tag{2}
\]

\[
\frac{\partial (\rho \omega)}{\partial t} + \nabla\cdot(\rho \omega \mathbf u)
= \nabla\cdot\left[\left(\mu + \sigma_\omega \mu_t\right)\nabla \omega\right]
+ \frac{\omega}{k}\left(\gamma P_k - \beta \rho k \omega\right) + D_\omega
\tag{3}
\]

其中 \(D_\omega\) 为 **SST 交叉扩散项**（仅 \(F_1<1\) 区域显著）：

\[
D_\omega = (1-F_1)\, 2\rho\,\frac{\sigma_{\omega 2}}{\omega}\,\nabla k\cdot\nabla\omega
\tag{4}
\]

**涡粘**（应变率模 \(S\)）：

\[
\mu_t = \rho\,\frac{a_1 k}{\max(a_1 \omega,\, S F_2)},\qquad
S = \sqrt{2 S_{ij}S_{ij}}
\tag{5}
\]

\[
S_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}+\frac{\partial u_j}{\partial x_i}\right)
\tag{6}
\]

**混合函数**（依赖壁面距离 \(y\)，见 §3）：

\[
F_1 = \tanh(\arg_1^4),\quad
\arg_1 = \min\left(\max\left(\frac{\sqrt{k}}{0.09\,\omega y},\, \frac{500\mu}{\rho y^2 \omega}\right),\, \frac{4\rho\sigma_{\omega 2} k}{\rho y^2 \omega}\right)
\tag{7}
\]

\[
F_2 = \tanh(\arg_2^2),\quad
\arg_2 = \max\left(\frac{2\sqrt{k}}{0.09\,\omega y},\, \frac{500\mu}{\rho y^2 \omega}\right)
\tag{8}
\]

**混合常数**（\(F_1=1\) → 内层 k-ω；\(F_1=0\) → 外层 k-ε 等价）：

\[
\phi = F_1 \phi_1 + (1-F_1)\phi_2,\quad \phi\in\{\sigma_k,\sigma_\omega,\beta,\gamma\}
\tag{9}
\]

\[
\gamma = \min\left(\frac{\rho S F_2}{a_1\omega}\,\max\left(\frac{a_1\omega}{S F_2},\, \frac{1}{\max(S F_2,10^{-10})}\right),\, 2.0\right)
\tag{10}
\]

**湍流生成**（与 \(P_k\) 一致）：

\[
P_k = \min(\tau_{ij}\,\frac{\partial u_i}{\partial x_j},\, 10\beta^* \rho k \omega)
\tag{11}
\]

\[
\tau_{ij} = 2\mu_t S_{ij} - \frac{2}{3}\rho k \delta_{ij}
\tag{12}
\]

## 2. 默认常数（Menter 2003）

| 符号 | 值 | 备注 |
|------|-----|------|
| \(a_1\) | 0.31 | 涡粘 |
| \(\beta^*\) | 0.09 | \(k\) 销毁 |
| \(\beta_1\) | 0.075 | 内层 \(\omega\) |
| \(\beta_2\) | 0.0828 | 外层 \(\omega\) |
| \(\sigma_{k1}\) | 0.85 | |
| \(\sigma_{k2}\) | 1.0 | |
| \(\sigma_{\omega 1}\) | 0.5 | |
| \(\sigma_{\omega 2}\) | 0.856 | |
| \(\kappa\) | 0.41 | 壁面 \(\omega\)（Wilcox） |

Case 可覆盖上表；变更须同步 [CHANGELOG.md](../../CHANGELOG.md) 与 benchmark 参考值。

## 3. 离散化（FVM）

- **网格**：与 NS 相同（结构 / 非结构混合单元）；\(k,\omega\) **单元中心** 存储（`TurbulenceFields` SoA）。
- **对流**：与标量输运一致，首版 **一阶 upwind**（面心 \(\mathbf u\cdot\mathbf n\)）；二阶延后。
- **扩散**：中心差分 + 现有梯度框架（结构：有限差分；非结构：IDWLS，见 [unstructured_fvm.md](unstructured_fvm.md)）。
- **源项**：\(P_k\)、\(\omega\) 销毁、交叉扩散 (4) 在单元中心显式求值；销毁项 **LU-SGS 对角隐式**（见 [ADR 0014](../adr/0014-turbulence-k-omega-sst-rans.md) §5）。
- **正性**：\(k,\omega \ge\) `k_floor` / `omega_floor` 单元 clip；clip 次数写入 tracing（非唯一状态）。
- **壁距 \(y\)**：`WallDistanceField`，求解前由壁面 patch 预计算（结构化法向 / 非结构 BFS）。

**退化验证**（T2）：全域 \(F_1=1\)、\(D_\omega=0\) → 纯 k-ω 极限，用于平板边界层 golden。

## 4. 边界条件

| BC | \(k\) | \(\omega\) | \(\mathbf U,p,T\) | 代码（规划） |
|----|-------|------------|-------------------|--------------|
| **Wall**（无滑移） | \(0\) | \(\displaystyle \frac{6\nu}{\beta_1 y^2}\)（首版解析壁面，非壁函数） | 现有粘性壁 | `apply_turbulence_wall` |
| **turbulent_inlet** | 指定 | 指定 | `inlet_ghost`（已有） | `apply_turbulent_inlet` |
| **Farfield** | \(k_\infty\) 或零梯度 | \(\omega_\infty\) 或零梯度 | 现有 farfield | `apply_turbulence_farfield` |
| **Outlet** | 零梯度 | 零梯度 | 现有 outlet | `apply_turbulence_outlet` |
| **Symmetry** | Neumann | Neumann | 现有 symmetry | `apply_turbulence_symmetry` |

**远场初值**（缺省时）：

\[
k = \frac{3}{2}(U_{\mathrm{ref}} I)^2,\quad
\omega = \frac{\sqrt{k}}{C_\mu^{1/4} L_{\mathrm{ref}}},\quad C_\mu=0.09
\tag{13}
\]

\(I\) 为湍流强度；\(L_{\mathrm{ref}}\) 取 [nondimensional.md](nondimensional.md) 长度参考或 case 参数。

## 5. 与 NS / LU-SGS 耦合

单 LU-SGS 伪时间步（分裂扫，顺序固定以便回归）：

```text
1. U^n → primitive, ∇u, ∇T
2. k^n, ω^n → y, F1, F2, μ_t, P_k, sources
3. assemble NS residual（μ_eff）
4. assemble k, ω transport residual
5. LU-SGS sweep: NS → k → ω（各用局部 Δt_i）
6. clip k, ω; 记录收敛
```

谱半径：NS 现有 \(\sigma_i\) + \(k,\omega\) 对流 \(|\mathbf u|\) 与扩散 \(\nu_k=(\mu+\sigma_k\mu_t)/\rho\)、\(\nu_\omega\) 贡献（实现：`cell_spectral_radius_*` 扩展）。

## 6. 实现映射

| 式 / 步骤 | 代码位置 | 阶段 |
|-----------|----------|:----:|
| (1) \(\mu_{\mathrm{eff}}\) | `physics::viscosity::ViscousPhysicsConfig::face_transport_coefficients` | T1+ |
| (5)–(12) 闭包 | `physics::turbulence::MenterKOmegaSst` | T2+ |
| (7)(8) \(F_1,F_2\) | `physics::turbulence::sst_blending` | T3 |
| \(y\) | `mesh::WallDistanceField` / cache | T1.5 |
| (2)(3) FVM 装配 | `discretization::turbulence::assemble_*` | T2+ |
| §4 BC | `discretization::turbulence::bc` | T3 |
| §5 编排 | `solver::compressible_*` + LU-SGS | T2+ |
| Case 解析 | `io::case` `[physics.turbulence]` | T0→T2 |

## 7. 参考文献

1. Menter, F. R. (1994). *Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications*. AIAA Journal, 32(8), 1598–1605. DOI [10.2514/3.12149](https://doi.org/10.2514/3.12149)
2. Menter, F. R. (2003). *Review of the SST Turbulence Model*. ERCOFTAC Series (Hybrid RANS/LES 综述章节；可压 RANS 常数集引用此版).
3. Wilcox, D. C. (2006). *Turbulence Modeling for CFD*. DCW Industries. ISBN 978-1928729082. Ch. 4（k-ω 基模与壁面 \(\omega\)）.
4. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications*. Elsevier. ISBN 978-0080449509. §10（湍流模型 FVM 离散与边界条件）.

## 8. 相关算例

| 算例 | 阶段 | 说明 |
|------|:----:|------|
| `tests/benchmarks/flat_plate_turbulent/` | T2–T3 | \(C_f\)、\(u^+\)（待建） |
| `tests/benchmarks/channel_re_tau_395/` | T3 | 槽道（待建） |
| `tests/benchmarks/cylinder_mach8/` | T3+ | 降 Re RANS 扩展 |
| dual_ellipsoid | T4 | 非结构工程回归 |

---

**维护**：实现 T1 起更新「状态」列与 §6 代码路径；常数 / 壁面 \(\omega\) 公式变更须更新 benchmark 参考值。
