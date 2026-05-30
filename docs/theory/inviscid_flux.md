# 无粘通量与 Roe Riemann 求解

> 模块：`src/discretization/inviscid.rs`、`roe.rs`、`residual/` · 版本：v1.x · 状态：**已实现（物理通量 + Roe + Harten 熵修正）**

## 1. 控制方程

三维可压缩 Euler（守恒形式，理想气体）：

$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla\cdot \mathbf{F}(\mathbf{U}) = 0 \tag{1}
$$

$$
\mathbf{U} = \begin{bmatrix}\rho \\ \rho u \\ \rho v \\ \rho w \\ \rho E\end{bmatrix}, \quad
\mathbf{F} = \begin{bmatrix}\rho \mathbf{u} \\ \rho \mathbf{u}\mathbf{u} + p\mathbf{I} \\ (\rho E + p)\mathbf{u}\end{bmatrix}
$$

状态方程：\(p = (\gamma-1)\rho e\)，\(E = e + \tfrac{1}{2}|\mathbf{u}|^2\)。

---

## 2. 物理面通量 \(\mathbf{F}\cdot\mathbf{n}\)

令 \(u_n = \mathbf{u}\cdot\mathbf{n}\)，\(p\) 为静压，\(\rho E\) 为单元总能量密度：

$$
\begin{aligned}
F_{\mathrm{mass}} &= \rho u_n \\
\mathbf{F}_{\mathrm{mom}} &= \rho u_n \mathbf{u} + p \mathbf{n} \\
F_{\mathrm{energy}} &= (\rho E + p)\, u_n
\end{aligned}
\tag{2}
$$

代码结构体 `InviscidFlux { mass, momentum[3], energy }` 对应式 (2) 在法向 \(\mathbf{n}\) 上的分量。

---

## 3. FVM 残差装配

单元 \(i\) 的时间导数（与 [time_integration.md](time_integration.md) 式 (1) 一致）：

$$
\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t}
= -\frac{1}{V_i}\sum_{f} A_f\, \hat{\mathbf{F}}_f \tag{3}
$$

| 面类型 | owner 贡献 | neighbor 贡献 |
|--------|------------|---------------|
| 内部面 | \(-A/V_i\,\hat{\mathbf{F}}\) | \(+A/V_j\,\hat{\mathbf{F}}\) |
| 边界面 | \(-A/V_i\,\hat{\mathbf{F}}\) | — |

\(\hat{\mathbf{F}}\) 沿 owner → neighbor 法向定义；`accumulate_interior_face` / `accumulate_boundary_face` 实现符号。

---

## 4. 数值通量：Roe 近似 Riemann 解

### 4.1 通量公式

$$
\hat{\mathbf{F}} = \tfrac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \tfrac{1}{2}\sum_{k=1}^{5} |\lambda_k|\,\alpha_k \mathbf{r}_k \tag{4}
$$

- \(\mathbf{F}_L,\mathbf{F}_R\)：左右态物理通量（式 (2)）
- \(\lambda_k\)：Jacobian 特征值（声学 \(\lambda_1=u_n-a\)，接触 \(\lambda_2=u_n\)，剪切 \(\lambda_3,\lambda_4\)，声学 \(\lambda_5=u_n+a\)）
- \(\alpha_k\)：波强度（密度/压力/法向速度跳跃展开）
- \(\mathbf{r}_k\)：右特征向量（在 Roe 平均态上计算）

实现：`roe_flux` → `combine_fluxes`（半和减半耗散）。

### 4.2 Roe 平均

$$
\begin{aligned}
\tilde{\rho} &= \sqrt{\rho_L\rho_R} \\
\tilde{\mathbf{u}} &= \frac{\sqrt{\rho_L}\mathbf{u}_L + \sqrt{\rho_R}\mathbf{u}_R}{\sqrt{\rho_L}+\sqrt{\rho_R}} \\
\tilde{h} &= \frac{\sqrt{\rho_L}h_L + \sqrt{\rho_R}h_R}{\sqrt{\rho_L}+\sqrt{\rho_R}}, \quad
h = \frac{\rho E + p}{\rho} \\
\tilde{a} &= \sqrt{(\gamma-1)\left(\tilde{h} - \tfrac{1}{2}|\tilde{\mathbf{u}}|^2\right)}
\end{aligned}
\tag{5}
$$

### 4.3 波强度（法向）

令 \(\Delta p = p_R-p_L\)，\(\Delta \rho = \rho_R-\rho_L\)，\(\Delta u_n = (u_n)_R-(u_n)_L\)：

$$
\alpha_1 = \frac{\Delta p - \tilde{\rho}\tilde{a}\,\Delta u_n}{2\tilde{a}^2}, \quad
\alpha_5 = \frac{\Delta p + \tilde{\rho}\tilde{a}\,\Delta u_n}{2\tilde{a}^2}
\tag{6}
$$

接触波与剪切波强度见 `wave_strengths`（\(\alpha_2,\alpha_3,\alpha_4\)）。

### 4.4 Harten 熵修正

弱激波附近 Roe 解可能违反熵条件。对声学特征值 \(\lambda\in\{\lambda_1,\lambda_5\}\)：

$$
|\lambda| \leftarrow
\begin{cases}
|\lambda| & \text{if } |\lambda| \ge \delta \\
\dfrac{\lambda^2 + \delta^2}{2\delta} & \text{if } |\lambda| < \delta
\end{cases}
\tag{7}
$$

默认 \(\delta = 0.2(|\tilde{u}_n| + \tilde{a})\)（`RoeFluxConfig::entropy_delta = None`）。接触/剪切波仍用 \(|\lambda_2|=|\tilde{u}_n|\) 等。

配置：`RoeFluxConfig { entropy_fix: true, entropy_delta: Option<f64> }`。

---

## 5. 面通量管线

```text
owner, neighbor (± MUSCL 模板点)
    ↓ reconstruct_face_states   ← 见 interface_reconstruction.md
InterfaceStates { left, right }
    ↓ roe_flux / hllc_flux (InviscidFluxConfig)
InviscidFlux
    ↓ accumulate_*_face
ConservedResidual  (= dU/dt)
```

入口函数：`face_inviscid_flux`（`FaceFluxInput` + `InviscidFluxConfig`）。

---

## 6. 实现映射

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (2) 物理通量 | `physical_inviscid_flux` | **已实现** |
| (4) Roe 通量 | `roe_flux` | **已实现** |
| (5) Roe 平均 | `roe_averages` | **已实现** |
| (6) 波强度 | `wave_strengths` | **已实现** |
| (7) 熵修正 | `harten_entropy_fix`、`fixed_eigenvalue` | **已实现** |
| HLLC（Toro §10） | `hllc_flux` | **已实现** |
| 面 dispatch | `face_inviscid_flux` | **已实现** |
| (3) 1D/3D 装配 | `assemble_inviscid_residual_1d` / `_3d` | **已实现** |
| AUSM+ | — | **规划** |

配置：`InviscidFluxConfig`（`reconstruction` / `limiter` / `scheme`）；`CompressibleEulerConfig::inviscid`。

---

## 7. 参考文献

1. Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43(2), 357–372. DOI [10.1016/0021-9991(81)90128-5](https://doi.org/10.1016/0021-9991(81)90128-5).
2. Harten, A. (1983). On the symmetric form of the Godunov-type schemes. *Journal of Computational Physics*, 49(3), 357–393.
3. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer. Ch. 10–11（Roe、熵修正）。
4. 精确 Riemann 验证：`physics::riemann_exact`（Toro §4）；算例 `tests/benchmarks/sod_1d/`。

---

## 8. 相关算例

- `tests/benchmarks/sod_1d/` — Sod 激波管，100 单元，\(t=0.2\)
- `discretization::roe::tests::sod_interface_roe_flux_matches_reference_values`
- `discretization::hllc::tests::sod_interface_hllc_flux_matches_reference_values`
- `discretization::residual::assembly_1d::tests::two_cell_discontinuity_has_opposing_mass_rhs`

精确解采样：相对隔膜坐标 \(x' = x - x_{\mathrm{diaphragm}}\)（`sample_exact(problem, x', t)`）。
