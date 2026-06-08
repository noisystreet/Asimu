# 边界条件

本文描述 asimu 标量扩散与可压缩流边界条件的数据模型与数值施加，架构对照 CFL3D 的 `bc.F` + `bcXXXX.F` 分层。

---

## 1. 架构分层

| 层 | asimu 模块 | CFL3D 类比 |
|----|-----------|-----------|
| 数据 | `boundary::{BoundaryKind, BoundaryPatch, BoundarySet}` | `cfl3d.inp` BC 段、`ibcinfo` |
| 调度 | `boundary::BoundaryRegistry` | `bc.F` 主循环 |
| 网格映射 | `mesh::BoundaryMesh` | 面索引 `I/J/K` + 段范围 |
| 数值施加 | `discretization::bc` | `bc1000.F`（Dirichlet）、`bc2004.F`（Neumann）等 |
| 解析 | `io::case` | 输入文件读取 |

**约束**：`io` 只产出 `BoundaryPatch` 数据；不在 I/O 层修改线性系统。

---

## 2. 标量扩散边界类型

### 2.1 Dirichlet

固定值 \(\phi = \phi_b\)（在边界**面**上给定）。FVM 采用 ghost 单元：

\[
\phi_{\text{ghost}} = 2\phi_b - \phi_{\text{owner}}
\]

对 owner 行累加：对角 \(+2D/d\)，右端 \(+2D/d \cdot \phi_b\)（\(d\) 为面到单元中心距离）。

另提供 `apply_dirichlet` **强施加**（整行替换），用于特殊场景，默认 patch 调度使用 `apply_dirichlet_face`。

### 2.2 Neumann

法向通量条件：

\[
-D \frac{\partial \phi}{\partial n} = q
\]

通过 ghost 单元消元：距 owner 中心距离 \(d\) 处，

\[
\phi_{\text{ghost}} = \phi_{\text{owner}} + \frac{d\, q}{D}
\]

对 owner 行累加：对角 \(+D/d\)，右端 \(+q\)。

---

## 3. 可压缩边界通量

3D 可压缩无粘残差在边界面使用显式边界通量接口：

```text
owner primitive + boundary exterior state
    ↓ reconstruct_face_primitives
InterfacePrimitiveStates
    ↓ face_inviscid_flux
F_boundary
```

`BoundaryInviscidFluxInput` 中的 `exterior` 是边界模型给出的面外侧状态。它可以由传统 ghost cell、特征边界条件或后续边界 Riemann 模型生成；残差装配只消费边界面通量，不依赖边界模型内部实现。

### 3.1 Farfield 特征状态

令面法向指向域外，\(u_n=\mathbf{u}\cdot\mathbf{n}\)，声速 \(a=\sqrt{\gamma p/\rho}\)。亚声速 farfield 使用一维 Riemann 不变量：

\[
R^+ = u_{n,o} + \frac{2a_o}{\gamma-1}, \qquad
R^- = u_{n,\infty} - \frac{2a_\infty}{\gamma-1}
\tag{1}
\]

\[
u_n = \frac{R^+ + R^-}{2}, \qquad
a = \frac{\gamma-1}{4}(R^+ - R^-)
\tag{2}
\]

若 \(u_n<0\)（边界入流），熵与切向速度取远场；若 \(u_n\ge0\)（边界出流），熵与切向速度取内侧 owner。超声速入流直接使用远场，超声速出流直接外推 owner。

### 3.2 Inlet / Outlet 特征状态

亚声速入口给定总压 \(p_0\)、总温 \(T_0\) 与流向 \(\mathbf{d}\)，保留内侧出射特征 \(R^+\)。入口马赫数 \(M\in[0,1)\) 由下式求解：

\[
R^+ =
\sqrt{\frac{\gamma R T_0}{1+\frac{\gamma-1}{2}M^2}}
\left(\frac{2}{\gamma-1}+(\mathbf{d}\cdot\mathbf{n})M\right)
\tag{3}
\]

随后用等熵关系恢复静温和静压：

\[
T=\frac{T_0}{1+\frac{\gamma-1}{2}M^2}, \qquad
p=p_0\left(\frac{T}{T_0}\right)^{\gamma/(\gamma-1)}
\tag{4}
\]

亚声速出口给定静压 \(p_b\)，保留 owner 的熵、切向速度与出射特征 \(R^+\)，由 \(p_b/\rho^\gamma=p_o/\rho_o^\gamma\) 恢复密度和声速，再用

\[
u_n = R^+ - \frac{2a}{\gamma-1}
\tag{5}
\]

恢复法向速度。超声速入口/出口分别退化为全指定来流/零梯度外推。

实现：`farfield_ghost`、`inlet_ghost`、`outlet_ghost` 生成边界外侧状态；`inviscid_boundary_face_flux` 和 `BoundaryInviscidFluxInput` 负责边界面通量。

---

## 4. 施加顺序

与 CFL3D `bc.F` 一致，按 `BoundarySet` 中 patch **声明顺序**遍历：

```
内部面装配 → apply_boundary_conditions(patches) → 线性求解
```

1D 扩散实现见 `discretization::diffusion_1d::assemble_diffusion_1d`。

---

## 5. 1D 逻辑边界名

| TOML 键 | 面 | owner 单元 |
|---------|-----|-----------|
| `left` | `FaceId(0)` | `CellId(0)` |
| `right` | `FaceId(1)` | `CellId(n-1)` |

算例格式见 [CASE_FORMAT.md](../CASE_FORMAT.md) §5。

---

---

## 7. 验证（扩散）

`tests/benchmarks/1d_diffusion_analytical`：\(D=1\)，\(\phi(0)=0\)，\(\phi(L)=1\)，解析解 \(\phi(x)=x/L\)。

集成测试：`tests/boundary_1d_diffusion.rs`。

可压缩边界单元测试：`discretization::bc_compressible::tests::*`。

---

## 8. 后续

- 2D 逻辑名 `bottom` / `top`
- CGNS `ZoneBC` → `BoundaryPatch`
- `Inlet` / `Outlet` / `Wall` / `Symmetry`（见 DATA_MODEL §5）
- 不可压 `convective_outlet`（I5+，ADR 0015）

---

## 9. 不可压缩 Navier-Stokes 边界（ADR 0015）

不可压 NS 在 `discretization/incompressible/bc.rs` 施加；**不使用** §3 特征/Riemann 边界。

### 9.1 架构

```text
BoundaryPatch (IncompressibleInlet / Outlet / Wall / ...)
    ↓ BoundaryRegistry → BcHandler::Incompressible*
    ↓ refresh_incompressible_ghosts
ghost u, v, w, p  →  Rhie-Chow m_dot  →  动量对流/扩散  →  压力 Poisson BC
```

### 9.2 Ghost 公式

距 owner 中心法向距离 \(d_f\)：

**Dirichlet**（速度入口、动壁）：

\[
\phi_g = 2\phi_b - \phi_o
\]

**Neumann**（壁面压力、出口速度、对称）：

\[
\phi_g = \phi_o + d_f \left(\frac{\partial \phi}{\partial n}\right)_b
\]

零梯度：\(\phi_g = \phi_o\)。

### 9.3 类型与方程分工

| 类型 | \(\mathbf{u}\) | \(p\) | \(p'\) | \(\dot{m}_f\) |
|------|----------------|-------|--------|---------------|
| 无滑移壁 | \(\mathbf{u}_g = -\mathbf{u}_o\) | Neumann | Neumann | 0 |
| 动壁 \(U_w\) | \(\mathbf{u}_g = 2U_w - \mathbf{u}_o\) | Neumann | Neumann | \(\rho U_w\cdot\mathbf{S}\) |
| 速度入口 | \(\mathbf{u}_g = 2u_b - u_o\) | Neumann | Neumann | upwind \(u_b\) |
| 压力出口 | \(\partial u/\partial n=0\) | \(p=p_b\) | \(p'=0\) | upwind owner |
| 对称 | \(u_n=0\), \(\partial u_t/\partial n=0\) | Neumann | Neumann | \(u_n=0\) |
| 压力参考 | — | \(p=p_{\mathrm{ref}}\) | \(p'=0\) | — |

### 9.4 Case 与可压分流

`solver.type = "incompressible_ns"` 时：

- `kind = "inlet"` → 必须 `velocity = [u,v,w]`（**非** `total_pressure`）；
- `kind = "outlet"` → `static_pressure`（gauge）；
- `kind = "farfield"` → **Validate 失败**。

`solver.type = "compressible_ns"` 时保持 §3 语义不变。

### 9.5 施加顺序

与 §4 相同：ghost 刷新 → Rhie-Chow → 动量装配 → 压力 Poisson；patch 按声明顺序。

---

## 10. 参考文献

1. Hirsch, C. (2007). *Numerical Computation of Internal and External Flows*, 2nd ed. Ch. 19（特征边界条件）。
2. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications*, 3rd ed. §8（Euler / Navier-Stokes 边界条件）。
3. Thompson, K. W. (1987). Time dependent boundary conditions for hyperbolic systems. *Journal of Computational Physics*, 68(1), 1–24.
4. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Ch. 6–7（不可压 ghost 单元）。
5. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Ch. 8–9.
