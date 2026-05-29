# 边界条件（v0.2）

本文描述 asimu v0.2 标量扩散问题的边界条件数据模型与数值施加，架构对照 CFL3D 的 `bc.F` + `bcXXXX.F` 分层。

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

## 2. 边界类型（v0.2）

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

## 3. 施加顺序

与 CFL3D `bc.F` 一致，按 `BoundarySet` 中 patch **声明顺序**遍历：

```
内部面装配 → apply_boundary_conditions(patches) → 线性求解
```

1D 扩散实现见 `discretization::diffusion_1d::assemble_diffusion_1d`。

---

## 4. 1D 逻辑边界名

| TOML 键 | 面 | owner 单元 |
|---------|-----|-----------|
| `left` | `FaceId(0)` | `CellId(0)` |
| `right` | `FaceId(1)` | `CellId(n-1)` |

算例格式见 [CASE_FORMAT.md](../CASE_FORMAT.md) §5。

---

## 5. 验证

`tests/benchmarks/1d_diffusion_analytical`：\(D=1\)，\(\phi(0)=0\)，\(\phi(L)=1\)，解析解 \(\phi(x)=x/L\)。

集成测试：`tests/boundary_1d_diffusion.rs`。

---

## 6. 后续（v0.3+）

- 2D 逻辑名 `bottom` / `top`
- CGNS `ZoneBC` → `BoundaryPatch`
- `Inlet` / `Outlet` / `Wall` / `Symmetry`（见 DATA_MODEL §5）
