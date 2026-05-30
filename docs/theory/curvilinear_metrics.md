# 曲线坐标与贴体网格度量

> 模块：`src/mesh/`、`src/discretization/residual/` · 版本：v1.x · 状态：**规划**

## 1. 连续形式（贴体 FVM）

三维守恒方程：

$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = 0 \tag{1}
$$

对控制体 \(V_i\) 积分并高斯散度：

$$
\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t} = -\frac{1}{V_i}\oint_{\partial V_i} \mathbf{F}\cdot\mathrm{d}\mathbf{S}
= -\frac{1}{V_i}\sum_f \mathbf{F}_f\cdot\mathbf{S}_f \tag{2}
$$

其中：

- \(V_i\)：**物理**体积（m³）
- \(\mathbf{S}_f = \hat{\mathbf{n}}_f A_f\)：有向面积向量（m²），方向为 owner → neighbor
- \(\mathbf{F}_f\)：数值通量（kg/(m²·s) 等量纲），Roe/HLLC 输出笛卡尔分量

**关键**：式 (2) 中 **不出现** 逻辑 Δξ、Δη；曲线坐标只用于 **索引邻居**，几何来自 \(\mathbf{x}(i,j,k)\)。

---

## 2. 逻辑坐标 vs 物理坐标

| 概念 | 含义 | asimu 中的对象 |
|------|------|----------------|
| 逻辑 (i,j,k) | 结构化网格索引 | `StructuredMesh3d::cell_index` |
| 物理 (x,y,z) | 笛卡尔坐标 | `node_x/y/z(i,j,k)` |
| 计算空间 (ξ,η,ζ) | 贴体曲线坐标 | **不单独存储**；用节点坐标隐式表达 |

CFL3D、OVERFLOW 等也采用类似策略：**预计算或按需计算 metric**，而非在 PDE 中显式写出变换后的方程。

### 2.1 C 形圆柱网格

C 形块常见拓扑：

```
        外边界 (farfield)
       ┌─────────────────┐
  入口 │    圆柱 (wall)   │  出口 x = const
       │       ╭──╮       │
       └───────┴──┴───────┘
              尾迹 / 切缝
```

- **nz = 1**：准 2D 挤出；z 方向有真实厚度
- **j = 0 外边界段**：节点沿 i 排列，但物理 x 相同 → 逻辑 Δx=0，**物理上边仍是有限长度曲线/直线**
- 必须用 **四边形面** 计算 \(\mathbf{S}\)，不能用 Δx

---

## 3. 几何度量算法

### 3.1 单元体积 \(V_{i,j,k}\)

六面体单元由 8 个顶点张成。常用做法：

1. **五/六四面体分解**：将六面体拆成 5 或 6 个四面体，体积代数和
2. **顶点顺序**：与 (i,j,k) → (i+1,j,k) → … 一致，保证右手系

```text
V = (1/6) | (x1-x0) · ((x2-x0) × (x3-x0)) |   // 单个四面体
V_cell = Σ V_tet
```

**验收**：均匀盒子 \(1\times1\times1\)、\(N^3\) 单元 → 每单元体积 \(1/N^3\)。

### 3.2 面面积向量 \(\mathbf{S}_f\)

以 **i 面**（位于 cell i 与 i+1 之间）为例，四顶点（节点平面 i+1）：

$$
\mathbf{x}_{00}=(i+1,j,k),\;
\mathbf{x}_{10}=(i+1,j+1,k),\;
\mathbf{x}_{11}=(i+1,j+1,k+1),\;
\mathbf{x}_{01}=(i+1,j,k+1)
$$

两三角分解：

$$
\mathbf{S} = \frac{1}{2}(\mathbf{x}_{10}-\mathbf{x}_{00})\times(\mathbf{x}_{01}-\mathbf{x}_{00})
+ \frac{1}{2}(\mathbf{x}_{11}-\mathbf{x}_{10})\times(\mathbf{x}_{01}-\mathbf{x}_{10})
\tag{3}
$$

j 面、k 面、边界 i\_min/i\_max 等同理，仅顶点索引不同。

派生量：

$$
A = |\mathbf{S}|,\quad \hat{\mathbf{n}} = \mathbf{S}/A
\tag{4}
$$

**法向约定**：owner 单元指向 neighbor 单元（与现有 `LogicalFace3d` 一致）。

### 3.3 面间距（CFL 用）

$$
h_f = \frac{|\mathbf{r}_{\mathrm{nb}}-\mathbf{r}_{\mathrm{ow}}|\cdot A}{|\mathbf{r}_{\mathrm{nb}}-\mathbf{r}_{\mathrm{ow}}|}
\quad\text{或}\quad
h_f = \frac{V_{\mathrm{ow}}+V_{\mathrm{nb}}}{2A}
\tag{5}
$$

取 owner/neighbor **单元中心** \(\mathbf{r}\)（8 顶点平均或 6 面中心平均）。

CFL 时间步：

$$
\Delta t = \mathrm{CFL}\cdot \min_f \frac{h_f}{|u_n|+a}
\tag{6}
$$

**注意**：忽略数值零间距（\(\|\mathbf{S}\|\) 或 \(V\) 相对全场极小），避免 \(\Delta t\to 0\)。

---

## 4. 与当前实现的差异

### 4.1 现状（Cartesian 路径）

| 组件 | 文件 | 现状 |
|------|------|------|
| 体积 | `structured.rs` / `structured_3d_boundary.rs` | `cell_volume_at` = Δx·Δy·Δz |
| i 面法向 | `assembly_3d.rs` | 固定 `(1,0,0)` |
| 边界面几何 | `boundary_face_geometry` | 用 `cell_dx/dy/dz` 估算 |
| 残差 | `residual/mod.rs` | `-F·A/V`，A 标量 |
| Riemann | `roe.rs` / `hllc.rs` | **已支持**任意 `Vector3` 法向 |
| MUSCL | `muscl_stencil_3d.rs` | 逻辑 i/j/k 邻居 |

### 4.2 目标（Curvilinear 路径）

| 组件 | 规划 |
|------|------|
| `mesh/metrics.rs` | `CellMetric { volume, center }`、`FaceMetric { area_vector, area, normal }` |
| `StructuredMesh3d` | `cell_metric(i,j,k)`、`i/j/k_face_metric(...)` |
| `assembly_3d.rs` | 用 metric 替代 Δx/固定法向 |
| `face_geometry_3d` | 边界 patch 走同一套 metric |
| `compressible.rs` | CFL 用 \(h_f\) 与 min 正间距 |
| `case.toml` | `[mesh] metric = "curvilinear"` |

### 4.3 残差累加（通量与面积向量）

Roe/HLLC 返回 `InviscidFlux { mass, momentum[3], energy }`，已是 **笛卡尔** 动量通量分量。

守恒更新：

$$
\Delta U = -\frac{\Delta t}{V}\,\mathbf{F}^{\mathrm{flux}}
$$

其中质量方程用标量通量；动量方程在曲线网格上，**法向通量** \(\mathbf{F}_{\mathrm{mom}}\) 已为全局分量，与标量 `area` 相乘等价于 \(\mathbf{F}\cdot\mathbf{S}\) 当 \(\mathbf{F}\) 定义在法向时——需与 `inviscid.rs` 中式 (2) 一致。

**实现检查点**：uniform box 上 curvilinear 路径与 Cartesian 路径 L2(ρ̇) 差异 < 1e-10。

---

## 5. 界面重构（MUSCL）在曲线网格上

逻辑邻居 **不等于** 物理法向邻居。分阶段策略：

| 阶段 | 策略 | 精度 | 复杂度 |
|------|------|------|--------|
| C0 | 一阶（cell 值） | 1 | 低 |
| C1 | 逻辑 MUSCL + curvilinear metric | 1–2 | 中（当前代码 + metric） |
| C2 | 沿 \(\hat{\mathbf{n}}\) 或中心连线距离 limiter | 2 | 中高 |
| C3 | 完整 \(\partial\mathbf{x}/\partial\xi\) Jacobian 变换 | 2+ | 高（CFL3D 级） |

**建议**：M1–M3 用 **C0 或 C1** 跑通 cylinder Mach 8 smoke test；C2 作为后续精度项。

---

## 6. C 形块折叠线与尾迹切缝

贴体 metric **不能** 单独解决所有 C 形拓扑问题：

| 类型 | 特征 | 处理 |
|------|------|------|
| 外边界退化线 | j=0，Δx=0 但 \(\|\mathbf{S}\|>0\) | metric 可修复 |
| 尾迹切缝 / 块内缝 | 逻辑缝、双侧单元 | 专用 **cut BC** 或 periodic |
| 极小的物理体积 | 折叠线附近 \(V\to 0\) | 检测 + 特殊 BC 或 2D 模式 |
| nz=1 准 2D | 仅一层 z | 优先 **2D curvilinear** 求解器 |

检测建议：

```text
degenerate if V_i < ε_V * V_ref
         or cond(J) > ε_cond
         or A_f < ε_A * A_ref
```

---

## 7. 配置与 API 草案

### 7.1 case.toml

```toml
[mesh]
kind = "cgns"
path = "cylinder.cgns"
zone = 1
scale = 0.001
metric = "curvilinear"   # "cartesian" | "curvilinear"（CGNS 默认 curvilinear）
```

### 7.2 Rust API（规划）

```rust
// mesh/metrics.rs
pub struct FaceMetric {
    pub area_vector: Vector3,
    pub area: Real,
    pub normal: Vector3,
}

pub struct CellMetric {
    pub volume: Real,
    pub center: Vector3,
}

impl StructuredMesh3d {
    pub fn cell_metric(&self, i: usize, j: usize, k: usize) -> CellMetric;
    pub fn i_face_metric(&self, i: usize, j: usize, k: usize) -> FaceMetric;
    pub fn j_face_metric(&self, i: usize, j: usize, k: usize) -> FaceMetric;
    pub fn k_face_metric(&self, i: usize, j: usize, k: usize) -> FaceMetric;
}
```

可选：`MetricCache` 在 `load_cgns_zone` 后预计算，避免每步重复。

---

## 8. 实施路线图

| 阶段 | 内容 | 验收 |
|------|------|------|
| **M1** | `mesh/metrics.rs`：四面体/三角分解算 V、S | 均匀盒子体积误差 < 1e-12 |
| **M2** | `assembly_3d` + `face_geometry_3d` 走 metric | 盒子均匀来流 L2(ρ̇)≈0 |
| **M3** | CFL 用 \(h_f\)；忽略相对零 metric | cylinder dt 合理，无 1e-21 |
| **M4** | CGNS 默认 `metric=curvilinear` | cylinder 零体积单元 = 0 |
| **M5** | 贴体 MUSCL 或一阶 fallback | Mach 8 smoke：step1 残差有限 |
| **M6** | 折叠线检测 + 2D 模式 / cut BC | C 形尾迹 V&V |

---

## 9. 实现映射

| 式 / 步骤 | 规划代码位置 | 状态 |
|-----------|--------------|------|
| (2) FVM 积分 | `discretization/residual/mod.rs` | **已实现**（标量 A） |
| (3)(4) 面积向量 | `mesh/metrics.rs` | 规划 |
| 单元体积 | `mesh/metrics.rs` | 规划 |
| (6) CFL | `solver/compressible.rs` | **部分**（`min_positive_spacing`） |
| i/j/k 面通量装配 | `discretization/residual/assembly_3d.rs` | **Cartesian** |
| 边界 ghost 法向 | `discretization/bc_compressible.rs` | **部分**（用 `FaceGeometry3d`） |
| Roe/HLLC | `discretization/roe.rs`、`hllc.rs` | **已实现** |
| CGNS 读入 | `io/cgns/read.rs` | **已实现** |
| 网格诊断 | `mesh/diagnostics.rs` | **已实现**（Δx 非正警告） |

---

## 10. 参考文献

1. Thomas, J. L., & Lombard, C. K. (1987). Geometric conservation law and its application to flow computations on moving grids. *AIAA Journal*, 35(8), 1410–1417.
2. Vinokur, M. (1989). An analysis of finite-difference and finite-volume formulations of conservation laws. *NASA CR-177512*.
3. Rumsey, C. L., Biedron, R. T., & Thomas, J. L. (2010). CFL3D: Its history and some recent applications. *NASA/TM-2010-216758*.
4. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics* (6th ed.). Springer. Ch. 8（FVM on structured grids）.
5. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer. Ch. 16（curvilinear 简要讨论）.

---

## 11. 相关算例与文档

- `cylinder.cgns` + `tests/benchmarks/cylinder_mach8/case.toml` — C 形圆柱 Mach 8（metric 修复后的 smoke test）
- [inviscid_flux.md](inviscid_flux.md) — 无粘通量与残差符号
- [interface_reconstruction.md](interface_reconstruction.md) — MUSCL 与面状态
- [boundary_conditions.md](boundary_conditions.md) — 可压缩 BC ghost
- [adr/0008-cgns-io.md](../adr/0008-cgns-io.md) — CGNS 读入
- [adr/0009-compressible-navier-stokes.md](../adr/0009-compressible-navier-stokes.md) — 可压缩 NS 架构

---

## 12. 术语对照

| 中文 | 英文 | 备注 |
|------|------|------|
| 曲线坐标 | curvilinear coordinates | 计算空间 (ξ,η,ζ) |
| 贴体坐标 | body-fitted coordinates | 贴物面网格 |
| 度量 / metric | geometric metrics | \(V\), \(\mathbf{S}\), 可选 Jacobian |
| C 形网格 | C-grid | 圆柱外流常用块拓扑 |
| 折叠线 | collapsed line / degenerate line | 逻辑方向零厚度、物理为线 |
| 面积向量 | area vector | \(\mathbf{S}=\hat{\mathbf{n}}A\) |
