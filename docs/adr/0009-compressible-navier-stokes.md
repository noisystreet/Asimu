# ADR 0009: 三维可压缩 Navier-Stokes 求解器架构

- **状态**: 已接受（规划基线）
- **日期**: 2026-05-29
- **关联**: [ADR 0002](0002-layered-cfd-architecture.md)、[ADR 0005](0005-time-integration.md)、[ADR 0008](0008-cgns-io.md)、[ARCHITECTURE.md](../ARCHITECTURE.md)、[DATA_MODEL.md](../DATA_MODEL.md)

## 背景

asimu 演进路线为 v0.2 对流-扩散 → v0.3 不可压 NS（SIMPLE）→ v0.4 瞬态方腔。工程侧已通过 [ADR 0008](0008-cgns-io.md) 具备 **3D 结构化网格读入**（`StructuredMesh3d`），但当前 `ARCHITECTURE.md` §2.2 将「完整三维可压 NS 生产级求解」列为非目标，**缺少**可压 NS 的模块边界、数值基线与演进节奏定案。

三维可压缩 Navier-Stokes 与不可压 NS 在架构上存在本质差异：

| 维度 | 不可压 NS（v0.3） | 可压 NS（本 ADR） |
|------|-------------------|-------------------|
| 主变量 | \(p, \mathbf{u}\) | 守恒变量 \(\mathbf{U}=[\rho,\rho u,\rho v,\rho w,\rho E]^T\) |
| 闭合 | \(\nabla\cdot\mathbf{u}=0\) | 理想气体 EOS \(p=p(\rho,e)\) |
| 空间离散 | 中心差分为主 | Riemann 求解器捕捉激波 |
| 时间推进 | 隐式/半隐式常见 | 显式 RK + CFL 为主 |
| 线性代数 | 压力 Poisson / 耦合系统 | 显式路径通常无大型隐式系统 |

若不提前定案，易出现：通量公式堆入 `solver`、原始变量破坏守恒性、I/O 层渗透离散假设——与 [ADR 0002](0002-layered-cfd-architecture.md) 分层原则冲突。

## 决策

### 1. 沿用 FVM + 数据/算法分层

复用 ADR 0002 依赖方向，**不**为可压 NS 另起炉灶：

```
core ← mesh ← field ← discretization
core ← physics
core ← linalg          # 隐式/稳态路径按需
mesh + field + discretization + physics + boundary ← solver
io → mesh + field + boundary   # 只产出数据，不做离散假设
app/case → 串联流水线
```

`solver` **仅**负责时间推进与非线性迭代编排；无粘/粘性通量、梯度重构、EOS 均不属于 `solver`。

### 2. 控制方程与主变量

三维可压缩 NS 守恒形式：

\[
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \nabla \cdot \mathbf{F}_v(\mathbf{U}, \nabla \mathbf{U}) + \mathbf{S}
\]

**主存储为守恒变量 SoA**（`field::ConservedFields`）：

| 分量 | 字段名 |
|------|--------|
| \(\rho\) | `density` |
| \(\rho u, \rho v, \rho w\) | `momentum_x/y/z` |
| \(\rho E\) | `total_energy` |

原始变量（\(p, T, \mathbf{u}, a\)）由 `physics` 从 \(\mathbf{U}\) **派生**，不作为时间推进主状态，避免激波处非物理振荡。

类型契约（`physics` / `field`）：

```rust
pub struct ConservedState { pub density, pub momentum: [Real; 3], pub total_energy }
pub struct PrimitiveState { pub density, pub velocity: [Real; 3], pub pressure, pub temperature }
pub struct IdealGasEoS { pub gamma, pub gas_constant }
```

### 3. 模块职责

| 模块 | 路径（规划） | 职责 |
|------|-------------|------|
| `physics` | `src/physics/` | EOS（`IdealGasEoS`）、粘性模型（Sutherland）、来流参数（`FreestreamParams`）；湍流模型 trait 延后 |
| `field` | `src/field/conserved.rs` | `ConservedFields` SoA；`uniform` / `from_freestream` 初始化 |
| `mesh` | `src/mesh/structured.rs` | 3D FVM 几何：`cell_volume`、`face_area`、`face_normal`、`owner`/`neighbor` |
| `discretization` | `src/discretization/` | 梯度、无粘通量、粘性通量、残差装配 |
| `boundary` | `src/boundary/` | 可压 BC 类型与 patch 调度 |
| `solver` | `src/solver/` | `CompressibleNSSolver` 编排；CFL；委托 `TimeIntegrator` |
| `linalg` | `src/linalg/` | 显式路径不强制；稳态隐式 / 湍流隐式源项时 CSR + GMRES |

#### 3.1 `discretization` 子模块划分

```
discretization/
├── gradient.rs         # 格林-高斯 / 最小二乘
├── reconstruction.rs   # MUSCL / 限制器
├── inviscid.rs         # Riemann 数值通量
├── viscous.rs          # 粘性通量（梯度 + 中心差分）
└── residual.rs         # dU/dt = -1/V Σ(F·S) + S/V
```

**扩展点 trait**（热路径 enum dispatch 优先，trait 用于测试 mock 与后期替换）：

| Trait | 职责 | 首版实现 |
|-------|------|----------|
| `FluxScheme` | 无粘面通量 | HLLC |
| `GradientScheme` | 单元梯度 | 格林-高斯 |
| `ReconstructionScheme` | 面左右状态 | MUSCL-2 + van Leer 限制器 |
| `ViscosityModel` | \(\mu(T), \lambda(T)\) | Sutherland |

#### 3.2 单步数据流

```
ConservedFields U
  → physics: U → primitive (ρ, p, T, u, μ, λ)
  → gradient: ∇U / ∇u, ∇T
  → face loop:
       UL, UR ← reconstruction + BC ghost
       F_inv  ← FluxScheme::numerical_flux(UL, UR, n)
       F_visc ← viscous_flux(∇u, ∇T, μ, λ)
       R[owner]     += (F_inv + F_visc) · S
       R[neighbor]  -= (F_inv + F_visc) · S
  → TimeIntegrator: U += dt * (-R/V + S)
```

### 4. 边界条件

扩展现有 `BoundaryRegistry` 两阶段模式（registry → apply），新增可压类型：

| 类型 | 用途 | 离散处理 |
|------|------|----------|
| `Wall` | 无滑移/滑移壁 | 等距镜像 ghost：无滑移 \(\mathbf{u}_g=-\mathbf{u}_o\)；滑移 \(u_{n,g}=-u_{n,o}\)、\(\mathbf{u}_{t,g}=\mathbf{u}_{t,o}\) |
| `Farfield` | 远场 | Riemann 特征边界 |
| `Inlet` | 总压/总温入口 | 指定下游状态 |
| `Outlet` | 静压出口 | 外推或指定 \(p\) |
| `Symmetry` | 对称面 | 法向速度为零、切向梯度为零 |

BC 数学与离散细节见 `docs/theory/compressible_ns.md`（规划）；v0.2 标量 BC（Dirichlet/Neumann）保持不变。

### 5. 时间推进

复用 [ADR 0005](0005-time-integration.md) 的 `TimeIntegrator`，可压 NS 典型配置：

| 场景 | 实现 | 版本 |
|------|------|------|
| 瞬态 | 显式 RK3 | v1.x 首版 |
| 稳态收敛 | 局部时间步（LUS）/ 双时间步 | v1.x+ |
| 刚性 / 细网格 | 隐式 BDF + GMRES | v1.x+ 评估 |

CFL 约束独立为 `solver::cfl`：

\[
\Delta t = \mathrm{CFL} \cdot \min_i \frac{V_i}{\sum_f (|\lambda_{\max}|_f \cdot A_f)}
\]

其中 \(\lambda_{\max} = |u_n| + a\)（法向）。

配置示例（`case.toml` 扩展，实现期定案）：

```toml
[solver]
type = "compressible_ns"

[time]
mode = "transient"
cfl_max = 0.5

[physics.eos]
gamma = 1.4
gas_constant = 287.052871936417

[freestream]
mach = 0.3
pressure = 101325.0
temperature = 288.15
```

### 6. 网格与 I/O

| 项 | 决策 |
|----|------|
| 首版网格 | 3D 结构化六面体（`StructuredMesh3d`） |
| 几何来源 | [ADR 0008](0008-cgns-io.md) CGNS zone 读入 |
| FVM 几何 | 由节点坐标**预计算**体积/面积/法向，不在 I/O 层做通量假设 |
| 场输出 | VTS PointData（`io-vtk`）；CGNS 场写出后续 ADR |
| BC 读入 | CGNS BC 段映射 → `BoundaryPatch`（v1.x） |

当前 `StructuredMesh3d` 仅有节点坐标；**可压 NS 首版 PR 必须先补全 FVM 几何查询接口**（见 [DATA_MODEL.md](../DATA_MODEL.md) §3）。

### 7. 数值基线（首版可压 NS）

| 项 | 选择 |
|----|------|
| 空间离散 | FVM + MUSCL-2 + HLLC |
| 粘性项 | 中心差分 + 格林-高斯梯度 |
| 时间积分 | 显式 RK3 + CFL |
| EOS | 理想气体 |
| 粘性 | Sutherland 空气 |
| 并行 | 单线程验证 → v1.0 `rayon` 面循环 |
| 湍流 | 层流首版；**RANS 见 [ADR 0014](0014-turbulence-k-omega-sst-rans.md)（Menter k-ω SST）** |

### 8. 验证演进路线

按难度递增，每步须有 `tests/benchmarks/` 算例：

| 阶段 | 算例 | 验证量 |
|------|------|--------|
| 1 | 1D Sod 激波管 | 密度/压力剖面 |
| 2 | 2D 双马赫反射 | 激波位置 |
| 3 | 2D 平板边界层（可压） | 速度剖面 |
| 4 | 3D 方盒 / 喷管 | 质量守恒、总压恢复 |

算例登记见 [BENCHMARKS.md](../BENCHMARKS.md)（实现时追加条目）。

### 9. 与不可压路线关系

- **不替换** v0.3 不可压 SIMPLE 规划；二者共享 `mesh`、`field` 基础设施与 `boundary` 框架。
- `case` 层通过 `solver.type` 选择 `IncompressibleNavierStokes` 或 `CompressibleNavierStokes`。
- 可压 NS **不阻塞** v0.2–v0.4 交付；首版目标版本 **v1.x**（不可压路径稳定后）。

### 10. 架构反模式（禁止）

- 在 `solver` 内实现 Roe/HLLC 通量公式
- 以原始变量作为时间推进主状态
- `io` 解析阶段假设离散格式或 CFL
- 全热路径 trait 对象化（扩展点除外）
- 首版引入 MPI / GPU 可压路径（先单线程数值验证）

## 后果

### 正面

- 可压 NS 与现有分层架构一致，模块可独立单测
- 守恒变量 + Riemann 通量为社区主流，文献与 V&V 资源丰富
- CGNS 3D 网格 I/O 已有，几何链路可复用
- `physics` / `ConservedFields` 骨架可先行落地，不依赖完整通量实现

### 负面

- 3D FVM 几何预计算与面循环实现量大
- HLLC + MUSCL 调试成本高，需分级算例覆盖
- 显式 CFL 限制细网格时间步长，生产场景可能需隐式扩展
- 与不可压 NS 并行开发时，`boundary` / `case` schema 需协调

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 原始变量有限差分 | 激波稳定性差；守恒性差 |
| 不可压 SIMPLE 扩展至可压 | 低速可行，跨音速/激波不适用 |
| 首版非结构化 + GMRES | 复杂度过高，拖慢验证（ADR 0002 已否决） |
| 分布式 MPI 首版 | 先正确后并行；MPI 单独 ADR |
| 全耦合隐式 Newton-Krylov 首版 | 实现与调参成本高；显式路径先验证 |
