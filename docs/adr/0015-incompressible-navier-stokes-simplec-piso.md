# ADR 0015: 三维不可压缩 Navier-Stokes（SIMPLEC + PISO）

- **状态**: 已接受（规划基线，实现分阶段 I0–I6）
- **日期**: 2026-06-08
- **关联**: [ADR 0002](0002-layered-cfd-architecture.md)、[ADR 0005](0005-time-integration.md)、[ADR 0008](0008-cgns-io.md)、[ADR 0009](0009-compressible-navier-stokes.md)、[ADR 0010](0010-unstructured-mixed-mesh.md)、[ADR 0014](0014-turbulence-k-omega-sst-rans.md)、[ARCHITECTURE.md](../ARCHITECTURE.md)、[DATA_MODEL.md](../DATA_MODEL.md)、[boundary_conditions.md](../theory/boundary_conditions.md)

## 背景

[ARCHITECTURE.md](../ARCHITECTURE.md) §10 原规划 v0.3 交付 **2D 不可压 NS 原型（SIMPLE）**，v0.4 方腔 Re=100。工程侧已具备：

- **3D 结构化网格**：`StructuredMesh3d`、`MultiBlockStructuredMesh3d`、CGNS 读入（[ADR 0008](0008-cgns-io.md)）；
- **1D 稳态扩散 FVM + 三对角求解**：`assemble_diffusion_1d`、`LinearSystem::solve_tridiagonal` — 压力 Poisson 离散模板；
- **`linalg` CSR + CG/GMRES + ILU(0)**：2D/3D Poisson 线性求解可复用骨架；
- **边界数据层**：`BoundaryKind::Wall/Inlet/Outlet/Symmetry`（当前语义偏可压，需不可压专用施加）。

可压缩路径（[ADR 0009](0009-compressible-navier-stokes.md)）以守恒变量 + Riemann 通量为主，**与不可压压力-速度耦合算法正交**，不宜从可压代码「降 Ma 扩展」得到生产级不可压求解器。

本 ADR 定案：

1. **首版直接三维**（结构化六面体；2D 算例通过 `nz = 1` 或退化验证，不单独维护 2D 求解器 fork）；
2. **双算法**：**SIMPLEC**（稳态 / 伪瞬态默认）与 **PISO**（瞬态默认）；
3. **不采纳**首版 SIMPLE（非 Consistent 形式）：SIMPLEC 收敛更快、压力欠松弛需求更低，实现增量相对 SIMPLE 可忽略。

## 决策

### 1. 控制方程与主变量

常密度、等温、层流 Newtonian 流体（首版）：

\[
\nabla\cdot\mathbf{u} = 0 \tag{1}
\]

\[
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\mathbf{u}) = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} \tag{2}
\]

| 项 | 定案 |
|----|------|
| 主存储 | **原始变量 SoA**：\(p\)、\(u,v,w\)（cell-centered） |
| 密度 | 常数 \(\rho\)（`physics::IncompressibleFluid`） |
| 粘性 | \(\nu = \mu/\rho\) 常数（首版）；变 \(\mu(T)\) 后续扩展 |
| 能量 | 首版 **不解**；等温假设 |
| 湍流 | 层流首版；RANS（\(k,\omega\)）经 [ADR 0014](0014-turbulence-k-omega-sst-rans.md) **I6+ 评估**，非阻塞项 |

类型契约（`field` / `physics`）：

```rust
/// 不可压 NS 主状态（cell-centered SoA）。
pub struct IncompressibleFields {
    pub pressure: ScalarField,      // p
    pub velocity_x: ScalarField,    // u
    pub velocity_y: ScalarField,    // v
    pub velocity_z: ScalarField,    // w
}

pub struct IncompressibleFluid {
    pub density: Real,
    pub kinematic_viscosity: Real,  // ν
}
```

**禁止**以 \(\rho\mathbf{u}\) 或守恒变量作为不可压时间推进主状态（与 ADR 0009 对称、相反）。

### 2. 空间离散：collocated FVM + 通量格式

| 项 | 定案 |
|----|------|
| 方法 | 有限体积法（FVM），与 ADR 0002 一致 |
| 布局 | **同位（collocated）** 压力与速度共 cell center |
| 网格 | **结构化 3D 六面体** `StructuredMesh3d` / `MultiBlockStructuredMesh3d` 首版 |

**不采纳**首版 staggered（MAC）网格：与现有 `ScalarField` SoA 与 VTK 场输出一致性好，Rhie-Chow 为社区标准 collocated 方案（Ferziger et al. Ch. 9；Versteeg & Malalasekera Ch. 8）。

**贴体/曲线网格**：复用 `MeshMetricMode::Curvilinear` 与 `MetricCache3d` 面法向/面积（与可压 3D 共享几何预计算）；Rhie-Chow 距离 \(d_f\) 取 owner 中心至面心距离。

#### 2.1 面通量分解

动量方程 FVM 通过面 \(f\) 的净通量进入 owner/neighbor：

\[
\Phi_f = \Phi_f^{\mathrm{conv}} + \Phi_f^{\mathrm{visc}} - V_P \left(\frac{\partial p}{\partial n}\right)_f \delta_{f,P}
\]

| 通量项 | 离散 | 模块 |
|--------|------|------|
| 质量通量 \(\dot{m}_f = \rho\,\mathbf{u}_f\cdot\mathbf{S}_f\) | **Rhie-Chow**（§2.2） | `rhie_chow.rs` |
| 对流 \(\Phi^{\mathrm{conv}} = \dot{m}_f\,\phi_f\) | **ConvectionScheme**（§2.3） | `convection.rs` |
| 扩散 \(\Phi^{\mathrm{visc}} = -\rho\nu (\nabla\phi)_f\cdot\mathbf{S}_f\) | 中心差分（§2.4） | `diffusion.rs` |
| 压力梯度 | 面心差分，**不**经 Rhie-Chow（动量装配内） | `momentum.rs` |

**禁止**对不可压对流使用 Riemann / HLLC（[ADR 0009](0009-compressible-navier-stokes.md) 可压专用）；`FluxScheme` trait 在不可压路径 **不复用**。

#### 2.2 Rhie-Chow 面速度（质量通量）

面 \(f\)（owner \(O\)，neighbor \(N\)；边界仅 \(O\)）：

\[
\mathbf{u}_f = \overline{\mathbf{u}}_f - \overline{\mathbf{D}}_f\left(\overline{\nabla p}_f - \frac{p_N - p_O}{|\mathbf{x}_N-\mathbf{x}_O|}\,\mathbf{e}_{ON}\right) \tag{R1}
\]

\[
\dot{m}_f = \rho\,\mathbf{u}_f\cdot\mathbf{S}_f \tag{R2}
\]

| 项 | 定案 |
|----|------|
| \(\overline{\mathbf{u}}_f,\overline{\mathbf{D}}_f\) | 线性插值（结构化：面心权重） |
| \(\overline{\nabla p}_f\) | Green-Gauss 或中心差分（与 `gradient.rs` 一致） |
| 边界 | \(\mathbf{u}_f\) 由 BC ghost 速度 + 壁面 \(\dot{m}_f=0\)（无滑移） |
| 单元测试 | 均匀 \(\mathbf{u}\)、\(\nabla p=0\) → \(\mathbf{u}_f=\mathbf{u}\)，\(\dot{m}_f\) 与解析一致 |

Rhie-Chow **仅**用于 \(\dot{m}_f\) 与压力修正方程源项 \(\nabla\cdot(\rho\mathbf{u}^*)\)；动量方程中的 **压力梯度项** 仍用 cell-centered \(p\) 面差分，避免双重修正。

#### 2.3 对流通量格式（`ConvectionScheme`）

动量分量 \(\phi\in\{u,v,w\}\) 的对流项 \(\nabla\cdot(\mathbf{u}\phi)\) 离散为 \(\sum_f \dot{m}_f \phi_f\)。面值 \(\phi_f\) 由 **质量通量方向** 决定：

\[
\phi_f =
\begin{cases}
\phi_O, & \dot{m}_f \ge 0 \quad\text{（upwind-from-owner）} \\
\phi_N, & \dot{m}_f < 0
\end{cases}
\tag{C1}
\]

| 格式 | Case 枚举 | 阶段 | 说明 |
|------|-----------|------|------|
| **一阶 upwind** | `upwind` | **I1–I4 默认** | 稳定、单调；Re 高时耗散大 |
| **中心差分** | `central` | I4+（调试） | 无耗散；**仅**低 Re 或调试，生产默认关闭 |
| **MINMOD 线性** | `minmod` | I6 | 二阶 TVD；需 \(\phi_f=\phi_O+\frac{1}{2}\psi(\nabla\phi)\cdot(\mathbf{x}_f-\mathbf{x}_O)\) |
| **QUICK** | `quick` | I6 | 三阶；非结构推迟 |

**扩展点**（enum dispatch，非 trait object）：

```rust
pub enum ConvectionScheme {
    Upwind,
    Central,
    Minmod,
    Quick,
}
```

高阶格式（`minmod` / `quick`）在 **边界面临近单元** 降阶为 upwind（缺少完整 stencil 时）。结构化 3D 使用逻辑邻居 stencil；与可压 `SlopeLimiter` **独立** 实现（标量动量分量，非守恒变量限制器）。

**Parse → Validate**：`Re > 10_000` 且 `convection = "central"` → case 加载失败或警告（实现期定案为 **hard error**）。

#### 2.4 扩散通量

Newtonian 粘性，常 \(\nu\) 首版：

\[
\Phi_f^{\mathrm{visc}} = -\rho\nu \left(\nabla \phi\right)_f \cdot \mathbf{S}_f \tag{D1}
\]

\[
(\nabla\phi)_f = \frac{\phi_N - \phi_O}{|\mathbf{x}_N - \mathbf{x}_O|}\,\mathbf{e}_{ON} \quad\text{（正交网格）} \tag{D2}
\]

| 项 | 定案 |
|----|------|
| 内面 | 中心差分 (D2)；曲线网格用法向投影梯度（`MetricCache3d`） |
| 壁面 | ghost \(\phi_g\) 满足 Dirichlet / Neumann（§6） |
| 交叉导数 | 首版 **省略** \(\partial^2 u/\partial x\partial y\) 等交叉项（Cartesian 均匀网格 exact）；贴体 I6 补全完整应力张量离散 |

#### 2.5 压力梯度（动量方程）

单元 \(P\) 上压力梯度进入动量源项（非 Rhie-Chow 路径）：

\[
(\partial p / \partial x)_P \approx \frac{p_E - p_W}{\Delta x_P}, \quad \text{etc.} \tag{G1}
\]

边界单元：壁面/对称面 **零梯度 Neumann**；入口 **零梯度**；出口 **Dirichlet** \(p=p_b\)（若指定）。

### 3. 压力-速度耦合算法

#### 3.1 共用动量离散骨架

对单元 \(P\)，动量方程离散（矢量分量独立装配）：

\[
a_P \mathbf{u}_P = \sum a_{nb}\mathbf{u}_{nb} + \mathbf{H}(\mathbf{u}) - \mathbf{D}_P \nabla p
\]

其中 \(\mathbf{D}_P = V_P / a_P\)（SIMPLEC 中 \(a_P\) 定义见下），\(\mathbf{H}\) 为显式处理的邻点与源项组合。

#### 3.2 SIMPLEC（稳态 / 伪瞬态）

**用途**：`[time].mode = "steady"` 或 `"pseudo_transient"`（本地时间步加速收敛）。

**外层循环**（`solver::incompressible::SimplecDriver`）：

```
repeat until 收敛:
  1. 组装动量方程 → 以 α_u 欠松弛 Gauss-Seidel 求解 u*
  2. 计算 d_P = V_P / a_P^c   （SIMPLEC 一致系数，见 §3.4）
  3. 组装压力修正方程 ∇·(ρ d ∇p') = ∇·(ρ u*)
  4. 求解 p'（CG + ILU(0) 或 GS）
  5. p ← p + α_p p'；  u ← u* - d ∇p'  （α_p 通常 1.0，SIMPLEC 允许省略压力欠松弛）
  6. 检查 ||∇·u||、动量残差
```

#### 3.3 PISO（瞬态）

**用途**：`[time].mode = "transient"` 默认耦合算法。

**单时间步**（`solver::incompressible::PisoDriver`）：

```
1. 组装动量 → 求解 u*（含旧 p 梯度，或 PISO 分裂形式）
2. for k = 1 .. n_piso_correctors:
     a. 组装 ∇·(ρ d ∇p') = ∇·(ρ u*)   （u* 每步更新或不更新，见理论页 PISO-2）
     b. 求解 p'
     c. u ← u* - d ∇p'；  p ← p + p'
     d. 更新 u* 用于下一步 corrector（n ≥ 2 时）
3. 物理时间 t ← t + Δt
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `n_piso_correctors` | `2` | 工业常用 2–3；Re 高 / 大 Δt 可增至 3 |
| 动量欠松弛 | **无**（α_u = 1） | 瞬态 PISO 标准形式 |
| 压力欠松弛 | **无** | 与 SIMPLEC 对比的关键差异 |

**稳态算例若选 PISO**：允许外层伪时间迭代包裹 PISO 步（`max_outer_iterations`），但 **默认仍推荐 SIMPLEC**。

#### 3.4 SIMPLEC 一致系数

SIMPLE 使用 \(d_P = V_P / a_P\)（\(a_P\) 含欠松弛）。SIMPLEC 采用：

\[
a_P^c = a_P - \sum a_{nb} \quad (\text{或等价 consistent 形式}), \qquad d_P = \frac{V_P}{a_P^c}
\]

实现须在 `discretization::incompressible::momentum` 导出 **两套** 对角元：`a_P`（GS 求解）与 `a_P_c`（压力修正），并在理论页 `[incompressible_simplec_piso.md](../theory/incompressible_simplec_piso.md)` 锁定与 Patankar (1980) / Ferziger et al. (2020) 一致的符号。

**不实现**经典 SIMPLE 作为生产路径；单元测试可保留 SIMPLE 1D 类比用于回归对比。

### 4. 模块职责（遵守 ADR 0002 分层）

```
field/incompressible.rs           # IncompressibleFields
physics/incompressible.rs         # IncompressibleFluid, ν, ρ
discretization/incompressible/
├── momentum.rs                   # 动量 FVM 装配、H(u)、a_P、a_P^c
├── pressure_correction.rs        # ∇·(ρ d ∇p') 装配（对称 SPD）
├── rhie_chow.rs                  # 面速度 u_f、质量通量 ρ u·S
├── convection.rs                 # upwind / 高阶（I4+）
├── diffusion.rs                  # ν∇²u 扩散项
└── bc.rs                         # 不可压 BC ghost / 通量修正
solver/incompressible/
├── mod.rs                        # IncompressibleNsSolver 编排
├── simplec.rs                    # SIMPLEC 外层循环
├── piso.rs                       # PISO 时间步
└── linear.rs                     # 动量 GS / 压力 CG 调度
linalg/                           # 复用 CsrMatrix、CG、ILU(0)、（可选）多重网格
case/incompressible_3d.rs         # case.toml dispatch
```

| 模块 | 负责 | 不负责 |
|------|------|--------|
| `discretization/incompressible` | 动量/压力方程系数、Rhie-Chow、BC 数值施加 | 外层 SIMPLEC/PISO 循环 |
| `solver/incompressible` | 算法编排、欠松弛、收敛判据、TimeIntegrator 对接 | 通量公式细节 |
| `physics` | \(\rho, \nu\) 常数 | 网格遍历 |
| `io` | 解析 `solver.type`、BC、物性 | 压力修正装配 |

**扩展点**（enum dispatch 优先，trait 用于测试 mock）：

```rust
pub enum PressureVelocityCoupling {
    SimpleC(SimplecConfig),
    Piso(PisoConfig),
}
```

**禁止**（与 ADR 0009 §10 对称）：

- 在 `solver` 内写 Rhie-Chow 或 Poisson 五点/七点模板公式；
- `io` 解析阶段假设 SIMPLEC/PISO 或设置欠松弛；
- 用可压 `bc_compressible` ghost 直接套不可压入口；
- 首版非结构 + PISO + 湍流 **三合一**（分阶段，见 §8）。

### 5. 线性求解器

| 子问题 | 矩阵性质 | 首版求解器 | 备注 |
|--------|----------|------------|------|
| 动量 \(u,v,w\) | 非对称、对角占优 | **Gauss-Seidel**（分分量） | 每 SIMPLEC/PISO 步 1–若干次扫描 |
| 压力修正 \(p'\) | 对称弱对角占优（Poisson） | **CG + ILU(0)**（`linalg` CSR） | 结构化 7 点模板；奇异时固定 \(p_\mathrm{ref}\) |
| 大规模 / 病态 | 同上 | **代数多重网格 AMG** | I5+ 评估；非阻塞 |

压力方程参考点：全域 Neumann 型边界时，在 **一个** cell 强施加 \(p = p_\mathrm{ref}\)（Case 可配置 `pressure_reference` patch 或 cell index）。

**复用**：1D 扩散 BC ghost 模式 → 压力修正方程边界行；`LinearSystem::add_coupling` 思路推广至 3D CSR 装配。

### 6. 边界条件

沿用 `BoundaryRegistry` 两阶段模式（registry → handler → apply）；**不可压专用 handler** 位于 `discretization/incompressible/bc.rs`，与可压 `bc_compressible` **分文件**。

#### 6.1 数据模型（`boundary::kind`）

新增 / 扩展（`solver.type = "incompressible_ns"` 时解析）：

```rust
pub enum BoundaryKind {
    // ... 扩散 / 可压 variant 保持不变 ...

    /// 不可压速度入口（Case: kind = "inlet" + velocity = [...]）
    IncompressibleInlet {
        velocity: [Real; 3],
    },
    /// 不可压压力出口（Case: kind = "outlet" + static_pressure）
    IncompressibleOutlet {
        pressure: Real,
        /// I5+：对流出口零梯度 vs 固定 p
        mode: OutletMode,  // FixedPressure | ZeroGradientVelocity
    },
    /// 壁面（可压/不可压共用；不可压忽略 heat 或仅 adiabatic）
    Wall {
        no_slip: bool,
        wall_velocity: Option<[Real; 3]>,  // 动壁（方腔顶盖）
        heat: WallHeat,                    // 不可压首版忽略
    },
    Symmetry,
    Periodic { partner: String },
    /// 固定压力参考（单 cell / 单 patch，Poisson 定解）
    PressureReference { pressure: Real },
}
```

**Case 分流**：同一 TOML `kind = "inlet"` 在 `compressible_ns` 下解析 `total_pressure/total_temperature`；在 `incompressible_ns` 下 **必须** 提供 `velocity`，否则 Validate 失败。

#### 6.2 边界类型与方程分工

| 类型 | 动量 \(\mathbf{u}\) | 压力 \(p\) | 压力修正 \(p'\) | Rhie-Chow \(\dot{m}_f\) |
|------|---------------------|------------|-----------------|-------------------------|
| **无滑移壁** `Wall { no_slip: true, wall_velocity: None }` | \(\mathbf{u}_g = -\mathbf{u}_o\)（ghost） | \(\partial p/\partial n = 0\) | \(\partial p'/\partial n = 0\) | \(\dot{m}_f = 0\) |
| **动壁** `wall_velocity = U_w` | \(\mathbf{u}_g = 2U_w - \mathbf{u}_o\) | Neumann | Neumann | \(\dot{m}_f = \rho U_w\cdot\mathbf{S}_f\) |
| **滑移壁** `no_slip: false` | \(u_{n,g}=-u_{n,o}\)，\(u_{t,g}=u_{t,o}\) | Neumann | Neumann | \(u_n=0\) |
| **速度入口** `IncompressibleInlet` | \(\mathbf{u}_g = 2\mathbf{u}_b - \mathbf{u}_o\) | \(\partial p/\partial n = 0\) | Neumann | upwind \(\phi_f=\mathbf{u}_b\) |
| **压力出口** `IncompressibleOutlet` | \(\partial \mathbf{u}/\partial n = 0\)（零梯度） | \(p = p_b\)（Dirichlet） | \(p' = 0\)（齐次 Dirichlet） | upwind 外推 owner |
| **对称面** `Symmetry` | \(u_n=0\)，\(\partial u_t/\partial n=0\) | Neumann | Neumann | \(u_n=0\) |
| **周期** `Periodic` | partner patch 场互换 | 周期一致 | 周期一致 | 同内部面 |
| **压力参考** `PressureReference` | — | 单点 \(p=p_{\mathrm{ref}}\) | 单点 \(p'=0\) | — |

#### 6.3 Ghost 单元公式（动量 / 扩散）

距 owner 中心法向距离 \(d_f\)（至 ghost 中心），速度分量 \(\phi\)：

**Dirichlet（入口 / 动壁）**：

\[
\phi_g = 2\phi_b - \phi_o \tag{B1}
\]

**Neumann（壁面压力 / 出口速度 / 对称）**：

\[
\phi_g = \phi_o + d_f \left(\frac{\partial \phi}{\partial n}\right)_b \tag{B2}
\]

零梯度：\((\partial\phi/\partial n)_b = 0 \Rightarrow \phi_g = \phi_o\)。

扩散通量 (D1) 与动量对流 upwind 均消费 ghost 值；**压力修正 Poisson** 在 Dirichlet 面强施加 \(p'\) 行（出口 \(p'=0\)；参考点 \(p'=0\)）。

#### 6.4 施加顺序

```
1. 刷新边界 ghost 缓冲（u, v, w, p, p'）
2. 计算 Rhie-Chow m_dot（含 BC）
3. 装配动量对流/扩散 + 压力梯度源
4. SIMPLEC/PISO 内：装配压力 Poisson + BC
5. 修正 u, p 后下一迭代/时间步
```

Patch 顺序：按 `BoundarySet` **声明顺序**（与 [boundary_conditions.md](../theory/boundary_conditions.md) §4 一致）。重叠 patch **Validate 阶段拒绝**。

#### 6.5 CGNS / 多块映射

| CGNS `BCType_t` | 不可压默认映射 |
|-----------------|----------------|
| `BCWall*` / `BCWallInviscid` | `Wall { no_slip: true }` |
| `BCInflow` / `BCInflowSubsonic` | `IncompressibleInlet`（需 Case 覆盖 `velocity`） |
| `BCOutflow` / `BCOutflowSubsonic` | `IncompressibleOutlet` |
| `BCSymmetryPlane` | `Symmetry` |
| `BCFarfield` | **不支持**（不可压无 farfield Riemann）；加载失败并提示改 inlet/outlet |

多块接口面：**内部面**，无 BC；\(\mathbf{u},p\) 连续（首版块间 conformal，非 conformal 单独 ADR）。

#### 6.6 验证要求

每种 BC 至少一个单元测试：均匀场 + 单 patch 不破坏常值解；方腔四壁 + 动顶盖 golden；入口-出口质量守恒 \(|\sum \dot{m}_f|/|\sum \dot{m}_{\mathrm{in}}| < 10^{-6}\)（I4 benchmark）。

### 7. 时间积分与 [ADR 0005](0005-time-integration.md) 对接

不可压 **无** 声速 CFL；时间推进由 **对流 CFL** + **扩散 Fourier 数** 约束，外层由 SIMPLEC（稳态）或 PISO（瞬态）完成压力-速度耦合。

#### 7.1 模式矩阵

| `[time].mode` | 压力-速度耦合 | 时间推进实现 | `TimeIntegrator` |
|---------------|---------------|--------------|------------------|
| `steady` | **SIMPLEC** 外层直到收敛 | 无物理 \(\Delta t\)；迭代计数 | `SteadyStateIntegrator` + `SimplecDriver` |
| `pseudo_transient` | **SIMPLEC** | 局部 \(\Delta t_i\) 加速（Ferziger §7.3） | `PseudoTransientIntegrator`（扩展 ADR 0005） |
| `transient` | **PISO** 每步 | 固定或自适应 \(\Delta t\) | `Bdf1Integrator` 或 `ExplicitEulerIntegrator` + `PisoDriver` |

**默认组合**（Validate 可自动修正并 `tracing::warn`）：

| mode | 默认 `coupling` |
|------|-----------------|
| `steady` | `simplec` |
| `pseudo_transient` | `simplec` |
| `transient` | `piso` |

#### 7.2 时间步长约束

**对流 CFL**（向量形式，单元 \(P\)）：

\[
\Delta t \le \mathrm{CFL} \cdot \min_P \frac{V_P^{1/3}}{|\mathbf{u}|_P + \varepsilon} \tag{T1}
\]

**扩散 Fourier 数**（均匀网格 \(\Delta x_P = \min(\Delta x,\Delta y,\Delta z)\)）：

\[
\Delta t \le \mathrm{CFL}_\nu \cdot \min_P \frac{(\Delta x_P)^2}{\nu} \tag{T2}
\]

\[
\Delta t = \min(\Delta t_{\mathrm{conv}}, \Delta t_{\mathrm{visc}}) \tag{T3}
\]

| 参数 | 默认 | 说明 |
|------|------|------|
| `cfl_max` | `0.5` | 对流 CFL（瞬态 / 伪瞬态） |
| `cfl_visc` | `0.25` | 扩散 Fourier 上限 |
| `dt` | — | 若显式给定且 `<` CFL 限制，**以用户为准**；否则 `suggested_dt` |

**伪瞬态局部时间步**（`pseudo_transient`）：

\[
\Delta t_P = \mathrm{CFL}_{\mathrm{pseudo}} \cdot \frac{V_P^{1/3}}{|\mathbf{u}|_P + \varepsilon} \tag{T4}
\]

每 SIMPLEC 外层迭代用 \(\Delta t_P\) 缩放动量对角（局部收敛加速）；**不**推进物理时间 \(t\)（manifest 中 `time` 仍单调递增迭代索引或保持 0，实现期在 manifest 区分 `pseudo_time`）。

**不采用**可压 Blazek \(\Lambda^c + \Lambda^v\) 公式（含声速 \(a\)）。

#### 7.3 瞬态 PISO 时间离散

**默认**：**一阶向后 Euler（BDF1）** 动量半离散 + PISO 压力修正（Issa 1986 算子分裂）。

动量 BDF1（单元 \(P\)）：

\[
\frac{\rho V_P}{\Delta t}(\mathbf{u}_P - \mathbf{u}_P^n) + \sum_f \dot{m}_f \mathbf{u}_f = -\nabla p + \mu\nabla^2\mathbf{u} \tag{T5}
\]

**PISO 步内顺序**（`PisoDriver::advance`）：

```
U^n, p^n 已知
  → 组装 BDF1 动量 → GS 得 u*
  → for k in 1..n_piso_correctors:
        组装 Poisson(p') 源 = div(rho u*)
        解 p' → 更新 u, p（无 α_p）
  → U^{n+1} = u, p^{n+1} = p
  → t += dt
```

| 配置 | 默认 | 阶段 |
|------|------|------|
| `time.scheme = "bdf1"` | **是** | I3 首版 |
| `time.scheme = "euler"` | 显式欧拉动量（调试） | I3 |
| `time.scheme = "bdf2"` | 二阶 BDF | I6+ 评估 |
| `n_piso_correctors` | `2` | I3 |
| `n_orthogonality_correctors` | `0` | I6+（非正交网格） |

**BDF2 / Crank-Nicolson** 非 I0–I5 承诺；需历史步存储与启动步 BDF1。

#### 7.4 稳态 SIMPLEC 收敛与残差

外层迭代（非物理时间步）监控：

| 残差 | 定义 | 收敛阈值（默认） |
|------|------|------------------|
| 连续性 | \(\|\nabla\cdot(\rho\mathbf{u}^*)\|_\infty / \rho U_{\mathrm{ref}}\) | `continuity_tolerance = 1e-5` |
| 动量 | \(\|\mathbf{R}_u\|_\infty / (\rho U_{\mathrm{ref}}^2)\)（分分量 max） | `momentum_tolerance = 1e-5` |

\(U_{\mathrm{ref}}\) 来自 Case `[physics.incompressible].reference_velocity` 或入口速度模。

**Run Manifest / metrics**（对接 [ADR 0005](0005-time-integration.md) §4）：

```json
{
  "time": { "mode": "steady", "coupling": "simplec", "outer_iteration": 42 },
  "residual": { "continuity": 1.2e-6, "momentum": 3.4e-6 },
  "log10_continuity": -5.92
}
```

瞬态 PISO 每步记录 `step`, `dt`, `cfl`, `n_piso_corrector` 内层 \(p'\) 残差。

#### 7.5 双时间步（可选，I5+）

稳态难收敛算例可启用 **双时间步**（Blazek §6.2 类比，**不含**声速项）：

\[
\frac{\partial \mathbf{u}}{\partial \tau} + \nabla\cdot(\mathbf{u}\mathbf{u}) = -\nabla p + \nu\nabla^2\mathbf{u}, \quad
\frac{\partial p}{\partial \tau} = -\rho c_p \nabla\cdot\mathbf{u}
\]

伪时间 \(\tau\) 与 SIMPLEC 外层合并评估；**非** I0–I4 默认路径。

#### 7.6 与可压 `TimeIntegrator` 关系

| 项 | 可压（ADR 0009） | 不可压（本 ADR） |
|----|------------------|------------------|
| 主循环 | RK4 / LU-SGS / GMRES on \(\mathbf{U}\) | SIMPLEC / PISO on \(\mathbf{u},p\) |
| CFL | \(\|\mathbf{u}\|+a\) | \(\|\mathbf{u}\|\), \(\nu\) |
| 模块 | `solver/time/rk4.rs` 等 | `solver/incompressible/` + 扩展 `solver/time/pseudo_transient.rs` |
| 共享 | `TimeMode`, `TimeStepInfo`, manifest 字段名 | enum 分支，**不**共用 RK4 stage 缓冲 |

CFL 模块复用 `solver::time` 框架（`CflSchedule`, `min_positive_dt`），**不**使用 `max_wave_speed` / 声速。

### 8. 网格与 I/O

| 项 | 定案 |
|----|------|
| 首版网格 | `StructuredMesh3d`；多块 `MultiBlockStructuredMesh3d`（I5） |
| 几何 | 预计算 `cell_volume`、面 `area`、`normal`、面心（`MetricCache3d`） |
| 读入 | CGNS 结构化 zone（[ADR 0008](0008-cgns-io.md)）；VTS 仅输出 |
| 2D 验证 | `nz = 1` 单层 3D 网格；**不**维护独立 `StructuredMesh2d` 求解器 |
| 非结构 3D | **I6+** 评估；须面拓扑 + Rhie-Chow 非结构形式，**不在 I0–I5 承诺** |

### 9. Case 配置

```toml
[solver]
type = "incompressible_ns"

[solver.incompressible]
coupling = "simplec"          # simplec | piso
convection = "upwind"         # upwind | central | minmod | quick
n_piso_correctors = 2
# n_orthogonality_correctors = 0   # I6+

[solver.incompressible.simplec]
max_outer_iterations = 500
momentum_relaxation = 0.7     # α_u
pressure_relaxation = 1.0     # α_p；SIMPLEC 默认 1.0
momentum_tolerance = 1.0e-5
continuity_tolerance = 1.0e-5

[physics.incompressible]
density = 1.0
kinematic_viscosity = 0.01    # ν；Re = U L / ν 由算例给定
reference_velocity = 1.0      # 残差归一化 U_ref

[time]
mode = "steady"               # steady | pseudo_transient | transient
scheme = "bdf1"                 # bdf1 | euler（transient）；steady 忽略
dt = 1.0e-3                   # transient / pseudo_transient
cfl_max = 0.5
cfl_visc = 0.25
cfl_pseudo = 5.0              # pseudo_transient 局部加速
max_steps = 100000

[solver.incompressible.pressure_linear]
solver = "cg"                 # cg | gauss_seidel（调试）
max_iterations = 200
tolerance = 1.0e-8
preconditioner = "ilu0"

[[boundary]]
name = "inlet"
kind = "inlet"
velocity = [1.0, 0.0, 0.0]

[[boundary]]
name = "outlet"
kind = "outlet"
static_pressure = 0.0         # gauge pressure

[[boundary]]
name = "walls"
kind = "wall"
no_slip = true
# wall_velocity = [0.0, 1.0, 0.0]   # 动壁（方腔顶盖）
```

**Parse → Validate**：

- `coupling = "piso"` 且 `time.mode = "steady"` → 警告并回退 SIMPLEC，或要求显式 `allow_piso_steady = true`；
- `transient` 且 `coupling = "simplec"` → 允许（伪瞬态），但日志提示 PISO 更适瞬态；
- 出口 `static_pressure` 与参考点二选一，避免双重约束。

`case` 层通过 `solver.type` 在 `IncompressibleNavierStokes` 与 `CompressibleNavierStokes` 间 dispatch（与 ADR 0009 §9 一致）。

### 10. 数值基线（首版不可压 3D）

| 项 | 选择 |
|----|------|
| 耦合 | SIMPLEC（稳态）+ PISO-2（瞬态 BDF1） |
| 布局 | Collocated + Rhie-Chow |
| 空间 | FVM，结构化六面体 |
| 对流 | 一阶 upwind（默认）；MINMOD/QUICK @ I6 |
| 扩散 | 中心差分 |
| 边界 | Ghost 单元；入口速度 / 出口压力 / 壁面无滑移 |
| 瞬态 | BDF1 + PISO-2；CFL 对流 + Fourier 扩散 |
| 稳态 | SIMPLEC；可选 pseudo_transient 局部 Δt |
| 物性 | 常 \(\rho\)、常 \(\nu\) |
| 动量求解 | Gauss-Seidel |
| 压力求解 | CG + ILU(0) |
| 并行 | 单线程验证（I0–I5）；I6 评估 `rayon` 单元/面着色 |
| 湍流 | 层流；RANS 见 ADR 0014 **后续** |

### 11. 分阶段交付（I0–I6）

| 阶段 | 交付 | 网格 | 验证 |
|:----:|------|------|------|
| **I0** | 本 ADR + [incompressible_simplec_piso.md](../theory/incompressible_simplec_piso.md) + DATA_MODEL / CASE_FORMAT 草案 | — | — |
| **I1** | Stokes 3D（无对流）：动量扩散 + 压力 Poisson 链；Rhie-Chow 均匀场零修正单测 | 均匀 `StructuredMesh3d` | 3D Poiseuille / Couette（\(n_z=1\) 退化 2D 亦可） |
| **I2** | **SIMPLEC** 稳态完整链；`Wall` + 参考压力 | 均匀块 | 顶盖方腔 Re=100（Ghia 2D 数据；3D 顶盖 \(n_z\ge 8\) 或 \(n_z=1\) 对比） |
| **I3** | **PISO** 瞬态；`TimeIntegrator` transient；`n_piso_correctors` 可配 | 均匀块 | Taylor–Green 3D 涡衰减；或 2D 方腔启动流 \(n_z=1\) |
| **I4** | `IncompressibleInlet/Outlet`；一阶 upwind 稳定通道流 | 均匀 / 拉伸 | 3D 管道 Poiseuille \(Re=100\) 质量守恒 |
| **I5** | `MultiBlockStructuredMesh3d` + CGNS BC；VTU/VTS 写出 \(\mathbf{u},p\) | 多块 | 多块通道 smoke；manifest 回归 |
| **I6** | 贴体 `MetricCache3d`；可选 QUICK；`parallel-fvm` 评估（动量/压力独立） | 曲线块 | S 弯管层流（低 Re） |

**POC 不以加速比为合入条件**；每阶段须单元测试 + `tests/benchmarks/` 至少一项通过方可进入下一阶段。

### 12. V&V 算例（`tests/benchmarks/`）

| ID | 阶段 | 验证量 |
|----|:----:|--------|
| `poiseuille_3d` | I1 | 速度剖面、压降 |
| `lid_cavity_re100` | I2 | \(u,v\) 中心线 vs Ghia (1982) |
| `taylor_green_3d` | I3 | 动能衰减率 |
| `channel_re100_3d` | I4 | 流量、压降、质量守恒 |
| `multiblock_channel` | I5 | 接口连续、manifest |

登记见 [BENCHMARKS.md](../BENCHMARKS.md)（实现时追加）。

### 13. 与可压 / 湍流路线关系

| 关系 | 说明 |
|------|------|
| vs [ADR 0009](0009-compressible-navier-stokes.md) | **并行独立**；共享 `mesh` 几何、`boundary` 框架、`linalg`、case dispatch |
| vs [ADR 0014](0014-turbulence-k-omega-sst-rans.md) | 可压 RANS 首版；不可压 \(\nu_t\) 耦合 **I6+ 单独里程碑**，复用 `face_transport_coefficients` 思路 |
| vs [ADR 0010](0010-unstructured-mixed-mesh.md) | 非结构不可压 **晚于** 结构 3D SIMPLEC/PISO 验证 |
| ARCHITECTURE §10 | v0.3 数值能力修订为 **3D 不可压 SIMPLEC/PISO 原型**（取代原「2D SIMPLE」表述） |

### 14. 架构反模式（禁止）

- 在 `solver` 内实现 Poisson 七点模板或 Rhie-Chow 公式
- 用可压 HLLC 通量处理不可压对流
- 跳过 Rhie-Chow 直接 collocated 中心插值（必然棋盘格）
- 首版同时交付非结构 + PISO + 湍流
- 全局 `static mut` 缓存 SIMPLEC 迭代状态
- 生产路径 `unwrap` 压力 CG 不收敛

## 后果

### 正面

- 直接对齐 3D 工程网格（CGNS / 多块），避免 2D→3D 二次重构
- SIMPLEC + PISO 覆盖稳态与瞬态主流需求，算法选择清晰
- Collocated + 现有 SoA `ScalarField` 一致，VTK 输出简单
- 扩散/Poisson/`linalg` 链路可复用 v0.2 成果
- 与可压路径模块边界对称，便于 `case` 统一编排

### 负面

- Rhie-Chow + 3D CSR Poisson 实现与调试工作量大
- Collocated 高 Re 需更细网格或高阶格式，首版 upwind 耗散偏大
- CG 压力求解在病态网格上可能慢，I5 前需固定 \(p_\mathrm{ref}\) 与 ILU 调参
- 非结构 Rhie-Chow 推迟至 I6+，短期不支持混合网格不可压
- `BoundaryKind` 可压/不可压并存，Case 校验必须严格

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 2D SIMPLE 首版再升 3D | 与 CGNS 3D 资产重复建设；\(n_z=1\) 可验证 2D |
| 经典 SIMPLE（非 C） | SIMPLEC 收敛更好；实现成本几乎相同 |
| 首版仅 PISO | 稳态工业算例 SIMPLEC 更省算力 |
| 首版仅 SIMPLEC | 瞬态方腔/涡街需 PISO 或多步投影 |
| Staggered MAC 网格 | 与现有 field/IO 不一致；3D 索引复杂 |
| 可压低 Ma 代替不可压 | 密度耦合、时间步、BC 均不匹配 |
| 首版非结构 collocated | Rhie-Chow 与 Poisson 在非结构更复杂；ADR 0010 后评估 |
| 压力方程直接 AMG 首版 | 实现成本高；CG+ILU 足够 I0–I4 |
| 可压 Riemann BC 套不可压 | 无意义；入口/出口语义不同 |
| 瞬态 Crank-Nicolson 首版 | 与 PISO 分裂耦合复杂；BDF1 先验证 |
| 中心差分对流作默认 | 高 Re 振荡；upwind 作生产默认 |

## 修订记录

| 日期 | 内容 |
|------|------|
| 2026-06-08 | 初版：定案 3D collocated FVM、SIMPLEC + PISO、模块边界、I0–I6 |
| 2026-06-08 | 补充 §2 通量格式、§6 边界条件、§7 时间积分；Case 扩展 `convection` / `time.scheme` |

修订时 **不删除** 已有条目；变更算法族（如 PRIME / PIMPLE）或网格布局须新开 ADR 或修订 §3/§8。
