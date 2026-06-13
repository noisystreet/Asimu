# asimu Case 文件格式（v0.2）

> 数据模型背景：[DATA_MODEL.md](DATA_MODEL.md) §8、§9 · I/O 实现：`src/io/`（v0.2 起逐步落地）

## 1. 概述

| 项 | v0.2 约定 |
|----|-----------|
| 格式 | **TOML**（`.toml` 或 `case.toml`） |
| 编码 | UTF-8 |
| v0.1 遗留 | `name=...;cells=...` 单行格式仍可读，**新算例请用 TOML** |

解析流程：**Parse → Validate → Trust**（见 AGENTS.md）。校验在 `io` 层完成；数值热路径信任已验证结构。

---

## 2. 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 算例名称 |
| `benchmark_id` | string | 否 | 对应 `tests/benchmarks/{id}/`；写入 Run Manifest（v0.3+） |
| `mesh` | table | 是 | 网格描述（§3） |
| `physics` | table | 是 | 物性（§4） |
| `boundary` | table | 是 | 边界条件（§5） |
| `initial` | table | 否 | 初始条件（§5.5）；缺省为全零 |
| `numerics` | table | 否 | 核心计算精度（§4.5，ADR 0016）；默认 `compute_precision = "f64"` |
| `time` | table | 否 | 时间推进（§6）；默认 `mode = "steady"` |

---

## 3. `[mesh]`

v0.2 首版支持 **1D 结构化均匀网格**；2D 在 v0.2.x 后续 PR 扩展，字段预留如下。

### 3.1 1D（v0.2 必实现）

```toml
[mesh]
kind = "structured_1d"
cells = 32
length = 1.0          # 域长度 [m]，均匀划分
origin = 0.0          # 可选，默认 0.0
```

| 字段 | 类型 | 约束 |
|------|------|------|
| `kind` | string | 必须为 `structured_1d` |
| `cells` | integer | ≥ 1，≤ `io.limits.max_cells`（规划） |
| `length` | float | > 0 |
| `origin` | float | 可选 |

### 3.2 2D（v0.2.x 规划）

```toml
[mesh]
kind = "structured_2d"
nx = 32
ny = 32
lx = 1.0
ly = 1.0
```

### 3.3 外部 VTK VTS（v0.3.x，二进制 appended）

```toml
[mesh]
kind = "vts"
path = "mesh/cavity.vts"
```

| 项 | 约定 |
|----|------|
| 格式 | VTK XML StructuredGrid，**仅** `format="appended"` + base64 `AppendedData` |
| 不支持 | ASCII、inline binary、多 Piece |
| Feature | 库集成须启用 `io-vtk` |
| API | `asimu::io::load_vts(&path)` → `StructuredMesh`（2D/3D） |
| 写出 | `asimu::io::write_vts(&mesh, &path)`（Float64 appended，未压缩） |

详见 [adr/0007-vts-binary-io.md](adr/0007-vts-binary-io.md)。

### 3.4 多块结构化 3D（首版：读入 / 诊断）

```toml
[mesh]
kind = "multi_block_structured_3d"
metric = "cartesian" # 可选：cartesian | curvilinear
scale = 0.001        # 可选：统一缩放全部 block 坐标

[[mesh.blocks]]
name = "inlet"
nx = 16
ny = 8
nz = 4
lx = 1.0
ly = 0.5
lz = 0.25

[[mesh.blocks]]
name = "outlet"
nx = 16
ny = 8
nz = 4
lx = 1.0
ly = 0.5
lz = 0.25
```

| 项 | 约定 |
|----|------|
| 支持 | 多个均匀 `StructuredMesh3d` block，block 名称必须唯一 |
| 求解 | 与单块 3D 共用 `BlockRunState` 推进框架；`structured_3d` 与单 zone CGNS 读入为 1-block `MultiBlockStructured3d`；无接口 block 可使用 GMRES/RK4/LU-SGS sweep，有 1-to-1 接口时要求 LU-SGS 对角隐式（**case 解析阶段校验**） |
| 统计 | `CaseMesh::num_cells()` 返回所有 block 单元总数 |
| 诊断 | `mesh_check` 可做逐 block 几何预检与整体范围统计 |
| 支持 | `[boundary]` 可按 `block_name/patch` 绑定到单个 block；1-to-1 接口通量可在 LU-SGS 多块路径中守恒装配；`[restart]` 支持 version=2 多块 TOML 初场 |
| 暂不支持 | 非 1-to-1 / overset / sliding 接口、合并 VTK 输出 |

### 3.5 外部 CGNS（features `io-cgns` + `io-vtk`）

```toml
[mesh]
kind = "cgns"
path = "mesh/wing.cgns"
scale = 0.001 # 可选：统一缩放全部 zone 坐标
```

| 项 | 约定 |
|----|------|
| 依赖 | 系统 `libcgns-dev`（`build.rs` 链接 `-lcgns`） |
| 支持 | Structured zone；单 zone 与多 zone 均读为 `MultiBlockStructured3d`（单 zone 为 1-block 容器）；ADF / HDF5 由 libcgns 处理 |
| 边界 | `ZoneBC` 自动读入；`FamilyName` 为 `IN` / `OUT` / `WALL` 时映射为入口 / 出口 / 壁面 |
| 求解 | 多 zone CGNS 可进入 3D 可压缩求解路径；当前按 block 同步推进，1-to-1 接口通过共享无粘通量守恒装配，最终 `solution_cgns` 写为单个多 Zone CGNS 文件；严格守恒多块路径要求 `time.scheme = "lu_sgs"` 且 `lusgs_sweep = false` |
| 导出 | `export_cgns_zone_to_vts` 或 `make cgns-to-vts IN=... OUT=...` |

详见 [adr/0008-cgns-io.md](adr/0008-cgns-io.md)。

---

## 4. `[physics]`

v0.2 稳态对流-扩散 / 纯扩散：

```toml
[physics]
diffusivity = 1.0     # 分子扩散系数 D
# velocity = [1.0, 0.0]   # v0.2.x 对流项启用后
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `diffusivity` | float | ≥ 0 |
| `velocity` | float 数组 | 可选；2D 为 `[ux, uy]` |

---

## 4.5 `[numerics]`

核心求解热路径精度（ADR [0016](adr/0016-runtime-compute-precision.md)）。网格几何、I/O 与配置数值仍使用 `f64`。

```toml
[numerics]
compute_precision = "f64"   # f64（默认）| f32
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `compute_precision` | string | `f64`（默认）或 `f32`。`f32` 在 Validate 阶段校验能力矩阵（ADR 0016）；未覆盖的组合会报错，不会静默回退 `f64` |

**`f32` 当前支持范围（P2–P5）**

| 项 | 支持 |
|----|------|
| 方程 / 网格 | 3D 可压缩 Euler；结构化单块或多块（无 1-to-1 接口）；非结构 |
| 空间离散 | `reconstruction = first_order`；非结构另支持 `muscl` + `unstructured_limiter` |
| 时间推进 | `euler`、`rk4`；`lu_sgs`（对角，须 `local_time_step = true`）；结构化另支持 `gmres` |
| 粘性 / NS | 非结构 3D：`[physics.viscous]` 与 `[navier_stokes]`（梯度/通量仍为 mixed `f64`） |
| 隐式限制 | 不支持 `lusgs_sweep = true`；非结构不支持 `gmres` |
| exec scatter（P5） | 非结构 typed 无粘/粘性内面：`parallel-fvm` 下 `f32` 走 `AtomicU32` CAS scatter（与 `f64` 同色桶契约）；无 `parallel-fvm` 时串行回退 |
| 其它 | 不支持 `residual_smoothing` |

网格几何、谱半径监控、I/O 与 manifest 仍使用 `f64`；`f32` 求解结果在写出前转换为 `f64`。

---

## 5. `[boundary]`

v0.2 支持 **Dirichlet** 与 **Neumann**。键名为逻辑边界名（`left` / `right` / `bottom` / `top`）。

```toml
[boundary.left]
kind = "dirichlet"
value = 0.0

[boundary.right]
kind = "dirichlet"
value = 1.0
```

| `kind` | 字段 | 说明 |
|--------|------|------|
| `dirichlet` | `value` | 固定值 |
| `neumann` | `flux` | 法向通量（扩散问题为 `-D ∂φ/∂n`） |

1D 默认映射：`left` → 首端面，`right` → 末端面。

含 `[euler]` 段时，所有 `wall` patch 自动改为**无粘滑移壁**（`no_slip = false`），即使 CGNS 映射为有滑移壁。可用同名 `[boundary.<patch>]` 覆盖 CGNS patch 的其他参数，但 `[euler]` 仍会在解析末将壁面设为滑移。

| `wall` 字段 | 说明 |
|-------------|------|
| `no_slip` | `true` 无滑移（\(\mathbf{u}_g=-\mathbf{u}_o\)，面心 \(\mathbf{u}=0\)）；`false` 滑移（\(u_{n,g}=-u_{n,o}\) 面心 \(u_n=0\)，切向 \(\mathbf{u}_{t,g}=\mathbf{u}_{t,o}\)） |
| `heat` | `adiabatic` / `isothermal` / `heat_flux`（须 `[navier_stokes]`） |
| `wall_temperature` | 等温壁温度 (K)，`heat = "isothermal"` |
| `heat_flux` | 进入流体的热流密度 (W/m²)，`heat = "heat_flux"` |

| `outlet` 字段 | 说明 |
|---------------|------|
| `static_pressure` | 亚声速出口静压；超声速出口可省略 |
| `supersonic` | `true` 时出口 ghost 全变量零梯度外推 owner；`false` 时施加 `static_pressure` |
| `mach` | 兼容字段；未写 `supersonic` 时 `mach >= 1` 视为超声速出口 |

不可压缩 `[incompressible]` 可使用以下结构化 3D 边界 kind（逻辑名支持 `i_min/i_max/j_min/j_max/k_min/k_max` 及别名 `left/right/bottom/top/front/back`）。当前 I1 skeleton 先施加到边界 owner 单元；后续 SIMPLEC/PISO 会下沉为 ghost/面通量。

```toml
[boundary.i_min]
kind = "velocity_inlet"
velocity = [1.0, 0.0, 0.0]       # m/s，解析后除以 U_ref

[boundary.i_max]
kind = "pressure_outlet"
pressure = 0.0                   # Pa，解析后除以 rho * U_ref^2

[boundary.j_max]
kind = "moving_wall"
velocity = [1.0, 0.0, 0.0]       # m/s，解析后除以 U_ref
```

| 不可压 `kind` | 字段 | 说明 |
|---------------|------|------|
| `wall` | `no_slip` | `true` 置 owner 速度为零；`false` 去除法向速度 |
| `moving_wall` | `velocity` | 指定动壁速度 |
| `velocity_inlet` | `velocity` | 指定入口速度 |
| `pressure_outlet` | `pressure` 或 `static_pressure` | 指定出口压力 |
| `symmetry` | 无 | 去除 owner 法向速度 |
| `periodic` | `partner` | 当前不可压缩 SIMPLEC 支持 `i_min`/`i_max` 成对周期 wrap |

周期充分发展方向示例：

```toml
[boundary.i_min]
kind = "periodic"
partner = "i_max"

[boundary.i_max]
kind = "periodic"
partner = "i_min"
```

---

## 5.5 `[initial]`（可选）

v0.2 支持标量场初始条件。键名为场名（如 `phi`）。未声明时，求解器以**全零**场作为初值。

```toml
[initial.phi]
kind = "uniform"
value = 0.0

[initial.phi]
kind = "linear"
left = 0.0
right = 1.0

[initial.phi]
kind = "values"
data = [0.0, 0.25, 0.5, 0.75]
```

| `kind` | 字段 | 说明 |
|--------|------|------|
| `uniform` | `value` | 常值 |
| `linear` | `left`, `right` | 沿域长线性插值（单元中心） |
| `values` | `data` | 逐单元数组，长度 = `mesh.cells` |

API：`CaseSpec::build_initial_fields()` / `initial_scalar("phi")`。

---

## 6. `[time]`

见 ADR 0005、[DATA_MODEL.md](DATA_MODEL.md) §11。

```toml
[time]
mode = "steady"       # steady | transient
# scheme = "rk4"      # rk4（默认）| euler | lu_sgs | gmres | simplec（不可压缩稳态）| piso（不可压缩瞬态）
# lusgs_omega = 1.0   # 可选；lu_sgs 松弛因子 ω∈(0,1]
# lusgs_sweep = false # 可选；false=阶段C对角隐式（默认），true=阶段D双扫
# lusgs_sweep_backward_damping = 0.5 # 可选；后扫耦合阻尼，建议 0.3–0.7
# gmres_preconditioner = "scalar_diagonal" # scalar_diagonal | cell_block_diagonal
# residual_smoothing = false
# residual_smoothing_epsilon = 0.5
# residual_smoothing_sweeps = 1
# dt = 1.0e-3
# cfl = 0.4
# final_time = 0.2
max_steps = 1000      # 时间推进步数上限（稳态伪时间 / 瞬态物理时间共用）
# min_steps = 0       # 可选；允许 tolerance 早停前必须完成的最小步数
# tolerance = -6.0    # 可选；log₁₀(RMS(ρ̇)) 阈值，满足则早停
```

`scheme = "gmres"` 启用 3D 可压缩 matrix-free GMRES 隐式伪时间步：
求解 \((D_{\Delta t}-J_R)\Delta U=R(U)\)，默认使用 LU-SGS 标量对角预条件器，并在有限差分扰动与更新守恒量前做正性限制。`gmres_preconditioner = "cell_block_diagonal"` 可切换为每单元 5×5 局部无粘 Jacobian 块预条件器（更强耦合，构造成本更高）。
每个 GMRES 步会在日志事件 `GMRES 隐式步诊断` 中输出 profiling 字段：`profile_compute_dt_ms`、`profile_preconditioner_build_ms`、`profile_linear_solve_ms`、`profile_line_search_ms`、`profile_step_total_ms` 与 `log10_residual`（步初 \(\|R(U^0)\|\)，与 `profile_base_residual_ms` 同源装配）。

| 字段 | 说明 |
|------|------|
| `mode` | `steady` 稳态伪时间推进；`transient` 瞬态物理时间推进 |
| `scheme` | 可选；`rk4`（默认）、`euler`、`lu_sgs`、`gmres`、`simplec` 或 `piso`；`simplec` / `piso` 仅用于不可压缩 pressure-velocity 路径。**3D 可压缩局部时间步均用 Blazek §6.1.4/§9.1** \(\Delta t_i=\mathrm{CFL}/\sigma_i\)，\(\sigma_i=(\Lambda_i^c+C_v\Lambda_i^v)/V_i\) |
| `local_time_step` | 可选；`true` 时逐单元 CFL 时间步（稳态加速；`lu_sgs` / `gmres` **必须**为 `true`） |
| `lusgs_omega` | 可选；`lu_sgs` 松弛因子 \(\omega\in(0,1]\)，默认 1 |
| `lusgs_sweep` | 可选；`false`（默认）仅用对角隐式（阶段 C）；`true` 启用 i/j/k 双扫（阶段 D，含正性限制与线搜索） |
| `lusgs_sweep_backward_damping` | 可选；后扫邻居耦合阻尼 \(\in(0,1]\)，默认 0.5 |
| `gmres_preconditioner` | 可选；`scalar_diagonal`（默认）或 `cell_block_diagonal`（每单元 5×5 局部无粘 Jacobian 块） |
| `residual_smoothing` | 可选；`true` 启用 3D 稳态方向分裂隐式残差光顺（瞬态忽略） |
| `residual_smoothing_epsilon` | 可选；光顺系数，默认 0.5，建议 0.2–0.6 |
| `residual_smoothing_sweeps` | 可选；i→j→k 光顺轮数，默认 1 |
| `max_steps` | 最大推进步数（稳态与瞬态共用，不再使用 `max_iterations`） |
| `min_steps` | 可选；满足 `tolerance` 后仍至少执行的最小步数，用于避免初始场假收敛 |
| `cfl` | CFL 初值（3D Euler 时间步控制；默认 0.4） |
| `cfl_max` | 可选；CFL 终值，从 `cfl` 线性增至 `cfl_max` |
| `cfl_ramp_steps` | 可选；线性爬升步数（第 1 步…`cfl_ramp_steps`）；未设则在 `max_steps` 全程爬升；爬升结束后保持 `cfl_max`；**若 `max_steps` 更短，仍按 `cfl_ramp_steps` 爬升，末步 CFL 可能低于 `cfl_max`** |
| `tolerance` | 可选；log₁₀(RMS(ρ̇)) 阈值，与 `max_steps` 成对用于残差早停 |
| `dt` | 固定时间步（设正数时覆盖 CFL 估算） |
| `final_time` | 可选物理终止时刻（Sod 等算例亦可在 `[sod]` 指定） |

含 `[sod]` 段时若省略 `[time]`，默认 `mode = "transient"`。

### 6.1 `[sod]`（Sod 激波管 benchmark）

```toml
[sod]
diaphragm = 0.5      # 间断位置（域坐标）
final_time = 0.2     # 物理终止时刻
cfl = 0.4            # CFL 数（固定 dt 时可在 [time] 指定 dt）

# 可选：无粘离散（省略时为一阶 Roe）
flux = "roe"              # roe | hllc | van_leer | hanel_van_leer | slau2
reconstruction = "muscl"  # first_order | muscl
limiter = "van_albada"    # 仅 muscl：minmod | van_leer | van_albada
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `flux` | `roe`（一阶） | 指定 `roe`/`hllc` 且未写 `reconstruction` 时默认 **MUSCL** 重构 |
| `reconstruction` | 随 `flux` | `first_order`：单元内分段常数，Godunov 型通量**单调**，勿配 `limiter` |
| `limiter` | `minmod` | **仅** `reconstruction = "muscl"`；一阶时忽略并告警 |

须配合 `structured_1d` 网格与 `[physics] gamma/gas_constant`。CLI：`asimu --case tests/benchmarks/sod_1d/case.toml`。

HLLC 变体示例：`tests/benchmarks/sod_1d/case_muscl_hllc.toml`。

---

## 6.5 无量纲求解（可压缩算例）

3D 可压缩 Euler/NS **仅在 \(*\) 变量下求解**；**输入 TOML 仍为 SI 有量纲**，`CaseSpec` 解析完成后自动调用 `apply_nondimensionalization` 缩放。须配置 `[freestream]`。

| 参考量 | 取值（自动，不可覆盖） |
|--------|------------------------|
| 长度 \(L_{\mathrm{ref}}\) | 1 m |
| 速度 \(U_{\mathrm{ref}}\) | 来流声速 \(a_\infty\) |
| 温度 \(T_{\mathrm{ref}}\) | 来流静温 \(T_\infty\) |
| 粘度 \(\mu_{\mathrm{ref}}\) | \(\mu(T_\infty)\)（Sutherland/常数） |

派生：\(\rho_{\mathrm{ref}}=p_\infty/(RT_\infty)\)，\(p_{\mathrm{ref}}=\rho_{\mathrm{ref}}U_{\mathrm{ref}}^2\)，
\(\mathrm{Re}=\rho_{\mathrm{ref}}U_{\mathrm{ref}}L_{\mathrm{ref}}/\mu_{\mathrm{ref}}\)。
NS 粘性项含 \(1/\mathrm{Re}\)；流场 CGNS/VTK 输出自动还原 SI。

**理论手册**（公式编号与代码一一对应）：[docs/theory/nondimensional.md](theory/nondimensional.md)。

### 来流 \(*\) 与实现入口

| 量 | 来流 \(*\) 值 | 构造入口 |
|----|---------------|----------|
| \(p^*\) | \(1/\gamma\) | 缩放后 `[freestream].pressure` |
| \(T^*\) | \(1\) | 缩放后 `[freestream].temperature` |
| \(\rho^*\) | \(1\) | **`FreestreamContext::primitive`**（勿用 `p/(RT)`） |
| \(u^*\) | \(M_\infty\)（\(a^*=1\)） | 同上 |
| 初场 | 均匀来流守恒量 | `ConservedFields::from_freestream_context` |
| BC ghost | 与来流一致 | `apply_compressible_boundary_conditions(..., &FreestreamContext, ...)` |

静温：有量纲 \(T=p/(\rho R)\)；无量纲 \(T^*=p^*\gamma/\rho^*\) → `ViscousPhysicsConfig::static_temperature`（见理论手册式 (1)(2)）。

须配合 `[freestream]` 或 `[restart]`；与 `[mesh].scale` 独立（后者仅做网格单位换算）。

### `[restart]` 初场

单 block 与多块均通过 case TOML 的 `[restart]` 段指定 restart 文件路径（相对算例目录）：

```toml
[restart]
path = "restart.toml"
```

| 版本 | 适用 | 格式 |
|------|------|------|
| `version = 1` | 单 block 3D / 1D | 顶层 `num_cells` + `density` / `momentum_*` / `total_energy`；可选 `compute_precision = "f32"`（缺省 `f64`） |
| `version = 2` | 多块 3D | `[[blocks]]` 数组，每项含 `name`（须与 mesh block 名一致）及守恒量数组；顶层可选 `compute_precision` |

`compute_precision` 须与 case `[numerics].compute_precision` 一致；跨精度 restart（`f32` 文件 + `f64` case 或反向）**暂不支持**，加载时报错。旧版无 `compute_precision` 字段的 restart 视为 `f64`。

多块 restart 仍须配置 `[freestream]` 以驱动边界 ghost；初场守恒量来自 restart 文件。API：`CaseSpec::build_multiblock_conserved_fields()`、`io::load_multiblock_conserved_fields()`、`io::write_multiblock_conserved_fields()`。

## 6.6 `[incompressible]`（I1 skeleton）

I1 阶段仅支持单 block `structured_3d`，可初始化不可压主变量、装配/求解伪瞬态动量预测与压力校正 skeleton，并写出 CGNS；边界条件与完整 SIMPLEC/PISO 循环仍在后续阶段。

```toml
[incompressible]
pressure = 0.0                  # Pa，解析后除以 rho * U_ref^2
velocity = [1.0, 0.0, 0.0]       # m/s，解析后除以 U_ref
body_force = [0.0, 0.0, 0.0]     # m/s^2，每单位质量体力；解析后乘 L_ref/U_ref^2
convection_scheme = "upwind"      # upwind（默认）| central
piso_correctors = 1                # 每个外层步的压力校正次数；time.scheme="simplec" 时强制为 1，PISO 常用 2
density = 1.0
kinematic_viscosity = 0.01       # m^2/s，解析后为 1/Re
velocity_under_relaxation = 0.7   # 可选，(0, 1]，默认 1.0
pressure_under_relaxation = 0.3   # 可选，(0, 1]，默认 1.0

[incompressible.linear.momentum]
solver = "gmres"                  # 当前仅 gmres
restart = 30                      # 可选，默认 30
max_iters = 100                   # 可选，默认 100
tolerance = 1.0e-8                # 可选，默认 1.0e-8

[incompressible.linear.pressure]
solver = "pcg"                    # pcg（默认）| gmres
restart = 64                      # 可选，默认 64
max_iters = 500                   # 可选，默认 500
tolerance = 1.0e-10               # 可选，默认 1.0e-10

[incompressible.reference]
length = 1.0                     # L_ref，m
velocity = 1.0                   # U_ref，m/s
```

不可压缩求解内部使用无量纲星号量：\(x^*=x/L_{\mathrm{ref}}\)、\(u^*=u/U_{\mathrm{ref}}\)、\(p^*=p/(\rho U_{\mathrm{ref}}^2)\)、\(\nu^*=1/Re\)、\(\mathbf{f}^*=\mathbf{f}L_{\mathrm{ref}}/U_{\mathrm{ref}}^2\)。`[incompressible.reference]` 必填，不自动从网格或速度猜测。默认 CGNS Vertex 输出字段为 `Pressure`、`VelocityX`、`VelocityY`、`VelocityZ`，输出会还原 SI。`[incompressible]` 不能与 `[sod]`、`[euler]`、`[navier_stokes]` 同时出现。

---

## 7. `[output]` 与 `[observability]`

### 7.1 `[output]`

```toml
[output]
dir = "output"                    # 相对算例目录
residual_csv = "residual.csv"
solution_cgns = "flow.cgns"
solution_every = 100
solution_vtk = false              # 为 true 时额外写 .vtu/.vts（需 feature io-vtk）
```

相对路径均相对 **算例文件所在目录**；写出文件落在 `dir` 下。

`solution_cgns` / `solution_vtk` 流场含：可压缩输出 `Density`、`VelocityX/Y/Z`、`Pressure`、`MachNumber`、`Temperature`；不可压缩输出 `Pressure`、`VelocityX/Y/Z`（CGNS 为 Vertex 插值，VTK 为单元中心）。多块 3D case 的最终流场与 `solution_every` 间隔快照均写为单个 CGNS 文件、每个 block 一个 Structured Zone。

配置 `residual_csv`（及可选 `residual_plot`）时，算例结束会写出残差 CSV 与曲线图；可压缩多步路径在 `solution_every` 间隔也会同步刷新。不可压缩 SIMPLEC/PISO 会在最终输出时写出逐步 pressure-velocity residual CSV；同时配置 `solution_cgns` 与 `solution_every` 时，还会写出 `flow_step000100.cgns` 形式的间隔快照。

### 7.2 `[observability]` — Chrome trace

```toml
[observability]
chrome_trace = "profiling/trace.json"
```

| 字段 | 说明 |
|------|------|
| `chrome_trace` | 相对 `[output].dir`（未设 `[output]` 时默认 `output/`）的 Chrome trace JSON 路径；省略或空字符串表示关闭 |

算例运行结束后写出 trace；用 [ui.perfetto.dev](https://ui.perfetto.dev) 或 Chrome `chrome://tracing` 打开。时间轴上的 span 来自 `tracing`（如每步 `advance_step_3d`）。日志级别仍由 CLI `--log-level` 控制。

**CLI（优先于算例文件）**：

```bash
asimu --case case.toml --chrome-trace                      # 默认 <算例>/output/profiling/trace.json
asimu --case case.toml --chrome-trace case_cylinder/trace.json  # 相对**当前工作目录**
asimu --case case.toml --chrome-trace /tmp/run.trace.json # 绝对路径
# 或环境变量 ASIMU_CHROME_TRACE=profiling/trace.json
```

示例（圆柱算例性能分析）：

```toml
[output]
dir = "output"

[observability]
chrome_trace = "profiling/trace.json"
```

---

## 8. 全局 `[solver]`（`config/default.toml`，非算例）

算例时间推进与收敛见 `[time].max_steps` / `[time].tolerance`。全局 `config/default.toml` 的 `[solver]` 仅保留 CLI 占位求解器步数：

```toml
[solver]
max_steps = 100
```

CLI：`--max-steps` / `ASIMU_MAX_STEPS`。

---

## 9. 完整示例（1D 扩散）

见 `tests/benchmarks/1d_diffusion_analytical/case.toml`。

---

## 10. 与 v0.1 占位格式迁移

| v0.1 | v0.2 TOML 等价 |
|------|----------------|
| `name=demo;cells=256` | `name = "demo"` + `[mesh] kind = "structured_1d" cells = 256 length = 1.0` |

`io::load_mesh_from_case` 在 v0.2 将检测扩展名 / 内容：`.toml` 走新解析器，遗留单行格式保持兼容至 v0.3。

---

## 11. 相关文档

- [BENCHMARKS.md](BENCHMARKS.md) — V&V 算例与 `expected.json`
- [theory/fvm_diffusion.md](theory/fvm_diffusion.md) — 扩散方程离散
- [SECURITY.md](../SECURITY.md) — 文件大小与路径限制
