# 热传导 — 无量纲有限体积法（设计）

> 模块（规划）：`src/physics/`、`src/io/`、`src/discretization/`、`src/linalg/`、`src/solver/`、`src/case/`
> 版本：v0.3+ · 状态：**设计**（实现未开始）
> 网格：结构化 1D/3D + 非结构混合单元（`UnstructuredMesh3d`）
> 关联：[fvm_diffusion.md](fvm_diffusion.md)（1D 有量纲骨架）、[boundary_conditions.md](boundary_conditions.md) §2、[nondimensional.md](nondimensional.md)、[ADR 0010](../adr/0010-unstructured-mixed-mesh.md)、[ADR 0016](../adr/0016-runtime-compute-precision.md)

---

## 1. 目标与范围

### 1.1 物理问题

稳态导热（固体或流体纯传导区）：

\[
\nabla \cdot (k \nabla T) = 0 \tag{1}
\]

可选体源项（规划 H3）：

\[
\nabla \cdot (k \nabla T) + \dot{q} = 0 \tag{1b}
\]

其中 \(T\) 为温度 (K)，\(k\) 为热导率 (W/(m·K))，\(\dot{q}\) 为体积热源 (W/m³)。

瞬态形式（规划 H3）：

\[
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{q} \tag{2}
\]

### 1.2 设计原则

| 原则 | 要求 |
|------|------|
| **无量纲求解** | 与不可压 / 可压路径一致：TOML **写 SI**，Validate 后缩放为 \(*\) 量，热路径只读已验证的 \(*\) 配置 |
| **FVM 守恒** | 面通量连续；稳态全局热流闭合作为 V&V 指标 |
| **显式依赖** | `mesh`、`ScalarField`、`BoundarySet`、`ConductionReferenceScales` 经参数传入；禁止模块级缓存上次算例 |
| **分层** | `discretization` 只装配；`linalg` 只求解；`case` 编排 Parse → Validate → 无量纲 → Trust → run |
| **双精度** | **H0–H2 同时支持 `f32` / `f64`**（[ADR 0016](../adr/0016-runtime-compute-precision.md)）：`[numerics].compute_precision` 在 case 边界分发，`run_typed::<T>` 内单态化；几何/I/O 保持 `f64` |
| **网格** | 节点坐标保持 **SI f64**（`mesh` 层）；\(*\) 仅作用于场值、物性、BC 与装配导出的有效几何比（\(d/L_{\mathrm{ref}}\)、\(A/L_{\mathrm{ref}}^2\)） |

### 1.3 首版范围（H0–H2）

- 常数 \(k\) 稳态式 (1)
- 网格：非结构 `UnstructuredMesh3d`（tet/hex/pyramid/prism）；结构化 3D 为后续薄封装
- BC：`Dirichlet`（等温）、`Neumann`（给定热流 / 绝热 \(q=0\)）
- 线性求解：CSR + Jacobi-preconditioned PCG（**`CsrMatrix<T>` / `PcgSolver` typed**，\(T\in\{\texttt{f32},\texttt{f64}\}\)）
- 算例：**H2 优先 CGNS**（非结构 zone 读入 + 温度场 CellCenter 写回）；VTU 读/写列为 H2+
- 精度：默认 `compute_precision = "f64"`；同一 binary 可通过 TOML 切换 `f32`（V&V 容差分档，见 §6）

**不在首版**：变 \(k(T)\)、Robin 对流边界、瞬态 Fo 推进、与 NS 能量方程耦合（见 §9）。

### 1.4 与现有 1D 扩散的关系

`tests/benchmarks/1d_diffusion_analytical/` 使用 `[physics].diffusivity` 与 **有量纲** 三对角求解（`assemble_diffusion_1d`）。热传导路径引入 **独立算例段** `[heat_conduction]` 与 mandatory `[heat_conduction.reference]`；1D 算例在 H4 可迁移至同一无量纲框架，首版保持向后兼容。

---

## 2. 参考量与无量纲化

### 2.1 参考量

用户必须在 TOML 中声明（不自动从网格猜测）：

| 符号 | 含义 | TOML 字段 | 约束 |
|------|------|-----------|------|
| \(L_{\mathrm{ref}}\) | 特征长度 (m) | `[heat_conduction.reference].length` | \(>0\) |
| \(T_{\mathrm{ref}}\) | 特征温度 (K) | `[heat_conduction.reference].temperature` | \(>0\) |
| \(k_{\mathrm{ref}}\) | 特征热导率 (W/(m·K)) | `[heat_conduction.reference].conductivity` | \(>0\)；默认 = `[heat_conduction].thermal_conductivity` |

派生热流尺度（用于 Neumann BC 与通量诊断）：

\[
q_{\mathrm{ref}} = \frac{k_{\mathrm{ref}}\, T_{\mathrm{ref}}}{L_{\mathrm{ref}}} \tag{3}
\]

派生热扩散系数（瞬态用，H3）：

\[
\alpha_{\mathrm{ref}} = \frac{k_{\mathrm{ref}}}{\rho_{\mathrm{ref}} c_{p,\mathrm{ref}}}, \qquad
t_{\mathrm{ref}} = \frac{L_{\mathrm{ref}}^2}{\alpha_{\mathrm{ref}}} \tag{4}
\]

（\(\rho_{\mathrm{ref}}, c_{p,\mathrm{ref}}\) 在瞬态段 `[heat_conduction.material]` 声明。）

### 2.2 星号变量

\[
x^* = \frac{x}{L_{\mathrm{ref}}}, \quad
T^* = \frac{T}{T_{\mathrm{ref}}}, \quad
k^* = \frac{k}{k_{\mathrm{ref}}} \tag{5}
\]

常数 \(k\) 首版取 \(k^* \equiv 1\)。

将 (5) 代入 (1)，除以 \(k_{\mathrm{ref}} T_{\mathrm{ref}} / L_{\mathrm{ref}}^2\)，得 **控制方程的 \(*\) 形式**（与有量纲方程同形）：

\[
\nabla^* \cdot (k^* \nabla^* T^*) = 0 \tag{6}
\]

因此 **离散模板与有量纲 FVM 相同**，只需：

- 面间距 \(d^* = d / L_{\mathrm{ref}}\)
- 面积 \(A^* = A / L_{\mathrm{ref}}^2\)
- 装配系数用 \(k^*\)
- 未知量与 BC 为 \(T^*\)

### 2.3 边界条件的 \(*\) 形式

| BC | 有量纲 | 无量纲（求解器内） |
|----|--------|-------------------|
| Dirichlet | \(T = T_b\) | \(T^* = T_b / T_{\mathrm{ref}}\) |
| Neumann | \(-k\,\partial T / \partial n = q\) | \(-k^*\,\partial T^* / \partial n^* = q^*\)，\(q^* = q / q_{\mathrm{ref}}\) |
| 绝热 | \(q = 0\) | \(q^* = 0\) |

现有 `apply_dirichlet_face` / `apply_neumann`（[boundary_conditions.md](boundary_conditions.md) §2）将 **`diffusivity` 参数解释为 \(k^*\)**，`mesh.face_spacing` 返回 SI 距离 \(d\)；装配前使用 \(d^* = d / L_{\mathrm{ref}}\)（或在 `BoundaryMesh` 适配层传入已缩放间距）。

### 2.4 Parse → Validate → Trust

```
TOML (SI)
  → CaseSpec::validate()          # 字段、patch 覆盖、正性
  → apply_conduction_nondimensionalization()
       · 缩放 BC 值为 T*, q*
       · 写入 ConductionReferenceScales（只读）
  → discretization / linalg       # 信任 * 量，不再读 TOML
  → 输出还原 T = T* * T_ref      # CGNS（H2 首选）/ manifest；VTU 见 H2+
```

规划类型（`src/physics/reference.rs` 旁）：

```rust
pub struct ConductionReferenceScales {
    pub length: Real,
    pub temperature: Real,
    pub conductivity: Real,
    pub heat_flux: Real,   // k_ref * T_ref / L_ref
}
```

### 2.5 计算精度（f32 / f64，ADR 0016）

热传导路径遵循项目 **Compute Precision** 模型：仅在求解热路径使用运行时可选的 `f32` / `f64`；**不在**面循环内读配置分支。

#### 2.5.1 配置与分发

```toml
[numerics]
compute_precision = "f64"   # f64（默认）| f32
```

```rust
pub fn run(case: &CaseSpec) -> Result<CaseRunResult> {
    case::validate::compute_precision(case)?; // 扩展 conduction 能力矩阵
    match case.numerics.compute_precision {
        ComputePrecision::F64 => run_conduction_typed::<f64>(case),
        ComputePrecision::F32 => run_conduction_typed::<f32>(case),
    }
}
```

Validate 阶段（`case::validate::compute_precision`）在 `[heat_conduction]` 算例上 **同时登记** `f32` 与 `f64` 为已实现组合；未实现组合报错，禁止静默回退。

#### 2.5.2 精度分界

| 数据 | 存储类型 | 说明 |
|------|----------|------|
| 网格坐标、体积、面积、法向 | **`f64` / `Real`** | `UnstructuredMesh3d` 构造期固定 |
| `ConductionReferenceScales` | **`Real`（f64）** | 配置与 I/O；进入 typed 路径时 `T::from_real` |
| 温度场 \(T^*\) | **`ScalarFieldT<T>`** | \(T=\texttt{f32}\) 或 \(\texttt{f64}\) |
| CSR 系数 / RHS | **`CsrMatrix<T>`** | 与场同精度 |
| PCG 向量 \(x,r,p,z\) | **`Vec<T>`** | 与矩阵同精度 |
| PCG 相对残差归约 | **`f64` 累加** | 输入来自 `T`，避免 `f32` 归约漂移 |
| manifest / CGNS 输出温度 | **`f64` 写出** | `T` 场 `to_real()` 后还原 SI (K)；文件格式不变 |
| 收敛历史、V&V 误差 | **`f64`** | 日志与 `expected.json` 对比在 orchestration 层 |

#### 2.5.3 装配：几何 f64 → 系数 T

内部面电导在 **f64 几何** 上计算无量纲比，再落入 `T`（避免 `f32` 坐标差分）：

```text
d_O* = (f64) d_O / L_ref
A_f* = (f64) A_f / L_ref²
G_f* = T::from_real(k_f*) * T::from_real(A_f*) / T::from_real(d_O* + d_N*)
```

边界 BC：`apply_conduction_boundary_conditions_typed<T>` 包装现有 ghost 公式；`d` 用 f64 间距除以 `L_ref` 后转 `T`，\(k^*\) 与 \(T_b^*\)、\(q^*\) 在 Validate 后已为 `T` 或可 `from_real` 的 `Real` 缓存。

#### 2.5.4 线性求解默认容差

| 精度 | PCG 相对残差默认 | 备注 |
|------|------------------|------|
| `f64` | \(10^{-8}\) | 式 (10) |
| `f32` | \(10^{-5}\) | 不得复用 `f64` golden；见 [ADR 0016 §4](../adr/0016-runtime-compute-precision.md) |

用户可通过 `[heat_conduction.linear] tolerance` 覆盖；Validate 时 `f32` tolerance 下限 \(\ge 10^{-7}\)（避免无效迭代）。

#### 2.5.5 非目标（首版）

- 同一 time step 内动态切换精度
- mixed precision（场 `f32` + 矩阵 `f64` 等）
- `f32` restart 与 `f64` restart 互转（加载时报错）
- CUDA 导热 kernel（H2 仅 CPU；GPU 列为远期）

---

## 3. 有限体积离散

### 3.1 半离散守恒式

对单元 \(P\)，积分 (6) 得：

\[
\sum_{f \in \partial P} F_f^* = 0, \qquad
F_f^* = -k_f^*\, A_f^*\, \frac{T_N^* - T_O^*}{d_O^* + d_N^*} \tag{7}
\]

- 内部面：\(O\)=owner，\(N\)=neighbor；\(k_f^*\) 首版取算术平均 \( (k_O^* + k_N^*)/2 \) 或常数 1。
- 边界面：由 §2.3 ghost 公式并入 owner 行（与 1D 相同）。

**装配顺序**（与 ARCHITECTURE §8.5.3 一致）：先内部面，再按 patch 顺序施加 BC。

### 3.2 非结构内部面

遍历 `UnstructuredMesh3d` 面列表（与 [unstructured_fvm.md](unstructured_fvm.md) 面循环同拓扑）：

| 量 | 来源 |
|----|------|
| \(A_f\) | `face_metric(face).area` |
| \(\mathbf{x}_f\) | `face_metric(face).center` |
| \(\mathbf{x}_P\) | `cell_metric(owner).center` |
| \(d_O\) | \(\|\mathbf{x}_P - \mathbf{x}_f\|\) |
| \(d_N\) | \(\|\mathbf{x}_N - \mathbf{x}_f\|\)（内部面） |

\[
G_f^* = \frac{k_f^*\, A_f^*}{d_O^* + d_N^*} \tag{8}
\]

CSR 累加：

\[
A_{OO} \mathrel{+}= G_f^*,\; A_{ON} \mathrel{-}= G_f^*,\;
A_{NN} \mathrel{+}= G_f^*,\; A_{NP} \mathrel{-}= G_f^* \tag{9}
\]

**非正交修正**：首版省略（与 1D 骨架一致）；skew 网格 V&V 失败时再增 ADR 修订。

### 3.3 结构化 3D

逻辑 I/J/K 面与 `StructuredMesh3d` 度量；复用 (8)(9)，\(d_O,d_N\) 由 `face_spacing(owner_volume, neighbor_volume, area)` 再除以 \(L_{\mathrm{ref}}\)（见 `mesh/metrics.rs`）。

### 3.4 1D 结构化

均匀网格：\(G^* = k^* / \Delta x^*\)，\(\Delta x^* = \Delta x / L_{\mathrm{ref}}\)。与现有 `assemble_diffusion_1d` 同模板，参数改为 \(*\) 量（H4 统一）。

---

## 4. 线性系统与求解

| 网格 | 矩阵结构 | 求解器 |
|------|----------|--------|
| 1D 均匀 | 三对角 | Thomas（`LinearSystem::solve_tridiagonal`，H4 增 `LinearSystemT<T>` 或 typed 包装） |
| 非结构 / 一般 3D | **`CsrMatrix<T>`** 对称正定 | **`PcgSolver::solve`**（typed CSR + Jacobi；残差归约 f64） |

**预分配**：构造期由 `UnstructuredMesh3d` 的 owner/neighbor 建立固定 CSR sparsity（`ConductionCsrPattern`，精度无关）；`f32` / `f64` **共享同一 sparsity**，仅 `values`/`rhs` 元素类型不同。

**收敛判据**（默认，可 `[heat_conduction.linear]` 覆盖）：

\[
\frac{\|r\|_2}{\|b\|_2} < \tau_{\mathrm{pcg}}, \quad \text{max iter} = 2000 \tag{10}
\]

| `compute_precision` | \(\tau_{\mathrm{pcg}}\) 默认 |
|---------------------|-------------------------------|
| `f64` | \(10^{-8}\) |
| `f32` | \(10^{-5}\) |

---

## 5. 模块与 API 设计

### 5.1 依赖方向

```
core ← mesh ← field ← discretization ← linalg
core ← physics
mesh + field + discretization + physics + linalg ← solver::steady_conduction
case → io, solver, config
```

### 5.2 规划入口

| 函数 | 模块 | 职责 |
|------|------|------|
| `assemble_conduction_internal_faces_unstructured_typed::<T>` | `discretization::conduction` | 内部面 CSR 累加 (8)(9) |
| `apply_conduction_boundary_conditions_typed::<T>` | `discretization::conduction` / `bc` | ghost BC；\(k^*, d^*, T_b^*, q^*\) |
| `solve_steady_conduction_3d_typed::<T>` | `solver::steady_conduction` | 装配 + BC + typed PCG |
| `apply_conduction_nondimensionalization` | `io::nondimensional` | SI → \(*\)（`Real`；进 typed 前 `from_real`） |
| `run` / `run_conduction_typed::<T>` | `case::conduction_3d` | 精度分发 + 编排 |

```rust
// 规划签名（显式输入；T = f32 | f64）
pub fn solve_steady_conduction_3d_typed<T: ComputeFloat>(
    mesh: &UnstructuredMesh3d,
    boundary: &BoundarySet,
    reference: &ConductionReferenceScales,
    config: &ConductionSolverConfig,
) -> Result<(ScalarFieldT<T>, ConductionSolveReport)>;

// case 边界薄包装
pub fn solve_steady_conduction_3d(...) -> Result<(ScalarField, ...)>
where /* 委托 F64，兼容 rustdoc / 过渡 API */;
```

`ScalarFieldT<T>` 存 **\(T^*\)**；manifest / CGNS 写出：`T::to_real` → `reference.dimensional_temperature` → **f64** 写盘。

### 5.3 Case 路由

`detect_run_kind` 新增分支（优先于默认 1D 扩散）：

```text
[heat_conduction] 存在
  且 mesh ∈ { Unstructured3d, Structured3d, Structured1d }
  且 无 [euler] / [navier_stokes] / [incompressible]
→ CaseRunKind::ConductionSteady
```

**精度校验**：`case::validate::compute_precision` 扩展 `[heat_conduction]` + `Unstructured3d` 分支；`f32`/`f64` 均允许（与当前 1D `diffusion` 拒绝 `f32` 不同，见 §1.4 H4 迁移）。

### 5.4 TOML 示例（设计，H2：CGNS 非结构 zone）

```toml
name = "unstructured_conduction_slab"
benchmark_id = "unstructured_conduction_slab"

[mesh]
kind = "cgns"
path = "slab_hex.cgns"          # 单 zone Unstructured（tet/hex/pyramid/prism）
# zone_index = 1                # 可选，默认 1；多 zone 文件时指定
# scale = 1.0                   # 可选，坐标统一缩放

[heat_conduction]
thermal_conductivity = 16.2   # k (W/(m·K))，SI

[heat_conduction.reference]
length = 0.1                  # L_ref (m)，必填
temperature = 300.0           # T_ref (K)，必填
# conductivity 省略时 = thermal_conductivity

# CGNS ZoneBC 提供 patch 面列表；热 BC 类型与数值在 TOML 按 patch 名绑定
[boundary.cold]
kind = "dirichlet"
value = 280.0                 # T_b (K)

[boundary.hot]
kind = "dirichlet"
value = 320.0

[boundary.insulated]
kind = "neumann"
flux = 0.0                    # q (W/m²)，进入域为正

[output]
solution_cgns = "out/solution.cgns"   # H2：CellCenter Temperature (K)，SI 还原后写出

[numerics]
compute_precision = "f64"           # f64 | f32；输出 CGNS 仍为 Float64

[heat_conduction.linear]
# tolerance = 1.0e-8                # 可选；f32 默认 1e-5

[time]
mode = "steady"
```

**网格读入**：复用 `io::load_cgns_unstructured_zone`（[ADR 0008](../adr/0008-cgns-io.md)、[API.md](../API.md)）；`mesh.kind = "cgns"` 在 structured 读失败时自动回落非结构 zone（与可压非结构 case 相同，`mesh_load` 现有逻辑）。

**边界 patch**：CGNS `ZoneBC` / `FamilyName` 产出 `BoundarySet` 面 ID；`[boundary.<patch>]` 键名须与 CGNS patch 名一致（或可压 case 已有的 `block/patch` 规则，首版单 zone 无 block 前缀）。CFD 语义映射（`IN`/`OUT`/`WALL`）**不**自动等同热 BC；导热算例必须在 TOML 显式声明 `dirichlet` / `neumann`。

### 5.5 I/O 优先级（H2）

| 优先级 | 格式 | 读 | 写 | 说明 |
|--------|------|----|----|------|
| **P0** | **CGNS Unstructured** | `load_cgns_unstructured_zone`（**已实现**） | `write_temperature_cgns_unstructured`（**H2 新增**） | benchmark 网格与 `solution_cgns` 均用 CGNS；须 `io-cgns` feature |
| P1 | VTU | `load_vtu`（已实现） | `write_scalar_vtu_unstructured`（H2+） | ParaView 快捷查看；非 H2 阻塞项 |
| P2 | 内置 `structured_*` | 生成器 | CGNS / manifest | H4 与 1D 迁移 |

**H2 写出约定**（规划 API）：

```rust
pub fn write_temperature_cgns_unstructured(
    path: &Path,
    mesh: &UnstructuredMesh3d,
    temperature_k: &[Real],   // 已还原 SI (K)，f64；与 compute_precision 无关
) -> Result<()>;
```

- 场名 `Temperature`，位置 `CellCenter`，与 `write_flow_cgns_unstructured` section 布局一致，避免 CFD/导热双轨 CGNS 结构。
- CI benchmark 在 `tests/benchmarks/unstructured_conduction_slab/` 存放 `slab_hex.cgns`（或 `scripts/` 生成后提交）；`.gitignore` 仅忽略 `out/` / `output/`。

**VTU（H2+）**：读入可作为 dev 便利路径；V&V 算例与 manifest golden **不以 VTU 为必需输入**。

实现 H2 时同步 [CASE_FORMAT.md](../CASE_FORMAT.md)：`[heat_conduction]`、`[output].solution_cgns` 与 CGNS 非结构 zone 说明。

---

## 6. V&V 与算例

| benchmark_id | 网格来源 | 验证量 |
|--------------|----------|--------|
| `unstructured_conduction_slab` | **`slab_hex.cgns`**（H2 P0） | 一维解析 \(T^*(x^*) = x^*\)（两面 Dirichlet） |
| `unstructured_conduction_sphere` | CGNS 径向网格（H2+） | 球壳解析 |
| 迁移 | `1d_diffusion_analytical` | H4 与 \(*\) 框架对齐后 L2 回归 |

H2 CI 算例须满足：`mesh.kind = "cgns"`、`make check` 在 `io-cgns` 启用环境通过；无 libcgns 时 conduction 集成测试 `#[ignore]` 或 feature-gate（与现有 CGNS 测试一致）。

**精度 V&V**：同一 `case.toml` 跑两遍——默认 `f64` 与 `[numerics] compute_precision = "f32"`；`expected.json` 分档容差（`f64` 收紧 / `f32` 放宽），manifest 记录 `compute_precision`。

**Metrics（manifest）**：

| 量 | 定义 |
|----|------|
| `max_abs_temperature_error` | \(\max |T^* - T^*_{\mathrm{exact}}|\) |
| `l2_temperature_error` | RMS 误差 |
| `heat_flux_imbalance_ratio` | 稳态边界热流闭合；定义见 **§10.3** |
| `linear_solver` | PCG 迭代数、相对残差（归约 f64） |
| `compute_precision` | `"f32"` / `"f64"`（manifest 必填） |

**`expected.json` 容差示例**（slab smoke）：

| 量 | `f64` max | `f32` max |
|----|-----------|-----------|
| `l2_temperature_error` | \(10^{-2}\) | \(5\times 10^{-2}\) |
| `heat_flux_imbalance_ratio` | \(10^{-2}\) | \(2\times 10^{-2}\) |
| PCG 相对残差 | \(< 10^{-8}\) | \(< 10^{-5}\) |

Reference golden 以 **`f64` 生成**；`f32` 仅验证相对误差/守恒阈值，不复制同一绝对容差。

---

## 7. 实现阶段

| 阶段 | 交付 | 出口标准 |
|------|------|----------|
| **H0** | `ConductionReferenceScales` + `apply_conduction_nondimensionalization` + **`run_conduction_typed::<T>` 骨架** + 单元测试 | SI↔\(*\) 往返、BC 缩放；`f32`/`f64` 参考量 `from_real`/`to_real` |
| **H1** | `assemble_*_typed::<T>` + **`CsrMatrix<T>` / typed PCG** + 2-cell tet/hex 单测 | 手算 \(G_f^*\) 与 CSR 一致；`f32`/`f64` 双套单测 |
| **H2** | `solve_steady_conduction_3d_typed` + `CaseRunKind::ConductionSteady` + **CGNS** benchmark | `make check`（`io-cgns`）；slab **`f64` + `f32` smoke**（分档 `expected.json`） |
| **H2+** | VTU 读/写、可选 `export_cgns_zone_to_vts` 后处理 | ParaView 便利路径 |
| **H3** | 瞬态 (2)、变 \(k(T)\)、Robin BC | Fo 衰减 golden |
| **H4** | 1D 迁移、`[physics].diffusivity` 文档弃用路径 | 旧 benchmark 不回归 |

---

## 8. 实现映射（规划）

| 式 / 步骤 | 代码位置 | 状态 |
|-----------|----------|------|
| (3)–(5) 参考量 | `physics::ConductionReferenceScales` | 规划 |
| SI → \(*\) | `io::nondimensional::apply_conduction_nondimensionalization` | 规划 |
| (7)–(9) 内部面 | `discretization::conduction::assemble_internal_faces_unstructured_typed::<T>` | 规划 |
| BC §2.3 | `discretization::conduction::apply_conduction_boundary_conditions_typed::<T>` | 规划（ghost 公式复用） |
| (10) PCG | `linalg::PcgSolver` + `CsrMatrix<T>` | **部分**（CSR/PCG 现仅 `Real`；H1 增 typed） |
| 精度分发 | `case::conduction_3d::run` + `validate::compute_precision` | 规划 |
| 编排 | `solver::steady_conduction::solve_steady_conduction_3d_typed::<T>` | 规划 |
| case | `case::conduction_3d::run` | 规划 |
| CGNS 读 | `io::load_cgns_unstructured_zone` / `mesh_load` cgns 回落 | **已实现** |
| CGNS 写 | `io::write_temperature_cgns_unstructured` | 规划（H2 P0） |
| VTU 读/写 | `load_vtu` / `write_scalar_vtu_unstructured` | 规划（H2+） |
| 热流诊断 | `discretization::conduction::compute_boundary_heat_balance_*` | 规划（§10.3） |
| patch 覆盖率 | `case::validate::conduction_boundary` | 规划（§10.2） |
| PCG 初值 | `solver::steady_conduction`（默认 §10.1） | 规划 |
| 输出还原 | `ComputeFloat::to_real` + `ConductionReferenceScales::dimensional_temperature` → f64 CGNS | 规划 |

---

## 9. 与可压 NS 能量方程 Fourier 项的关系

可压路径中壁面热流 / 等温壁已在 `discretization::compressible::wall_thermal` 实现（有量纲 / 可压 \(*\) 混合）。**本设计**为 **独立标量求解器**，用于：

- 纯固体导热、流固解耦预热；
- 非结构网格上验证 FVM 扩散模板与 PCG 栈。

与 NS 耦合（共轭传热）不在本设计范围；耦合时 NS 侧仍用 `wall_thermal`，固体区调用 `solve_steady_conduction_3d` 或瞬态等价物，界面通量 \(q\) 连续作为接口条件（远期 ADR）。

---

## 10. 实现前定案（补充）

以下条目在 H0 Validate / H2 V&V 中**必须**落实，避免实现阶段歧义。

### 10.1 PCG 初值

| 项 | 定案 |
|----|------|
| H0–H2 默认 | **\(T^* \equiv 0\)**（全单元）；稳态线性问题，PCG 仍收敛 |
| 可选（H2+） | `[heat_conduction.initial] kind = "uniform", value = …`（SI → \(T^*\)）或 Dirichlet patch 常值外推 |
| 禁止 | 从 `[initial.phi]` legacy 段隐式读取；导热与 1D 扩散初值段分离 |

实现：`PcgSolver` 的 `x` 初猜在 `solve_steady_conduction_3d_typed` 内显式 `fill(0)` 或按 config；不依赖未初始化的 `ScalarFieldT` 内存。

### 10.2 边界 patch 覆盖率

Validate 阶段（`case::validate::conduction_boundary`，在 `run` 之前）检查：

1. **每个边界面**（`face_neighbor == None`）恰好出现在 **一个** `BoundaryPatch.face_ids` 中；
2. patch 并集无重复、无遗漏（与 `mesh::check_unstructured_mesh3d` 的边界覆盖率一致）；
3. CGNS 读入时：面列表来自 CGNS `ZoneBC`；TOML `[boundary.<name>]` **只覆盖 `kind`/数值**，`<name>` 必须与 CGNS patch 名一致，不得凭空新增 patch；
4. **至少一个 Dirichlet patch**；若全部为 Neumann（含绝热），矩阵奇异 → Validate **报错**（纯 Neumann 稳态不定）。

非结构网格 **禁止** 使用 `i_min` / `j_max` 等逻辑边界名（见 `case_boundary::resolve_mesh_logical_boundary`）。

### 10.3 边界热流诊断与符号

求解完成后，用**收敛的 \(T^*\)** 与 **§3.1 同一公式** 在边界面上重算面热流（不读装配 RHS），符号与 I4 `mass_flux` 类比：

**面热流（owner 视角，无量纲）**：

\[
F_f^* = -k_f^*\, A_f^*\, \frac{T_{\mathrm{ext}}^* - T_O^*}{d_O^*} \tag{11}
\]

- 边界面：\(T_{\mathrm{ext}}^*\) 由 ghost 关系给出（与 `apply_conduction_boundary_conditions_typed` 一致）；内部面仅用式 (7)。
- **\(F_f^* > 0\)**：沿 owner 外法向、**流出 owner 单元**（对边界 owner 即流出计算域）。

**patch 与全域汇总**：

\[
\dot Q_{\mathrm{patch}}^* = \sum_{f \in \mathrm{patch}} F_f^*, \qquad
\dot Q_{\mathrm{net}}^* = \sum_{\mathrm{all\ BC\ patches}} \dot Q_{\mathrm{patch}}^* \tag{12}
\]

稳态守恒应 \(\dot Q_{\mathrm{net}}^* \approx 0\)。

**驱动尺度**（分母，类比 I4 `inlet_magnitude`）：

\[
\dot Q_{\mathrm{ref}}^* = \max\!\left(
  \max_{\mathrm{Dirichlet\ patches}} |\dot Q_{\mathrm{patch}}^*|,\;
  \max_{\mathrm{Neumann\ patches}} \textstyle\sum_f |q_f^*\, A_f^*|,\;
  \varepsilon
\right) \tag{13}
\]

\[
\mathrm{heat\_flux\_imbalance\_ratio} =
  \frac{|\dot Q_{\mathrm{net}}^*|}{\dot Q_{\mathrm{ref}}^*} \tag{14}
\]

| 场景 | 期望 |
|------|------|
| slab 两面 Dirichlet | \(\dot Q_{\mathrm{ref}}^*\) 为较大侧 patch 热流；\(\mathrm{ratio} \ll 1\) |
| 一面 Dirichlet + 绝热 Neumann | 仍用 Dirichlet 侧 \(\|\dot Q^*\|\) 作尺度 |
| 仅 Neumann | Validate 拒绝，不产出 ratio |

代码：`compute_incompressible_boundary_mass_balance_3d` 的 patch 循环结构可对照；实现为 `compute_conduction_boundary_heat_balance_3d_typed::<T>`，**禁止**在 case 层手写通量公式。

实现后于 [DEBUG_CHECKLIST.md](../DEBUG_CHECKLIST.md) 增加 conduction 行：先查 patch 名 / Dirichlet 是否存在，再查 `heat_flux_imbalance_ratio` 与 PCG 残差。

---

## 11. 参考文献

1. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere. ISBN 978-0891165224. Ch. 5–6（FVM 扩散与边界处理）。
2. Ferziger, J. H., Perić, M., & Street, R. L. (2020). *Computational Methods for Fluid Dynamics*. Springer. DOI [10.1007/978-3-319-55774-2](https://doi.org/10.1007/978-3-319-55774-2). Ch. 8.
3. Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2011). *Fundamentals of Heat and Mass Transfer* (7th ed.). Wiley. Ch. 3（稳态导热与边界条件）。
4. Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Butterworth-Heinemann. ISBN 978-0-08-099995-1. Ch. 3（FVM 扩散模板；非结构面循环）。

---

## 12. 相关文档

- [fvm_diffusion.md](fvm_diffusion.md) — 1D 有量纲扩散骨架
- [boundary_conditions.md](boundary_conditions.md) §2 — Dirichlet / Neumann 离散
- [unstructured_fvm.md](unstructured_fvm.md) — 非结构面拓扑与度量
- [adr/0008-cgns-io.md](../adr/0008-cgns-io.md) — CGNS 读入 / 写出约定（H2 P0）
- [adr/0016-runtime-compute-precision.md](../adr/0016-runtime-compute-precision.md) — f32/f64 运行时精度模型
- [BENCHMARKS.md](../BENCHMARKS.md) — 算例库（实现后登记）
- [DEBUG_CHECKLIST.md](../DEBUG_CHECKLIST.md) — 排查清单（实现后补充 conduction 行）
