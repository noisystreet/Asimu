# asimu 数据模型

> 本文描述核心数据结构的设计意图与字段约定。实现状态见 [ARCHITECTURE.md](ARCHITECTURE.md) §1.3。  
> 多精度与 GPU：[ARCHITECTURE.md](ARCHITECTURE.md) §8.4 · Run Manifest / Restart：[ARCHITECTURE.md](ARCHITECTURE.md) §8.5

---

## 1. 数值标量类型（`core::Real`）

自 v0.2 起，场变量与线性系统系数统一使用 `Real`，而非散落 `f64`：

```rust
/// 默认 f64；feature `precision-f32` 下为 f32
pub type Real = f64;

pub enum PrecisionMode {
    F64,           // 默认
    F32,           // v0.5+，编译期 feature
    Mixed,         // v0.6+：场 f32，归约/残差 f64
}
```

| 类型 | 使用场景 |
|------|----------|
| `Real` | 场值、矩阵系数、通量、残差 |
| `f64` | 网格节点坐标、I/O 文本解析中间值（几何精度固定双精度） |

**约束**：`ScalarField::new` 构造时校验；`Mixed` 模式下内部可能持有 `values_f32` + 归约 buffer，对外 API 仍通过 `Real` 视图或显式转换（实现期定案）。

---

## 2. 标识符

```rust
// 概念类型 — 实现时可用 newtype 包装 usize
pub struct CellId(pub u32);
pub struct FaceId(pub u32);
pub struct NodeId(pub u32);
```

使用 newtype 而非裸 `usize`，可在编译期区分 ID 语义，减少混用错误。

---

## 3. 网格（`mesh`）

### 3.1 通用接口

所有网格类型实现 `Mesh` trait（或共享基类结构），供 `field` 与 `discretization` 查询规模：

| 方法 / 属性 | 说明 |
|-------------|------|
| `num_nodes()` | 节点数 |
| `num_cells()` | 单元数 |
| `num_faces()` | 面数（含边界） |
| `cell_volume(id)` | 单元体积/面积 |
| `face_normal(id)` | 面法向（单位向量） |
| `face_owner(id)` | 面所属单元（内部面：owner/neighbor） |

### 3.2 结构化网格（v0.2 首选）

```rust
pub struct StructuredMesh2d {
    pub nx: usize,
    pub ny: usize,
    pub x: Vec<f64>,   // 节点 x 坐标，长度 (nx+1)*(ny+1)
    pub y: Vec<f64>,   // 节点 y 坐标
    // 预计算：cell_volumes, face_areas, ...
}
```

**索引约定**：cell `(i, j)` → `CellId = j * nx + i`，`i ∈ [0, nx)`, `j ∈ [0, ny)`。

### 3.3 当前占位类型

v0.1 的 `Mesh { name, cell_count }` 将在 v0.2 替换为上述结构，保留 `name` 作为调试标识。

---

## 4. 物理场（`field`）

### 4.1 标量场

```rust
pub struct ScalarField {
    pub name: String,
    pub values: Vec<Real>,  // len == mesh.num_cells()
}
```

### 4.2 矢量场

```rust
pub struct VectorField2d {
    pub name: String,
    pub x: Vec<Real>,
    pub y: Vec<Real>,
}
```

### 4.3 场集合

```rust
pub struct Fields {
    pub scalars: Vec<ScalarField>,
    pub vectors: Vec<VectorField2d>,
}
```

**约束**：所有场的长度必须与关联网格的 `num_cells()` 一致；构造时校验，违反返回 `AsimuError::Field`。

### 4.4 守恒变量场（v1.x 可压 NS）

可压缩 NS 以守恒变量 SoA 为主状态（见 [adr/0009-compressible-navier-stokes.md](adr/0009-compressible-navier-stokes.md)）：

```rust
pub struct ConservedFields {
    pub density: ScalarField,       // ρ
    pub momentum_x: ScalarField,    // ρu
    pub momentum_y: ScalarField,    // ρv
    pub momentum_z: ScalarField,    // ρw
    pub total_energy: ScalarField,  // ρE
}
```

| 类型 | 说明 |
|------|------|
| `ConservedState` | 单单元 \([\rho, \rho u, \rho v, \rho w, \rho E]\) |
| `PrimitiveState` | 派生原始变量 \((\rho, \mathbf{u}, p, T)\) |
| `IdealGasEoS` | 理想气体闭合 \(p=\rho R T\) |

原始变量**不作为**时间推进主存储；由 `physics::eos` 从 `ConservedState` 计算。

---

## 5. 边界条件

```rust
pub enum BoundaryKind {
    Dirichlet { value: Real },
    Neumann { flux: Real },
    // v0.3+
    Inlet { velocity: VectorField2d },
    Outlet { pressure: Real },
    Wall { no_slip: bool },
    Symmetry,
}

pub struct BoundaryPatch {
    pub name: String,
    pub face_ids: Vec<FaceId>,
    pub kind: BoundaryKind,
}
```

### 5.1 应用框架（规划）

BC 在 **独立阶段** 应用，顺序：内部面装配 → 按 `patches` 顺序 `apply`：

```rust
pub trait BoundaryCondition {
    fn apply(
        &self,
        patch: &BoundaryPatch,
        mesh: &dyn Mesh,
        fields: &mut Fields,
        system: Option<&mut LinearSystem>,
    ) -> Result<()>;
}

/// v0.3+ — 名称 → 构造，供 io 与扩展
pub struct BoundaryRegistry {
    /* ... */
}
```

**约束**：`io` 只解析为 `BoundaryPatch` 数据；数值操作仅在 `discretization::bc` 或专用 `bc` 模块。

---

## 6. 线性系统（`linalg`）

```rust
pub struct CsrMatrix {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<Real>,
}

pub struct LinearSystem {
    pub matrix: CsrMatrix,
    pub rhs: Vec<Real>,
    pub solution: Vec<Real>,
}
```

v0.2 结构化网格可先用对角或三对角专用存储，接口与 `CsrMatrix` 对齐以便后续泛化。

---

## 7. 求解状态（`solver`）

```rust
pub struct SolveResult {
    pub iterations: u32,
    pub residual: Real,
    pub converged: bool,
}

pub struct SolverState {
    pub iteration: u32,
    pub residual_history: Vec<Real>,
    /// 瞬态模式（ADR 0005）
    pub time: Real,
    pub step: u64,
}
```

---

## 8. 算例描述（`case` / `io`）

```rust
pub struct CaseSpec {
    pub name: String,
    pub mesh_path: Option<PathBuf>,
    pub boundary_patches: Vec<BoundaryPatch>,
    pub initial_fields: Fields,
    pub solver_config: SolverConfig,
    pub numerics: NumericsConfig,   // precision, backend（v0.5+）
}
```

### 8.1 数值与执行配置（规划）

```rust
pub struct NumericsConfig {
    pub precision: PrecisionMode,     // F64 | F32 | Mixed
    pub backend: ExecBackendKind,     // Cpu | GpuWgpu | GpuCuda
}

pub enum ExecBackendKind {
    Cpu,
    GpuWgpu,   // feature gpu-wgpu
    GpuCuda,   // feature gpu-cuda
}
```

TOML 预留：

```toml
[numerics]
precision = "f64"      # f64 | f32 | mixed
backend = "cpu"        # cpu | gpu-wgpu | gpu-cuda
```

### 8.2 占位 case 文件格式（v0.1，遗留）

```
name=<mesh_name>;cells=<count>
```

v0.2 起新算例请使用 TOML；完整 schema 见 [CASE_FORMAT.md](CASE_FORMAT.md)。遗留单行格式兼容至 v0.3。

### 8.3 时间推进配置（规划，ADR 0005）

```rust
pub enum TimeMode {
    Steady,
    Transient,
}

pub struct TimeConfig {
    pub mode: TimeMode,
    pub dt: Real,
    pub cfl_max: Option<Real>,
}
```

```toml
[time]
mode = "steady"    # steady | transient
dt = 1.0e-3
cfl_max = 0.5
```

### 8.4 参数扫描 Study（规划，v0.5+）

```rust
pub struct StudyConfig {
    pub parameter: String,
    pub values: Vec<Real>,
    pub output_dir: PathBuf,
}
```

```toml
[study]
parameter = "Re"
values = [100, 400, 1000]
output_dir = "output/study_re"
```

---

## 9. Case 文件格式（v0.2）

v0.2 算例采用 **TOML**（`case.toml`），字段包括 `name`、`mesh`、`physics`、`boundary`、可选 `solver` / `time`。

| 文档 | 内容 |
|------|------|
| [CASE_FORMAT.md](CASE_FORMAT.md) | 完整 schema、示例、迁移说明 |
| `tests/benchmarks/*/case.toml` | V&V 算例实例 |

解析实现：`io` 模块（Parse → Validate → Trust）。

---

## 10. Run Manifest（运行清单，v0.3+）

每次运行写入 `output/run-manifest.json`（路径可配置）。详见 [ARCHITECTURE.md](ARCHITECTURE.md) §8.5.1、[OBSERVABILITY.md](OBSERVABILITY.md)。

```rust
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub asimu_version: String,
    pub git_commit: Option<String>,
    pub config_hash: String,
    pub precision: PrecisionMode,
    pub backend: ExecBackendKind,
    pub case_name: String,
    pub benchmark_id: Option<String>,
    pub time: TimeConfig,
    pub started_at: String,
    pub finished_at: String,
    pub solve: SolveResult,
    pub observability: ObservabilitySummary,
    pub output_paths: Vec<PathBuf>,
}

pub struct ObservabilitySummary {
    pub wall_time_sec: Real,
    pub metrics_path: Option<PathBuf>,
    pub phases_ms: Option<PhaseTimings>,
}

pub struct PhaseTimings {
    pub assemble: u64,
    pub linear_solve: u64,
    pub io: u64,
}
```

MCP Resource（v1.2+）：`asimu://run/latest` → 最近一次 manifest JSON。

---

## 11. 时间推进抽象（`TimeIntegrator`，ADR 0005）

```rust
pub enum TimeMode {
    Steady,
    Transient,
}

pub struct TimeStepInfo {
    pub dt: Real,
    pub physical_time: Real,
    pub step: u64,
    pub is_final: bool,
}

pub trait TimeIntegrator {
    fn mode(&self) -> TimeMode;
    fn advance(&mut self, state: &mut SolverState) -> Result<TimeStepInfo>;
    fn suggested_dt(&self, mesh: &dyn Mesh, fields: &Fields) -> Result<Real>;
}
```

| 实现 struct | 版本 |
|-------------|------|
| `SteadyStateIntegrator` | v0.2 |
| `ExplicitEulerIntegrator` | v0.4 |
| `RungeKutta4Integrator` | v0.5+（评估） |

模块路径：`solver/time/`。`solver` 持有 `Box<dyn TimeIntegrator>` 或 enum dispatch（v0.2 优先 enum 避免 trait object 开销）。

---

## 12. Checkpoint / Restart（v0.4+）

```rust
pub struct RestartSnapshot {
    pub schema_version: u32,
    pub mesh_fingerprint: String,
    pub fields: Fields,
    pub solver_state: SolverState,
    pub time_config: TimeConfig,
    /// 默认不保存完整 LinearSystem
    pub linear_system: Option<LinearSystem>,
}
```

| 字段 | 说明 |
|------|------|
| `schema_version` | 用于向前/向后兼容策略 |
| `mesh_fingerprint` | 网格哈希，防止错配 restart |
| 文件扩展名 | `.asimu-restart`（规划） |

`io::restart::{read,write}` 负责序列化；见 [ARCHITECTURE.md](ARCHITECTURE.md) §8.5.2。

---

## 13. I/O 资源上限（规划）

```rust
pub struct IoLimits {
    pub max_file_bytes: u64,      // 默认 256 MiB
    pub max_cells: u64,           // 默认 1e8
    pub max_case_lines: u64,
}
```

在 `io` 解析入口强制；与 [SECURITY.md](../SECURITY.md) 及 MCP 沙箱一致。

---

## 14. 执行后端（`exec`，v1.2+ 规划）

`exec` 层封装设备存储与 kernel 调度，算法层只持有 `ExecutionContext`：

```rust
pub struct ExecutionContext {
    pub backend: Box<dyn ExecBackend>,
    pub precision: PrecisionMode,
}

pub trait ExecBackend: Send + Sync {
    fn name(&self) -> &'static str;
    /// 在设备上分配场 buffer（CPU 即 Vec<Real>）
    fn alloc_field(&self, len: usize) -> Result<FieldBuffer>;
    /// 通量装配 / SpMV 等热算子入口
    fn spmv(&self, matrix: &CsrMatrix, x: &FieldBuffer, y: &mut FieldBuffer) -> Result<()>;
}

pub enum FieldBuffer {
    Cpu(Vec<Real>),
    #[cfg(feature = "gpu-wgpu")]
    GpuWgpu(/* 设备 buffer 句柄 */),
}
```

**约束**：

- `discretization` / `linalg` 通过 `ExecutionContext` 调用，不直接依赖 wgpu/CUDA API
- GPU buffer 与 CPU `ScalarField` 同步经 `exec::copy_host_to_device` / `copy_device_to_host` 显式执行
- 主 crate 默认 `unsafe_code = forbid`；GPU 底层驱动封装在 `exec/gpu/` 或独立 crate

---

## 15. 错误映射

| 场景 | 错误变体 |
|------|----------|
| 场长度与网格不匹配 | `AsimuError::Field` |
| 无效 cell/face 索引 | `AsimuError::Mesh` |
| 矩阵维度不一致 | `AsimuError::Linalg` |
| 线性求解不收敛 | `AsimuError::Solver` |
| GPU 设备不可用 / kernel 失败 | `AsimuError::Exec`（v1.2+ 规划） |
| 精度模式与 feature 不匹配 | `AsimuError::Config` |
| restart 与 mesh 不匹配 | `AsimuError::Io`（规划） |
| 输入文件超限 / 路径非法 | `AsimuError::Io`（规划） |
| manifest 写入失败 | `AsimuError::Io`（规划） |

规划在 `error.rs` 中扩展 `Exec`；`Field` / `Linalg` 已在 v0.2 骨架实现。

---

## 16. 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 模块职责与依赖
- [CASE_FORMAT.md](CASE_FORMAT.md) — v0.2 算例 TOML schema
- [OBSERVABILITY.md](OBSERVABILITY.md) — 性能与 metrics
- [API.md](API.md) — 公开 Rust API
- [BENCHMARKS.md](BENCHMARKS.md) — 验证算例库
- [adr/0003-multi-precision-and-gpu.md](adr/0003-multi-precision-and-gpu.md)
- [adr/0005-time-integration.md](adr/0005-time-integration.md)
- [adr/0006-ffi-interop.md](adr/0006-ffi-interop.md)
- [adr/0009-compressible-navier-stokes.md](adr/0009-compressible-navier-stokes.md)
