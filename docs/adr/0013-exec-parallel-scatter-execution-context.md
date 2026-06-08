# ADR 0013: `exec` 并行 scatter 与 `ExecutionContext`

- **状态**: 已接受（规划基线，实现分阶段）
- **日期**: 2026-06-08
- **关联**: [ADR 0003](0003-multi-precision-and-gpu.md)、[ADR 0011](0011-parallel-fvm-face-coloring.md)、[ARCHITECTURE.md](../ARCHITECTURE.md) §8.4、[unstructured_fvm.md](../theory/unstructured_fvm.md)

## 背景

### 1. 当前并行模型的 Amdahl 上限

[ADR 0011](0011-parallel-fvm-face-coloring.md) 已落地 **graph coloring + 桶内 compute 并行 + scatter 串行**：

| 阶段 | 现状（P8/P8′/P9 后） | 约束 |
|------|----------------------|------|
| compute | `rayon` 桶内并行；SIMD batch4 在 `exec::cpu` | 同色面无单元冲突 |
| scatter | **主线程串行**写 `&mut [Real]` | 主 crate `unsafe_code = forbid` |

dual_ellipsoid（221 万单元 / ~475 万内面 / 9 色）Chrome trace 表明：粘性/无粘 **fused interior flux** 仍占 LU-SGS RHS 约 **70–80%**；其中 **scatter 约占 flux 的 35–40%**。在 compute 已并行且 P9 去掉 gather 物化后，**scatter 串行**成为下一步架构级瓶颈。

### 2. `exec` 层现状与 ADR 0003 缺口

[ADR 0003](0003-multi-precision-and-gpu.md) 规划了 `ExecutionContext { backend: Cpu | Gpu }`，但 v0.x 实现仅为：

- `src/exec/cpu/` 下若干 **无状态** SIMD 内核（viscous batch4、Roe/HVL、LSQ、LU-SGS 对角）
- `discretization` 仍直接持有 `rayon`、残差 `&mut` slice，并内联 scatter

缺少：

1. **统一的执行上下文**（buffer 生命周期、并行策略、backend 选择）
2. **可并行的 residual 累加原语**（在不污染 `discretization` 的前提下突破 `unsafe_code = forbid`）
3. **GPU 与 CPU 共用**的「面通量 → 单元残差」接口形态

### 3. 目标

在不破坏 [ARCHITECTURE.md](../ARCHITECTURE.md) §7 依赖方向的前提下：

- 将 **并行 scatter** 与 **（可选）原子/归约** 下沉 `exec`
- 引入 **`ExecutionContext`** 作为 `discretization` / `linalg` 访问热算子的唯一边界
- 保持 **Parse → Validate → Trust**：着色正确性仍在 mesh cache 构造期验证；`exec` 热路径信任已着色输入
- 为 v1.2+ GPU 面循环预留 **同一套** `FaceFluxScatter` 语义

## 决策

### 1. `ExecutionContext` 职责边界

```text
discretization / linalg
        │  （只读 mesh 拓扑 + 场；不直接 rayon/unsafe）
        ▼
  ExecutionContext
        ├── backend: ExecBackend（CpuScalar | CpuParallel | Gpu…）
        ├── buffers: ExecBuffers（场 alias / 设备 buffer / scratch）
        └── kernels: ExecKernels（flux batch4、scatter、SpMV…）
        ▼
   cpu/ · gpu/（feature-gated，允许 approved unsafe）
```

| 层 | 负责 | 不负责 |
|----|------|--------|
| **discretization** | 遍历面/单元顺序、BC 语义、调用 `ctx.kernels().…` | `rayon` 线程池、`AtomicU64`、CUDA/wgpu API |
| **exec** | 并行策略、scatter 原语、SIMD/GPU kernel、scratch 复用 | 通量公式物理含义、Riemann 格式选型 |
| **solver / case** | 持有 `ExecutionContext` 与步间 `ExecScratch` | 面循环内层实现 |

**构造**：`ExecutionContext` 由 **case / solver 编排层**在算例启动时从 `ExecConfig` 构建（只读 config，不在热路径读环境变量）。

### 2. 核心 API（规划名，v1.0 落地时可微调）

#### 2.1 上下文与 backend

```rust
/// 执行后端枚举；默认 CpuParallel（与 parallel-fvm 对齐）。
pub enum ExecBackend {
    CpuScalar,
    CpuParallel,   // rayon + 0013 scatter
    #[cfg(feature = "gpu-wgpu")]
    GpuWgpu,
}

pub struct ExecutionContext {
    backend: ExecBackend,
    // 步间复用；长度由 mesh 规模推导，case 层 create / reset
    scratch: ExecScratch,
}

pub struct ExecConfig {
    pub backend: ExecBackend,
    pub parallel_min_len: usize,  // 默认 1024；桶内并行 compute/scatter 下限
    pub scatter_mode: ScatterMode, // 默认 Auto（见 §2.4）
}
```

#### 2.2 着色面装配：compute + scatter 分离但同属 exec

```rust
/// 单面通量对 owner/neighbor 的 scatter 贡献（粘性/无粘共用布局）。
pub struct FaceScatterContribution {
    pub owner: CellId,
    pub neighbor: CellId,
    pub owner_scale: Real,
    pub neighbor_scale: Real,
    pub flux: [Real; NVAR],  // NVAR = 该装配核守恒变量数（粘性 4，无粘 5…）
}

/// 同色桶内：并行 compute，并行 scatter（0013 核心）。
pub trait ColoredFaceScatterKernel {
    fn compute_face(&self, face_idx: usize) -> Option<FaceScatterContribution>;
}

impl ExecutionContext {
    pub fn scatter_colored_bucket<K: ColoredFaceScatterKernel>(
        &mut self,
        kernel: &K,
        face_indices: &[usize],
        residual: &mut dyn ResidualWriter,  // exec 侧 trait object 或泛型 monomorphize
    ) -> Result<()>;
}
```

**语义**：

- **桶间仍串行**（不同色可写同一单元，必须顺序累加各色桶）
- **桶内** compute 与 scatter 均由 `exec` 调度；`discretization` 仅提供 `ColoredFaceScatterKernel` 实现（通量公式）

#### 2.3 `ScatterMode`（并行 scatter 策略）

| 模式 | 机制 | 适用 | 备注 |
|------|------|------|------|
| **`Auto`** | **算例启动时**按 §2.4 解析为 `Serial` 或 `ParallelUnsafeAtomics` | **默认** | 配置默认；trace 记录解析结果 |
| `Serial` | 单线程 indexed add | golden 基线 / 显式强制 | 用户显式指定时不再自动升级 |
| `ParallelUnsafeAtomics` | `AtomicU64` CAS（`exec` 内 approved unsafe） | 大网格、`CpuParallel` | 用户显式指定时 **不**因网格小降级 |
| `ParallelLocalBuffer` | 线程局部 buffer + 桶末 reduction | 无原子平台 / 调试 | feature 或平台回退 |
| `GpuScatter` | device atomic / reduction | v1.2+ | feature `gpu-wgpu` |

**配置默认（v1.0）**：`scatter_mode = Auto`（**不是**裸 `ParallelUnsafeAtomics`）。`Auto` 在大网格上解析为 `ParallelUnsafeAtomics`，在小网格上解析为 `Serial`（见 §2.4）。

**显式 override**（TOML / `ExecConfig` 构造）：

```toml
# 可选；省略时 = "auto"
[exec]
scatter = "auto"   # auto | serial | atomic
# parallel_min_faces = 65536   # 可选，覆盖 Auto 阈值（Parse → Validate）
```

主 crate 仍 `unsafe_code = forbid`；原子实现位于 `src/exec/`（见 §4）。

**浮点原子**：x86_64 / aarch64 上对 `f64` 使用 `AtomicU64` + `to_bits`/`from_bits` 循环 CAS；非原生 `AtomicF64` 平台走 `ParallelLocalBuffer` 回退或编译期拒绝该 backend。

#### 2.4 默认 `Auto`：小网格自动降级为 `Serial`

**决策**：**是**——小网格自动走 `Serial`，避免 rayon 调度 + 原子 CAS 固定开销超过收益；**不在热路径每步分支**，仅在 `ExecutionContext` **构造时**解析一次，结果缓存于 `ExecutionContext` 内（`resolved_scatter_mode`）。

**解析时机**：`ExecutionContext::new(config, mesh_exec_metrics)`（mesh cache 已有着色与 `num_interior_faces`）。

**`Auto` → `ParallelUnsafeAtomics` 当且仅当**以下 **全部** 成立：

| # | 条件 | 默认阈值 | 说明 |
|---|------|----------|------|
| 1 | `config.backend == CpuParallel` | — | `CpuScalar` 恒为 `Serial` |
| 2 | `interior_faces >= scatter_parallel_min_faces` | **`65_536`**（\(2^{16}\)） | 命名常量 `EXEC_SCATTER_PARALLEL_MIN_FACES` |
| 3 | `max_color_bucket_faces >= parallel_min_len` | **`1024`** | 最大着色桶内面数；与 rayon `with_min_len` 同量级 |

否则 `Auto` → **`Serial`**。

**桶级二次降级**（仅当已解析为 `ParallelUnsafeAtomics` 时）：

- 对 **单个着色桶**，若 `bucket.len() < parallel_min_len` → 该桶 **串行** scatter（与现有 `with_min_len` 语义一致）
- 不改变 context 级 `resolved_scatter_mode`；trace span `exec_colored_bucket_scatter` 标注 `bucket_serial=true`

**不降级的情况**：

- 用户显式 `scatter_mode = ParallelUnsafeAtomics` → **始终** atomic（含小网格；CI/benchmark 可强制对比）
- 用户显式 `scatter_mode = Serial` → **始终** serial（golden / 回归基线）

**可观测性**（构造时一次 `tracing::info!`）：

```text
exec_scatter_mode_resolved mode=atomic|serial reason=auto|explicit
  interior_faces=… max_bucket_faces=… parallel_min_len=…
```

**阈值依据**（POC 假设，E5 用 benchmark 修订）：

- dual_ellipsoid（~475 万面）远高于 65536，必走 atomic
- 单元测试 / 3×3×3 小网格（\(O(10^2)\) 面）走 serial，与 ADR 0011「小网格可能无并行收益」一致
- 65536 面 ≈ 中等网格下限；可在 `config` 暴露 `scatter_parallel_min_faces` 覆盖，**禁止**热路径读环境变量

**与 `parallel-fvm` 的关系**：

| `parallel-fvm` | `Auto` 典型解析 | compute |
|----------------|----------------|---------|
| 启用（默认） | 大网格 → atomic；小网格 → serial | 仍可按 `parallel_min_len` 并行 |
| 禁用 | 恒 `Serial` | 串行 |

小网格 `Auto → Serial` **仅影响 scatter**；compute 是否并行仍由 `ExecBackend` / feature 决定（小网格 compute 并行也可因 `with_min_len` 自然退化为串行）。

### 3. 与现有 `parallel-fvm` / `simd-fvm` 的关系

| 现有 feature | v1.0 映射 | 说明 |
|--------------|-----------|------|
| `parallel-fvm` | `ExecBackend::CpuParallel` | 默认启用；`discretization` 逐步移除直接 `rayon` 依赖 |
| `simd-fvm` | `ExecKernels::cpu_simd()` | batch4 仍在 `exec::cpu`；由 `ExecutionContext` 统一 dispatch |
| 无 `parallel-fvm` | `CpuScalar` | 串行 scatter + 串行 compute |

**迁移原则**：

1. **Phase A**：引入 `ExecutionContext` + `Serial` scatter；行为与 P8 一致，golden 不变
2. **Phase B**：桶内改 `ParallelUnsafeAtomics` scatter；新增 golden「并行 scatter vs 串行 scatter」
3. **Phase C**：删除 `discretization` 内 `rayon` 与 `parallel_bucket_*` 缓冲；仅保留 kernel trait 实现

`Cargo.toml` 过渡期可保留 `parallel-fvm = ["dep:rayon"]`，但 **rayon 仅 `exec` 依赖**（`discretization` 不再直接依赖 `rayon`）。

### 4. unsafe 与 crate 边界

| 位置 | `unsafe` | 说明 |
|------|----------|------|
| 主 crate `asimu` | **禁止**（保持） | AGENTS / `[lints.rust]` |
| `src/exec/` | **允许**（本 ADR 批准，限于 scatter atomics + GPU FFI） | 模块顶部 `#![allow(unsafe_code)]` + rustdoc 说明 |
| 规划 `asimu-exec-unsafe` | 可选拆分 | 若 clippy/audit 需隔离；对外只 re-export safe API |

**禁止**：

- 在 `discretization` 内为 scatter 打开 unsafe
- `Mutex`/`RwLock` 包裹残差数组（AGENTS 热路径禁令）

### 5. `ExecScratch` 与步间缓冲

P8′ 引入的 **桶级 flat buffer**（`parallel_bucket_geoms` 等）迁入 `ExecScratch`：

```rust
pub struct ExecScratch {
    /// 每桶 batch×4 + remainder 槽；按 max_bucket_faces 扩容
    colored_face_buffer: ColoredFaceBuffer,
    /// IDWLS / SpMV 等其它 exec 级 scratch（与 discretization 解耦）
    // ...
}
```

- **case / solver** 在网格加载后 `ExecScratch::with_mesh(&mesh_cache)` 一次分配
- **discretization** 通过 `&mut ExecutionContext` 借用 scratch，不拥有 Vec 生命周期

### 6. GPU 路径（v1.2+，本 ADR 仅定接口）

GPU 面循环沿用同一 **`FaceScatterContribution`** 布局：

1. Host：着色桶顺序 launch kernel
2. Device：batch compute flux → scatter（atomic 或 warp reduction）
3. Host：BC / 收敛判断不变

`ExecutionContext::sync()` 仅在 GPU backend 需要；CPU 路径为零开销空操作。

### 7. 依赖方向（修订 ADR 0003 §4）

```
core ← exec
discretization → exec（trait 调用，无反向依赖）
linalg → exec（SpMV 等，v1.2+）
exec ↛ discretization / solver / io / case
```

`exec` **不得**引用 `InteriorFaceBatchStatic4` 等 discretization 类型。batch 几何经 **exec 自有** `ExecFaceBatchGeom4` 传入（init-time 从 mesh cache 转换一次）。

### 8. 验证与 CI

| 测试 | 判据 |
|------|------|
| `scatter_serial_matches_atomic_parallel` | 同色桶：Serial vs ParallelUnsafeAtomics，残差 `approx_eq`（1e-12） |
| `colored_bucket_atomic_matches_full_serial_face_order` | 与 ADR 0011 线性面序 golden 一致 |
| `exec_context_cpu_scalar_matches_legacy_path` | Phase A 迁移期：新 API vs 旧 `#[cfg]` 路径 |
| dual_ellipsoid benchmark | manifest 记录 `lusgs_rhs` / `visc_flux_fused`；相对 P9 基线回归 < 5% |
| GPU（可选） | `#[ignore = "gpu"]`；Cpu vs Gpu `approx_eq` |

**CI 矩阵**（Makefile / GitHub Actions）：

- `make check`：`scatter_mode = Auto`（单测小网格自动 `Serial`；与现行为一致）
- `make check-exec-parallel-scatter`：显式 `ParallelUnsafeAtomics` 或 dual_ellipsoid 级 fixture（验证 atomic 路径）
- 无 GPU runner 时跳过 gpu job

### 9. 可观测性

Chrome trace span 迁移：

| 旧 span | 新 span（exec 内） |
|---------|-------------------|
| `unstructured_*_interior_flux_fused` | 保留；内层增加 `exec_colored_bucket_scatter`（`mode=atomic\|serial`） |
| — | `exec_scratch_ensure`（init / 扩容，仅首步或非热路径） |

## 后果

### 正面

- 打破 ADR 0011 scatter 串行 Amdahl 上限（dual_ellipsoid 预期 flux 再降 **~30–40%**）
- `discretization` 瘦身：移除 `rayon`、flat buffer、`mem::take` 并行技巧
- CPU/GPU 共用 scatter 语义，符合 ARCHITECTURE §8.4.3
- unsafe 隔离在 `exec`，主库可审计、可 forbid

### 负面 / 限制

- 引入 `ExecutionContext` 增加一层 indirection（热路径需 monomorphize / inline 避免虚调用）
- 原子 scatter 在 **单桶面数 < parallel_min_len** 时桶级退化为 serial（§2.4）；极端小网格由 **`Auto` 整体解析为 Serial**
- `ParallelLocalBuffer` 内存峰值 = O(threads × n_cells × n_var)（仅作回退）
- GPU scatter 与 CPU 浮点结合顺序差异可能导致末位差异（golden 分 backend）

## 未采纳

| 方案 | 原因 |
|------|------|
| 主 crate 内 `unsafe` 原子 scatter | 违反现有 lint；破坏 audit 边界 |
| `discretization` 内 `Mutex` 并行写残差 | AGENTS 禁止热路径锁；性能不可预测 |
| 取消着色、改用 Metis 分区 + halo exchange | 范围过大；MPI 远期 ADR；不着色无法保证桶内无冲突 |
| 仅优化 compute、不碰 scatter | trace 显示 scatter 已是 flux 内第二大项；收益递减 |
| 运行时 `enum` dispatch 每面选 backend | 热路径分支过多；编译期 / 算例级选定 backend |
| 默认强制全网格 `ParallelUnsafeAtomics`（无 Auto） | 小网格固定开销；与 ADR 0011 POC 结论不符 |
| 每步动态检测网格规模切换 scatter | 隐式状态、热路径分支；改为构造时 `Auto` 解析一次 |

## 实现里程碑

| 阶段 | 交付 | 目标版本 |
|------|------|----------|
| **E0** | `ExecutionContext` + `ExecScratch` 骨架；`Serial` scatter 包装现有 P8 路径 | v1.0-alpha |
| **E1** | `ParallelUnsafeAtomics` scatter；粘性/无粘 unstructured interior；golden | v1.0 |
| **E2** | `discretization` 移除 `rayon` 依赖；`ExecFaceBatchGeom4` 与 mesh cache 转换 | v1.0 |
| **E3** | IDWLS RHS / SpMV 经 `ExecutionContext`（与 flux 共用 scratch 池） | v1.1 |
| **E4** | `GpuWgpu` prototype：batch4 flux + device scatter | v1.2 |
| **E5** | dual_ellipsoid + 小网格 benchmark；修订 `EXEC_SCATTER_PARALLEL_MIN_FACES` 若需 | v1.0+ |

## 对关联 ADR 的修订指引

### ADR 0011（追加段落，不删除历史条目）

在 **实现追踪** 表追加：

| 项 | 状态 |
|----|------|
| **`exec` 并行 scatter（0013）** | **规划**（v1.0，`Auto` 默认，大网格 → `ParallelUnsafeAtomics`） |
| `discretization` 直接 `rayon` | **过渡期**；E2 后 deprecated |

**§2 compute/scatter 分离** 补充：scatter 串行为主 crate 过渡策略；v1.0 起 scatter **并行原语** 由 [ADR 0013](0013-exec-parallel-scatter-execution-context.md) 在 `exec` 实现。

### ADR 0003

**§2 执行后端** 中 `ExecutionContext` 由「规划」更新为「**接口已定义（0013），GPU 仍 v1.2+**」。

## 文档与理论同步

| 文档 | 变更 |
|------|------|
| [docs/theory/unstructured_fvm.md](../theory/unstructured_fvm.md) | 面着色 § 增加 exec scatter 模式说明 |
| [docs/API.md](../API.md) | 公开 `exec::ExecutionContext`（若对外暴露） |
| [docs/en/ARCHITECTURE.md](../en/ARCHITECTURE.md) | §8.4 摘要同步 |
| [CHANGELOG.md](../../CHANGELOG.md) | E0/E1 落地时记录 |

## 实现追踪

| 项 | 状态 |
|----|------|
| ADR 0013 文本 | **已接受（规划）** |
| **`ScatterMode::Auto` + 小网格降级策略（§2.4）** | **2026-06-08 定案** |
| **E0** `ExecutionContext` / `ExecScratch` / Serial scatter | **2026-06-08 已实现** |
| **E1** `ParallelUnsafeAtomics` scatter + golden | **2026-06-08 已实现** |
| **E2** `discretization` 去 `rayon`；`ExecFaceBatchStatic4` | **2026-06-08 已实现** |
| **E3** IDWLS / SpMV 经 `ExecutionContext` | **2026-06-08 已实现** |
| **E5** dual_ellipsoid benchmark + scatter 契约测试 | **2026-06-08 已实现**（`tests/benchmarks/dual_ellipsoid/`；`make check-exec-parallel-scatter`） |
| **E4** GPU scatter | v1.2+ 规划 |

修订本 ADR 时 **不删除** 已有条目；默认 scatter 模式变更或主 crate unsafe 政策变更须新开修订段落。
