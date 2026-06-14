# ADR 0017: CUDA 执行后端（`cudarc`）与 `exec` 多 Backend 模型

- **状态**: 已接受（规划基线，实现分阶段 G0–G4）
- **日期**: 2026-06-13
- **关联**: [ADR 0003](0003-multi-precision-and-gpu.md)、[ADR 0013](0013-exec-parallel-scatter-execution-context.md)、[ADR 0016](0016-runtime-compute-precision.md)、[ARCHITECTURE.md](../ARCHITECTURE.md) §8.4、[DATA_MODEL.md](../DATA_MODEL.md) §14

## 背景

### 1. 选型结论

在 NVIDIA 服务器场景下，非结构可压缩 **f32** 热路径（着色桶面通量、device scatter、CSR SpMV）优先采用 **CUDA**，而非 wgpu。公开基准与 asimu 面循环 workload 均表明：深度优化的 CUDA 路径在峰值吞吐上通常领先跨平台 compute shader 一个数量级量级（见项目讨论与 [TUM 多面几何对比](https://mediatum.ub.tum.de/doc/1781596/oe6dqmi581t5aormklgwj6hge.pdf)）。

[ADR 0003](0003-multi-precision-and-gpu.md) 已规划 `cuda` feature，但尚未批准具体 Rust 绑定与目录结构。本 ADR 定案：

- 宿主侧 CUDA 绑定：**`cudarc`**（crates.io，MIT/Apache-2.0）
- 设备 kernel 源语言：**CUDA C++**（`.cu`），经 **build 时 `nvcc` 预编译** 为 PTX/CUBIN
- 运行时 NVRTC（`cudarc::nvrtc`）仅作 **开发/调试** 可选路径，非生产默认

### 2. 现有 `exec` 与多 Backend 缺口

[ADR 0013](0013-exec-parallel-scatter-execution-context.md) 已实现 `ExecutionContext`、CPU scatter（E0–E3）、`ExecScratch`、SpMV 入口。当前局限：

| 项 | 现状 | 缺口 |
|----|------|------|
| `ExecBackend` | 仅 `CpuScalar` / `CpuParallel` | 无 `GpuCuda`；CPU 并行策略与设备族混在同一枚举 |
| Case `[numerics]` | 已解析 `compute_precision`（ADR 0016） | **未**解析 `backend` |
| 热算子分发 | scatter / SpMV / IDWLS 在 CPU 分支实现 | 无设备 buffer、无 H2D/D2H 契约 |
| 依赖 | 无 CUDA crate | 需 ADR 批准 `cudarc` + 系统 CUDA toolkit |

若不先统一 **多 Backend 模型**，直接在 `discretization` 内引入 `cudarc` 将破坏分层（ADR 0003 §4）并导致 CPU/GPU 双路径散落。

### 3. 目标

1. 在 **`exec` 层** 接入 CUDA（`cudarc`），`discretization` / `linalg` / `solver` 仅经 `ExecutionContext` 调用。
2. 定义可扩展的 **多 Backend 配置与分发**，使 CPU、CUDA（及远期 wgpu）共用同一 scatter / SpMV **语义**。
3. 首版覆盖 **非结构可压缩 f32** typed 路径的一阶无粘内面通量 + device scatter；SpMV、粘性、MUSCL、GMRES 分阶段跟进。
4. 默认构建 **零 CUDA 依赖**；CI 无 GPU 时可跳过 `#[ignore = "gpu"]` 测试。

## 决策

### 1. 依赖：`cudarc` 与 Cargo feature

| 项 | 决策 |
|----|------|
| Crate | **`cudarc`**（版本在实现期 pin 至当前 stable，如 `0.19.x`） |
| Feature 名 | **`cuda`**（与 ADR 0003 / `config/default.toml` 注释一致） |
| `cudarc` features | `driver`、`nvrtc`（可选 dev）、`cusparse`；CUDA 版本用 `cuda-version-from-build-system` 或显式 `cuda-12080` 等 |
| 引入位置 | **仅** `src/exec/gpu/cuda/`（及可选 `kernels/*.cu`、`build.rs` 片段）；主 crate `Cargo.toml` 以 `optional = true` 声明 |
| 许可证 | MIT/Apache-2.0，与 asimu 兼容；**不**引入 GPL 系 CUDA 封装 |
| 系统依赖 | 运行：NVIDIA 驱动 + CUDA toolkit（`libcuda`、`libcudart`、`libnvrtc` 等）；构建：`nvcc`（预编译 kernel 时） |

**禁止**：在 `discretization`、`solver`、`linalg` 生产路径 `use cudarc::…`。

**`unsafe` 边界**：主 crate 保持 `forbid unsafe_code`；`src/exec/mod.rs` 已 `#![allow(unsafe_code)]`（ADR 0013）。CUDA driver 调用封装在 `exec::gpu::cuda` 内。

### 2. 多 Backend 配置模型

将「**设备族**」与「**CPU 并行策略**」分离，避免 `GpuCuda` 与 `CpuParallel` 语义纠缠。

#### 2.1 配置类型（Parse → Validate 后只读）

```rust
/// 执行设备族（算例级选定，热路径不切换）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    // 远期，本 ADR 不实现：
    // #[cfg(feature = "gpu-wgpu")]
    // Wgpu,
}

/// 仅当 device = Cpu 时生效。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecCpuPolicy {
    Scalar,
    Parallel,   // 需 feature parallel-fvm
}

pub struct ExecConfig {
    pub device: ExecDevice,
    pub cpu_policy: ExecCpuPolicy,
    pub compute_precision: ComputePrecision,
    pub scatter_mode: ScatterMode,
    pub parallel_min_len: usize,
    pub scatter_parallel_min_faces: usize,
}
```

**过渡期兼容**：保留 `ExecBackend` 作为 **解析/日志用扁平视图**（deprecated 别名），由 `(device, cpu_policy)` 映射：

| `ExecDevice` | `ExecCpuPolicy` | 等价旧 `ExecBackend` |
|--------------|-----------------|----------------------|
| `Cpu` | `Scalar` | `CpuScalar` |
| `Cpu` | `Parallel` | `CpuParallel` |
| `Cuda` | `_`（忽略） | `GpuCuda`（新增） |

实现期可先扩展 `ExecBackend` 枚举添加 `GpuCuda`，再逐步迁移到 `ExecDevice`；**对外 case TOML 使用 `backend` 字符串**，内部统一为 `ExecConfig`。

#### 2.2 Case TOML

```toml
[numerics]
compute_precision = "f32"   # f64 | f32（ADR 0016）
backend = "cuda"            # cpu | cuda（首版）；gpu-wgpu 远期
```

| `backend` | 要求 |
|-----------|------|
| `cpu`（默认） | 无额外 feature；`cpu_policy` 由 `parallel-fvm` 默认 `Parallel`，否则 `Scalar` |
| `cuda` | 编译启用 `cuda` feature；运行期检测到 NVIDIA 设备；Validate 校验精度/求解器组合 |

解析入口：`io::case_numerics` 扩展 `CaseNumericsConfig { backend: ExecDeviceKind }`；`case::validate::exec_backend` 校验 feature 与求解器能力矩阵。

#### 2.3 Validate 规则（首版）

| 组合 | 结果 |
|------|------|
| `backend = cuda`，未编译 `cuda` feature | `AsimuError::Config` |
| `backend = cuda`，`compute_precision = f64` | 首版 **报错**（G1 仅 f32）；G2+ 可扩展 |
| `backend = cuda`，非结构 + GMRES | 报错（与 ADR 0016 typed 路径一致） |
| `backend = cuda`，无可用 CUDA 设备 | 启动时 `AsimuError::Exec`（或 Config，实现期定案） |
| `backend = cpu` | 行为与当前一致 |

### 3. `ExecutionContext` 多 Backend 内部结构

```text
ExecutionContext
├── config: ExecConfig          // 只读
├── resolved_scatter: …         // CPU：ADR 0013；CUDA：device atomic / 桶内无冲突直写
├── metrics: MeshExecMetrics
├── scratch: ExecScratch        // CPU 步间 scratch
└── backend_state: BackendState
        ├── Cpu(CpuBackendState)     // 无额外状态（现状）
        └── Cuda(CudaBackendState)     // cudarc CudaContext、Stream、模块、设备池
```

**原则**：

- **构造时** 根据 `ExecConfig.device` 初始化 `BackendState` 一次；热路径 **不** 读 TOML、**不** 按面/单元 `match` 设备类型。
- 对外方法（`scatter_inviscid_pairs_f32`、`csr_spmv`、IDWLS 累加等）在 `exec` 模块边界做 **单次** `match &self.backend_state`，委托到 `cpu::` 或 `gpu::cuda::`。
- **禁止** 在 `discretization` 热循环内 `if ctx.is_cuda()`；应调用统一 API（如 `exec::scatter::scatter_inviscid_pairs_f32` 已具备的入口）。

#### 3.1 设备缓冲与同步

```rust
/// exec 自有场缓冲；算法层不持有 cudarc 类型。
pub enum ExecFieldBuffer<T> {
    Host(Vec<T>),
    #[cfg(feature = "cuda")]
    Device(CudaSlice<T>),   // 封装 cudarc 设备分配
}

impl ExecutionContext {
    /// GPU 路径：BC/收敛/I/O 前将相关场拉回 host；CPU 为零开销。
    pub fn sync_to_host(&mut self) -> Result<()>;

    /// GPU 路径：BC 更新后写回 device；CPU 为零开销。
    pub fn sync_to_device(&mut self) -> Result<()>;
}
```

| 数据 | 存放 | 说明 |
|------|------|------|
| `ConservedFieldsT<f32>` / residual / primitive | CUDA：`Device` | 步间常驻 GPU |
| 面几何、着色桶索引、`ExecFaceBatchStatic4` | CUDA：init 一次 H2D | 来自 `UnstructuredSolverMeshCache` |
| 网格坐标、体积、BC ghost 编排 | CPU `f64` | ADR 0016；BC 在 CPU 算完后 `sync_to_device` |
| RMS 残差累加 | CPU `f64` | ADR 0016 §4 |

### 4. CUDA Kernel 与 `cudarc` 集成

#### 4.1 源码与编译策略

```text
kernels/cuda/              # CUDA C++ 源（真源）
  inviscid_flux_f32.cu
  inviscid_scatter_f32.cu
src/exec/gpu/
  mod.rs                   # #[cfg(feature = "cuda")]
  cuda/
    mod.rs                 # CudaBackendState、ExecutionContext 扩展
    module.rs              # 加载 PTX/CUBIN（cudarc::driver）
    flux.rs                # launch 参数、着色桶 dispatch
    buffers.rs             # H2D/D2H 封装
    spmv.rs                # cusparse（G3+）
```

| 路径 | 用途 |
|------|------|
| **build.rs + `nvcc`** | **生产默认**：`OUT_DIR/*.ptx` 或 fatbin，`include_str!` / `env!("OUT_DIR")` 加载 |
| **`cudarc::nvrtc`** | **仅** `cuda` + `cuda-nvrtc-dev`（可选 feature）或 `#[cfg(debug_assertions)]`：快速改 `.cu` 无需重编 crate |

NVRTC 输入为 **CUDA C++ 字符串**，不是 Rust。禁止首版仅依赖 NVRTC（启动延迟、复现性差）。

#### 4.2 首版 GPU 算子范围（G1）

| 算子 | GPU | CPU 保留 |
|------|-----|----------|
| 无粘一阶内面通量 compute | ✓ 着色桶 kernel | BC、MUSCL 梯度 |
| 无粘内面 scatter → residual | ✓ device atomic 或桶内直写 | — |
| 边界通量 | ✗ 首版 CPU | 不规则访问 |
| 粘性通量 / IDWLS | G2 内面 CUDA / G2+ IDWLS device | G2 内面已实现 |
| CSR SpMV | **G3** `cusparse` 经 `ExecutionContext::csr_spmv` | — |
| 收敛 / CFL / I/O | ✗ CPU | ADR 0003 |

数值语义与 `exec::scatter::scatter_inviscid_pairs_f32`、CPU Roe/HLLC 参考实现 **同公式同着色契约**；golden 容差宽于 f64（ADR 0016）。

#### 4.3 `cudarc` API 使用约定

```rust
// 初始化（solver 构造 ExecutionContext 时一次）
let ctx = CudaContext::new(device_id)?;
let stream = ctx.default_stream();

// 加载模块（启动时一次）
let ptx = /* include_bytes!(env!("OUT_DIR/inviscid.ptx")) */;
let module = ctx.load_module(ptx)?;

// 每 RHS 评估
// 1. sync_to_device（若 BC 刚更新）
// 2. launch kernel(grid, block, stream, params...)
// 3. 仅在 observe/converge 时 sync_to_host
```

- **Stream**：首版单默认 stream；G4 评估多 stream overlap H2D/compute。
- **错误**：`cudarc` 错误映射为 `AsimuError::Exec`（`error.rs` 扩展变体，v1.3 规划项在本 ADR 实现期落地）。
- **日志**：`tracing` 记录 device 名、PTX 架构、kernel 名；禁止 `println!`。

### 5. Scatter 与 CPU 策略在 CUDA 下的关系

| `ExecDevice` | `ScatterMode` / 解析结果 | 行为 |
|--------------|--------------------------|------|
| `Cpu` | ADR 0013 不变 | Serial / ParallelUnsafeAtomics |
| `Cuda` | `ScatterMode` **不映射到 CPU atomic** | 设备侧 scatter；`ParallelUnsafeAtomics` 等价于 device `atomicAdd`；`Serial` 等价于桶内单线程 block 或顺序 dispatch |

`resolve_scatter_mode` 扩展：`device == Cuda` 时 **不** 要求 `parallel-fvm`；GPU scatter 能力由 `CudaBackendState` 声明。

`uses_parallel_cell_loops()` 语义修订：

```rust
pub fn uses_parallel_cell_loops(&self) -> bool {
    match self.device() {
        ExecDevice::Cpu => matches!(self.cpu_policy(), ExecCpuPolicy::Parallel),
        #[cfg(feature = "cuda")]
        ExecDevice::Cuda => false, // 单元环在 GPU kernel 内并行，不经 rayon
    }
}
```

### 6. 依赖方向与模块边界

```text
core ← exec ← discretization
core ← exec ← linalg
solver / case → exec（持有 ExecutionContext，不引用 cudarc）
exec::gpu::cuda → cudarc（可选）
exec ↛ discretization / solver / io
```

`discretization` 继续通过 `InviscidTypedScatterBackend`、`ExecutionContext` 与 `exec::scatter::*` 交互；**不得**新增 `#[cfg(feature = "cuda")]` 分支于面循环内。

### 7. 验证与 CI

| 测试 | 判据 |
|------|------|
| `cpu_f32_matches_cuda_f32_inviscid_single_tet` | 相对误差 < ADR 0016 f32 tol |
| `cuda_matches_cpu_dual_ellipsoid_smoke` | `#[ignore = "gpu"]`；守恒 / 残差趋势一致 |
| 无 GPU CI | 默认 `cargo test` 跳过；可选 self-hosted job `make test-cuda` |
| 构建矩阵 | `make check` 不启用 `cuda`；CI 增加 `check-cuda`（仅编译，可无设备） |

Manifest 记录：`exec_device`、`cuda_device_name`、`kernel_ptx_arch`（Run Manifest 扩展，实现期同步 [DATA_MODEL.md](../DATA_MODEL.md)）。

## 后果

### 正面

- CUDA 路径与 ADR 0013 `ExecutionContext` 自然延伸，CPU 行为不变。
- `ExecDevice` + `ExecCpuPolicy` 为远期 wgpu 留出槽位，无需再改 case schema。
- `cudarc` 生态成熟（driver、nvrtc、cusparse），SpMV 可复用 cuSPARSE。
- f32 非结构生产算例可在同一 binary 用 `backend = "cuda"` 切换。

### 负面

- 维护 **CPU + CUDA** 双路径测试与数值对齐成本。
- 构建/CI 需 CUDA toolkit 工具链（可选 job）；开发者笔记本无 NVIDIA 时仅能测 CPU。
- `ExecutionContext` 体积增大（设备状态、缓冲池）；需严格控制公开 API 面。
- 首版不支持 f64 GPU、边界 GPU 化、GMRES GPU。

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| **wgpu 作为首版 GPU** | 用户定案 NVIDIA HPC；公开基准峰值劣势 |
| **baracuda-nvrtc** 替代 cudarc | 生态较新（alpha）；cudarc 下载量与 cuSPARSE 集成更成熟 |
| **仅 NVRTC、无 nvcc 预编译** | 启动编译延迟；CI/复现性差 |
| **cuda-oxide（Rust 写 kernel）** | 与现有 CPU/CUDA C++ 通量对照成本高；首版 YAGNI |
| **`cust` / 手写 `cuda-sys`** | 维护分散；cudarc 已为 Rust CUDA 事实标准 |
| **每算子 `dyn ExecKernel` trait 对象** | 热路径虚调用；改用具名 `match` + 单态化 CPU 路径 |
| **在 `discretization` 内直接 `cudarc`** | 违反 ADR 0003 依赖方向 |

## 对关联 ADR 的修订指引

### ADR 0003

- §2 执行后端：`cuda` 定案为 **`cudarc` + `exec::gpu::cuda`**；wgpu 仍为远期可选项，**不**与 CUDA 互斥于同一 binary（可同时编译两 feature，算例每次仅选其一）。
- §3 Feature：`cuda` 依赖 **已批准**（本 ADR）。

### ADR 0013

- §2.1 `ExecBackend` 规划中的 `GpuWgpu` 仍保留；**`GpuCuda` 由本 ADR 定义**，优先级高于 wgpu（v1.3 调整路线图：CUDA 先于 wgpu 原型）。
- E4「GpuWgpu prototype」与 G1「GpuCuda 一阶无粘」**并行独立**里程碑；asimu 生产路径以 **G1–G4** 为准。

### ADR 0016

- f32 typed 路径为 CUDA 首版精度；`compute_precision = f64` + `backend = cuda` 首版 Validate 拒绝。
- 归约 / 几何 / BC 策略不变（§4）。

## 实现里程碑

| 阶段 | 交付 | 验证 |
|------|------|------|
| **G0** | `cuda` feature、`cudarc` 依赖、`ExecDevice`/`backend` 解析与 Validate、空 `CudaBackendState` 占位 | config 单测；无设备时友好报错 |
| **G1** | `nvcc` 编译一阶无粘 f32 kernel；着色桶 flux + device scatter；`sync_*` 骨架 | single tet CPU≈CUDA |
| **G2** | 粘性内面 + dual_ellipsoid f32 smoke | benchmark manifest |
| **G3** | `cusparse` SpMV 经 `ExecutionContext::csr_spmv` | implicit 路径预研 |
| **G4** | 多 stream、性能回归文档 | perf < 5% 回归基线 |

## 文档同步

| 文档 | 变更 |
|------|------|
| [docs/README.md](../README.md) | ADR 列表增加 0017 |
| [docs/ARCHITECTURE.md](../ARCHITECTURE.md) §8.4、§11 | 摘要与 ADR 表 |
| [docs/en/ARCHITECTURE.md](../en/ARCHITECTURE.md) | GPU 小节摘要 |
| [docs/API.md](../API.md) | G0 落地时公开 `ExecDevice`、`backend` 解析 |
| [docs/CASE_FORMAT.md](../CASE_FORMAT.md) | `[numerics] backend` 字段 |
| [CHANGELOG.md](../../CHANGELOG.md) | G0/G1 落地时记录 |

## 实现追踪

| 项 | 状态 |
|----|------|
| ADR 0017 文本 | **已接受（规划）** |
| **G0** feature + 配置 + 多 Backend 类型 | **2026-06-13 已实现** |
| **G1** 一阶无粘 CUDA kernel | **2026-06-13 已实现**（Roe/HVL + 着色桶 scatter；边界仍 CPU；含 Makefile/benchmark/sync/端到端 smoke） |
| **G2** dual_ellipsoid GPU smoke | **2026-06-13 已实现**（粘性内面 f32 CUDA kernel + device scatter；IDWLS/边界面仍 CPU；`case_cuda_f32.toml` + CPU≈CUDA 单 tet） |
| **G3** cuSPARSE SpMV | **2026-06-13 已实现**（`ExecutionContext::csr_spmv` f64 分发；CSR 结构 device 缓存 + CPU≈CUDA 单测） |
| **G4** 性能与 manifest 字段 | 规划 |
