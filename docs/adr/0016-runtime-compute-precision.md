# ADR 0016: 核心计算模块运行时精度选择

- **状态**: 已接受（规划基线，实现分阶段 P0–P5）
- **日期**: 2026-06-09
- **关联**: [ADR 0003](0003-multi-precision-and-gpu.md)、[ADR 0013](0013-exec-parallel-scatter-execution-context.md)、[ARCHITECTURE.md](../ARCHITECTURE.md) §8.4、[DATA_MODEL.md](../DATA_MODEL.md) §10

## 背景

[ADR 0003](0003-multi-precision-and-gpu.md) 原定多精度路线为：

1. v0.x 继续全局 `core::Real = f64`；
2. 后续通过 Cargo feature `precision-f32` 在**编译期**选择单一 `Real`；
3. 更远期评估混合精度。

当前非结构可压缩路径已经把热点逐步集中到 `discretization`、`linalg` 与 `exec`：

- 面通量装配、IDWLS 梯度、谱半径、LU-SGS / GMRES 更新；
- `ExecutionContext` 已成为并行 scatter、scratch 与 SpMV 的统一入口；
- I/O、case 解析、日志、观测、边界 patch registry 等模块仍以工程数据编排为主。

因此，**全库运行时动态精度**没有必要，且会把 `enum PrecisionValue` 分支污染到非热路径和公共 API。真正需要的是：对大规模算例，允许用户在 case 层选择核心求解计算使用 `f32` 或 `f64`，同时让非核心模块保持稳定、可读、可测试。

本 ADR 修订 ADR 0003 的“精度模型”部分：不采用单纯 Cargo feature 切换全局 `Real`，而是在核心计算模块内引入**运行时选择、边界分发、内部单态化**的精度模型。

## 决策

### 1. 精度只作用于核心计算域

新增概念：**Compute Precision**，仅覆盖求解热路径的数据与算子。

| 范围 | 是否运行时可选精度 | 说明 |
|------|--------------------|------|
| `field` 中求解场、残差、梯度 scratch | 是 | `f32` / `f64` 双实例；首批覆盖可压缩守恒变量与 primitive cache |
| `discretization` 装配 | 是 | 对 `Scalar: FloatLike` 单态化，case 启动时选择具体实例 |
| `linalg` CSR、SpMV、Krylov 向量 | 是 | `CsrMatrix<T>` / `Vector<T>`；归约策略见 §4 |
| `exec` CPU kernel / scatter / scratch | 是 | `ExecutionContext` 持有 resolved precision，并分发到 typed kernel |
| `mesh` 几何坐标、拓扑、边界 patch | 否，保留 `f64` / `Real` 过渡 | 几何构造与质量检查重视稳健性；拓扑无精度语义 |
| `io`、`case`、`config`、`app`、日志/trace | 否 | 输入输出按文件格式解析；进入计算域时转换 |

**非目标**：

- 不支持同一进程中一个 time step 内动态切换精度。
- 不把所有公开 API 泛型化为 `T: Float`。
- 不在首版实现 mixed precision（例如 field `f32` + residual `f64`），仅预留。

### 2. Case 配置与运行时分发

配置入口：

```toml
[numerics]
compute_precision = "f64"  # f64 | f32
```

默认值为 `f64`。高马赫可压缩算例（例如 `mach >= 5`）首版规则：

- 未显式设置时使用 `f64`；
- 显式 `f32` 时允许运行，但在 case 加载日志中发出 `warn`，提示激波/强压缩问题可能更敏感；
- V&V benchmark 的 reference 默认仍以 `f64` 生成。

分发方式：

```rust
pub enum ComputePrecision {
    F64,
    F32,
}

pub fn run_case(case: &CaseSpec) -> Result<()> {
    match case.numerics.compute_precision {
        ComputePrecision::F64 => run_typed::<f64>(case),
        ComputePrecision::F32 => run_typed::<f32>(case),
    }
}
```

运行时分支只出现在 **case / solver 编排边界**。进入 `run_typed::<T>` 后，装配、时间推进、线性代数与 exec kernel 均由 Rust 单态化，热路径内不出现每个单元/每个面的精度分支。

### 3. 类型模型：保留 `Real`，新增计算标量 trait

`core::Real` 在过渡期继续表示默认工程标量（`f64`），用于：

- public config 数值；
- mesh 几何与 I/O；
- 文档与旧 API 兼容。

核心计算新增 sealed trait（命名可在实现期微调）：

```rust
pub trait ComputeFloat:
    Copy + Send + Sync + 'static
{
    const PRECISION: ComputePrecision;

    fn from_real(value: Real) -> Self;
    fn to_real(self) -> Real;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, rhs: Self) -> Self;
}
```

实现仅允许 `f32` 与 `f64`。禁止对外开放任意 `num_traits::Float` 泛型，以免引入额外依赖和不可控实现。

字段容器演进：

```rust
pub struct ConservedFieldsT<T: ComputeFloat> { /* SoA<T> */ }
pub type ConservedFields = ConservedFieldsT<Real>; // 过渡兼容

pub struct ConservedResidualT<T: ComputeFloat> { /* SoA<T> */ }
pub type ConservedResidual = ConservedResidualT<Real>;
```

模块内部优先使用 `ConservedFieldsT<T>`；公开旧 API 可在 P0–P2 保留 `Real` 别名版本。

### 4. 归约、容差与稳定性策略

`f32` 不是简单把所有 `f64` 改成 `f32`。首版采用以下规则：

| 项 | `f64` | `f32` |
|----|-------|-------|
| 场变量 / residual / 通量 | `f64` | `f32` |
| 几何坐标 / 体积 / 面法向 cache | `f64` | `f64`，进入 kernel 前按需转换或预打包 |
| RMS 残差归约 | `f64` | **累加用 `f64`**，输入值来自 `f32` |
| CFL / dt / 收敛历史 | `f64` | `f64` 编排值，写回更新时转换 |
| 正性检查 | `f64` 阈值 | 阈值按 `f32` epsilon 放宽下限 |
| 输出 | `f64` 写出 | `f32` 计算结果转换为 `f64` 写出，文件格式不变 |

`f32` 的默认容差不得复用 `f64` golden：

- 单元测试：使用按精度分派的 `tol(precision)`；
- benchmark：保存 `f64` reference，`f32` 用相对误差/守恒误差阈值；
- 残差收敛：若用户未显式给出 tolerance，`f32` 默认 tolerance 不小于 `1e-5` 量级（实现期按无量纲残差定义定案）。

### 5. `exec` 集成

`ExecutionContext` 增加精度字段：

```rust
pub struct ExecConfig {
    pub backend: ExecBackend,
    pub compute_precision: ComputePrecision,
    // ...
}
```

CPU kernel 组织：

```text
exec/
  cpu/
    f64/  或 generic<T=f64> 单态化入口
    f32/  或 generic<T=f32> 单态化入口
```

SIMD 策略：

| 精度 | 首版 SIMD |
|------|-----------|
| `f64` | 继续使用当前 `wide::f64x4` 路径 |
| `f32` | 首版可先走 scalar / rayon；后续增加 `wide::f32x8` 或等价 kernel |

scatter 策略：

- `Serial`：按 `T` 直接累加。
- `ParallelUnsafeAtomics`：`f64` 继续 `AtomicU64` CAS；`f32` 需新增 `AtomicU32` CAS，不允许把 `f32` 强行扩成 `f64` residual。
- colored bucket 契约不因精度改变。

### 6. I/O 与 restart

文件格式不因计算精度改变：

- VTU / CGNS 输出默认写 `Float64`，保证 ParaView 与现有后处理脚本稳定；
- restart 需要记录 `compute_precision`；
- `f32` restart 重新以 `f32` 恢复，`f64` restart 以 `f64` 恢复；
- 跨精度 restart（`f32` 文件用 `f64` 恢复，或反向）首版不支持，加载时报错；后续可增加显式 `--convert-restart-precision`。

### 7. Parse → Validate → Trust

case 加载阶段完成：

1. 解析 `compute_precision`；
2. 校验求解器、后端、精度组合是否实现；
3. 构造 typed solver / typed fields；
4. 热路径信任 typed buffer 长度与精度一致。

禁止在面循环或单元循环内读取配置判断精度。

## 后果

### 正面

- 用户可在同一 binary 内通过 case 选择 `f32` / `f64`，适合快速筛选和生产性能评估。
- 热路径仍由 Rust 单态化，避免每次算术都经 enum dispatch。
- 非核心模块保持 `f64` 与现有 API，降低迁移成本。
- 为 GPU / mixed precision 留出清晰边界：先解决 typed compute island，再扩展后端。

### 负面

- 核心数据结构需要 `T` 泛型，编译时间和测试矩阵增加。
- `f32` 数值稳定性需要单独调参，尤其是高 Mach 激波、LU-SGS、GMRES 预条件。
- `exec` kernel 与 scatter 原语要维护 `f32` / `f64` 两套测试。
- public API 在过渡期会同时存在 `ConservedFields` 与 `ConservedFieldsT<T>`，需要严格控制公开面。

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 继续 ADR 0003 的 Cargo feature `precision-f32` | 同一 binary 不能按 case 切换；CI / 发布包矩阵复杂；用户体验差 |
| 全库 `T: Float` 泛型 | 污染 `io` / `case` / `mesh` 等非核心模块，编译与维护成本过高 |
| 每个数值用 `enum RealValue { F32, F64 }` | 热路径每次运算分支，破坏性能目标 |
| 只在输出或存储用 `f32`，计算仍 `f64` | 节省 I/O/内存有限，不能验证单精度求解性能 |
| 首版 mixed precision | 调试难度高，难以判断误差来自存储、通量还是归约 |

## 实现里程碑

| 阶段 | 内容 | 验证 |
|------|------|------|
| P0 | 新增 `ComputePrecision`、case 解析、`ComputeFloat` trait 骨架；默认仍 `f64` | config 单测；旧 case 行为不变 |
| P1 | `field` / residual / primitive cache 泛型化；保留 `Real` type alias 兼容 | `f64` 单测不变；字段长度/转换测试 |
| P2 | 结构化可压缩 RHS typed 化（无 SIMD f32） | 结构化 smoke：`f32` vs `f64` 相对误差 |
| P3 | 非结构一阶无粘 + 粘性 RHS typed 化；`ExecutionContext` 记录 precision | single tet / dual_ellipsoid smoke |
| P4 | `linalg` / GMRES / LU-SGS typed 化；归约 `f64` 累加 | implicit solver regression |
| P5 | `f32` SIMD / atomic scatter 优化；benchmark 文档 | perf benchmark + V&V tolerance |

## 兼容性

- 默认 `compute_precision = "f64"`，现有 case 不需要修改。
- 未实现 typed 化的 solver 若收到 `f32`，必须在 Validate 阶段报错，而不是静默回退 `f64`。
- 旧 `core::Real = f64` 保留到核心路径迁移完成；是否最终删除或仅作为默认别名，另开 ADR 决定。
