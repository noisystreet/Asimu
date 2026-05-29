# AGENTS.md — AI 协作者必读

> 人类协作者请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 项目身份

| 项 | 值 |
|----|-----|
| 名称 | **asimu** |
| 类型 | Rust CFD 求解器（binary + library） |
| 技术栈 | Rust 1.85+, clap, tracing, thiserror, serde/toml |
| MSRV | **1.85**（`Cargo.toml` → `rust-version`） |
| 开发工具链 | **stable**（`rust-toolchain.toml`，含 rustfmt + clippy） |

### 目录结构

```
src/{main,lib,error,config}.rs
src/app/                       # CLI 应用编排（已实现）
src/{core,mesh,solver,io}/     # 库 API（v0.1）
src/{field,discretization,physics,linalg,case}/  # v0.2+ 规划
src/exec/                      # v1.2+ CPU/GPU 后端
src/mcp/                       # v1.1+ MCP（asimu-mcp）
config/          tests/          docs/           scripts/
docs/theory/     # 数值理论手册（离散、BC、时间推进）
```

**库 vs 应用 vs MCP**：数值模块为库 API；`app` 服务 CLI；`asimu-mcp`（规划）服务 Agent。

**横向能力**（Run Manifest、时间推进、V&V 算例库、可观测性）：见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §4.3、[docs/en/CROSS_CUTTING.md](docs/en/CROSS_CUTTING.md)。

## 硬约束

### 模块依赖方向

目标结构（v0.2+，详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §7）：

```
core ← mesh ← field ← discretization
core ← physics
core ← linalg
core ← io
mesh + field + discretization + physics + linalg ← solver
discretization + linalg → exec（CPU/GPU 热算子，v1.2+）
case → io, solver, config（编排层）
config → 各层只读，不依赖 solver 实现
```

v0.1 过渡期间：`solver` 可临时依赖 `mesh`；**新增代码**应遵循目标结构。

**数值类型**：公开 API 使用 `core::Real`（默认 `f64`），避免散落裸 `f64`（几何坐标除外）。**GPU**：仅经 `exec` 模块；禁止在 `discretization` 内直接调用 wgpu/CUDA（见 ADR 0003）。

**禁止**反向依赖；**禁止** `core` 引入 `solver` / `mesh` / `io` / `field` 等领域模块。

### 禁止引入的依赖（未经 ADR 批准）

| 依赖 | 原因 |
|------|------|
| GPL 系库 | 许可证传染性 |
| 未维护 crate（>2 年无更新且无 fork） | 安全风险 |
| 网络请求默认库（如 `reqwest`） | 与离线 CFD 工作流不符 |

### 文档修改权限

| 文档 | Agent 权限 |
|------|------------|
| `docs/API.md`、模块 rustdoc | 功能变更时**必须**同步更新 |
| `docs/theory/*` | 新增/变更离散、BC、时间推进、本构、非平凡求解器时**必须**新增或更新 |
| `docs/ARCHITECTURE.md` | 小改动可更新；**分层/依赖方向变更需人工审批** |
| `docs/adr/*` | 重大选型时新增 ADR，不删除已有 ADR |

### 代码标注

Agent 生成的大段新模块（>100 行）建议在文件头注释：`// Generated with AI assistance`（可选，非强制）。

### 安全红线

- **不得**提交密钥、Token、证书、`.env`
- **不得**绕过错误处理使用裸 `unwrap`（测试除外）
- **不得**在公开 Issue 讨论安全漏洞（见 [SECURITY.md](SECURITY.md)）

### 测试要求

新增功能**必须**包含测试：

- 纯逻辑 → 单元测试（同文件 `#[cfg(test)]`）
- CLI / 多模块 → `tests/` 集成测试
- 遵循 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 测试分层

## 编程风格约束

本节与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §3 设计原则一致，侧重 **Agent 写代码时的可执行规则**。

### 总原则

| 原则 | 要求 |
|------|------|
| **显式优于隐式** | 读函数签名即可知道依赖什么、产出什么 |
| **数据与行为分离** | 网格/场/矩阵是数据；算法函数尽量只读输入、写明确输出 |
| **纯函数优先** | `core`、`discretization` 中可纯化的逻辑不写副作用 |
| **局部可变** | 可变状态收窄到最小作用域；禁止「为了方便」扩大 `&mut` 传播 |

### 减少隐式状态（重点）

**隐式状态**指：影响程序行为、但未出现在函数参数或返回值中的数据（全局变量、环境变量读写、未文档化的内部缓存、初始化时注册的回调等）。

#### 必须遵守

1. **禁止**模块级 / 静态可变业务状态（`static mut`、全局 `OnceLock<RefCell<…>>` 存求解进度等）。
2. **禁止**在库代码中调用 `std::env::set_var`；配置在 `Cli::load_config()` 完成后视为**只读**。
3. **禁止**「初始化副作用」：如 `io` 解析时向全局 registry 注册网格、在 `mod` 加载时启动后台任务。
4. **求解器状态**必须封装在命名 struct 中（如 `SolverState`、`LinearSystem`），字段含义在 [DATA_MODEL.md](docs/DATA_MODEL.md) 或 rustdoc 中可查。
5. **函数契约**：凡影响数值结果的输入（`mesh`、`field`、`config` 切片、边界条件）必须作为**参数**传入，不得从函数内部「偷偷读取」模块私有静态或上一次调用的残留。
6. **迭代与收敛**：残差历史、迭代计数放在 `SolverState` 或返回值中；`tracing` 仅用于观测，**不能**作为唯一的状态存储。
7. **可重复性**：同一组输入参数多次调用，应得到相同输出（除非文档说明允许随机性或浮点非结合性）。

#### 推荐做法

```rust
// 推荐：输入输出清晰，无隐藏依赖
pub fn assemble_diffusion(
    mesh: &StructuredMesh2d,
    field: &ScalarField,
    diffusivity: f64,
    system: &mut LinearSystem,
) -> Result<()>;

// 避免：从模块内部 cache 或 static 取 mesh/配置
pub fn assemble_diffusion(system: &mut LinearSystem) -> Result<()>; // ❌
```

```rust
// 推荐：编排层持有状态，算法层无持久化
pub struct SteadyDiffusionSolver {
    config: SolverConfig,
}

impl SteadyDiffusionSolver {
    pub fn run(
        &self,
        mesh: &Mesh,
        field: &mut ScalarField,
    ) -> Result<SolveResult>;
}
```

#### 允许的例外（须注释说明原因）

- `tracing` 订阅器在 `init_tracing` 中一次性安装（应用入口）。
- 只读的全局物理常量（`core` 中 `pub const`），不得含运行时 mutable 状态。
- 测试模块 `#[cfg(test)]` 内的局部 fixture，不得泄漏到生产路径。

### 可变性与并发

- 默认 `let` 不可变；仅装配、时间推进等阶段对 `field` / `LinearSystem` 使用 `&mut`。
- **禁止**在 `discretization`、`linalg` 热路径使用 `RefCell` / `Mutex` / `RwLock`（除非 ADR 批准）。
- v0.x 单线程假设：不引入「为将来并行预留」的隐式线程局部存储。

### 函数与类型

- **单一职责**：一个函数只做一件事（装配 / 求解 / 应用 BC 分开）。
- **命名反映副作用**： mutating 函数用动词（`apply_`、`assemble_`、`update_`）；纯函数用名词或 `compute_` / `flux_`。
- **newtype 标识**：`CellId`、`FaceId` 等不用裸 `usize` 混用（见 DATA_MODEL）。
- **错误不吞没**：禁止空 `catch` 或 `_ = expr_that_can_fail`；可恢复错误用 `Result` 向上传。

### 数值与 CFD 专项

- `discretization` **不得**依赖 `solver`；所需参数由调用方传入。
- 边界条件以 `BoundaryPatch`（或等价 struct）显式传入，不在函数内部硬编码「壁面 = 第 0 面」。
- 浮点比较用 `core` 提供的容差工具（如 `approx_eq`），禁止裸 `==` 比较 `f64`（测试与常量 0 除外）。
- 魔数（CFL 数、松弛因子等）放入 `config` 或命名常量，禁止未命名字面量散落。

### 数值理论与参考文献

实现**有物理或数值含义**的功能时，须补充可追溯的理论说明与参考文献，便于审查、V&V 与后续维护。

#### 必须补充（满足任一即触发）

| 类别 | 示例 | 文档位置 |
|------|------|----------|
| 空间离散 / 通量格式 | FVM 扩散、对流格式、梯度重构 | `docs/theory/{topic}.md` |
| 边界条件类型 | Dirichlet、Neumann、入口/壁面 | 同上，或 BC 模块 rustdoc |
| 时间推进 | `TimeIntegrator`、CFL 条件 | `docs/theory/` + ADR 0005 |
| 物理本构 / 源项 | 粘性、湍流闭包（远期） | `docs/theory/` |
| 线性求解算法 | CG、预条件子（非 trivial 调用） | `docs/theory/` 或 `linalg` rustdoc |
| V&V 验证算例 | `tests/benchmarks/*` | 算例 `README.md`（见 [BENCHMARKS.md](docs/BENCHMARKS.md)） |

#### 理论说明最低内容

1. **控制方程或算法步骤**（LaTeX 或清晰 ASCII，编号与代码注释一致）
2. **离散假设**（网格类型、阶数、稳定性/守恒性说明）
3. **参考文献**：书名/论文 + 作者 + 年份；有 DOI/ISBN 须列出
4. **实现映射**：rustdoc 或理论页注明对应函数/模块（如「式 (3) → `assemble_diffusion`」）

#### 不必单独写理论页

- 纯工程改动（CLI、config、manifest schema、日志、MCP 适配）
- 不改变数值语义的重构、重命名、性能优化
- 已在 `docs/theory/` 或 ADR 中覆盖且本次未改公式/算法

#### 维护

- 修改离散公式、容差判据或 benchmark 参考值时，**同步**更新理论页/算例 README 与 [CHANGELOG.md](CHANGELOG.md)
- 参考值变更须在 PR 说明中引用文献章节或表格编号

### 类型与 API 设计

- **非法状态不可表示**：用 `enum` / newtype 在类型层表达约束（如 `BoundaryKind`、`CellId`），避免 `u8` + 魔法值；`ScalarField::new` 构造时校验长度，成功后内层不再重复断言。
- **最小公开面**：默认 private；`pub(crate)` 用于 crate 内跨模块；新增 `pub` 须同步 [docs/API.md](docs/API.md) 或 PR 说明理由。
- **`#[must_use]`**：`Result<SolveResult>` 及不可静默丢弃的返回值必须标注；禁止忽略收敛结果。

### 边界与错误

- **Parse → Validate → Trust**：`io` / `mesh` / `field` 构造阶段完成全部校验；`discretization`、`linalg` 热路径信任已验证输入，不再做 io 级检查（除非 ADR 说明）。
- **错误带上下文**：错误信息包含可定位信息（cell/face id、迭代步、路径、数值），避免 `"invalid"` / `"failed"` 裸消息；底层包装，上层用 `?` 传递。
- **可重复性**：同输入同输出；若使用随机性，`seed` 来自 config 并写入日志。

### 性能与分配

- **热路径零分配**：面循环 / 装配内层禁止 `Vec::push`、`format!`、`clone()`、`to_string()`。
- **预分配**：已知规模时用 `Vec::with_capacity(mesh.num_faces())` 等，避免反复扩容。
- **传引用不传大对象**：热路径参数优先 `&Mesh`、`&ScalarField`、`&[f64]`，避免按值传递大 `Vec`。

### 可测试性

- **小网格可测**：离散算子须能用 3×3 等极小网格单测，不依赖读文件。
- **解析解优先**：验证用 manufactured solution 或已知解析解；数值行为变更须更新 golden test，禁止悄悄放宽容差。
- **测试命名**：描述行为（`uniform_field_produces_zero_flux`），禁止 `test1`；测试间不得共享可变 static 或依赖执行顺序。

### 抽象纪律（YAGNI）

- **规则 of Three**：同一模式出现 ≥2 次再抽象 trait / 泛型；首版通量格式可写具体函数，不 premature 建 `FluxScheme` 体系。
- **trait 在边界**：扩展点（通量、线性求解器、BC）用 trait；热路径 mesh 遍历保持具体类型 + SoA。

### 注释与文档

- 注释写 **为什么** 与 **数值假设**（离散格式、BC 顺序、稳定性条件），不复述代码本身。
- 数值实现须链到 [docs/theory/](docs/theory/) 或算例 README 中的方程编号与参考文献（见上文「数值理论与参考文献」）。
- 公开函数 rustdoc 写清前置/后置条件（如「前置：`field.len == mesh.num_cells()`」）。
- 公开 API 或数值行为变更 → 更新 [docs/API.md](docs/API.md) 与 [CHANGELOG.md](CHANGELOG.md)。

### 反面模式（禁止）

| 模式 | 问题 |
|------|------|
| 全局 `static mut CURRENT_MESH` | 不可测试、不可并发、隐藏依赖 |
| `Solver::default()` 内部读环境变量 | 配置来源不透明 |
| 在 `Drop` 里改求解状态或写文件 | 副作用不可见 |
| 「上帝 struct」持有 mesh+field+solver+io | 破坏分层，参见复杂度门禁 |
| 用 `lazy_static` 缓存上次算例结果 | 跨调用隐式依赖 |
| 忽略 `Result` / `#[must_use]` 返回值 | 静默失败或误判收敛 |
| 热路径内 `format!` / 隐式分配 | 性能不可预测 |
| 未更新 golden test 即改离散公式 | 数值回归失效 |

- MCP：`docs/MCP.md`（v1.1+ 规划）
- 运行产物 / V&V：`docs/BENCHMARKS.md`、Run Manifest、Restart（ARCHITECTURE §8.5）

## 验证命令（修改后必须运行）

```bash
make check
# 等价于: cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

可选：`make audit`（需安装 `cargo-audit`）

## 已启用的工程质量门限

| 项 | 命令 / 配置 |
|----|-------------|
| 格式化 | `rustfmt.toml`，CI `cargo fmt --check` |
| Clippy | `-D warnings`，见 `Cargo.toml [lints.clippy]` |
| unsafe | 禁止（`[lints.rust] unsafe_code = forbid`） |
| 代码复杂度 | `scripts/complexity_check.py`：文件 ≤800 行、函数 ≤150 行、参数 ≤8 个 |
| 提交说明 | `scripts/commit_msg_check.py`（英文第 1 行 + 中文第 2 行） |

完整工程约定见 [agent_workflow.md](agent_workflow.md) 第 12 节；本项目仅启用上表子集。

## 文档与语言约定

| 内容 | 语言 |
|------|------|
| 代码注释、rustdoc | 中文（术语可保留英文） |
| 提交说明 | 英文第 1 行 + 中文第 2 行 |
| `docs/ARCHITECTURE.md` | 中文为主，`docs/en/` 英文摘要 |
| 改中文架构文档 | 同步更新 `docs/en/ARCHITECTURE.md` |

## MCP 集成（规划 v1.1+）

- MCP 服务端为 **适配层**，与 CLI 并列；详见 [docs/MCP.md](docs/MCP.md)
- **禁止**在 MCP handler 内实现数值逻辑；仅调用 `app`/`case` 与库 API
- **禁止** MCP Tool 执行 shell、读取 `.env`、访问沙箱外路径
- **io 解析**须遵守 [SECURITY.md](SECURITY.md) 资源上限（文件大小、单元数、路径）
- 实现前须库 API v1.0 稳定；MCP SDK 引入需 ADR 批准

## 协作入口

- PR 模板：`.github/pull_request_template.md`
- Issue 模板：`.github/ISSUE_TEMPLATE/`
- 安全：[SECURITY.md](SECURITY.md)
- 行为准则：[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## 运行时说明

- **CLI（`asimu`）**：一次性进程，跑完即退出；健康检查/优雅退出 **不适用**。
- **MCP（`asimu-mcp`，规划 v1.1+）**：stdio 长连接服务；需处理客户端断开、请求超时与日志级别；详见 [docs/MCP.md](docs/MCP.md)。
