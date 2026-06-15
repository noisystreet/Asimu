# 文档索引

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | **架构设计文档**（分层、依赖、演进路线） |
| [DATA_MODEL.md](DATA_MODEL.md) | 网格、场、BC、Run Manifest、TimeIntegrator |
| [CASE_FORMAT.md](CASE_FORMAT.md) | v0.2 算例 TOML schema |
| [API.md](API.md) | 公开 library API |
| [MCP.md](MCP.md) | MCP 集成规划（Tools / Resources） |
| [BENCHMARKS.md](BENCHMARKS.md) | V&V 验证算例库规划 |
| [DEBUG_CHECKLIST.md](DEBUG_CHECKLIST.md) | V&V / 无量纲 metrics 排查清单 |
| [OBSERVABILITY.md](OBSERVABILITY.md) | 性能与可观测性规划 |
| [SLIDING_MESH.md](SLIDING_MESH.md) | 滑移网格 / MRF / ALE 分阶段规划 |
| [theory/](theory/) | 数值理论手册（离散、BC、时间推进等） |
| [adr/](adr/) | 架构决策记录（ADR） |
| [en/CROSS_CUTTING.md](en/CROSS_CUTTING.md) | 四大横向能力英文摘要 |
| [en/](en/) | 其他英文文档摘要 |

## ADR 列表

| 编号 | 标题 |
|------|------|
| [0001](adr/0001-rust-cfd-foundation.md) | 以 Rust 构建 CFD 求解器基础 |
| [0002](adr/0002-layered-cfd-architecture.md) | CFD 分层架构与 v0.2 数值基线 |
| [0003](adr/0003-multi-precision-and-gpu.md) | 多精度与 CPU/GPU 执行后端规划 |
| [0004](adr/0004-mcp-integration.md) | MCP（Model Context Protocol）集成规划 |
| [0005](adr/0005-time-integration.md) | 时间推进与稳态/瞬态统一模型 |
| [0006](adr/0006-ffi-interop.md) | FFI / Python 互操作原则 |
| [0007](adr/0007-vts-binary-io.md) | VTK VTS 二进制读入（feature `io-vtk`） |
| [0008](adr/0008-cgns-io.md) | CGNS 读入与 VTS 导出（feature `io-cgns-vts`，系统 libcgns） |
| [0009](adr/0009-compressible-navier-stokes.md) | 三维可压缩 Navier-Stokes 求解器架构（规划基线） |
| [0010](adr/0010-unstructured-mixed-mesh.md) | 非结构混合单元网格（面拓扑路线；M1–M4 分阶段） |
| [0011](adr/0011-parallel-fvm-face-coloring.md) | 非结构 FVM 面着色 + `parallel-fvm` |
| [0012](adr/0012-unstructured-gradient-limiters.md) | 非结构二阶线性重构与梯度限制器 |
| [0013](adr/0013-exec-parallel-scatter-execution-context.md) | `ExecutionContext` + `exec` 并行 scatter |
| [0014](adr/0014-turbulence-k-omega-sst-rans.md) | 可压 RANS 湍流闭包（k-ω SST） |
| [0015](adr/0015-incompressible-navier-stokes-simplec-piso.md) | 三维不可压 NS（SIMPLEC + PISO） |
| [0016](adr/0016-runtime-compute-precision.md) | 核心计算模块运行时精度选择 |
| [0017](adr/0017-gpu-cuda-cudarc-multi-backend.md) | CUDA 后端（`cudarc`）与 `exec` 多 Backend |

维护策略：修改中文架构/数据模型文档时，同步更新 `docs/en/` 对应摘要（见 [AGENTS.md](../AGENTS.md)）。
