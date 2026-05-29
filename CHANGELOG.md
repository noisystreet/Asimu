# 变更日志

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- VTK VTS **二进制 appended** 读入：`io::load_vts`（feature `io-vtk`）；支持 zlib + 3D；ADR 0007；`StructuredMesh`
- v0.2 启动准备：`agent_workflow.md`、`docs/CASE_FORMAT.md`；`docs/theory/fvm_diffusion.md`
- v0.2 模块骨架：`field`、`discretization`、`linalg`、`solver/time`；`core::Real` 与 ID newtype
- 首个 V&V 算例目录 `tests/benchmarks/1d_diffusion_analytical/`（case + expected + README）
- AGENTS「数值理论与参考文献」约束；`docs/theory/` 索引
- 运行产物 / V&V / 可观测性：`docs/BENCHMARKS.md`、`docs/OBSERVABILITY.md`、`docs/en/CROSS_CUTTING.md`；**四大横向能力**写入 ARCHITECTURE §1.4、§4.3、§8.5–§8.6
- ADR 0005（时间推进）、ADR 0006（FFI/Python）
- `SECURITY.md` 不可信输入限制；`config/default.toml` 预留 `[output]`/`[time]`/`[study]`
- MCP 集成规划：`docs/MCP.md`、ADR 0004、`.cursor/mcp.json.example`
- 架构设计文档 `docs/ARCHITECTURE.md`（含多精度/GPU §8.4、MCP §4.3）
- `src/app/` 应用编排层；库 API 与 `prelude` 分离
- 数据模型文档 `docs/DATA_MODEL.md`
- ADR 0002：CFD 分层架构与 v0.2 数值基线
- ADR 0003：多精度与 CPU/GPU 执行后端规划
- AGENTS.md 编程风格约束
- 项目骨架：Rust binary + library 结构
- 模块化占位实现：`core`、`mesh`、`solver`、`io`、`config`
- CLI 入口与 TOML 配置加载
- 单元测试与集成测试目录
- CI、pre-commit、Makefile 统一命令入口
- AGENTS.md 与协作模板
