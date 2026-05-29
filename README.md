# asimu

面向工程与研究场景的 **Rust 计算流体力学（CFD）求解器**。

## 一句话定位

为需要可复现、可扩展 CFD 工作流的开发者与研究人员，提供模块化、类型安全的求解器骨架与 CLI 入口。

## 文档索引

| 文档 | 说明 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计（分层、依赖、演进路线） |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | 核心数据结构（网格、场、BC、Run Manifest） |
| [docs/CASE_FORMAT.md](docs/CASE_FORMAT.md) | v0.2 算例 TOML 格式 |
| [docs/API.md](docs/API.md) | 公开 API 与模块边界 |
| [docs/MCP.md](docs/MCP.md) | MCP 集成规划（v1.1+） |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | V&V 验证算例库 |
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | 性能与可观测性 |
| [docs/theory/](docs/theory/) | 数值理论手册 |
| [docs/en/CROSS_CUTTING.md](docs/en/CROSS_CUTTING.md) | 四大横向能力英文摘要 |
| [docs/adr/](docs/adr/) | 架构决策记录（ADR） |
| [AGENTS.md](AGENTS.md) | AI Agent 协作必读 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 |
| [SECURITY.md](SECURITY.md) | 安全漏洞上报 |
| [CHANGELOG.md](CHANGELOG.md) | 变更日志 |

英文文档见 [docs/en/](docs/en/)。

## 快速开始

### 前置条件

- Rust ≥ 1.85（见 `rust-toolchain.toml`）
- `make`（可选，推荐作为统一命令入口）

### 构建与运行

```bash
make build
make run
# 或
cargo run -- --log-level info
```

### 测试与检查

```bash
make test      # 单元 + 集成测试
make lint      # fmt + clippy
make check     # fmt + clippy + test（提交前推荐）
```

## 项目结构

```
asimu/
├── src/           # 源码：core / mesh / solver / io / config
├── config/        # 默认 TOML 配置
├── tests/         # 集成测试与 fixtures
├── docs/          # 架构与 API 文档
└── scripts/       # 开发脚本（如 commit-msg 校验）
```

## 许可证

Dual-licensed under [Apache-2.0](LICENSE) OR [MIT](LICENSE).

## 贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。公开协作请遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
