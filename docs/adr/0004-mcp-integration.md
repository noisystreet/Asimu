# ADR 0004: MCP（Model Context Protocol）集成规划

- **状态**: 已接受（规划基线，实现分阶段）
- **日期**: 2026-05-29
- **关联**: [MCP.md](../MCP.md)、[ARCHITECTURE.md](../ARCHITECTURE.md) §4.3

## 背景

asimu 目标用户包含 **AI Agent 协作者**（见 `AGENTS.md`、Cursor 规则）。除 CLI 外，需让 IDE / Agent 通过标准协议：

- 查询算例与文档
- 触发求解、读取状态与结果
- 在不变更核心数值代码的前提下扩展交互方式

[MCP（Model Context Protocol）](https://modelcontextprotocol.io/) 是 AI 客户端与外部工具/inter 服之间的开放协议，适合作为 **CLI 并列的适配层**。

## 决策

### 1. 定位：适配层，非求解核心

```
AI Client (Cursor 等)
        ↓ MCP (stdio / SSE)
   asimu-mcp 服务
        ↓ 调用
   app / case / 库 API (mesh, solver, io)
        ↓
   数值内核（不变）
```

- **禁止**在 MCP  handler 内实现离散格式或线性求解
- MCP 层只做：协议编解码、参数校验、路径沙箱、调用已有 API

### 2. 交付形态

| 阶段 | 形态 |
|------|------|
| v1.1（规划） | 独立 binary `asimu-mcp`（`src/bin/asimu-mcp.rs` 或 workspace crate `crates/asimu-mcp`） |
| 默认构建 | **不**包含 MCP；`cargo build` 仅 `asimu` CLI + lib |
| 可选 feature | `mcp-server` 或单独 package，避免默认依赖 MCP SDK |

传输：**stdio** 为首（Cursor 本地 MCP 常见模式）；SSE 为后续扩展。

### 3. 首批能力（v1.1 范围）

**Tools（Agent 可调用）**

| Tool | 说明 |
|------|------|
| `validate_config` | 校验 TOML / case，不运行求解 |
| `run_case` | 加载算例并执行（封装 `case` / `app`） |
| `get_run_summary` | 返回最近运行：迭代数、残差、是否收敛 |
| `list_fixtures` | 列出 `tests/fixtures/` 可用算例 |

**Resources（Agent 可读取）**

| URI 模式 | 说明 |
|----------|------|
| `asimu://docs/architecture` | 架构摘要（Markdown） |
| `asimu://docs/api` | 库 API 摘要 |
| `asimu://fixture/{name}` | 只读 fixture 内容 |

**Prompts（可选，v1.2+）**

| Prompt | 说明 |
|--------|------|
| `analyze_convergence` | 根据残差历史生成排查提示模板 |

### 4. 安全与权限

- **路径沙箱**：仅允许读取项目根、`config/`、`tests/fixtures/` 下路径；禁止任意 `../` 逃逸
- **无密钥**：Tool 参数与 Resource 响应不得包含 Token、`.env` 内容
- **无 shell**：禁止暴露「执行任意 shell 命令」类 Tool
- **写操作**：`run_case` 输出仅写入指定 `output/` 或临时目录（实现期定案）
- 与 [SECURITY.md](../../SECURITY.md)、[AGENTS.md](../../AGENTS.md) 安全红线一致

### 5. 依赖策略

- MCP Rust SDK（如 `rmcp` 或官方生态 crate）在 **实现 ADR 前不引入**
- 新增 MCP 依赖须：MIT/Apache 兼容、可 stdio 传输、CI 可 headless 测试
- MCP 集成测试：mock client 或协议级 snapshot，**不**要求 CI 连接 Cursor

### 6. 依赖方向

```
mcp-server (asimu-mcp) → app / case → 库 API
mcp-server → io, config（只读为主）
mcp-server 不得被 core / mesh / solver / discretization 依赖
```

## 后果

### 正面

- Agent 与 IDE 以标准协议驱动 asimu，无需伪造 CLI 解析
- 数值内核与交互协议解耦，CLI / MCP / 未来 REST 可并列
- 文档 Resource 减少 Agent 幻觉

### 负面

- 多一个 binary 与协议兼容维护面
- MCP SDK 生态仍在演进，需跟进 breaking changes
- 双路径测试（CLI + MCP）增加 CI 成本

## 备选方案

| 方案 | 未采纳原因 |
|------|------------|
| 仅文档 + CLI，不做 MCP | Agent 集成体验差，需反复 exec CLI |
| 在 `lib.rs` 内嵌 MCP | 污染库 API 与默认依赖 |
| 第一版即 SSE 云服务 | 超出离线/本地优先目标；stdio 足够 |

## 里程碑

见 [MCP.md](../MCP.md) 与 [ARCHITECTURE.md](../ARCHITECTURE.md) §10。
