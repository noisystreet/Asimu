# asimu MCP 集成规划

> Model Context Protocol（MCP）服务端设计备忘。  
> ADR：[adr/0004-mcp-integration.md](adr/0004-mcp-integration.md) · 架构总览：[ARCHITECTURE.md](ARCHITECTURE.md) §4.3

**状态**：规划（v1.1+ 实现）· 当前仓库 **无** MCP 代码。

---

## 1. 目标

让 Cursor 等 AI 客户端通过 MCP：

1. **读** — 架构/API 摘要、fixture 算例、运行结果
2. **验** — 校验配置与 case 文件
3. **跑** — 触发占位/真实求解（经 `app` / `case`，非重复实现数值逻辑）

**非目标（当前阶段）**

- 远程 HTTP MCP 云服务
- 通过 MCP 修改源码或执行任意 shell
- 替代 CLI 成为唯一入口

---

## 2. 架构位置

```
┌─────────────────────────────────────────┐
│  AI Client（Cursor、Claude Desktop 等）   │
└──────────────────┬──────────────────────┘
                   │ MCP (stdio 优先)
┌──────────────────▼──────────────────────┐
│  asimu-mcp（独立 binary / crate）         │
│  · tools / resources / prompts          │
│  · 路径沙箱 · 参数校验                    │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  app / case（应用编排）                    │
│  mesh · solver · io · config（库 API）    │
└─────────────────────────────────────────┘
```

与 **CLI** 并列，同属**适配层**；共享库 API，不共享协议代码。

---

## 3. 交付物

| 项 | 规划 |
|----|------|
| Binary | `asimu-mcp` |
| 构建 | 默认 `cargo build` **不**构建；`cargo build --bin asimu-mcp` 或 workspace feature |
| 配置 | 项目根 `.cursor/mcp.json` 或用户 MCP 配置指向 `asimu-mcp` 路径 |
| 文档 | 本文 + ADR 0004；实现后补充 Tool 参数 schema |

### 3.1 规划目录（v1.1+）

```
asimu/
├── src/bin/asimu-mcp.rs       # 入口：stdio 传输
├── src/mcp/                   # 或 crates/asimu-mcp/
│   ├── mod.rs
│   ├── server.rs              # 协议注册
│   ├── tools/                 # validate_config, run_case, ...
│   ├── resources/             # docs, fixtures
│   └── sandbox.rs             # 路径校验
└── .cursor/
    └── mcp.json.example       # 本地 MCP 配置示例
```

---

## 4. Tools 规划

| Tool | 输入 | 输出 | 依赖模块 |
|------|------|------|----------|
| `validate_config` | `path: string` | 校验结果 / 错误列表 | `config`, `io` |
| `run_case` | `case_path`, 可选 `output_dir` | `SolveResult` 摘要 JSON | `app` / `case`, `solver` |
| `get_run_summary` | （会话内最近一次） | 迭代、残差、converged | `app` 会话状态 |
| `list_fixtures` | 无 | fixture 名称列表 | `tests/fixtures/` 只读 |

**约束**

- 所有 Tool 返回结构化 JSON + 人类可读 `message`
- 失败时返回 MCP 错误，映射 `AsimuError` 上下文
- `run_case` 超时上限可配置（防 Agent 长时间阻塞）

---

## 5. Resources 规划

| URI | MIME | 说明 |
|-----|------|------|
| `asimu://docs/architecture` | text/markdown | `docs/ARCHITECTURE.md` 摘要或全文 |
| `asimu://docs/api` | text/markdown | `docs/API.md` 摘要 |
| `asimu://docs/agents` | text/markdown | `AGENTS.md` |
| `asimu://fixture/{name}` | text/plain | `tests/fixtures/{name}` |
| `asimu://run/latest` | application/json | 最近一次 `RunManifest`（v1.2+） |

Resources **只读**；算例结果亦可通过 manifest JSON 获取。

---

## 6. Prompts 规划（v1.2+，可选）

| Prompt | 用途 |
|--------|------|
| `debug_divergence` | 给定残差历史，生成排查 checklist |
| `explain_case` | 解释 fixture 格式与边界条件 |

---

## 7. 安全

| 规则 | 说明 |
|------|------|
| 路径沙箱 | 拒绝 workspace 外路径与 `..` 逃逸 |
| 无 shell Tool | 不提供 `exec_command` |
| 无密钥 | 不读 `.env`；Tool schema 不含 secret 字段 |
| 写范围 | 仅允许写入 `{workspace}/output/` 或临时目录 |
| 日志 | MCP 请求 ID 写入 `tracing`；不记录敏感 env |

---

## 8. 测试与 CI

| 层级 | 方式 |
|------|------|
| 单元 | `sandbox` 路径校验、参数反序列化 |
| 集成 | 启动 `asimu-mcp`，mock client 发送 `tools/list`、`tools/call` |
| CI | 默认 job 跑 MCP 协议测试；**不**依赖 Cursor 实例 |

---

## 9. 演进里程碑

| 版本 | MCP 交付 |
|------|----------|
| v0.x–v1.0 | 无实现；仅本文 + ADR |
| **v1.1** | `asimu-mcp` stdio；`validate_config`、`list_fixtures`、`run_case`（占位求解） |
| **v1.2** | Resources（docs + fixtures）；`get_run_summary` |
| **v1.3** | Prompts；`.cursor/mcp.json.example`；可选 SSE 评估 |
| v2.x | 与真实 PDE 算例、VTK 输出 Tool 联动 |

**前置条件**：`case` 模块稳定、库 API v1.0 冻结后再实现 MCP，避免协议随内核频繁 breaking。

---

## 10. 本地配置示例（规划）

```json
{
  "mcpServers": {
    "asimu": {
      "command": "/path/to/asimu-mcp",
      "args": [],
      "env": {
        "ASIMU_LOG_LEVEL": "warn"
      }
    }
  }
}
```

实现后提供仓库内 `.cursor/mcp.json.example`。

---

## 11. 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) §4.3 — 适配层总览
- [API.md](API.md) — 库 API（MCP 不得绕过）
- [AGENTS.md](../AGENTS.md) — Agent 协作约束
- [adr/0004-mcp-integration.md](adr/0004-mcp-integration.md)
