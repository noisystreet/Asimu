# asimu 性能与可观测性规划

> 架构：[ARCHITECTURE.md](ARCHITECTURE.md) §8.6 · 运行清单：[DATA_MODEL.md](DATA_MODEL.md) §10

**状态**：规划（分阶段落地）· 当前仅 `tracing` 基础日志。

---

## 1. 目标

| 目标 | 说明 |
|------|------|
| **可定位** | 性能瓶颈可测量（面循环、SpMV、I/O） |
| **可对比** | 版本/配置变更前后有 baseline |
| **可关联** | 指标与 `RunManifest`、benchmark 算例 ID 关联 |
| **低开销默认** | 生产路径默认轻量；详细 profiling 显式开启 |

---

## 2. 三层可观测性

```
┌─────────────────────────────────────────────────┐
│  L3 运行产物 — Run Manifest + metrics.jsonl     │  持久化、可复现
├─────────────────────────────────────────────────┤
│  L2 结构化指标 — iteration / timing / CFL      │  DEBUG+ 或配置开启
├─────────────────────────────────────────────────┤
│  L1 日志 — tracing (stderr)                     │  已实现
└─────────────────────────────────────────────────┘
```

### 2.1 L1 日志（已实现）

- 框架：`tracing` + `tracing-subscriber`
- 级别约定见 [ARCHITECTURE.md](ARCHITECTURE.md) §13.2
- 禁止业务 `println!`

### 2.2 L2 结构化指标（规划）

**输出**：`output/metrics.jsonl`（每行一条 JSON，与 manifest 同目录）

| 字段（示例） | 说明 |
|--------------|------|
| `run_id` | 与 manifest 关联的 UUID |
| `iteration` / `step` | 非线性 / 时间步 |
| `residual` | 当前残差范数 |
| `cfl` | 瞬态 CFL 数（若有） |
| `phase` | `assemble` / `solve` / `bc` |
| `elapsed_ms` | 阶段 wall time |

**实现模块**：`observability/` 或 `solver/metrics.rs`（v0.4 原型，v0.5 默认开启选项）

```toml
[observability]
metrics = true              # 写 metrics.jsonl
metrics_level = "iteration" # iteration | phase | off
```

### 2.3 L3 运行产物

- **`run-manifest.json`** — 见 [DATA_MODEL.md](DATA_MODEL.md) §10
- 可选 **`output/profiling/`** — 本地 flamegraph（仅 `--profile`，不进 CI）

---

## 3. 性能工程

### 3.1 微基准（Micro-benchmarks）

| 工具 | 用途 | 版本 |
|------|------|------|
| `criterion` | 面循环、SpMV、梯度 | v0.4+ dev-dep |
| CI | 对比 baseline，超阈 warn（非 block） | v1.0 评估 |

目录规划：

```
benches/
├── flux_assembly.rs
├── spmv.rs
└── README.md
```

### 3.2 宏基准（Macro-benchmarks）

- 与 [BENCHMARKS.md](BENCHMARKS.md) 算例结合：记录 **wall time + 单元数**
- 写入 manifest 扩展字段 `performance.wall_time_sec`、`performance.cells_per_sec`

### 3.3 Profiling 开关

```bash
# 规划 CLI
asimu run --case foo.toml --profile
# 生成本地 flamegraph（tracing-chrome / pprof 选型待定）
```

- 仅开发/本地；CI 默认关闭
- GPU 路径（v1.2+）单独 ADR 评估 nsys/RenderDoc 工作流

---

## 4. 与 Run Manifest 的集成

`RunManifest` 扩展块（规划）：

```json
{
  "observability": {
    "metrics_path": "output/metrics.jsonl",
    "wall_time_sec": 12.34,
    "phases_ms": {
      "assemble": 800,
      "linear_solve": 9200,
      "io": 120
    }
  }
}
```

MCP `get_run_summary` / Resource `asimu://run/latest` 包含上述摘要。

---

## 5. 演进里程碑

| 版本 | 交付 |
|------|------|
| v0.1 | `tracing` stderr |
| v0.3 | Manifest 含基础 `wall_time_sec` |
| v0.4 | `metrics.jsonl` 原型；首个 criterion bench |
| v0.5 | `[observability]` 配置；manifest 含 `phases_ms` |
| v1.0 | 发布前 macro-benchmark 记录进 manifest |
| v1.2 | GPU 阶段 timing（若 exec 落地） |

---

## 6. 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) §8.5.1 Run Manifest
- [BENCHMARKS.md](BENCHMARKS.md) — V&V 与 macro 性能
- [adr/0005-time-integration.md](adr/0005-time-integration.md) — 时间步指标
