# asimu 验证算例库（V&V Benchmarks）

> 与 golden test（防回归）互补：本目录关注 **物理正确性** 与 **文献对比**。
> 架构：[ARCHITECTURE.md](ARCHITECTURE.md) §8.5.6 · 数据：[DATA_MODEL.md](DATA_MODEL.md)

**状态**：规划（v0.2 起 1D 算例，v0.4+ 2D 方腔）· 目录：`tests/benchmarks/`

---

## 1. 定位与原则

| 原则 | 说明 |
|------|------|
| **物理优先** | 验证离散与模型，不是防 typo |
| **可追溯** | 每个算例 README 含参考文献 / DOI |
| **可自动化** | `expected.json` 机器可读；CI 断言 |
| **Manifest 联动** | 运行 manifest 写入 `benchmark_id` |

与 golden test 的区别见 [ARCHITECTURE.md](ARCHITECTURE.md) §8.5.6 · 性能 macro 基准见 [OBSERVABILITY.md](OBSERVABILITY.md) · metrics 与文献不一致时的排查顺序见 [DEBUG_CHECKLIST.md](DEBUG_CHECKLIST.md)。

## 2. 与 golden test 的区别

| | Golden test | Benchmark（本文） |
|---|-------------|-------------------|
| 目的 | 防止代码变更导致数值漂移 | 验证物理模型与离散实现正确 |
| 参考 | 仓库内快照 / 解析解 | 文献、经典算例数据库 |
| CI | 每次 PR 必跑 | 小算例必跑；大算例 `#[ignore]` 或 nightly |
| 位置 | `tests/fixtures/` + 集成测试 | `tests/benchmarks/` |

---

## 3. 目录约定

```
tests/benchmarks/
├── README.md                 # 本文件副本 / 索引
├── 1d_diffusion_analytical/  # v0.2 — 解析解对比
│   ├── case.toml
│   ├── expected.json         # 参考值与容差
│   └── README.md
├── channel_poiseuille/       # v0.3 — 通道 Poiseuille 流
└── lid_driven_cavity_re100/  # v0.4 — Ghia et al. 1982 参考点
```

每个算例子目录 **必须** 包含：

| 文件 | 内容 |
|------|------|
| `README.md` | 物理描述、参考文献、网格要求 |
| `case.toml` | 可复现输入（或指向 fixture 路径） |
| `expected.json` | 标量/剖面参考值 + 相对/绝对容差 |

---

## 4. 规划算例列表

| ID | 版本 | 方程/场景 | 参考来源 | CI |
|----|------|-----------|----------|-----|
| `1d_diffusion_analytical` | v0.2 | 1D 稳态扩散，Dirichlet | 解析解 | 必跑 |
| `1d_advection_diffusion` | v0.2 | 1D 对流-扩散 | 解析 / manufactured | 必跑 |
| `channel_poiseuille` | v0.3 | 2D 不可压通道 | 解析速度剖面（当前 smoke 骨架） | 必跑 |
| `lid_driven_cavity_re100` | v0.4 | 方腔 Re=100 | Ghia 1982 中心线（16×16 稳态 SIMPLEC） | 必跑 smoke |
| `taylor_green_3d` | v0.3 **I3 完成** | 周期 TG 涡 | 动能衰减 + Rhie-Chow IC + 首步 coupling（CI \(\|E/E_0-\exp(-4\nu t)\| < 0.01\)） | 必跑 V&V |
| `backward_facing_step` | v1.x | 台阶流 | 实验/文献（待定） | ignore |

---

## 5. `expected.json` schema

当前 **schema_version = 1**（由 `case::benchmark_expected` 解析）：

```json
{
  "schema_version": 1,
  "benchmark_id": "1d_diffusion_analytical",
  "asimu_min_version": "0.2.0",
  "status": "optional_ci_tier_label",
  "quantities": [
    {
      "name": "L2_error",
      "value": 1.0e-4,
      "tolerance_abs": 1.0e-6,
      "source": "analytical"
    }
  ],
  "profiles": []
}
```

Run Manifest（`output/run-manifest.json`，**schema_version = 2**）在存在 `expected.json` 时写入 `benchmark_status`，不可压算例另写入 `time.incompressible_advance`（`steady_coupling` / `steady_pseudo_time` / `physical_transient`）。

---

## 6. 运行与 CI

```bash
# 规划命令（v0.4+ Makefile target）
make bench          # 跑非 ignore 的 benchmark
make bench-all      # 含大算例，本地/nightly
```

- PR CI：仅 `tests/benchmarks/` 中无 `#[ignore]` 标记的算例
- 参考值变更：须 PR 说明 + 文献引用 + CHANGELOG

---

## 7. 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) §14 测试策略
- [adr/0002-layered-cfd-architecture.md](adr/0002-layered-cfd-architecture.md)
