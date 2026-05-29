# 验证算例（Benchmarks）

本目录存放 **物理验证算例**，与 `tests/fixtures/`（冒烟 / golden）区分。

完整规划见 [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)。

## 状态

| 算例 | 版本 | 状态 |
|------|------|------|
| `1d_diffusion_analytical/` | v0.2 | 骨架（case + expected） |
| `1d_advection_diffusion/` | v0.2 | 规划 |
| `channel_poiseuille/` | v0.3 | 规划 |
| `lid_driven_cavity_re100/` | v0.4 | 规划 |

运行 benchmark 时，`RunManifest.benchmark_id` 应设为对应 ID（见 [OBSERVABILITY.md](../../docs/OBSERVABILITY.md)）。

新增算例请遵循 `docs/BENCHMARKS.md` 目录约定（`README.md` + `case.toml` + `expected.json`）。
