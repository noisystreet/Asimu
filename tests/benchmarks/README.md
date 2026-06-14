# 验证算例（Benchmarks）

本目录存放 **物理验证算例**，与 `tests/fixtures/`（冒烟 / golden）区分。

完整规划见 [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)。

## 状态

| 算例 | 版本 | 状态 |
|------|------|------|
| `1d_diffusion_analytical/` | v0.2 | 骨架（case + expected） |
| `sod_1d/` | v1.x | Sod 激波管 vs 精确 Riemann 解 |
| `unstructured_freestream/` | v0.2+ | 非结构均匀来流 RHS 近零（一阶 / 二阶线性重构 BJ·V） |
| `unstructured_cuda_freestream/` | v1.3 | CUDA G1：f32 非结构一阶 Roe device kernel 端到端 smoke |
| `channel_poiseuille/` | v0.3 | 不可压缩通道流 smoke V&V 骨架 |
| `lid_driven_cavity_re100/` | v0.4 | 不可压缩 Re=100 顶盖驱动方腔 smoke V&V 骨架 |
| `1d_advection_diffusion/` | v0.2 | 规划 |

运行 benchmark 时，`RunManifest.benchmark_id` 应设为对应 ID（见 [OBSERVABILITY.md](../../docs/OBSERVABILITY.md)）。

新增算例请遵循 `docs/BENCHMARKS.md` 目录约定（`README.md` + `case.toml` + `expected.json`）。
