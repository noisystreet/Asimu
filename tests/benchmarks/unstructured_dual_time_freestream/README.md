# 非结构双时间步均匀来流（freestream）smoke

**benchmark_id**: `unstructured_dual_time_freestream`

## 目的

验证 `time.scheme = "dual_time"` 在单四面体封闭远场网格上：

1. 均匀来流内层 \(\|R_{\mathrm{eff}}\|_{\mathrm{rms}}\) 可降至阈值以下（f64）；
2. f32 CPU 路径 smoke 通过（容差宽于 f64，见 ADR 0016）。

理论见 [dual_time_stepping.md](../../../docs/theory/dual_time_stepping.md)。

## 运行

```bash
cargo test runs_single_tet_unstructured_dual_time_freestream -- --nocapture
cargo test runs_single_tet_unstructured_dual_time_freestream_f32_step -- --nocapture
```

### CUDA f32（P3b）

约束：`compute_precision = "f32"`、`backend = "cuda"`、`lusgs_sweep = false`、`local_time_step = true`、正数 `dt`。

```bash
make test-cuda
cargo test runs_single_tet_unstructured_cuda_dual_time_smoke_step --features cuda -- --ignored gpu
cargo run --features cuda -- --case tests/benchmarks/unstructured_dual_time_freestream/case_cuda_f32.toml
```

见 `case_cuda_f32.toml` 与 `docs/theory/dual_time_stepping.md` §3.5。

`case.toml` 为 manifest / 文档骨架；网格由 `attach_single_tet_farfield` 注入。

## 参考值

见 `expected.json` 与 `src/case/compressible_unstructured_3d_tests.rs`。
