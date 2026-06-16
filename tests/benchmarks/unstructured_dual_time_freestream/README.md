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

`case.toml` 为 manifest / 文档骨架；网格由 `attach_single_tet_farfield` 注入。

## 参考值

见 `expected.json` 与 `src/case/compressible_unstructured_3d_tests.rs`。
