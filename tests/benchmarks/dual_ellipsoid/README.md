# dual_ellipsoid

**benchmark_id**: `dual_ellipsoid`

非结构混合网格外气动工程算例（约 221 万单元 / 475 万内面 / 9 色）。用于 ADR 0013 **E5** exec scatter 与 LU-SGS RHS 性能回归。

## 前置

- CGNS 网格 `mix.cgns`（不在仓库内）
- 默认路径：`output/case_dualellipsoid/mix.cgns`，或通过环境变量 `ASIMU_MIX_CGNS_PATH` 指定
- 构建：`cargo build --release`（默认 features 含 `io-cgns`、`io-vtk`、`parallel-fvm`、`simd-fvm`）

## 运行

```bash
# 使用 output/ 下完整算例（推荐）
asimu --case output/case_dualellipsoid/case.toml --log-level info

# Chrome trace（桶级 scatter 为 trace 级，见 OBSERVABILITY.md）
asimu --case output/case_dualellipsoid/case.toml --log-level info --chrome-trace
```

## E5 判据（手工 / CI slow-tests）

1. **数值**：`residual.csv` 单调下降，末步 `log10_residual` 有限
2. **性能**：相对 P9 基线（或上一 release tag）LU-SGS 步 `profile_time_integration_ms` **回归 < 5%**
3. **Trace 阶段**：Perfetto 中 `unstructured_lusgs_rhs` / `unstructured_viscous_interior_flux_fused` 占步耗时主导；**不应**出现百万级 `exec_colored_bucket_scatter`（每色桶 1 次，默认 trace 级）

## G2 CUDA smoke（ADR 0017）

`case_cuda_f32.toml`：`backend = cuda`、`compute_precision = f32`、显式 Euler 短步（2 步）。需 `cargo build --features cuda,io-cgns` 与 GPU。

```bash
asimu --case tests/benchmarks/dual_ellipsoid/case_cuda_f32.toml --log-level info
```

集成测试 `dual_ellipsoid_cuda_smoke_when_cgns_present`（`#[ignore = gpu]` + `slow-tests`）在 CGNS 可用时比对 CPU/CUDA 残差趋势。

日志字段见算例步末 `非结构时间步 profiling`（`profile_time_integration_ms` 等）。
