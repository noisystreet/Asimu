# 非结构均匀来流（freestream）验证

**benchmark_id**: `unstructured_freestream`

## 目的

验证非结构 FVM 在**均匀来流**下无粘 RHS 近零（离散守恒 / 重构一致性）。覆盖：

- 一阶 Godunov（`reconstruction = first_order`）
- 二阶线性重构（IDWLS 梯度外推 + Barth–Jespersen / Venkatakrishnan；TOML 仍写 `reconstruction = muscl` + `unstructured_limiter`）

理论见 [ADR 0012](../../../docs/adr/0012-unstructured-gradient-limiters.md)、[unstructured_fvm.md](../../../docs/theory/unstructured_fvm.md)。

## 网格

单四面体（4 节点、4 边界面），远场 BC 覆盖全部边界面。网格在 TOML 中无法内联定义，集成测试通过 `attach_single_tet_farfield` 注入 `UnstructuredMesh3d`。

## 运行

单元 / 集成测试（推荐）：

```bash
cargo test uniform_field_on_closed_tet uniform_freestream_linear_reconstruction -- --nocapture
cargo test runs_single_tet_unstructured -- --nocapture
```

`case.toml` 为 manifest / 文档骨架；完整非结构路径见 `src/case/compressible_unstructured_3d_tests.rs`。

## 参考值

| 量 | 期望 | 容差 |
|----|------|------|
| RMS(\(\dot\rho\)) | 0 | \(10^{-9}\)（二阶线性重构） / \(10^{-10}\)（一阶） |

见 `expected.json` 与 `assembly_unstructured` 内 golden 测试。

## f32 计算精度（ADR 0016 P5）

非结构 typed 路径在 `parallel-fvm` 下使用 exec **着色桶 + atomic scatter**；`f32` 残差经 `AtomicU32` CAS 累加（禁止扩成 `f64` residual）。一阶无粘、MUSCL 无粘与粘性内面均已接着色桶并行；LU-SGS 扫掠 source/耦合差分为原生 f32。

### 运行对比

在 `case.toml` 增加：

```toml
[numerics]
compute_precision = "f32"   # 或 "f64" 基线
```

集成测试（单四面体，V&V）：

```bash
cargo test f32_single_tet_uniform_freestream -- --nocapture
cargo test f32_single_tet_muscl_uniform_freestream -- --nocapture
cargo test scatter_inviscid_f32_serial_matches_atomic_parallel --features parallel-fvm
```

### 性能 benchmark（大网格）

与 `dual_ellipsoid` 相同构建（`parallel-fvm` + 可选 `simd-fvm`），对比 `compute_precision = "f32"` 与 `"f64"` 的步末 `profile_time_integration_ms` 与末步残差。回归判据：相对 P5 基线 LU-SGS/RHS 步耗时回归 **< 5%**（见 [dual_ellipsoid/README.md](dual_ellipsoid/README.md) E5 说明）。
