# 非结构均匀来流（freestream）验证

**benchmark_id**: `unstructured_freestream`

## 目的

验证非结构 FVM 在**均匀来流**下无粘 RHS 近零（离散守恒 / 重构一致性）。覆盖：

- 一阶 Godunov（`reconstruction = first_order`）
- 二阶 IDWLS + Barth–Jespersen / Venkatakrishnan（`reconstruction = muscl` + `unstructured_limiter`）

理论见 [ADR 0012](../../../docs/adr/0012-unstructured-gradient-limiters.md)、[unstructured_fvm.md](../../../docs/theory/unstructured_fvm.md)。

## 网格

单四面体（4 节点、4 边界面），远场 BC 覆盖全部边界面。网格在 TOML 中无法内联定义，集成测试通过 `attach_single_tet_farfield` 注入 `UnstructuredMesh3d`。

## 运行

单元 / 集成测试（推荐）：

```bash
cargo test uniform_field_on_closed_tet uniform_freestream_muscl -- --nocapture
cargo test runs_single_tet_unstructured -- --nocapture
```

`case.toml` 为 manifest / 文档骨架；完整非结构路径见 `src/case/compressible_unstructured_3d_tests.rs`。

## 参考值

| 量 | 期望 | 容差 |
|----|------|------|
| RMS(\(\dot\rho\)) | 0 | \(10^{-9}\)（MUSCL） / \(10^{-10}\)（一阶） |

见 `expected.json` 与 `assembly_unstructured` 内 golden 测试。
