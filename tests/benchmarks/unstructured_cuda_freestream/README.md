# 非结构 CUDA 均匀来流（G1）

**benchmark_id**: `unstructured_cuda_freestream`

## 目的

验证 ADR 0017 **G1** 端到端路径：非结构 f32 + `backend = "cuda"` + 一阶 Roe 内面 device kernel，在均匀来流下 RHS 近零且与 CPU f32 一致。

## 约束（G1 能力矩阵）

- `compute_precision = "f32"`
- `backend = "cuda"`（须 `--features cuda` 编译）
- `reconstruction = first_order`
- `flux = "roe"` 或 `"hanel_van_leer"`（HVL）
- `time.scheme = "rk4"` 或 `"euler"`
- 无粘性 / 无 `lu_sgs` / 无 GMRES

## 网格

单四面体，集成测试通过 `attach_single_tet_farfield` 注入。

## 运行

```bash
# 编译（需 nvcc + NVIDIA 驱动）
cargo build --release --features cuda

# 单元 / 集成（GPU 单测默认 ignore）
make test-cuda

# 算例 smoke（需 GPU）
cargo run --features cuda -- --case tests/benchmarks/unstructured_cuda_freestream/case.toml
```

无 GPU 环境：`make check-cuda` 仅编译 + clippy；默认 `make check` 不含 `cuda` feature。

## 参考

| 量 | 期望 | 容差 |
|----|------|------|
| RMS(\(\dot\rho\)) | 0 | \(10^{-4}\)（CUDA vs CPU f32） |

见 `expected.json` 与 `assembly_unstructured_typed_cuda` 测试。
