# 1D 稳态扩散 — 解析解对比

## 物理

一维稳态扩散，\(D = 1\)，域 \([0, 1]\)：

- 左端 Dirichlet：\(\phi(0) = 0\)
- 右端 Dirichlet：\(\phi(1) = 1\)

**解析解**：\(\phi(x) = x\)

## 验证量

| 量 | 定义 | 容差 |
|----|------|------|
| `L2_error` | \(\|\phi_h - \phi_{exact}\|_2 / \sqrt{N}\) | ≤ `1.0e-4`（32 单元均匀网格） |

## 参考文献

1. 解析解直接由式 (1) 积分得到（见 [docs/theory/fvm_diffusion.md](../../docs/theory/fvm_diffusion.md)）。
2. Patankar (1980) Ch. 5 — FVM 离散参考。

## 运行

```bash
asimu --case tests/benchmarks/1d_diffusion_analytical/case.toml
cargo test --test case_run
cargo test --test boundary_1d_diffusion
```

## 文件

| 文件 | 说明 |
|------|------|
| `case.toml` | 算例输入（[CASE_FORMAT.md](../../docs/CASE_FORMAT.md)） |
| `expected.json` | 参考值与容差 |
