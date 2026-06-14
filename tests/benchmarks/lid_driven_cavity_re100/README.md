# Lid-Driven Cavity Re=100 — 不可压缩顶盖驱动方腔（ADR 0015 I2）

**benchmark_id**: `lid_driven_cavity_re100`

## 目的

验证 **稳态 SIMPLEC** 在 \(16\times16\times1\) 结构化网格上与 Ghia et al. (1982) Re=100 中心线速度的定量对比。

- 顶盖 `moving_wall`：\(\mathbf{u}_{lid}=(1,0,0)\)；
- 侧壁 `wall no_slip`；\(k\) 方向 `symmetry`（二维方腔）；
- \(Re=100\)（\(\nu=0.01\)，\(L=U=1\)）；
- `time.mode = steady`，`time.scheme = simplec`，`convection_scheme = upwind`。

## 判据（CI smoke）

| 量 | 阈值 |
|----|------|
| `simplec_converged` | `true`（约 3000 外层步） |
| `max_abs_corrected_field_divergence_after_boundary` | \(< 10^{-5}\) |
| `max_abs_corrected_velocity_delta_interior` | \(< 10^{-6}\) |
| Ghia \(u(y)\) 中心线 `max_abs` / `l2` | \(< 0.22\) / \(< 0.12\) |
| Ghia \(v(x)\) 中心线 `max_abs` / `l2` | \(< 0.12\) / \(< 0.09\) |

更细网格（24×24+）与更高阶格式在后续里程碑单独登记。

## 运行

```bash
asimu --case tests/benchmarks/lid_driven_cavity_re100/case.toml
cargo test --test case_run lid_driven_cavity_re100
```

## 参考文献

Ghia, U., Ghia, K. N., Shin, C. T. (1982). *Journal of Computational Physics*, 48(3), 387–411.
