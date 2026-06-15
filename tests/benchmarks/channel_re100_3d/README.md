# Channel Re=100 — 入口/出口内流（ADR 0015 I4）

**benchmark_id**: `channel_re100_3d` · **状态**: I4 完成（质量守恒 + 充分发展剖面 smoke V&V）

## 物理

二维 Poiseuille 通道（\(n_z=1\) + `symmetry`）：

- \(x\)：`velocity_inlet` @ `i_min`，`pressure_outlet` @ `i_max`
- \(y\)：无滑移壁面
- \(Re = U H / \nu = 1\times 1 / 0.01 = 100\)

## 数值

| 项 | 值 |
|----|-----|
| 网格 | 32×8×1（\(L_x=8\)，入口发展段） |
| 求解 | 稳态 **SIMPLEC** + 一阶 **upwind** |
| 入口 | \(\mathbf{u}=(1,0,0)\) m/s（均匀） |
| 出口 | \(p=0\) Pa，面通量零梯度外推 |

## V&V（CI）

| 量 | 判据 |
|----|------|
| 质量守恒 | `mass_flux_imbalance_ratio` \(< 1.5\times 10^{-2}\)（32×8 upwind smoke；ADR 目标 \(10^{-6}\)） |
| 充分发展剖面 | `poiseuille_profile_error` @ \(x=3L/4\)：\(u/U_m=6(y/H)(1-y/H)\)，`max_abs` \(<0.35\)，`l2` \(<0.2\) |
| 连续性 | `max_abs_corrected_field_divergence_after_boundary` \(< 10^{-5}\)（非 \(p'=0\) owner） |
| 收敛 | `simplec_converged = true` |

metrics 排查：[docs/DEBUG_CHECKLIST.md](../../../docs/DEBUG_CHECKLIST.md)

## 运行

```bash
asimu --case tests/benchmarks/channel_re100_3d/case.toml
cargo test --test case_run channel_re100_3d
```

## 参考文献

1. White, F. M. (2011). *Fluid Mechanics*, 7th ed., Ch. 6.
2. Ferziger et al. (2020). *CFD*, 4th ed., Ch. 7–8.
