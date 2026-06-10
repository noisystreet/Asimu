# Lid-Driven Cavity Re=100 — 不可压缩顶盖驱动方腔

**benchmark_id**: `lid_driven_cavity_re100`

## 目的

验证不可压缩 SIMPLEC 路径能运行封闭腔体、移动壁、压力参考单元组合。当前 case 使用 \(8\times8\times1\) 粗网格作为 CI smoke benchmark：

- 顶盖 `moving_wall`：\(\mathbf{u}_{lid}=(1,0,0)\)；
- 其余侧壁 `wall no_slip = true`；
- 前后面 `symmetry` 表示二维方腔；
- \(L_{ref}=1\)、\(U_{ref}=1\)、\(\nu=0.01\)，因此 \(Re=100\)。

当前 runner 会在 `Incompressible3dRunMetrics.centerline_profiles` 中返回近似 \(x=0.5\) 和 \(y=0.5\) 的 cell-centered 中心线速度样本，并在 `lid_cavity_profile_error` 中给出相对 Ghia 等人 Re=100 表格数据的线性插值误差。该误差目前仅作 smoke 诊断；后续完整验证应在封闭腔体长迭代稳定后收紧剖面误差阈值。

当前 smoke 判据要求压力校正收敛，连续性与动量线性残差低于 `time.tolerance = 1.0e-8`；SIMPLEC 外层还会检查速度更新量，因此该粗网格 2 步 case 预计 `simplec_converged=false`，用于暴露“尚未稳态”的状态。

## 参考文献

1. Ghia, U., Ghia, K. N., Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411. DOI: 10.1016/0021-9991(82)90058-4.
2. Ferziger, J. H., Peric, M., Street, R. L. (2020). *Computational Methods for Fluid Dynamics*, 4th ed., Springer. Chapter 7.

## 运行

```bash
asimu --case tests/benchmarks/lid_driven_cavity_re100/case.toml
cargo test --test case_run lid_driven_cavity_re100
```
