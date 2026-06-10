# Lid-Driven Cavity Re=100 — 不可压缩顶盖驱动方腔

**benchmark_id**: `lid_driven_cavity_re100`

## 目的

验证不可压缩 SIMPLEC 路径能运行封闭腔体、移动壁、压力参考单元组合。当前 case 使用 \(8\times8\times1\) 粗网格作为 CI 长迭代诊断 benchmark：

- 顶盖 `moving_wall`：\(\mathbf{u}_{lid}=(1,0,0)\)；
- 其余侧壁 `wall no_slip = true`；
- 前后面 `symmetry` 表示二维方腔；
- \(L_{ref}=1\)、\(U_{ref}=1\)、\(\nu=0.01\)，因此 \(Re=100\)。

当前 runner 会在 `Incompressible3dRunMetrics.centerline_profiles` 中返回近似 \(x=0.5\) 和 \(y=0.5\) 的 cell-centered 中心线速度样本，并在 `lid_cavity_profile_error` 中给出相对 Ghia 等人 Re=100 表格数据的线性插值误差。该误差现在作为粗网格定量 V&V：中心线 `max_abs` 必须小于 `1.0`；后续完整验证应在真正的 ghost/face wall 边界、更稳健的对流离散与网格加密完成后继续收紧。

当前判据要求 100 步不可压缩 PISO smoke 路径稳定完成，压力校正与动量线性求解收敛，并把欠松弛后连续性残差压到 `3.0e-5` 以内；`time.min_steps = 100` 避免初始阶段假收敛，外层使用 `time.tolerance = 3.0e-5` 同时检查连续性、动量残差与非速度约束 owner 速度更新量，因此该粗网格 case 预计 `simplec_converged=true`，但仍保留总速度更新量略高的边界 owner 诊断。

排查收敛时同时查看 `max_abs_corrected_divergence`、`max_abs_underrelaxed_corrected_divergence` 与 `max_abs_corrected_field_divergence_after_boundary`：第一项是全量压力校正方程残差，第二项是按 `pressure_under_relaxation` 实际修正后的 SIMPLEC 连续性残差，第三项是修正速度并重施加边界后的边界感知 face-flux 散度。`max_abs_corrected_velocity_delta_interior` / `boundary` 会把速度更新量拆成非速度约束 owner 与速度约束边界 owner 两类；当前剩余超阈值项来自后者。若第一项很小而后两项仍大，优先检查速度修正、Rhie-Chow 通量和边界一致性。

## 参考文献

1. Ghia, U., Ghia, K. N., Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411. DOI: 10.1016/0021-9991(82)90058-4.
2. Ferziger, J. H., Peric, M., Street, R. L. (2020). *Computational Methods for Fluid Dynamics*, 4th ed., Springer. Chapter 7.

## 运行

```bash
asimu --case tests/benchmarks/lid_driven_cavity_re100/case.toml
cargo test --test case_run lid_driven_cavity_re100
```
