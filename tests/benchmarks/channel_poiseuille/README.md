# Channel Poiseuille — 不可压缩通道流

**benchmark_id**: `channel_poiseuille`

## 目的

验证结构化 3D 不可压缩 SIMPLEC 路径能运行典型内流通道算例，并输出解析速度剖面对比所需的 V&V 诊断。当前阶段使用单层 \(z\) 方向网格表示二维通道：

- \(x\) 方向两端 `pressure_outlet` 固定参考压力；
- 上下壁面 `wall no_slip = true`；
- 前后面 `symmetry` 表示二维挤出方向。
- `[incompressible].body_force = [0.08, 0, 0]` 提供每单位质量体力驱动。

完整 Poiseuille 解析验证目标为

\[
u(y)=\frac{1}{2\nu}\left(-\frac{\mathrm{d}p}{\mathrm{d}x}\right)y(H-y),
\]

runner 会在 `Incompressible3dRunMetrics.centerline_profiles` 中返回 \(x\) 中线的 \(u(y)\) 样本，并在 `poiseuille_profile_error` 中给出相对解析式的 `max_abs` 与 `l2` 误差。后续在稳态更新量收敛判据完善后，`expected.json` 中的诊断标记应替换为剖面误差阈值。

当前 smoke 判据要求压力校正收敛，且 SIMPLEC 外层在 `time.tolerance = 1.0e-8` 下报告收敛。

## 参考文献

1. White, F. M. (2011). *Fluid Mechanics*, 7th ed., McGraw-Hill. Chapter 3, internal viscous flows.
2. Ferziger, J. H., Peric, M., Street, R. L. (2020). *Computational Methods for Fluid Dynamics*, 4th ed., Springer. Chapter 7.

## 运行

```bash
asimu --case tests/benchmarks/channel_poiseuille/case.toml
cargo test --test case_run channel_poiseuille
```
