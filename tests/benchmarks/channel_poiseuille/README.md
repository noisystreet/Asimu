# Channel Poiseuille — 不可压缩通道流

**benchmark_id**: `channel_poiseuille`

## 目的

验证结构化 3D 不可压缩 SIMPLEC 路径能运行典型内流通道算例，并通过解析速度剖面做 V&V 检查。当前阶段使用单层 \(z\) 方向网格表示二维通道：

- \(x\) 方向两端 `periodic` 表示充分发展方向；
- 上下壁面 `wall no_slip = true`；
- 前后面 `symmetry` 表示二维挤出方向。
- `[incompressible].body_force = [0.08, 0, 0]` 提供每单位质量体力驱动。

完整 Poiseuille 解析验证目标为

\[
u(y)=\frac{1}{2\nu}\left(-\frac{\mathrm{d}p}{\mathrm{d}x}\right)y(H-y),
\]

runner 会在 `Incompressible3dRunMetrics.centerline_profiles` 中返回 \(x\) 中线的 \(u(y)\) 样本，并在 `poiseuille_profile_error` 中给出相对解析式的 `max_abs` 与 `l2` 误差。当前集成测试要求 SIMPLEC 外层按速度更新量收敛，并检查 `max_abs < 0.12`、`l2 < 0.08`。

当前 case 使用 `time.tolerance = 3.0e-5` 与最多 2000 次 SIMPLEC 外层迭代；该阈值用于粗网格 CI 验证，后续网格加密后应收紧剖面误差。

## 参考文献

1. White, F. M. (2011). *Fluid Mechanics*, 7th ed., McGraw-Hill. Chapter 3, internal viscous flows.
2. Ferziger, J. H., Peric, M., Street, R. L. (2020). *Computational Methods for Fluid Dynamics*, 4th ed., Springer. Chapter 7.

## 运行

```bash
asimu --case tests/benchmarks/channel_poiseuille/case.toml
cargo test --test case_run channel_poiseuille
```
