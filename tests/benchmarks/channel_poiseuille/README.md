# Channel Poiseuille — 不可压缩通道流

**benchmark_id**: `channel_poiseuille`

## 目的

验证结构化 3D 不可压缩 SIMPLEC 路径能运行典型内流通道算例，并为后续解析速度剖面对比预留 V&V 数据结构。当前阶段使用单层 \(z\) 方向网格表示二维通道：

- 左侧 `velocity_inlet` 给定均匀入口速度；
- 右侧 `pressure_outlet` 固定参考压力；
- 上下壁面 `wall no_slip = true`；
- 前后面 `symmetry` 表示二维挤出方向。

完整 Poiseuille 解析验证目标为

\[
u(y)=\frac{1}{2\nu}\left(-\frac{\mathrm{d}p}{\mathrm{d}x}\right)y(H-y),
\]

后续在入口剖面、压力梯度驱动或体力源项支持完善后，`expected.json` 中的 smoke 阈值应替换为剖面误差阈值。

## 参考文献

1. White, F. M. (2011). *Fluid Mechanics*, 7th ed., McGraw-Hill. Chapter 3, internal viscous flows.
2. Ferziger, J. H., Peric, M., Street, R. L. (2020). *Computational Methods for Fluid Dynamics*, 4th ed., Springer. Chapter 7.

## 运行

```bash
asimu --case tests/benchmarks/channel_poiseuille/case.toml
cargo test --test case_run channel_poiseuille
```
