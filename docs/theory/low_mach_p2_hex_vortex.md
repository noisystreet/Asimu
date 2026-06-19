# 低马赫 P2 验证（hex_vortex_street 定常）

> 日期：2026-06-19
> 前置：[low_mach_ab_hex_vortex.md](low_mach_ab_hex_vortex.md)（P1 A/B）、[low_mach_cutoff_sweep_hex_vortex.md](low_mach_cutoff_sweep_hex_vortex.md)（P1 扫参）
> 代码：`cb8b50e` — LU-SGS 扫掠 \(\lambda_{ij}\) 与 \(\sigma^\text{LM}\) 共用 \(\beta\) 缩放

## 1. 目的

验证 P2（扫掠面耦合与预处理谱半径一致）是否消除 P1 在高 CFL 下的发散，并对比 OFF/ON 收敛趋势。

## 2. 运行配置

与 A/B 探针相同（`Ma=0.1`，`lu_sgs + local_time_step`，800 步），三组：

| 标签 | 预处理 | `M_cut` | `cfl_max` |
|------|--------|---------|-----------|
| OFF | 关 | — | 100 |
| ON | 开 | 0.1 | 100 |
| cutoff | 开 | 0.1 | 20 |

运行命令（P2 二进制）：

```bash
cargo run --release --bin asimu -- --case output/case_hex_votexstreet/case_steady_probe_lm_off.toml
cargo run --release --bin asimu -- --case output/case_hex_votexstreet/case_steady_probe_lm_on.toml
cargo run --release --bin asimu -- --case output/case_hex_votexstreet/case_steady_probe_lm_cutoff_0p1.toml
```

## 3. 指标对比（`residual.csv`）

| Case | steps | first | last | drop | slope/step | min | max | reach -3 |
|------|------:|------:|-----:|-----:|-----------:|----:|----:|:--------:|
| OFF（P2） | 800 | 2.0022 | 1.0673 | +0.935 | -1.17e-3 | -0.22 | 2.00 | 否 |
| ON `cfl_max=100`（**P2**） | 800 | 2.0022 | 2.7491 | -0.747 | +9.35e-4 | 1.55 | 3.79 | 否 |
| ON `cfl_max=100`（P1，对照） | 800 | 2.0022 | **36.77** | -34.76 | +4.35e-2 | — | — | 否 |
| cutoff `cfl_max=20`（**P2**） | 800 | 2.0022 | 2.6036 | -0.601 | +7.53e-4 | 1.42 | 2.94 | 否 |
| cutoff `cfl_max=20`（P1，对照） | 800 | 2.0022 | 2.6692 | -0.667 | +8.35e-4 | 1.76 | 3.84 | 否 |

## 4. 结论

1. **P2 消除高 CFL 发散**：`cfl_max=100` + ON 时，P1 末步 log10 爆炸至 36.77；P2 稳定在 2.75，说明扫掠 \(\lambda_{ij}\) 与 \(\sigma^\text{LM}\) 不一致是 P1 发散主因。
2. **OFF 仍优于 ON**：P2 后 ON 不再爆炸，但 800 步内残差仍高于 OFF（1.07 vs 2.75），斜率为正，未达 -3。
3. **保守 CFL 略改善**：`cfl_max=20` 下 P2 末值 2.60 vs P1 2.67；中间最低残差 ~1.42 vs P1 ~1.76。

## 5. 下一步建议

- 在 P2 基础上扫更低 `cfl_max`（5–10）寻找 ON 可收敛区间；
- 或与 OFF 同 CFL 长跑（20000 步主 case）对比物理量与残差平台；
- 平滑退化（`low_mach_max_mach` / `blend`）仍待实现。
