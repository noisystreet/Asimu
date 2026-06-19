# 低马赫预处理 A/B（hex_vortex_street 定常）

> 日期：2026-06-19
> 目标：对比 `low_mach_preconditioning` 关闭/开启时，低马赫定常收敛趋势。

## 1. 运行配置

统一基线（`Ma=0.1`，`lu_sgs + local_time_step`）：

- 网格：`output/case_hex_votexstreet/project1.cgns`
- 精度/后端：`f64 + cpu`
- 时间推进：`scheme = "lu_sgs"`
- CFL：`cfl=0.01, cfl_max=100, cfl_ramp_steps=1000`
- 运行步数：`max_steps=800`

唯一差异：

- OFF：`low_mach_preconditioning = false`
- ON：`low_mach_preconditioning = true, low_mach_mach_cutoff = 0.1`

运行命令：

```bash
cargo run --release --bin asimu -- --case output/case_hex_votexstreet/case_steady_probe_lm_off.toml
cargo run --release --bin asimu -- --case output/case_hex_votexstreet/case_steady_probe_lm_on.toml
```

## 2. 指标对比（来自 residual.csv）

| Case | steps | inner1 log10 | innerN log10 | drop (inner1-innerN) | slope/step | reach -3 |
|---|---:|---:|---:|---:|---:|---|
| OFF | 800 | 2.0022 | 1.0673 | +0.9350 | -1.1702e-3 | 未达到 |
| ON (`M_cut=0.1`) | 800 | 2.0022 | 36.7671 | -34.7648 | +4.3510e-2 | 未达到 |

## 3. 结论

在“高 `cfl_max=100` + `M_cut=0.1`”组合下，P1 低马赫预处理显著放大了局部伪时间步，导致该工况残差发散。
因此在当前 P1 版本中，低马赫预处理需要与更保守的 CFL/截止参数联调后再使用。

`M_cut` 扫参结果见 [low_mach_cutoff_sweep_hex_vortex.md](low_mach_cutoff_sweep_hex_vortex.md)（`cfl_max=20`，三者均未稳定收敛）。
