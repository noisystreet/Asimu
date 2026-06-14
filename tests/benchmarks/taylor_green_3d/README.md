# Taylor–Green 3D 涡衰减（I3）

**benchmark_id**: `taylor_green_3d`

## 物理

周期域 \([0,2\pi]^2\times[0,L_z]\)（\(n_z=1\) 准 2D）上的 Taylor–Green 涡，Reynolds 数由 \(\nu\) 与 \(U_{\mathrm{ref}},L_{\mathrm{ref}}\) 决定。

初场（SI 输入，求解器内部无量纲化）：

\[
u=\sin x\cos y\cos z,\quad v=-\cos x\sin y\cos z,\quad w=0
\]

层流动能衰减（Brachet et al. 1983；见 ADR 0015 I3）：

\[
\frac{E(t)}{E(0)}=\exp(-4\,\nu^* t^*),\quad \nu^*=1/Re,\ t^*=t\,U_{\mathrm{ref}}/L_{\mathrm{ref}}
\]

## 数值

- `time.mode = transient`，`time.scheme = bdf1`（BDF1 动量 + PISO-2）
- 16×16×1，双周期 + z 对称
- 中心对流格式

## 验证（I3 smoke）

当前 16×16 collocated FVM + PISO-2 在粗网格上存在显著数值耗散，CI 断言：

- 瞬态 PISO-2 跑满 `max_steps`
- 动能单调衰减（\(\nu>0\)）
- spin-up 后 \(-\mathrm{d}\ln E/\mathrm{d}t\) 与 \(4\nu^*\) **同量级**（0.05×–50×，待网格/求解器改进后收紧）

完整 \(E/E_0=\exp(-4\nu^* t^*)\) 对照见 ADR 0015，计划在更细网格或离散散度投影初场后启用。

```bash
asimu --case tests/benchmarks/taylor_green_3d/case.toml
cargo test --test case_run taylor_green_3d
```

## 参考文献

1. Brachet, M. E., et al. (1983). Small-scale structure of the Taylor–Green vortex. *Journal of Fluid Mechanics*, 130, 411–452.
2. Ghia et al. (1982) — 方腔对照；本算例为周期 TG 衰减。
