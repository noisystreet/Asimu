#!/usr/bin/env python3
"""绘制可压缩算例残差 CSV（log10 RMS）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def plot_residual(csv_path: Path, output: Path | None) -> None:
    cols = load_csv(csv_path)
    step = cols["step"]
    t = cols["t"]
    log10_r = cols["log10_residual"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step, log10_r, "o-", color="tab:orange", label="log10(RMS(rho_dot))")
    ax.set_xlabel("step")
    ax.set_ylabel("log10(RMS(rho_dot))")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle(f"Residual history (t_end={t[-1]:.3e} s)")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"OK  wrote {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="residual.csv 路径")
    parser.add_argument("--output", "-o", type=Path, default=None, help="输出 PNG")
    args = parser.parse_args()
    plot_residual(args.csv, args.output)


if __name__ == "__main__":
    main()
