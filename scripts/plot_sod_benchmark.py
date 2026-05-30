#!/usr/bin/env python3
"""绘制 Sod benchmark 文本剖面（由 sod_benchmark_export 示例生成）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_metadata(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines:
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        for token in body.split():
            if "=" in token:
                key, value = token.split("=", 1)
                meta[key] = value
    return meta


def load_profile(path: Path) -> tuple[dict[str, str], np.ndarray]:
    text = path.read_text(encoding="utf-8").splitlines()
    meta = parse_metadata(text)
    data_lines = [
        line
        for line in text
        if line and not line.startswith("#") and not line.startswith("x ")
    ]
    data = np.loadtxt(data_lines, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return meta, data


def title_from_meta(meta: dict[str, str]) -> str:
    ncells = meta.get("ncells", "?")
    final_time = meta.get("final_time", "?")
    if meta.get("format") == "compare":
        roe_l1 = meta.get("roe_l1", meta.get("muscl_roe_l1", "?"))
        muscl_l1 = meta.get("muscl_hllc_l1", "?")
        return (
            f"Sod shock tube compare (ncells={ncells}, t={final_time})\n"
            f"MUSCL+Roe L1={roe_l1}, MUSCL+HLLC L1={muscl_l1}"
        )
    l1 = meta.get("l1_density", "?")
    l2 = meta.get("l2_density", "?")
    scheme = meta.get("scheme", "numeric")
    return f"Sod shock tube ({scheme}, ncells={ncells}, t={final_time})\nL1={l1}, L2={l2}"


def plot_single(meta: dict[str, str], data: np.ndarray, output: Path | None) -> None:
    x = data[:, 0]
    rho_num = data[:, 1]
    rho_exact = data[:, 2]
    rho_err = data[:, 3]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title_from_meta(meta))

    ax0 = axes[0]
    ax0.plot(x, rho_exact, "k-", linewidth=1.5, label="exact")
    ax0.plot(
        x,
        rho_num,
        "o",
        markersize=3,
        linestyle="none",
        color="tab:blue",
        label="numeric",
    )
    ax0.set_ylabel(r"$\rho$")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(x, rho_err, color="tab:red", linewidth=1.0)
    ax1.axhline(0.0, color="k", linewidth=0.6)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\rho_{\mathrm{num}} - \rho_{\mathrm{exact}}$")
    ax1.grid(True, alpha=0.3)

    save_or_show(fig, output)


def plot_compare(meta: dict[str, str], data: np.ndarray, output: Path | None) -> None:
    x = data[:, 0]
    rho_roe = data[:, 1]
    rho_muscl = data[:, 2]
    rho_exact = data[:, 3]
    err_roe = data[:, 4]
    err_muscl = data[:, 5]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title_from_meta(meta))

    ax0 = axes[0]
    ax0.plot(x, rho_exact, "k-", linewidth=1.5, label="exact")
    ax0.plot(
        x,
        rho_roe,
        "o",
        markersize=3,
        linestyle="none",
        color="tab:blue",
        label="MUSCL+Roe",
    )
    ax0.plot(
        x,
        rho_muscl,
        "s",
        markersize=3,
        linestyle="none",
        color="tab:orange",
        label="MUSCL+HLLC",
    )
    ax0.set_ylabel(r"$\rho$")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(x, err_roe, color="tab:blue", linewidth=1.0, label="MUSCL+Roe error")
    ax1.plot(x, err_muscl, color="tab:orange", linewidth=1.0, label="MUSCL+HLLC error")
    ax1.axhline(0.0, color="k", linewidth=0.6)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\rho_{\mathrm{num}} - \rho_{\mathrm{exact}}$")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    save_or_show(fig, output)


def save_or_show(fig: plt.Figure, output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"OK  wrote {output}")
    else:
        plt.show()


def plot_profile(meta: dict[str, str], data: np.ndarray, output: Path | None) -> None:
    if meta.get("format") == "compare" or data.shape[1] >= 6:
        if data.shape[1] < 6:
            raise SystemExit("对比格式数据列不足，期望 6 列")
        plot_compare(meta, data, output)
    else:
        if data.shape[1] < 4:
            raise SystemExit("数据列不足，期望: x rho_numeric rho_exact rho_error")
        plot_single(meta, data, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        type=Path,
        help="sod_benchmark_export 生成的文本文件",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 PNG（省略则弹出交互窗口）",
    )
    args = parser.parse_args()
    if not args.profile.is_file():
        raise SystemExit(f"文件不存在: {args.profile}")
    meta, data = load_profile(args.profile)
    plot_profile(meta, data, args.output)


if __name__ == "__main__":
    main()
