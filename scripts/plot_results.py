#!/usr/bin/env python3
"""Generate the README benchmark figure from the committed reference CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "reference-rtx3050.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmark-summary.png"

WORKLOADS = [
    ("N8_C3_H32_W32_M16_K3_S1", "C3, K3"),
    ("N4_C16_H32_W32_M32_K3_S1", "C16, K3"),
    ("N4_C32_H31_W29_M64_K3_S2", "C32, K3, S2"),
    ("N2_C64_H28_W28_M64_K5_S1", "C64, K5"),
]
SERIES = [
    ("tiled-gemm", "FP32 tiled", "#3f5f8f"),
    ("tensor-fp16", "Tensor FP16", "#2f8f83"),
    ("tensor-bf16", "Tensor BF16", "#c58a32"),
]


def read_results(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    results: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            results[(row["shape"], row["executed"])] = {
                "gflops": float(row["gflops"]),
                "max_abs": float(row["max_abs"]),
            }
    return results


def plot(results: dict[tuple[str, str], dict[str, float]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, (throughput_ax, tradeoff_ax) = plt.subplots(
        1, 2, figsize=(11.2, 4.15), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    positions = np.arange(len(WORKLOADS))
    width = 0.24
    for series_index, (algorithm, label, color) in enumerate(SERIES):
        values = [results[(shape, algorithm)]["gflops"] for shape, _ in WORKLOADS]
        bars = throughput_ax.bar(
            positions + (series_index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        throughput_ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)

    throughput_ax.set_title("Kernel throughput across convolution workloads")
    throughput_ax.set_ylabel("Throughput (GFLOP/s)")
    throughput_ax.set_xlabel("Input channels and kernel")
    throughput_ax.set_xticks(positions, [label for _, label in WORKLOADS])
    throughput_ax.set_ylim(0, 475)
    throughput_ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    throughput_ax.legend(frameon=False, ncols=3, loc="upper left")

    representative = WORKLOADS[-1][0]
    tradeoff_points = []
    for algorithm, label, color in SERIES:
        row = results[(representative, algorithm)]
        tradeoff_points.append((row["max_abs"], row["gflops"], label, color))

    tradeoff_ax.plot(
        [point[0] for point in tradeoff_points],
        [point[1] for point in tradeoff_points],
        color="#8a8a8a",
        linewidth=1,
        linestyle="--",
        zorder=1,
    )
    annotations = [((7, -17), "left"), ((-8, -30), "center"), ((0, 13), "center")]
    for (error, throughput, label, color), (offset, alignment) in zip(
        tradeoff_points, annotations
    ):
        tradeoff_ax.scatter(
            error,
            throughput,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        tradeoff_ax.annotate(
            f"{label}\n{throughput:.0f} GFLOP/s",
            (error, throughput),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha=alignment,
        )

    tradeoff_ax.set_xscale("log")
    tradeoff_ax.set_title("Accuracy–throughput tradeoff (C64, K5)")
    tradeoff_ax.set_xlabel("Maximum absolute error vs. FP32 CPU oracle")
    tradeoff_ax.set_ylabel("Throughput (GFLOP/s)")
    tradeoff_ax.set_xlim(3e-7, 3e-2)
    tradeoff_ax.set_ylim(330, 455)
    tradeoff_ax.grid(alpha=0.22, linewidth=0.7)

    figure.suptitle(
        "CUDA convolution: five-trial median on RTX 3050 Laptop GPU",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", metadata={"Creator": "Nippun Sabharwal"})
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    plot(read_results(arguments.csv), arguments.output)


if __name__ == "__main__":
    main()
