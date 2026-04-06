#!/usr/bin/env python3
"""Plot source x target raw-accuracy heatmaps across all knockout K values.

Input:
  - cross_task_transfer_matrix.json

Output:
  - cross_ablation_raw_accuracy_by_k.png

Usage:
  python scripts/evaluation/plot_transfer_raw_accuracy.py \
    --results_dir results/comparison_ablation \
    --output_dir results/comparison_ablation/qwen_figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _short_label(name: str) -> str:
    return name.replace("_", "\n")


def plot_raw_accuracy_panels(matrix_path: str, output_path: str):
    with open(matrix_path, encoding="utf-8") as f:
        data = json.load(f)

    sources = data["sources"]
    targets = data["targets"]
    ks = data["knockout_sizes"]

    n = len(ks)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), squeeze=False)

    im = None
    for idx, k in enumerate(ks):
        ax = axes[idx // cols][idx % cols]
        mat = np.zeros((len(sources), len(targets)))
        for r, src in enumerate(sources):
            for c, tgt in enumerate(targets):
                mat[r, c] = data["results"][src][tgt]["by_k"][str(k)]["accuracy"]

        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_title(f"K={k}", fontsize=12)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([_short_label(t) for t in targets], fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels([_short_label(s) for s in sources], fontsize=8)

        for r in range(len(sources)):
            for c in range(len(targets)):
                val = mat[r, c]
                txt_color = "white" if val < 0.3 or val > 0.8 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7, color=txt_color)

        if idx // cols == rows - 1:
            ax.set_xlabel("Target task (evaluated on)", fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel("Source task (heads knocked out)", fontsize=10)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Cross-Ablation Raw Accuracy by K", fontsize=17, y=0.98)
    # Keep a fixed right margin for a standalone colorbar axis.
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.10, top=0.90, wspace=0.25, hspace=0.35)
    if im is not None:
        cax = fig.add_axes([0.93, 0.16, 0.015, 0.68])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Raw accuracy")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot raw accuracy transfer heatmaps across K.")
    parser.add_argument(
        "--results_dir",
        default="results/comparison_ablation",
        help="Directory containing cross_task_transfer_matrix.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for figure (default: results_dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    matrix_path = os.path.join(args.results_dir, "cross_task_transfer_matrix.json")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Missing required file: {matrix_path}")

    output_path = os.path.join(output_dir, "cross_ablation_raw_accuracy_by_k.png")
    plot_raw_accuracy_panels(matrix_path, output_path)


if __name__ == "__main__":
    main()
