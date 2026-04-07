#!/usr/bin/env python3
"""Plot per-task raw accuracy or drop vs K for QRScore-SEC only.

Input:
  - QRScore-SEC_results.json

Output (mode=accuracy):
    - qrscore_sec_task_accuracy_vs_k.png
    - qrscore_sec_task_accuracy/          (one mini graph per task)
    - qrscore_sec_task_accuracy_grid.png  (8 mini graphs in one page)

Output (mode=drop):
    - qrscore_sec_task_drop_vs_k.png
    - qrscore_sec_task_drop/
    - qrscore_sec_task_drop_grid.png

Usage:
  python scripts/evaluation/plot_qrscore_sec_task_drops.py \
    --results_dir results/comparison_ablation \
    --output_dir results/comparison_ablation/qwen_figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="QRScore-SEC per-task raw accuracy/drop vs K")
    parser.add_argument(
        "--results_dir",
        default="results/comparison_ablation",
        help="Directory containing QRScore-SEC_results.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: results_dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["accuracy", "drop"],
        default="accuracy",
        help="Plot raw accuracy or drop from K=0.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    result_path = os.path.join(args.results_dir, "QRScore-SEC_results.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing required file: {result_path}")

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    per_task_curves = data.get("per_task_curves", {})
    if not per_task_curves:
        raise ValueError("per_task_curves missing in QRScore-SEC_results.json")

    ks = sorted(int(k) for k in data["accuracy_curve"].keys())

    fig, ax = plt.subplots(figsize=(10.5, 6.3))

    for task in sorted(per_task_curves.keys()):
        curve = per_task_curves[task]
        baseline = curve["0"]
        vals = [curve[str(k)] for k in ks]
        if args.mode == "drop":
            vals = [baseline - v for v in vals]
        ax.plot(
            ks,
            vals,
            marker="o",
            linewidth=2,
            label=task.replace("_", " "),
        )

    y_label = "Raw accuracy" if args.mode == "accuracy" else "Accuracy drop from K=0"
    ax.set_title(f"QRScore-SEC: Per-Task {y_label} vs K", fontsize=14)
    ax.set_xlabel("Knockout size (K)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(ks)
    if args.mode == "accuracy":
        ax.set_ylim(0.0, 1.0)
    else:
        ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper left")

    fig.tight_layout()
    prefix = "qrscore_sec_task_accuracy" if args.mode == "accuracy" else "qrscore_sec_task_drop"
    out_path = os.path.join(output_dir, f"{prefix}_vs_k.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

    # Export one mini graph per task.
    mini_dir = os.path.join(output_dir, prefix)
    os.makedirs(mini_dir, exist_ok=True)

    for task in sorted(per_task_curves.keys()):
        curve = per_task_curves[task]
        baseline = curve["0"]
        vals = [curve[str(k)] for k in ks]
        if args.mode == "drop":
            vals = [baseline - v for v in vals]

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(ks, vals, marker="o", linewidth=2, color="#1f77b4")
        ax.set_title(task.replace("_", " "), fontsize=11)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Accuracy" if args.mode == "accuracy" else "Drop", fontsize=10)
        ax.set_xticks(ks)
        if args.mode == "accuracy":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        suffix = "accuracy_vs_k" if args.mode == "accuracy" else "drop_vs_k"
        task_file = f"{task}_{suffix}.png"
        task_path = os.path.join(mini_dir, task_file)
        fig.savefig(task_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved mini graphs in: {mini_dir}")

    # Also export all 8 mini plots on one page (2x4 grid).
    tasks = sorted(per_task_curves.keys())
    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), squeeze=False)

    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        curve = per_task_curves[task]
        baseline = curve["0"]
        vals = [curve[str(k)] for k in ks]
        if args.mode == "drop":
            vals = [baseline - v for v in vals]
        ax.plot(ks, vals, marker="o", linewidth=2, color="#1f77b4")
        ax.set_title(task.replace("_", " "), fontsize=11)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Accuracy" if args.mode == "accuracy" else "Drop", fontsize=10)
        ax.set_xticks(ks)
        if args.mode == "accuracy":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes (for safety if task count changes).
    for j in range(len(tasks), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    title_label = "Raw Accuracy" if args.mode == "accuracy" else "Accuracy Drop from K=0"
    fig.suptitle(f"QRScore-SEC Per-Task {title_label} vs K", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = os.path.join(output_dir, f"{prefix}_grid.png")
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
