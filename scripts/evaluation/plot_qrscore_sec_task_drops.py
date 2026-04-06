#!/usr/bin/env python3
"""Plot per-task raw accuracy vs K for QRScore-SEC only.

Input:
  - QRScore-SEC_results.json

Output:
    - qrscore_sec_task_accuracy_vs_k.png
    - qrscore_sec_task_accuracy/          (one mini graph per task)

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
    parser = argparse.ArgumentParser(description="QRScore-SEC per-task raw accuracy vs K")
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
        accs = [curve[str(k)] for k in ks]
        ax.plot(
            ks,
            accs,
            marker="o",
            linewidth=2,
            label=task.replace("_", " "),
        )

    ax.set_title("QRScore-SEC: Per-Task Raw Accuracy vs K", fontsize=14)
    ax.set_xlabel("Knockout size (K)", fontsize=12)
    ax.set_ylabel("Raw accuracy", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "qrscore_sec_task_accuracy_vs_k.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

    # Export one mini graph per task.
    mini_dir = os.path.join(output_dir, "qrscore_sec_task_accuracy")
    os.makedirs(mini_dir, exist_ok=True)

    for task in sorted(per_task_curves.keys()):
        curve = per_task_curves[task]
        accs = [curve[str(k)] for k in ks]

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(ks, accs, marker="o", linewidth=2, color="#1f77b4")
        ax.set_title(task.replace("_", " "), fontsize=11)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xticks(ks)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        task_file = f"{task}_accuracy_vs_k.png"
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
        accs = [curve[str(k)] for k in ks]
        ax.plot(ks, accs, marker="o", linewidth=2, color="#1f77b4")
        ax.set_title(task.replace("_", " "), fontsize=11)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xticks(ks)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes (for safety if task count changes).
    for j in range(len(tasks), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("QRScore-SEC Per-Task Raw Accuracy vs K", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = os.path.join(output_dir, "qrscore_sec_task_accuracy_grid.png")
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
