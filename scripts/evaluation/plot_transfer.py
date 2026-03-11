#!/usr/bin/env python3
"""Visualize cross-task transfer ablation results.

Outputs:
  - transfer_drop_heatmap_K{k}.png   (source×target drop heatmap, one per K>0)
  - head_similarity_heatmaps.png     (Jaccard overlap panels at each top-K)
  - specificity_bars.png             (on-target vs off-target drop + specificity index)
  - specificity_table.csv            (same data in tabular form)

Usage:
  python scripts/evaluation/plot_transfer.py \
    --results_dir results/comparison_ablation
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── 1. Transfer drop heatmap (one per K) ──────────────────────────────────

def plot_transfer_heatmaps(data, output_dir):
    sources = data["sources"]
    targets = data["targets"]
    ks = data["knockout_sizes"]

    for k in ks:
        if k == 0:
            continue
        matrix = np.zeros((len(sources), len(targets)))
        for r, src in enumerate(sources):
            for c, tgt in enumerate(targets):
                matrix[r, c] = data["results"][src][tgt]["by_k"][str(k)]["drop_from_k0"]

        short_t = [t.replace("_", "\n") for t in targets]
        short_s = [s.replace("_", "\n") for s in sources]

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(short_t, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(short_s, fontsize=8)
        ax.set_xlabel("Target task (evaluated on)", fontsize=11)
        ax.set_ylabel("Source task (heads knocked out)", fontsize=11)
        ax.set_title(f"Cross-Task Transfer: Accuracy Drop at K={k}", fontsize=13)
        for r in range(len(sources)):
            for c in range(len(targets)):
                val = matrix[r, c]
                color = "white" if val > 0.6 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Drop from K=0")
        fig.tight_layout()
        out = os.path.join(output_dir, f"transfer_drop_heatmap_K{k}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ── 2. Head similarity (Jaccard) panels ───────────────────────────────────

def plot_head_similarity(sim_data, output_dir):
    tasks = sim_data["tasks"]
    top_ks = sorted(sim_data["top_k"].keys(), key=int)
    short = [t.replace("_", "\n") for t in tasks]

    n = len(top_ks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)

    for idx, tk in enumerate(top_ks):
        ax = axes[idx // cols][idx % cols]
        matrix = np.array(sim_data["top_k"][tk])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(f"Top-{tk}", fontsize=11)
        for r in range(len(tasks)):
            for c in range(len(tasks)):
                val = matrix[r, c]
                color = "white" if val > 0.5 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Head Overlap (Jaccard Similarity) Across Tasks", fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, "head_similarity_heatmaps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Specificity bar chart + CSV ────────────────────────────────────────

def plot_specificity(spec_data, output_dir):
    summary_k = spec_data["summary_k"]
    sources = sorted(spec_data["sources"].keys())
    on_target = [spec_data["sources"][s]["on_target_drop"] for s in sources]
    off_target = [spec_data["sources"][s]["off_target_mean_drop"] for s in sources]
    specificity = [spec_data["sources"][s]["specificity_index"] for s in sources]

    short = [s.replace("_", "\n") for s in sources]
    x = np.arange(len(sources))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(x - w / 2, on_target, w, label="On-target drop", color="#E53935")
    ax1.bar(x + w / 2, off_target, w, label="Off-target mean drop", color="#FFA726")
    ax1.set_ylabel("Accuracy drop", fontsize=11)
    ax1.set_title(f"Task Specificity at K={summary_k}", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.5)

    colors = ["#43A047" if v > 0 else "#E53935" for v in specificity]
    ax2.bar(x, specificity, w * 2, color=colors)
    ax2.set_ylabel("Specificity index", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=8)
    ax2.set_xlabel("Source task (heads knocked out)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    out = os.path.join(output_dir, "specificity_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # CSV table
    csv_path = os.path.join(output_dir, "specificity_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source Task", "On-Target Drop", "Off-Target Mean Drop",
                         "Specificity Index", "Surgicality Ratio"])
        for s in sources:
            d = spec_data["sources"][s]
            writer.writerow([s, f"{d['on_target_drop']:.4f}",
                             f"{d['off_target_mean_drop']:.4f}",
                             f"{d['specificity_index']:.4f}",
                             f"{d['surgicality_ratio']:.4f}"])
    print(f"Saved: {csv_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize cross-task transfer results.")
    parser.add_argument("--results_dir", required=True,
                        help="Directory with cross_task_*.json files.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots (default: same as results_dir).")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    transfer_path = os.path.join(args.results_dir, "cross_task_transfer_matrix.json")
    sim_path = os.path.join(args.results_dir, "cross_task_head_similarity_topk.json")
    spec_path = os.path.join(args.results_dir, "cross_task_specificity_metrics.json")

    count = 0
    if os.path.exists(transfer_path):
        with open(transfer_path, encoding="utf-8") as f:
            plot_transfer_heatmaps(json.load(f), output_dir)
        count += 1
    if os.path.exists(sim_path):
        with open(sim_path, encoding="utf-8") as f:
            plot_head_similarity(json.load(f), output_dir)
        count += 1
    if os.path.exists(spec_path):
        with open(spec_path, encoding="utf-8") as f:
            plot_specificity(json.load(f), output_dir)
        count += 1

    if count == 0:
        print(f"No cross_task_*.json files found in {args.results_dir}")
    else:
        print(f"\nDone. All transfer plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
