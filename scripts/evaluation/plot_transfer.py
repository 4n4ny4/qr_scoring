#!/usr/bin/env python3
"""Visualize cross-task transfer ablation results.

Outputs:
  - transfer_drop_heatmap_K{k}.png   (source×target drop heatmap, one per K>0)
    - specificity_raw_accuracy_heatmaps_qrscore_sec.png
        (source×target raw-accuracy heatmap panels, one panel per K>0)
    - specificity_drop_from_k0_heatmaps_qrscore_sec.png
        (source×target drop-from-K0 heatmap panels, one panel per K>0)
  - head_similarity_heatmaps.png     (Jaccard overlap panels at each top-K)
  - specificity_bars.png             (on-target vs off-target drop + specificity index)
  - specificity_table.csv            (same data in tabular form)
    - curve_{source}__to__{target}.png (raw-accuracy curve per source→target pair)
    - pair_curve_summary.csv           (quick stats for all source→target curves)

Usage:
  python scripts/evaluation/plot_transfer.py \
    --results_dir results/comparison_ablation

    python scripts/evaluation/plot_transfer.py \
        --results_dir results/comparison_ablation \
        --generate_pair_curves \
        --pair_curves_output_dir cross_ablation_curves
"""

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_name(text):
    """Convert labels into filesystem-safe file-name segments."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", text).strip("_")


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


def _plot_transfer_metric_panels(
    data,
    output_path,
    metric_key,
    title,
    cmap,
    include_k0=False,
):
    """Plot one sourcextarget heatmap panel per K for a metric.

    By default K=0 is omitted to match historical behavior for ablation-focused
    panels. Set include_k0=True to include a baseline panel.
    """
    sources = data["sources"]
    targets = data["targets"]
    if include_k0:
        ks = list(data["knockout_sizes"])
    else:
        ks = [k for k in data["knockout_sizes"] if k != 0]

    n = len(ks)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    short_t = [t.replace("_", "\n") for t in targets]
    short_s = [s.replace("_", "\n") for s in sources]
    im = None

    for idx, k in enumerate(ks):
        ax = axes[idx // cols][idx % cols]
        matrix = np.zeros((len(sources), len(targets)))
        for r, src in enumerate(sources):
            for c, tgt in enumerate(targets):
                matrix[r, c] = data["results"][src][tgt]["by_k"][str(k)][metric_key]

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(short_t, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(short_s, fontsize=7)
        ax.set_title(f"K={k}", fontsize=11)

        for r in range(len(sources)):
            for c in range(len(targets)):
                val = matrix[r, c]
                color = "white" if val > 0.6 or (metric_key == "accuracy" and val < 0.4) else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Target task (evaluated on)", fontsize=9)
    for row_axes in axes:
        first_ax = row_axes[0]
        if first_ax.get_visible():
            first_ax.set_ylabel("Source task (heads knocked out)", fontsize=9)

    if im is not None:
        cbar_label = "Raw accuracy" if metric_key == "accuracy" else "Drop from K=0"
        # Reserve a dedicated colorbar axis so it never overlays any subplot.
        cax = fig.add_axes([0.935, 0.18, 0.015, 0.64])
        fig.colorbar(im, cax=cax, label=cbar_label)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.90, wspace=0.35, hspace=0.38)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_transfer_specificity_heatmaps(data, output_dir, model_name=None):
    """Generate multi-panel cross-ablation heatmaps for specificity interpretation."""
    raw_out = os.path.join(
        output_dir,
        "specificity_raw_accuracy_heatmaps_qrscore_sec.png",
    )
    drop_out = os.path.join(
        output_dir,
        "specificity_drop_from_k0_heatmaps_qrscore_sec.png",
    )

    title_suffix = f" ({model_name})" if model_name else ""

    _plot_transfer_metric_panels(
        data,
        output_path=raw_out,
        metric_key="accuracy",
        title=f"Cross-Ablation Raw Accuracy by K{title_suffix}",
        cmap="RdYlGn",
        include_k0=True,
    )
    _plot_transfer_metric_panels(
        data,
        output_path=drop_out,
        metric_key="drop_from_k0",
        title=f"Cross-Ablation Drop from Baseline by K{title_suffix}",
        cmap="YlOrRd",
        include_k0=False,
    )


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
    on_target = [
        spec_data["sources"][s]["on_target_drop"]
        if spec_data["sources"][s]["on_target_drop"] is not None else np.nan
        for s in sources
    ]
    off_target = [
        spec_data["sources"][s]["off_target_mean_drop"]
        if spec_data["sources"][s]["off_target_mean_drop"] is not None else np.nan
        for s in sources
    ]
    specificity = [
        spec_data["sources"][s]["specificity_index"]
        if spec_data["sources"][s]["specificity_index"] is not None else np.nan
        for s in sources
    ]

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

    colors = [
        "#9E9E9E" if np.isnan(v) else ("#43A047" if v > 0 else "#E53935")
        for v in specificity
    ]
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
            writer.writerow([
                s,
                "N/A" if d["on_target_drop"] is None else f"{d['on_target_drop']:.4f}",
                "N/A" if d["off_target_mean_drop"] is None else f"{d['off_target_mean_drop']:.4f}",
                "N/A" if d["specificity_index"] is None else f"{d['specificity_index']:.4f}",
                "N/A" if d["surgicality_ratio"] is None else f"{d['surgicality_ratio']:.4f}",
            ])
    print(f"Saved: {csv_path}")


def plot_pair_accuracy_curves(transfer_data, output_dir):
    """Create one raw-accuracy curve per source→target pair."""
    os.makedirs(output_dir, exist_ok=True)

    ks = transfer_data["knockout_sizes"]
    sources = transfer_data["sources"]
    targets = transfer_data["targets"]
    results = transfer_data["results"]

    summary_rows = []
    count = 0

    for source in sources:
        for target in targets:
            by_k = results[source][target]["by_k"]
            y_acc = [by_k[str(k)]["accuracy"] for k in ks]

            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.plot(ks, y_acc, marker="o", linewidth=2, color="#1f77b4")
            ax.set_xlabel("Knockout size (K)", fontsize=10)
            ax.set_ylabel("Raw accuracy", fontsize=10)
            ax.set_title(
                f"Cross Ablation Curve: {source} -> {target}",
                fontsize=11,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(min(ks), max(ks))
            ax.grid(alpha=0.3)
            ax.set_xticks(ks)

            safe_source = _safe_name(source)
            safe_target = _safe_name(target)
            out_file = f"curve_{safe_source}__to__{safe_target}.png"
            out_path = os.path.join(output_dir, out_file)

            fig.tight_layout()
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            k0_acc = y_acc[0]
            last_acc = y_acc[-1]
            max_drop = max(k0_acc - val for val in y_acc)
            summary_rows.append([
                source,
                target,
                f"{k0_acc:.4f}",
                f"{last_acc:.4f}",
                f"{max_drop:.4f}",
                out_file,
            ])
            count += 1

    summary_path = os.path.join(output_dir, "pair_curve_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_task",
            "target_task",
            "accuracy_k0",
            "accuracy_k_last",
            "max_drop_from_k0",
            "plot_file",
        ])
        writer.writerows(summary_rows)

    print(f"Saved {count} pair curves to: {output_dir}")
    print(f"Saved: {summary_path}")


# ── helper functions ───────────────────────────────────────────────────────

def _get_random_baseline_results(results_dir, tasks, ks):
    """Load results from multiple random baseline runs, keeping them separate."""
    random_files = [
        f for f in os.listdir(results_dir)
        if f.startswith("Random-seed") and f.endswith("_results.json")
    ]
    if not random_files:
        return None

    all_random_results = {}

    for fname in random_files:
        with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
            method_name = data.get("method", os.path.basename(fname).replace("_results.json", ""))
            
            # {task: {k: {accuracy, drop_from_k0}}}
            task_accuracies = {t: {"by_k": {str(k): {} for k in ks}} for t in tasks}

            if "per_task_curves" in data:
                for task_name, k_accuracies_map in data["per_task_curves"].items():
                    if task_name in tasks:
                        for k_str, acc in k_accuracies_map.items():
                            if k_str in task_accuracies[task_name]["by_k"]:
                                task_accuracies[task_name]["by_k"][k_str]["accuracy"] = acc
            
            # Calculate drop_from_k0
            for task in tasks:
                if "0" in task_accuracies[task]["by_k"]:
                    k0_acc = task_accuracies[task]["by_k"]["0"].get("accuracy", 0)
                    for k_str, k_data in task_accuracies[task]["by_k"].items():
                        k_data["drop_from_k0"] = k0_acc - k_data.get("accuracy", 0)

            all_random_results[method_name] = task_accuracies

    return all_random_results


def add_random_baseline_to_transfer_data(transfer_data, all_random_results):
    """Inject multiple random baseline results into the transfer data structure."""
    for method_name, random_results in all_random_results.items():
        if method_name not in transfer_data["sources"]:
            transfer_data["sources"].append(method_name)
        
        transfer_data["results"][method_name] = {}
        for target_task in transfer_data["targets"]:
            if target_task in random_results:
                transfer_data["results"][method_name][target_task] = random_results[target_task]
            else:
                # Fill with empty data if a task is missing
                transfer_data["results"][method_name][target_task] = {"by_k": {
                    str(k): {"accuracy": 0, "drop_from_k0": 0} for k in transfer_data["knockout_sizes"]
                }}
    return transfer_data


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize cross-task transfer results.")
    parser.add_argument("--results_dir", required=True,
                        help="Directory with cross_task_*.json files.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots (default: same as results_dir).")
    parser.add_argument(
        "--generate_pair_curves",
        action="store_true",
        help="Generate one raw-accuracy curve per source->target pair.",
    )
    parser.add_argument(
        "--pair_curves_output_dir",
        default=None,
        help=(
            "Output directory for pair curves. "
            "Default: output_dir/cross_ablation_curves"
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    transfer_path = os.path.join(args.results_dir, "cross_task_transfer_matrix.json")
    sim_path = os.path.join(args.results_dir, "cross_task_head_similarity_topk.json")
    spec_path = os.path.join(args.results_dir, "cross_task_specificity_metrics.json")

    count = 0
    if os.path.exists(transfer_path):
        with open(transfer_path, encoding="utf-8") as f:
            transfer_data = json.load(f)
            model_name = transfer_data.get("model_slug") or transfer_data.get("model_name")

            # Load, average, and inject random baseline data
            random_results = _get_random_baseline_results(
                args.results_dir,
                transfer_data["targets"],
                transfer_data["knockout_sizes"]
            )
            if random_results:
                print("Found and processed random baseline results.")
                transfer_data = add_random_baseline_to_transfer_data(transfer_data, random_results)
            else:
                print("No random baseline results found in results directory.")

            plot_transfer_heatmaps(transfer_data, output_dir)
            plot_transfer_specificity_heatmaps(transfer_data, output_dir, model_name=model_name)
            if args.generate_pair_curves:
                pair_curves_dir = args.pair_curves_output_dir or os.path.join(
                    output_dir, "cross_ablation_curves"
                )
                plot_pair_accuracy_curves(transfer_data, pair_curves_dir)
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
