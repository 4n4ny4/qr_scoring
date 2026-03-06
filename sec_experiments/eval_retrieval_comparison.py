"""
Evaluate and compare retrieval results across multiple head configs.

Reads retrieval output files from run_retrieval_comparison.sh, computes
Recall@k for each method x task combination, and produces:
  - A summary JSON with per-task and pooled recall for each method
  - A printed table for quick inspection
  - A bar-chart PNG comparing methods

Usage:
  python sec_experiments/eval_retrieval_comparison.py
  python sec_experiments/eval_retrieval_comparison.py \
    --retrieval_dir results/retrieval_comparison \
    --niah_dir data/niah_input
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

METHODS = ["full_head", "qr_sec", "qr_lme", "qr_nq"]

METHOD_LABELS = {
    "full_head": "Full Head (1024)",
    "qr_sec": "QR-SEC (top-16)",
    "qr_lme": "QR-Paper-LME (top-16)",
    "qr_nq": "QR-Paper-NQ (top-16)",
}

METHOD_COLORS = {
    "full_head": "#607D8B",
    "qr_sec": "#2196F3",
    "qr_lme": "#4CAF50",
    "qr_nq": "#8BC34A",
}

RECALL_KS = [1, 3, 5, 10]


def get_top_k_docs(doc_scores, k):
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]


def compute_recall(samples, scores, k):
    recall_count = 0
    valid = 0
    for sample in samples:
        qid = str(sample["idx"])
        gt_docs = [str(d) for d in sample.get("gt_docs", [])]
        if not gt_docs or qid not in scores:
            continue
        top_k = get_top_k_docs(scores[qid], k)
        if any(g in top_k for g in gt_docs):
            recall_count += 1
        valid += 1
    return recall_count / valid if valid else 0, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_dir", default="results/retrieval_comparison")
    parser.add_argument("--niah_dir", default="data/niah_input")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.join(args.retrieval_dir, "retrieval_comparison_summary.json")

    summary = {"methods": {}, "tasks": TASKS, "recall_ks": RECALL_KS}

    for method in METHODS:
        method_results = {"per_task": {}, "pooled": {}}

        all_samples = []
        all_scores = {}

        for task in TASKS:
            retrieval_file = os.path.join(args.retrieval_dir, f"{task}_{method}.json")
            data_file = os.path.join(args.niah_dir, f"{task}_test.json")

            if not os.path.exists(retrieval_file):
                continue
            if not os.path.exists(data_file):
                continue

            with open(retrieval_file) as f:
                scores = json.load(f)
            with open(data_file) as f:
                samples = json.load(f)

            task_recalls = {}
            for k in RECALL_KS:
                recall, n = compute_recall(samples, scores, k)
                task_recalls[f"recall@{k}"] = recall

            method_results["per_task"][task] = {
                "n_instances": len(samples),
                **task_recalls,
            }

            all_samples.extend(samples)
            all_scores.update(scores)

        if all_scores:
            for k in RECALL_KS:
                recall, n = compute_recall(all_samples, all_scores, k)
                method_results["pooled"][f"recall@{k}"] = recall
            method_results["pooled"]["n_instances"] = n

        summary["methods"][method] = method_results

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.output_file}")

    # --- Print table ---
    print_comparison_table(summary)

    # --- Plot ---
    if HAS_MPL:
        plot_retrieval_comparison(summary, os.path.dirname(args.output_file))


def print_comparison_table(summary):
    methods = summary["methods"]
    available = [m for m in METHODS if m in methods and methods[m]["pooled"]]
    if not available:
        print("No results to display.")
        return

    print(f"\n{'='*90}")
    print("RETRIEVAL COMPARISON: Recall by method (pooled across tasks)")
    print(f"{'='*90}")

    header = f"{'Method':<28s}"
    for k in RECALL_KS:
        header += f"  {'R@'+str(k):>8s}"
    header += f"  {'N':>6s}"
    print(header)
    print("-" * len(header))

    for method in available:
        pooled = methods[method]["pooled"]
        row = f"{METHOD_LABELS.get(method, method):<28s}"
        for k in RECALL_KS:
            val = pooled.get(f"recall@{k}", 0)
            row += f"  {val:>8.4f}"
        row += f"  {pooled.get('n_instances', 0):>6d}"
        print(row)

    print(f"\n{'='*90}")
    print("Per-task Recall@1")
    print(f"{'='*90}")

    header = f"{'Task':<28s}"
    for method in available:
        header += f"  {method:>12s}"
    print(header)
    print("-" * len(header))

    for task in TASKS:
        row = f"{task:<28s}"
        for method in available:
            val = methods[method]["per_task"].get(task, {}).get("recall@1", None)
            row += f"  {val:>12.4f}" if val is not None else f"  {'---':>12s}"
        print(row)

    pooled_row = f"{'POOLED':<28s}"
    for method in available:
        val = methods[method]["pooled"].get("recall@1", 0)
        pooled_row += f"  {val:>12.4f}"
    print("-" * len(header))
    print(pooled_row)


def plot_retrieval_comparison(summary, output_dir):
    methods = summary["methods"]
    available = [m for m in METHODS if m in methods and methods[m]["pooled"]]
    if len(available) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pooled recall at each K
    ax = axes[0]
    x = np.arange(len(RECALL_KS))
    width = 0.8 / len(available)
    for i, method in enumerate(available):
        pooled = methods[method]["pooled"]
        vals = [pooled.get(f"recall@{k}", 0) for k in RECALL_KS]
        offset = (i - len(available) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, None), alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Pooled Retrieval Recall (all tasks)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{k}" for k in RECALL_KS])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Per-task Recall@1
    ax = axes[1]
    tasks_with_data = [t for t in TASKS
                       if any(t in methods[m]["per_task"] for m in available)]
    x = np.arange(len(tasks_with_data))
    width = 0.8 / len(available)
    for i, method in enumerate(available):
        vals = [methods[method]["per_task"].get(t, {}).get("recall@1", 0)
                for t in tasks_with_data]
        offset = (i - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method),
               color=METHOD_COLORS.get(method, None), alpha=0.85)

    ax.set_xlabel("Task")
    ax.set_ylabel("Recall@1")
    ax.set_title("Per-Task Recall@1")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks_with_data],
                       fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "retrieval_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
