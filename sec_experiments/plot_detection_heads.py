#!/usr/bin/env python3
"""
Plot top 20 QR heads per task from results/detection/*_heads.json.
Also plot overlap of retrieval heads across tasks.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "detection"
OUTPUT_TOP20 = RESULTS_DIR / "top20_heads_plot.png"
OUTPUT_OVERLAP = RESULTS_DIR / "head_overlap_plot.png"

TASKS = [
    "registrant_name",
    "headquarters_city",
    "headquarters_state",
    "incorporation_state",
    "incorporation_year",
    "employees_count_total",
    "ceo_lastname",
    "holder_record_amount",
]


def load_top20(task: str) -> list[tuple[str, float]]:
    path = RESULTS_DIR / f"{task}_heads.json"
    with open(path) as f:
        data = json.load(f)
    pairs = [tuple(x) for x in data]
    return pairs[:20]


def load_top20_heads_only(task: str) -> set[str]:
    """Set of head IDs in top 20 for this task."""
    return {h for h, _ in load_top20(task)}


def main():
    results = {}
    for task in TASKS:
        results[task] = load_top20(task)

    # ----- Figure 1: Top 20 per task with readable x-axis -----
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        heads, scores = zip(*results[task])
        heads = list(heads)[::-1]
        scores = list(scores)[::-1]
        y_pos = np.arange(len(heads))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scores)))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(heads, fontsize=9)
        ax.set_xlabel("QRScore", fontsize=10)
        ax.set_ylabel("")
        ax.set_title(task.replace("_", " ").title(), fontsize=11)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=False))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0e}" if x != 0 else "0"))
        ax.tick_params(axis="both", labelsize=9, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.suptitle("Top 20 QR heads per task (Llama-3.1-8B-Instruct)", fontsize=14, y=1.02)
    plt.tight_layout(pad=1.4, rect=(0.03, 0.02, 0.99, 0.96))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_TOP20, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    print(f"Saved: {OUTPUT_TOP20}")

    # ----- Figure 2: Overlap across tasks -----
    # 2a) Task x Task: number of shared top-20 heads
    task_sets = {task: load_top20_heads_only(task) for task in TASKS}
    n = len(TASKS)
    overlap_matrix = np.zeros((n, n))
    for i, t1 in enumerate(TASKS):
        for j, t2 in enumerate(TASKS):
            overlap_matrix[i, j] = len(task_sets[t1] & task_sets[t2])

    # 2b) Heads that appear in multiple tasks
    head_to_count = {}
    for task in TASKS:
        for h in task_sets[task]:
            head_to_count[h] = head_to_count.get(h, 0) + 1
    multi_task_heads = [(h, c) for h, c in head_to_count.items() if c >= 2]
    multi_task_heads.sort(key=lambda x: -x[1])
    multi_task_heads = multi_task_heads[:25]  # top 25 most shared

    fig2, (ax_heat, ax_shared) = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap: task x task overlap count
    short_labels = [t.replace("_", "\n")[:14] for t in TASKS]
    im = ax_heat.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=20, aspect="equal")
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_xticklabels(short_labels, fontsize=9)
    ax_heat.set_yticklabels(short_labels, fontsize=9)
    plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax_heat.tick_params(axis="both", labelsize=9, pad=6)
    ax_heat.set_xlabel("Task", fontsize=10)
    ax_heat.set_ylabel("Task", fontsize=10)
    for i in range(n):
        for j in range(n):
            ax_heat.text(j, i, int(overlap_matrix[i, j]), ha="center", va="center", fontsize=8, color="black")
    ax_heat.set_title("Number of shared top-20 heads\nbetween each pair of tasks")
    plt.colorbar(im, ax=ax_heat, label="Shared heads", shrink=0.8)

    # Bar chart: heads that appear in multiple tasks
    if multi_task_heads:
        heads_shared, counts_shared = zip(*multi_task_heads)
        y_pos = np.arange(len(heads_shared))[::-1]
        ax_shared.barh(y_pos, counts_shared, color="steelblue", edgecolor="navy", alpha=0.85)
        ax_shared.set_yticks(y_pos)
        ax_shared.set_yticklabels(heads_shared, fontsize=10)
        ax_shared.set_xlabel("Number of tasks (top-20)", fontsize=10)
        ax_shared.set_ylabel("Head (layer-head)", fontsize=10)
        ax_shared.set_title("Heads appearing in multiple tasks")
        ax_shared.set_xlim(0, max(counts_shared) + 1)
        ax_shared.tick_params(axis="both", labelsize=9, pad=4)
    else:
        ax_shared.text(0.5, 0.5, "No head appears in top-20 of more than one task", ha="center", va="center", transform=ax_shared.transAxes)
        ax_shared.set_title("Heads appearing in multiple tasks")

    plt.suptitle("Overlap of top-20 QR heads across tasks", fontsize=13, y=1.02)
    plt.tight_layout(pad=1.6, rect=(0.02, 0.02, 0.98, 0.96))
    plt.savefig(OUTPUT_OVERLAP, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Saved: {OUTPUT_OVERLAP}")


if __name__ == "__main__":
    main()
