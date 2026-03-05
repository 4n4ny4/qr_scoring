"""
Compare QRHead detection results across tasks.

Produces:
- Per-task top-K head lists
- Pairwise Jaccard overlap matrix
- Spearman rank correlation matrix
- Union importance analysis (which heads appear consistently)
- Heatmap visualizations saved to results/detection/
"""

import json
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np

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

TOP_K = 20


def load_head_scores(result_dir):
    """Load per-task head scores from JSON files produced by detect_qrhead_lme.py."""
    task_heads = {}
    for task in TASKS:
        path = os.path.join(result_dir, f"{task}_heads.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {task}")
            continue
        with open(path) as f:
            head_scores = json.load(f)
        task_heads[task] = head_scores
    return task_heads


def head_list_to_rank_dict(head_scores):
    """Convert [(head_id, score), ...] sorted list to {head_id: rank} dict."""
    return {head_id: rank for rank, (head_id, _) in enumerate(head_scores)}


def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def spearman_rank_corr(rank_a, rank_b, all_heads):
    """Spearman rank correlation between two task rankings over the full head set."""
    ranks_a = np.array([rank_a.get(h, len(rank_a)) for h in all_heads])
    ranks_b = np.array([rank_b.get(h, len(rank_b)) for h in all_heads])
    n = len(all_heads)
    d = ranks_a - ranks_b
    d_sq_sum = np.sum(d ** 2)
    rho = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return rho


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    result_dir = os.path.join(project_dir, "results", "detection")

    print(f"Loading results from {result_dir}\n")
    task_heads = load_head_scores(result_dir)

    if len(task_heads) < 2:
        print("Need at least 2 task results to compare. Exiting.")
        sys.exit(1)

    active_tasks = list(task_heads.keys())
    n_tasks = len(active_tasks)

    all_heads_set = set()
    for scores in task_heads.values():
        for head_id, _ in scores:
            all_heads_set.add(head_id)
    all_heads = sorted(all_heads_set)
    print(f"Total unique heads: {len(all_heads)}")

    print(f"\n{'='*60}")
    print(f"TOP-{TOP_K} HEADS PER TASK")
    print(f"{'='*60}")
    task_top_sets = {}
    for task in active_tasks:
        top_heads = task_heads[task][:TOP_K]
        task_top_sets[task] = set(h for h, _ in top_heads)
        print(f"\n{task}:")
        for i, (head_id, score) in enumerate(top_heads):
            print(f"  {i+1:3d}. {head_id:8s}  QRScore={score:.6f}")

    print(f"\n{'='*60}")
    print(f"PAIRWISE JACCARD OVERLAP (top-{TOP_K})")
    print(f"{'='*60}")
    max_task_len = max(len(t) for t in active_tasks)
    header = " " * (max_task_len + 2) + "  ".join(f"{t[:8]:>8s}" for t in active_tasks)
    print(header)
    jaccard_matrix = np.zeros((n_tasks, n_tasks))
    for i, t1 in enumerate(active_tasks):
        row = f"{t1:{max_task_len}s}  "
        for j, t2 in enumerate(active_tasks):
            j_val = jaccard(task_top_sets[t1], task_top_sets[t2])
            jaccard_matrix[i, j] = j_val
            row += f"{j_val:8.3f}  "
        print(row)

    print(f"\n{'='*60}")
    print(f"SPEARMAN RANK CORRELATION (full ranking)")
    print(f"{'='*60}")
    rank_dicts = {task: head_list_to_rank_dict(task_heads[task]) for task in active_tasks}
    spearman_matrix = np.zeros((n_tasks, n_tasks))
    print(header)
    for i, t1 in enumerate(active_tasks):
        row = f"{t1:{max_task_len}s}  "
        for j, t2 in enumerate(active_tasks):
            rho = spearman_rank_corr(rank_dicts[t1], rank_dicts[t2], all_heads)
            spearman_matrix[i, j] = rho
            row += f"{rho:8.3f}  "
        print(row)

    print(f"\n{'='*60}")
    print(f"HEADS APPEARING IN TOP-{TOP_K} ACROSS MULTIPLE TASKS")
    print(f"{'='*60}")
    head_task_count = Counter()
    head_task_map = {}
    for task in active_tasks:
        for head_id in task_top_sets[task]:
            head_task_count[head_id] += 1
            if head_id not in head_task_map:
                head_task_map[head_id] = []
            head_task_map[head_id].append(task)

    sorted_heads = sorted(head_task_count.items(), key=lambda x: (-x[1], x[0]))
    print(f"\n{'Head':>8s}  {'Count':>5s}  Tasks")
    print("-" * 70)
    for head_id, count in sorted_heads:
        if count >= 2:
            tasks_str = ", ".join(head_task_map[head_id])
            print(f"{head_id:>8s}  {count:5d}  {tasks_str}")

    universal_heads = [h for h, c in sorted_heads if c == n_tasks]
    if universal_heads:
        print(f"\nHeads in top-{TOP_K} for ALL {n_tasks} tasks: {universal_heads}")
    else:
        max_count = sorted_heads[0][1] if sorted_heads else 0
        near_universal = [h for h, c in sorted_heads if c == max_count]
        print(f"\nNo head appears in top-{TOP_K} for all tasks.")
        print(f"Most frequent ({max_count}/{n_tasks} tasks): {near_universal}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        short_labels = [t.replace("_", "\n") for t in active_tasks]

        im0 = axes[0].imshow(jaccard_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        axes[0].set_xticks(range(n_tasks))
        axes[0].set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        axes[0].set_yticks(range(n_tasks))
        axes[0].set_yticklabels(short_labels, fontsize=8)
        axes[0].set_title(f"Jaccard Overlap (top-{TOP_K})")
        for i in range(n_tasks):
            for j in range(n_tasks):
                axes[0].text(j, i, f"{jaccard_matrix[i,j]:.2f}",
                           ha="center", va="center", fontsize=7)
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(spearman_matrix, cmap="RdBu", vmin=-1, vmax=1)
        axes[1].set_xticks(range(n_tasks))
        axes[1].set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_yticks(range(n_tasks))
        axes[1].set_yticklabels(short_labels, fontsize=8)
        axes[1].set_title("Spearman Rank Correlation")
        for i in range(n_tasks):
            for j in range(n_tasks):
                axes[1].text(j, i, f"{spearman_matrix[i,j]:.2f}",
                           ha="center", va="center", fontsize=7)
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        plt.tight_layout()
        heatmap_path = os.path.join(result_dir, "head_comparison_heatmap.png")
        plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        print(f"\nHeatmap saved to {heatmap_path}")

    except ImportError:
        print("\nmatplotlib not available; skipping heatmap generation.")

    summary = {
        "top_k": TOP_K,
        "tasks": active_tasks,
        "jaccard_matrix": jaccard_matrix.tolist(),
        "spearman_matrix": spearman_matrix.tolist(),
        "universal_heads_top_k": [h for h, c in sorted_heads if c >= n_tasks],
        "head_frequency": {h: c for h, c in sorted_heads if c >= 2},
        "per_task_top_k": {
            task: task_heads[task][:TOP_K] for task in active_tasks
        },
    }
    summary_path = os.path.join(result_dir, "cross_task_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
