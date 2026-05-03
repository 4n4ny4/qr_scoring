#!/usr/bin/env python3
"""Single-panel heatmap: Llama + NQ-detected head ablation, raw accuracy by task × K.

Reads per-instance logs (default: results/comparison_ablation/QRScore-8B-NQ-TRAIN_token_log.jsonl)
and aggregates mean accuracy (y = target task, x = K heads ablated).

Usage:
  python scripts/evaluation/plot_nq_ablation_heatmap.py \\
    --output results/comparison_ablation/llama_nq_heads_ablation_accuracy.png
"""

import argparse
import json
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Order matches common paper / figure conventions (not alphabetical).
TASK_ORDER = [
    "ceo_lastname",
    "employees_count_total",
    "headquarters_city",
    "headquarters_state",
    "holder_record_amount",
    "incorporation_state",
    "incorporation_year",
    "registrant_name",
]

TASK_DISPLAY = {
    "ceo_lastname": "CEO last name",
    "employees_count_total": "Employee count",
    "headquarters_city": "HQ city",
    "headquarters_state": "HQ state",
    "holder_record_amount": "Holder record amount",
    "incorporation_state": "Incorp. state",
    "incorporation_year": "Incorp. year",
    "registrant_name": "Registrant name",
}

MODEL_LABEL = "meta-llama/Llama-3.1-8B-Instruct"
METHOD_TAG = "QRScore-8B-NQ-TRAIN"


def aggregate_token_log(path, method=METHOD_TAG):
    """Return (ks_sorted, per-task k->accuracy) from jsonl."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("method") != method:
                continue
            k = int(r["K"])
            t = r["task"]
            raw[t][k].append(int(r["correct"]))

    if not raw:
        raise ValueError(f"No rows for method={method} in {path}")

    all_ks = set()
    for by_k in raw.values():
        all_ks.update(by_k.keys())
    ks = sorted(all_ks)

    out = {}
    for t in TASK_ORDER:
        out[t] = {}
        for k in ks:
            arr = raw[t].get(k, [])
            if not arr:
                out[t][k] = float("nan")
            else:
                out[t][k] = sum(arr) / len(arr)
    return ks, out


def main():
    parser = argparse.ArgumentParser(description="Plot NQ head ablation heatmap for Llama.")
    parser.add_argument(
        "--token_log",
        default=os.path.join(
            PROJECT_DIR,
            "results",
            "comparison_ablation",
            "QRScore-8B-NQ-TRAIN_token_log.jsonl",
        ),
        help="jsonl from run_ablation for QRScore-8B-NQ-TRAIN",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            PROJECT_DIR,
            "results",
            "comparison_ablation",
            "llama_nq_heads_ablation_accuracy.png",
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override figure title (default: Llama + NQ heads + model id)",
    )
    args = parser.parse_args()

    ks, acc_by_task = aggregate_token_log(args.token_log)

    matrix = np.array([[acc_by_task[t][k] for k in ks] for t in TASK_ORDER])
    y_labels = [TASK_DISPLAY[t] for t in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("K heads ablated", fontsize=12)
    ax.set_ylabel("Target task", fontsize=12)

    title = args.title or f"Llama raw accuracy (NQ heads)\n{MODEL_LABEL}"
    ax.set_title(title, fontsize=13)

    for r in range(len(TASK_ORDER)):
        for c in range(len(ks)):
            val = matrix[r, c]
            color = "white" if val < 0.45 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, shrink=0.82, label="Raw accuracy")
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
