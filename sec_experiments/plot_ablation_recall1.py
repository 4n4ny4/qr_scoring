#!/usr/bin/env python3
"""Plot Recall@1 from ablation_summary.json to show qr_head_top16 > full > random."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_DIR / "results" / "ablation" / "ablation_summary.json"
OUT_PATH = PROJECT_DIR / "results" / "ablation" / "recall1_plot.png"


def main():
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    pooled = data["pooled"]
    conditions = ["full_head", "qr_head_top16", "random_head"]
    labels = ["Full (1024 heads)", "QR top-16", "Random 16 heads"]

    vals = [pooled[c]["recall@1"]["value"] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]  # blue, green, red
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Recall@1", fontsize=12)
    ax.set_title("Recall@1 by condition (pooled, n=244)\nQR top-16 > Full > Random", fontsize=13)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.spines["top"].set_visible(False)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
