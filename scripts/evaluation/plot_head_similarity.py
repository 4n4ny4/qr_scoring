import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def load_similarity(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_similarity(sim_data, output_path, title, model_name=None):
    tasks = sim_data["tasks"]
    top_k = sim_data["top_k"]
    ks = sorted((int(k) for k in top_k.keys()))

    num_plots = len(ks)
    cols = 4
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 9))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, k in enumerate(ks):
        ax = axes[idx]
        matrix = np.array(top_k[str(k)], dtype=float)
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title(f"Top-{k}", fontsize=10)
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tasks, fontsize=8)
        ax.set_aspect("equal")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = "white" if value >= 0.6 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    for idx in range(num_plots, len(axes)):
        axes[idx].axis("off")

    full_title = title
    if model_name:
        full_title = f"{title}\n{model_name}"
    fig.suptitle(full_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Plot Jaccard similarity heatmaps from head overlap data.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--title", default="Head Overlap (Jaccard Similarity) Across Tasks")
    parser.add_argument("--model_name", default=None)
    args = parser.parse_args()

    sim_data = load_similarity(args.input_json)
    title = args.title.replace("\\n", "\n")
    plot_similarity(sim_data, args.output_png, title, model_name=args.model_name)


if __name__ == "__main__":
    main()
