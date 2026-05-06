import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "qr_scoring_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib.pyplot as plt
import numpy as np


TASKS = [
    "ceo_lastname",
    "employees_count_total",
    "headquarters_city",
    "headquarters_state",
    "holder_record_amount",
    "incorporation_state",
    "incorporation_year",
    "registrant_name",
]

K_VALUES = [8, 16, 32, 48, 64, 96, 128]


# Mistral values are rounded to two decimals because they were transcribed from
# the supplied heatmap rather than loaded from a local JSON artifact.
MISTRAL_MATRICES = {
    8: [
        [1.00, 0.45, 0.60, 0.60, 0.45, 0.45, 0.45, 0.33],
        [0.45, 1.00, 0.60, 0.60, 0.33, 0.45, 0.60, 0.33],
        [0.60, 0.60, 1.00, 0.78, 0.45, 0.78, 0.78, 0.45],
        [0.60, 0.60, 0.78, 1.00, 0.45, 0.60, 0.60, 0.60],
        [0.45, 0.33, 0.45, 0.45, 1.00, 0.33, 0.33, 0.23],
        [0.45, 0.45, 0.78, 0.60, 0.33, 1.00, 0.78, 0.60],
        [0.45, 0.60, 0.78, 0.60, 0.33, 0.78, 1.00, 0.45],
        [0.33, 0.33, 0.45, 0.60, 0.23, 0.60, 0.45, 1.00],
    ],
    16: [
        [1.00, 0.28, 0.39, 0.39, 0.39, 0.52, 0.45, 0.45],
        [0.28, 1.00, 0.45, 0.52, 0.33, 0.52, 0.45, 0.39],
        [0.39, 0.45, 1.00, 0.88, 0.45, 0.78, 0.78, 0.68],
        [0.39, 0.52, 0.88, 1.00, 0.39, 0.78, 0.78, 0.68],
        [0.39, 0.33, 0.45, 0.39, 1.00, 0.39, 0.39, 0.45],
        [0.52, 0.52, 0.78, 0.78, 0.39, 1.00, 0.88, 0.60],
        [0.45, 0.45, 0.78, 0.78, 0.39, 0.88, 1.00, 0.60],
        [0.45, 0.39, 0.68, 0.68, 0.45, 0.60, 0.60, 1.00],
    ],
    32: [
        [1.00, 0.60, 0.49, 0.49, 0.42, 0.49, 0.45, 0.42],
        [0.60, 1.00, 0.52, 0.52, 0.52, 0.64, 0.64, 0.39],
        [0.49, 0.52, 1.00, 1.00, 0.52, 0.68, 0.64, 0.64],
        [0.49, 0.52, 1.00, 1.00, 0.52, 0.68, 0.64, 0.64],
        [0.42, 0.52, 0.52, 0.52, 1.00, 0.52, 0.49, 0.42],
        [0.49, 0.64, 0.68, 0.68, 0.52, 1.00, 0.78, 0.52],
        [0.45, 0.64, 0.64, 0.64, 0.49, 0.78, 1.00, 0.49],
        [0.42, 0.39, 0.64, 0.64, 0.42, 0.52, 0.49, 1.00],
    ],
    48: [
        [1.00, 0.57, 0.48, 0.50, 0.55, 0.45, 0.45, 0.55],
        [0.57, 1.00, 0.50, 0.55, 0.52, 0.57, 0.57, 0.39],
        [0.48, 0.50, 1.00, 0.92, 0.60, 0.66, 0.63, 0.55],
        [0.50, 0.55, 0.92, 1.00, 0.63, 0.68, 0.66, 0.52],
        [0.55, 0.52, 0.60, 0.63, 1.00, 0.66, 0.63, 0.43],
        [0.45, 0.57, 0.66, 0.68, 0.66, 1.00, 0.96, 0.45],
        [0.45, 0.57, 0.63, 0.66, 0.63, 0.96, 1.00, 0.43],
        [0.55, 0.39, 0.55, 0.52, 0.43, 0.45, 0.43, 1.00],
    ],
    64: [
        [1.00, 0.62, 0.52, 0.51, 0.62, 0.52, 0.56, 0.54],
        [0.62, 1.00, 0.52, 0.52, 0.56, 0.56, 0.58, 0.47],
        [0.52, 0.52, 1.00, 0.88, 0.51, 0.68, 0.68, 0.58],
        [0.51, 0.52, 0.88, 1.00, 0.49, 0.68, 0.68, 0.54],
        [0.62, 0.56, 0.51, 0.49, 1.00, 0.54, 0.58, 0.47],
        [0.52, 0.56, 0.68, 0.68, 0.54, 1.00, 0.91, 0.51],
        [0.56, 0.58, 0.68, 0.68, 0.58, 0.91, 1.00, 0.54],
        [0.54, 0.47, 0.58, 0.54, 0.47, 0.51, 0.54, 1.00],
    ],
    96: [
        [1.00, 0.61, 0.60, 0.63, 0.59, 0.59, 0.59, 0.56],
        [0.61, 1.00, 0.70, 0.70, 0.61, 0.66, 0.64, 0.59],
        [0.60, 0.70, 1.00, 0.94, 0.59, 0.71, 0.68, 0.67],
        [0.63, 0.70, 0.94, 1.00, 0.59, 0.71, 0.68, 0.67],
        [0.59, 0.61, 0.59, 0.59, 1.00, 0.60, 0.57, 0.49],
        [0.59, 0.66, 0.71, 0.71, 0.60, 1.00, 0.88, 0.61],
        [0.59, 0.64, 0.68, 0.68, 0.57, 0.88, 1.00, 0.57],
        [0.56, 0.59, 0.67, 0.67, 0.49, 0.61, 0.57, 1.00],
    ],
    128: [
        [1.00, 0.64, 0.66, 0.65, 0.63, 0.62, 0.61, 0.61],
        [0.64, 1.00, 0.66, 0.66, 0.66, 0.63, 0.62, 0.60],
        [0.66, 0.66, 1.00, 0.95, 0.62, 0.75, 0.73, 0.75],
        [0.65, 0.66, 0.95, 1.00, 0.61, 0.77, 0.72, 0.74],
        [0.63, 0.66, 0.62, 0.61, 1.00, 0.63, 0.61, 0.59],
        [0.62, 0.63, 0.75, 0.77, 0.63, 1.00, 0.88, 0.72],
        [0.61, 0.62, 0.73, 0.72, 0.61, 0.88, 1.00, 0.67],
        [0.61, 0.60, 0.75, 0.74, 0.59, 0.72, 0.67, 1.00],
    ],
}


def parse_head(row):
    head = row[0] if isinstance(row, list) else row.get("head")
    layer, head_idx = str(head).split("-", 1)
    return int(layer), int(head_idx)


def off_diagonal_values(matrix):
    matrix = np.asarray(matrix, dtype=float)
    values = []
    for row in range(matrix.shape[0]):
        for col in range(row + 1, matrix.shape[1]):
            values.append(float(matrix[row, col]))
    return np.asarray(values, dtype=float)


def reorder_matrix(matrix, source_tasks):
    source_index = {task: idx for idx, task in enumerate(source_tasks)}
    indices = [source_index[task] for task in TASKS]
    matrix = np.asarray(matrix, dtype=float)
    return matrix[np.ix_(indices, indices)]


def load_similarity_json(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        int(k): reorder_matrix(matrix, payload["tasks"])
        for k, matrix in payload["top_k"].items()
        if int(k) in K_VALUES
    }


def load_llama_from_rankings(project_dir):
    ranking_dir = project_dir / "results" / "detection" / "meta-llama__Llama-3.1-8B-Instruct"
    matrices = {}
    for k in K_VALUES:
        task_heads = {}
        for task in TASKS:
            topk_path = ranking_dir / "topk" / f"long_context_{task}_top{k}.json"
            full_path = ranking_dir / f"long_context_{task}_heads.json"
            path = topk_path if topk_path.exists() else full_path
            if not path.exists():
                raise FileNotFoundError(f"Missing Llama ranking for {task}: {topk_path} or {full_path}")
            with path.open("r", encoding="utf-8") as f:
                task_heads[task] = [parse_head(row) for row in json.load(f)[:k]]

        matrix = np.eye(len(TASKS), dtype=float)
        for src_idx, src in enumerate(TASKS):
            src_set = set(task_heads[src])
            for tgt_idx, tgt in enumerate(TASKS):
                if src_idx == tgt_idx:
                    continue
                tgt_set = set(task_heads[tgt])
                union = src_set | tgt_set
                matrix[src_idx, tgt_idx] = len(src_set & tgt_set) / len(union) if union else 0.0
        matrices[k] = matrix
    return matrices


def build_model_matrices(project_dir):
    qwen_path = (
        project_dir
        / "results"
        / "comparison_ablation"
        / "Qwen__Qwen2.5-7B-Instruct"
        / "cross_task_head_similarity_topk.json"
    )
    olmo_path = (
        project_dir
        / "results"
        / "comparison_ablation"
        / "allenai__OLMo-7B-Instruct-hf"
        / "cross_task_head_similarity_topk.json"
    )

    return {
        "Mistral": {
            "data_source": "transcribed_from_supplied_heatmap_rounded_2dp",
            "matrices": {k: np.asarray(v, dtype=float) for k, v in MISTRAL_MATRICES.items()},
        },
        "Llama-3.1-8B": {
            "data_source": "derived_from_local_topk_rankings",
            "matrices": load_llama_from_rankings(project_dir),
        },
        "Qwen2.5-7B": {
            "data_source": str(qwen_path),
            "matrices": load_similarity_json(qwen_path),
        },
        "OLMo-7B-Instruct": {
            "data_source": str(olmo_path),
            "matrices": load_similarity_json(olmo_path),
        },
    }


def write_stats_csv(path, model_data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "data_source",
                "k",
                "n_pairs",
                "min",
                "q1",
                "median",
                "mean",
                "q3",
                "max",
                "std",
            ],
        )
        writer.writeheader()
        for model, payload in model_data.items():
            for k in K_VALUES:
                values = off_diagonal_values(payload["matrices"][k])
                writer.writerow(
                    {
                        "model": model,
                        "data_source": payload["data_source"],
                        "k": k,
                        "n_pairs": len(values),
                        "min": f"{values.min():.6f}",
                        "q1": f"{np.quantile(values, 0.25):.6f}",
                        "median": f"{np.median(values):.6f}",
                        "mean": f"{values.mean():.6f}",
                        "q3": f"{np.quantile(values, 0.75):.6f}",
                        "max": f"{values.max():.6f}",
                        "std": f"{values.std(ddof=0):.6f}",
                    }
                )


def write_pair_csv(path, model_data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "data_source", "k", "task_a", "task_b", "jaccard"],
        )
        writer.writeheader()
        for model, payload in model_data.items():
            for k in K_VALUES:
                matrix = payload["matrices"][k]
                for i, task_a in enumerate(TASKS):
                    for j, task_b in enumerate(TASKS):
                        if j <= i:
                            continue
                        writer.writerow(
                            {
                                "model": model,
                                "data_source": payload["data_source"],
                                "k": k,
                                "task_a": task_a,
                                "task_b": task_b,
                                "jaccard": f"{matrix[i, j]:.6f}",
                            }
                        )


def plot_summary(path, model_data):
    colors = {
        "Mistral": "#4C78A8",
        "Llama-3.1-8B": "#F58518",
        "Qwen2.5-7B": "#54A24B",
        "OLMo-7B-Instruct": "#B279A2",
    }
    model_names = list(model_data.keys())

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0, 1.25]})

    ax = axes[0]
    for model, payload in model_data.items():
        means = []
        q1s = []
        q3s = []
        for k in K_VALUES:
            values = off_diagonal_values(payload["matrices"][k])
            means.append(values.mean())
            q1s.append(np.quantile(values, 0.25))
            q3s.append(np.quantile(values, 0.75))
        ax.plot(K_VALUES, means, marker="o", linewidth=2.0, markersize=4.5, label=model, color=colors[model])
        ax.fill_between(K_VALUES, q1s, q3s, color=colors[model], alpha=0.13, linewidth=0)
    ax.set_title("Mean overlap rises with K")
    ax.set_xlabel("Top-K heads per task")
    ax.set_ylabel("Off-diagonal Jaccard")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    k = 128
    box_values = [off_diagonal_values(model_data[model]["matrices"][k]) for model in model_names]
    box = ax.boxplot(box_values, patch_artist=True, widths=0.55, showfliers=False)
    for patch, model in zip(box["boxes"], model_names):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[model])
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)
    rng = np.random.default_rng(0)
    for idx, (model, values) in enumerate(zip(model_names, box_values), start=1):
        jitter = rng.normal(0, 0.025, size=len(values))
        ax.scatter(np.full_like(values, idx, dtype=float) + jitter, values, color=colors[model], s=12, alpha=0.65, linewidths=0)
    ax.set_title("Pairwise spread at K=128")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    selected_pairs = [
        ("headquarters_city", "headquarters_state", "HQ city/state"),
        ("incorporation_state", "incorporation_year", "Incorp. state/year"),
    ]
    width = 0.34
    x = np.arange(len(model_names))
    for pair_idx, (task_a, task_b, label) in enumerate(selected_pairs):
        values = []
        i = TASKS.index(task_a)
        j = TASKS.index(task_b)
        for model in model_names:
            values.append(model_data[model]["matrices"][128][i, j])
        offset = (pair_idx - 0.5) * width
        bars = ax.bar(x + offset, values, width=width, label=label, alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_title("Semantic pairs dominate")
    ax.set_ylabel("Jaccard at K=128")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("Cross-task retrieval-head overlap across models", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-model task-head Jaccard overlap.")
    parser.add_argument("--project_dir", default=".", help="Repository root.")
    parser.add_argument("--output_dir", default="results/comparison_ablation", help="Directory for figure and CSV outputs.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data = build_model_matrices(project_dir)
    figure_path = output_dir / "cross_model_jaccard_summary.png"
    stats_path = output_dir / "cross_model_jaccard_stats.csv"
    pair_path = output_dir / "cross_model_jaccard_pair_values.csv"

    write_stats_csv(stats_path, model_data)
    write_pair_csv(pair_path, model_data)
    plot_summary(figure_path, model_data)

    print(f"Wrote {figure_path}")
    print(f"Wrote {figure_path.with_suffix('.pdf')}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {pair_path}")


if __name__ == "__main__":
    main()
