"""
Plot accuracy vs knockout size for each retrieval method from a canonical
per-model ablation directory such as:
`results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct/`.

Outputs:
  - accuracy_vs_knockout.png        (overall curves)
  - per_task_accuracy_curves.png    (8 subplots, one per task)
  - per_task_heatmaps.png           (tasks × K heatmap per method)
  - accuracy_table.csv              (method × task × K)
  - drop_from_baseline_table.csv    (drop from K=0 for each cell)

Usage:
  python scripts/evaluation/plot_ablation.py
  python scripts/evaluation/plot_ablation.py \\
    --results_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct \\
    --output_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct
"""

import argparse
import csv
import json
import os
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Plots will be skipped.")


METHOD_COLORS = {
    "QRScore-SEC": "#2196F3",
    "QRScore-8B-LME-TRAIN": "#4CAF50",
    "QRScore-8B-NQ-TRAIN": "#8BC34A",
    "Random-avg": "#9E9E9E",
    "Random-seed42": "#BDBDBD",
    "Random-seed123": "#BDBDBD",
    "Random-seed456": "#BDBDBD",
}

METHOD_STYLES = {
    "QRScore-SEC": {"marker": "o", "linestyle": "-", "linewidth": 2.5},
    "QRScore-8B-LME-TRAIN": {"marker": "s", "linestyle": "-", "linewidth": 2.5},
    "QRScore-8B-NQ-TRAIN": {"marker": "D", "linestyle": "--", "linewidth": 2},
    "Random-avg": {"marker": "x", "linestyle": ":", "linewidth": 2},
}

DISPLAY_METHODS_DEFAULT = [
    "QRScore-SEC",
    "QRScore-8B-LME-TRAIN",
    "QRScore-8B-NQ-TRAIN",
    "Random-avg",
]


def load_results_json(path):
    """Load one comparison_ablation results JSON.

    Returns (method_name, knockout_sizes, accuracies).
    """
    with open(path) as f:
        data = json.load(f)

    method = data.get("method") or os.path.basename(path).replace("_results.json", "")

    knockout_sizes = data.get("knockout_sizes", [])
    accuracy_curve = data.get("accuracy_curve", {})

    if not knockout_sizes and accuracy_curve:
        knockout_sizes = sorted(int(k) for k in accuracy_curve.keys())

    accuracies = [accuracy_curve[str(k)] for k in knockout_sizes]
    return method, knockout_sizes, accuracies


def collect_method_curves(results_dir):
    """Collect per-method accuracy curves from all *_results.json files."""
    method_curves = {}

    for fname in os.listdir(results_dir):
        if not fname.endswith("_results.json"):
            continue
        path = os.path.join(results_dir, fname)
        method, ks, accs = load_results_json(path)
        method_curves[method] = (ks, accs)

    return method_curves


def filter_method_curves(method_curves, include_methods=None):
    """Filter method curve map to selected methods, preserving insertion order."""
    if not include_methods:
        return method_curves
    return {m: curve for m, curve in method_curves.items() if m in include_methods}


def build_display_curves(method_curves, average_random=True):
    """Build curves to display, optionally averaging Random-seed* methods."""
    display_curves = {}

    # First, keep the main methods if present.
    for m in ["QRScore-SEC", "QRScore-8B-LME-TRAIN", "QRScore-8B-NQ-TRAIN"]:
        if m in method_curves:
            display_curves[m] = method_curves[m]

    # Handle Random seeds.
    random_methods = [m for m in method_curves.keys() if m.startswith("Random-seed")]
    if random_methods and average_random:
        # Assume all random seeds share the same knockout_sizes.
        base_ks, _ = method_curves[random_methods[0]]
        sums = [0.0] * len(base_ks)
        counts = [0] * len(base_ks)

        for m in random_methods:
            ks, accs = method_curves[m]
            if ks != base_ks:
                raise ValueError(f"Knockout sizes mismatch for {m}: {ks} vs {base_ks}")
            for i, a in enumerate(accs):
                sums[i] += a
                counts[i] += 1

        avg_accs = [s / c if c else 0.0 for s, c in zip(sums, counts)]
        display_curves["Random-avg"] = (base_ks, avg_accs)
    else:
        for m in random_methods:
            display_curves[m] = method_curves[m]

    return display_curves


def plot_accuracy_curves(display_curves, output_path, model_name: str):
    if not HAS_MPL:
        return

    if not display_curves:
        print("No curves to plot.")
        return

    # Determine global knockout sizes for x-axis ticks.
    all_ks = set()
    for _, (ks, _) in display_curves.items():
        all_ks.update(ks)
    xticks = sorted(all_ks)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name in DISPLAY_METHODS_DEFAULT:
        if method_name not in display_curves:
            continue
        ks, accs = display_curves[method_name]
        style = METHOD_STYLES.get(method_name, {"marker": ".", "linestyle": "-", "linewidth": 1.5})
        color = METHOD_COLORS.get(method_name, None)
        ax.plot(ks, accs, label=method_name, color=color, markersize=7, **style)

    # Also plot any other methods that are not in the default list.
    for method_name, (ks, accs) in display_curves.items():
        if method_name in DISPLAY_METHODS_DEFAULT:
            continue
        style = {"marker": ".", "linestyle": "-", "linewidth": 1.5}
        color = METHOD_COLORS.get(method_name, None)
        ax.plot(ks, accs, label=method_name, color=color, markersize=6, **style)

    ax.set_xlabel("Number of Knocked-Out Heads (K)", fontsize=13)
    ax.set_ylabel("Answer Accuracy", fontsize=13)
    ax.set_title(
        f"{model_name}: Head Ablation Accuracy vs Knockout Size\n"
        "(Steeper drop = more effective detection method)",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(xticks)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy plot to {output_path}")


# ── full-data loader (for per-task plots / tables) ───────────────────────

def load_full_method_results(results_dir: str) -> dict:
    """Load all *_results.json files, returning full dicts keyed by method."""
    methods = {}
    for p in sorted(Path(results_dir).glob("*_results.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        methods[data["method"]] = data
    return methods


def filter_full_methods(methods: dict, include_methods=None) -> dict:
    """Filter full method result payloads to selected methods."""
    if not include_methods:
        return methods
    return {m: d for m, d in methods.items() if m in include_methods}


def infer_model_name(methods: dict, fallback: str = "Unknown Model") -> str:
    """Infer the model name from one loaded results payload."""
    if not methods:
        return fallback
    first = next(iter(methods.values()))
    return first.get("model_name") or first.get("model") or fallback


def _sorted_ks(accuracy_curve: dict) -> list:
    return sorted(int(k) for k in accuracy_curve)


# ── per-task subplot grid ─────────────────────────────────────────────────

def plot_per_task_curves(methods: dict, output_dir: str, model_name: str):
    if not HAS_MPL:
        return
    first = next(iter(methods.values()))
    tasks = list(first.get("per_task_curves", {}).keys())
    if not tasks:
        print("No per_task_curves found in results; skipping per-task plot.")
        return

    n = len(tasks)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                             squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        for method_name, data in methods.items():
            task_curve = data["per_task_curves"].get(task, {})
            ks = _sorted_ks(task_curve)
            accs = [task_curve[str(k)] for k in ks]
            color = METHOD_COLORS.get(method_name, None)
            style = METHOD_STYLES.get(method_name,
                                      {"marker": ".", "linestyle": "-", "linewidth": 1.5})
            ax.plot(ks, accs, color=color, label=method_name, markersize=6, **style)
        ax.set_title(task.replace("_", " ").title(), fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("K")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(methods), 4),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{model_name}: Per-Task Accuracy Curves", fontsize=14, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "per_task_accuracy_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-task curves to {path}")


# ── per-task heatmaps (one panel per method) ──────────────────────────────

def plot_heatmaps(methods: dict, output_dir: str, model_name: str):
    if not HAS_MPL:
        return
    first = next(iter(methods.values()))
    if not first.get("per_task_curves"):
        print("No per_task_curves found; skipping heatmaps.")
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods,
                             figsize=(6 * n_methods, 5), squeeze=False)
    im = None
    for col, (method_name, data) in enumerate(methods.items()):
        tasks = list(data["per_task_curves"].keys())
        ks = _sorted_ks(data["accuracy_curve"])
        matrix = np.array([
            [data["per_task_curves"][t][str(k)] for k in ks]
            for t in tasks
        ])
        ax = axes[0][col]
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([t.replace("_", " ") for t in tasks], fontsize=9)
        ax.set_xlabel("K")
        ax.set_title(method_name, fontsize=11)
        for r in range(len(tasks)):
            for c in range(len(ks)):
                val = matrix[r, c]
                color = "white" if val < 0.4 else "black"
                ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Accuracy")
    fig.suptitle(f"{model_name}: Per-Task Accuracy Heatmaps", fontsize=14, y=1.02)
    fig.subplots_adjust(wspace=0.4)
    path = os.path.join(output_dir, "per_task_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps to {path}")


# ── summary CSV tables ────────────────────────────────────────────────────

def write_summary_csv(methods: dict, output_dir: str):
    """Write accuracy_table.csv and drop_from_baseline_table.csv."""
    first = next(iter(methods.values()))
    ks = _sorted_ks(first["accuracy_curve"])
    method_names = list(methods.keys())
    tasks = list(first.get("per_task_curves", {}).keys())

    # accuracy table
    acc_path = os.path.join(output_dir, "accuracy_table.csv")
    with open(acc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Task"] + [f"K={k}" for k in ks])
        for m in method_names:
            data = methods[m]
            row = [m, "OVERALL"]
            row += [f"{data['accuracy_curve'][str(k)]:.4f}" for k in ks]
            w.writerow(row)
            for t in tasks:
                row = [m, t]
                row += [f"{data['per_task_curves'][t][str(k)]:.4f}" for k in ks]
                w.writerow(row)
    print(f"Saved accuracy table to {acc_path}")

    # drop-from-baseline table
    drop_path = os.path.join(output_dir, "drop_from_baseline_table.csv")
    with open(drop_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Task", "Baseline(K=0)"] +
                   [f"Drop@K={k}" for k in ks if k != 0])
        for m in method_names:
            data = methods[m]
            baseline = data["accuracy_curve"]["0"]
            row = [m, "OVERALL", f"{baseline:.4f}"]
            row += [f"{baseline - data['accuracy_curve'][str(k)]:.4f}"
                    for k in ks if k != 0]
            w.writerow(row)
            for t in tasks:
                tb = data["per_task_curves"][t]["0"]
                row = [m, t, f"{tb:.4f}"]
                row += [f"{tb - data['per_task_curves'][t][str(k)]:.4f}"
                        for k in ks if k != 0]
                w.writerow(row)
    print(f"Saved drop table to {drop_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs knockout size for comparison ablation methods."
    )
    parser.add_argument(
        "--results_dir",
        default="results/comparison_ablation",
        help="Directory containing *_results.json files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write the plot (default: same as results_dir).",
    )
    parser.add_argument(
        "--no_average_random",
        action="store_true",
        help="Do not average Random-seed* curves; plot each seed separately.",
    )
    parser.add_argument(
        "--method_filter",
        nargs="+",
        default=None,
        help=(
            "Optional method names to include (e.g., QRScore-SEC). "
            "If provided, only those methods are plotted and tabulated."
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    os.makedirs(output_dir, exist_ok=True)

    method_curves = collect_method_curves(results_dir)
    method_curves = filter_method_curves(method_curves, include_methods=args.method_filter)
    if not method_curves:
        if args.method_filter:
            print(
                "No matching methods found for --method_filter in "
                f"{results_dir}: {args.method_filter}"
            )
        else:
            print(f"No *_results.json files found in {results_dir}")
        return

    display_curves = build_display_curves(
        method_curves, average_random=not args.no_average_random
    )

    pooled_plot_name = "accuracy_vs_knockout.png"
    if args.method_filter == ["QRScore-SEC"]:
        pooled_plot_name = "qrscore_sec_pooled_accuracy_curve.png"
    output_path = os.path.join(output_dir, pooled_plot_name)

    # per-task plots, heatmaps, and CSV tables
    full_methods = load_full_method_results(results_dir)
    full_methods = filter_full_methods(full_methods, include_methods=args.method_filter)
    model_name = infer_model_name(full_methods)

    plot_accuracy_curves(display_curves, output_path, model_name)

    if full_methods:
        if args.method_filter == ["QRScore-SEC"]:
            # Save QRScore-SEC-only plot under an explicit name.
            plot_per_task_curves(full_methods, output_dir, model_name)
            src = os.path.join(output_dir, "per_task_accuracy_curves.png")
            dst = os.path.join(output_dir, "qrscore_sec_per_task_accuracy_curves.png")
            if os.path.exists(src):
                os.replace(src, dst)
                print(f"Saved per-task curves to {dst}")
        else:
            plot_per_task_curves(full_methods, output_dir, model_name)

        plot_heatmaps(full_methods, output_dir, model_name)
        write_summary_csv(full_methods, output_dir)

    print(f"\nDone. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
