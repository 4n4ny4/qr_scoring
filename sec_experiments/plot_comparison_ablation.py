"""
Plot accuracy vs knockout size for each retrieval method from
`results/comparison_ablation/*_results.json`.

Usage:
  python sec_experiments/plot_comparison_ablation.py
  python sec_experiments/plot_comparison_ablation.py \
    --results_dir results/comparison_ablation \
    --output_dir results/comparison_ablation
"""

import argparse
import json
import os
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Plots will be skipped.")


METHOD_COLORS = {
    "QRScore-SEC": "#2196F3",
    "QRScore-Paper-LME": "#4CAF50",
    "QRScore-Paper-NQ": "#8BC34A",
    "Random-avg": "#9E9E9E",
    "Random-seed42": "#BDBDBD",
    "Random-seed123": "#BDBDBD",
    "Random-seed456": "#BDBDBD",
}

METHOD_STYLES = {
    "QRScore-SEC": {"marker": "o", "linestyle": "-", "linewidth": 2.5},
    "QRScore-Paper-LME": {"marker": "s", "linestyle": "-", "linewidth": 2.5},
    "QRScore-Paper-NQ": {"marker": "D", "linestyle": "--", "linewidth": 2},
    "Random-avg": {"marker": "x", "linestyle": ":", "linewidth": 2},
}

DISPLAY_METHODS_DEFAULT = [
    "QRScore-SEC",
    "QRScore-Paper-LME",
    "QRScore-Paper-NQ",
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


def build_display_curves(method_curves, average_random=True):
    """Build curves to display, optionally averaging Random-seed* methods."""
    display_curves = {}

    # First, keep the main methods if present.
    for m in ["QRScore-SEC", "QRScore-Paper-LME", "QRScore-Paper-NQ"]:
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


def plot_accuracy_curves(display_curves, output_path):
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
        "Head Ablation: Accuracy vs Knockout Size\n"
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
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    os.makedirs(output_dir, exist_ok=True)

    method_curves = collect_method_curves(results_dir)
    if not method_curves:
        print(f"No *_results.json files found in {results_dir}")
        return

    display_curves = build_display_curves(
        method_curves, average_random=not args.no_average_random
    )

    output_path = os.path.join(output_dir, "accuracy_vs_knockout.png")
    plot_accuracy_curves(display_curves, output_path)


if __name__ == "__main__":
    main()

