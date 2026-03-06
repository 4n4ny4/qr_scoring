"""
Visualization and statistical analysis for the comparison ablation.

Generates:
  1. Multi-line accuracy curve (all methods on same plot)
  2. Per-task heatmap (methods x tasks, showing accuracy drop at K=16)
  3. Head overlap analysis (Jaccard similarity between top-K heads of each method)
  4. Statistical significance (paired bootstrap at K=16)

Usage:
  python sec_experiments/plot_comparison.py
  python sec_experiments/plot_comparison.py --summary results/comparison_ablation/comparison_summary.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Plots will be skipped.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


METHOD_COLORS = {
    "QRScore-SEC": "#2196F3",
    "QRScore-Paper-LME": "#4CAF50",
    "QRScore-Paper-NQ": "#8BC34A",
    "RETHEAD": "#FF9800",
    "Random-avg": "#9E9E9E",
    "Random-seed42": "#BDBDBD",
    "Random-seed123": "#BDBDBD",
    "Random-seed456": "#BDBDBD",
}

METHOD_STYLES = {
    "QRScore-SEC": {"marker": "o", "linestyle": "-", "linewidth": 2.5},
    "QRScore-Paper-LME": {"marker": "s", "linestyle": "-", "linewidth": 2.5},
    "QRScore-Paper-NQ": {"marker": "D", "linestyle": "--", "linewidth": 2},
    "RETHEAD": {"marker": "^", "linestyle": "-", "linewidth": 2.5},
    "Random-avg": {"marker": "x", "linestyle": ":", "linewidth": 2},
}

DISPLAY_METHODS = ["QRScore-SEC", "QRScore-Paper-LME", "QRScore-Paper-NQ", "RETHEAD", "Random-avg"]


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def load_heads_from_json(path):
    with open(path) as f:
        data = json.load(f)
    return [(int(h.split("-")[0]), int(h.split("-")[1])) for h, s in data]


def load_heads_from_yaml(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    heads = []
    for h in config["attn_head_set"].split(","):
        layer, head = map(int, h.strip().split("-"))
        heads.append((layer, head))
    return heads


def plot_accuracy_curves(summary, output_dir):
    """Plot accuracy vs knockout size for all methods."""
    if not HAS_MPL:
        return

    methods = summary.get("methods", {})
    knockout_sizes = summary.get("knockout_sizes", [])

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name in DISPLAY_METHODS:
        if method_name not in methods:
            continue
        curve = methods[method_name].get("accuracy_curve", {})
        ks = sorted(int(k) for k in curve.keys())
        accs = [curve[str(k)] for k in ks]

        style = METHOD_STYLES.get(method_name, {"marker": ".", "linestyle": "-", "linewidth": 1.5})
        color = METHOD_COLORS.get(method_name, None)
        ax.plot(ks, accs, label=method_name, color=color,
                markersize=7, **style)

    ax.set_xlabel("Number of Knocked-Out Heads (K)", fontsize=13)
    ax.set_ylabel("Answer Accuracy", fontsize=13)
    ax.set_title("Head Ablation: Accuracy vs Knockout Size\n(Steeper drop = more effective detection method)", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if knockout_sizes:
        ax.set_xticks(knockout_sizes)

    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_task_heatmap(summary, output_dir, target_k=16):
    """Plot methods x tasks heatmap showing accuracy drop at K=target_k."""
    if not HAS_MPL:
        return

    methods = summary.get("methods", {})
    display = [m for m in DISPLAY_METHODS if m in methods]
    if not display:
        return

    tasks = []
    for m in display:
        ptc = methods[m].get("per_task_curves", {})
        for t in ptc:
            if t not in tasks:
                tasks.append(t)
    tasks.sort()

    if not tasks:
        print("No per-task data available for heatmap.")
        return

    matrix = np.zeros((len(display), len(tasks)))
    for i, method in enumerate(display):
        ptc = methods[method].get("per_task_curves", {})
        for j, task in enumerate(tasks):
            if task in ptc:
                baseline = ptc[task].get("0", 0)
                at_k = ptc[task].get(str(target_k), 0)
                matrix[i, j] = baseline - at_k
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.5), max(4, len(display) * 0.8)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(display, fontsize=11)

    for i in range(len(display)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if val > matrix[~np.isnan(matrix)].max() * 0.6 else "black")

    ax.set_title(f"Accuracy Drop at K={target_k} (higher = method found more important heads)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Accuracy Drop")
    fig.tight_layout()
    path = os.path.join(output_dir, f"per_task_heatmap_k{target_k}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_head_overlap(output_dir, top_k=16):
    """Compute and plot Jaccard similarity between top-K heads of each method."""
    if not HAS_MPL:
        return
    if not HAS_YAML:
        print("WARNING: yaml not available, skipping head overlap analysis")
        return

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs_dir = os.path.join(project_dir, "src", "qrretriever", "configs")
    results_dir = os.path.join(project_dir, "results", "detection")

    method_heads = {}

    lc_combined = os.path.join(results_dir, "long_context_combined_heads.json")
    if os.path.exists(lc_combined):
        method_heads["QRScore-SEC"] = set(load_heads_from_json(lc_combined)[:top_k])
    else:
        niah_combined = os.path.join(results_dir, "niah_combined_heads.json")
        if os.path.exists(niah_combined):
            method_heads["QRScore-SEC"] = set(load_heads_from_json(niah_combined)[:top_k])

    lme_path = os.path.join(configs_dir, "Llama-3.1-8B-Instruct_qr_head_LME.yaml")
    if os.path.exists(lme_path):
        method_heads["QRScore-Paper-LME"] = set(load_heads_from_yaml(lme_path)[:top_k])

    nq_path = os.path.join(configs_dir, "Llama-3.1-8B-Instruct_qr_head_NQ.yaml")
    if os.path.exists(nq_path):
        method_heads["QRScore-Paper-NQ"] = set(load_heads_from_yaml(nq_path)[:top_k])

    rethead_path = os.path.join(results_dir, "rethead_combined_heads.json")
    if os.path.exists(rethead_path):
        method_heads["RETHEAD"] = set(load_heads_from_json(rethead_path)[:top_k])

    if len(method_heads) < 2:
        print("Need at least 2 methods with head rankings for overlap analysis.")
        return

    names = sorted(method_heads.keys())
    n = len(names)
    jaccard_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            s1 = method_heads[names[i]]
            s2 = method_heads[names[j]]
            union = s1 | s2
            intersection = s1 & s2
            jaccard_matrix[i, j] = len(intersection) / len(union) if union else 0

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
    im = ax.imshow(jaccard_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = jaccard_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if val > 0.5 else "black")

    ax.set_title(f"Head Overlap (Jaccard Similarity, top-{top_k})", fontsize=13)
    plt.colorbar(im, ax=ax, label="Jaccard Index")
    fig.tight_layout()
    path = os.path.join(output_dir, f"head_overlap_jaccard_top{top_k}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print(f"\nHead overlap (Jaccard, top-{top_k}):")
    for i, j in combinations(range(n), 2):
        s1 = method_heads[names[i]]
        s2 = method_heads[names[j]]
        overlap = s1 & s2
        print(f"  {names[i]} vs {names[j]}: "
              f"Jaccard={jaccard_matrix[i,j]:.3f}, "
              f"overlap={len(overlap)}/{top_k}")
        if overlap:
            print(f"    shared: {sorted(overlap)}")


def compute_significance(summary, output_dir, target_k=16,
                         n_bootstrap=1000, seed=42):
    """Paired bootstrap significance test between methods at K=target_k."""
    methods = summary.get("methods", {})
    display = [m for m in DISPLAY_METHODS if m in methods]

    method_details = {}
    for method_name in display:
        results_path = os.path.join(
            output_dir,
            f"{method_name.replace(' ', '_')}_results.json"
        )
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            data = json.load(f)
        details = data.get("details", {}).get(str(target_k), [])
        if details:
            method_details[method_name] = {d["idx"]: d["correct"] for d in details}

    if len(method_details) < 2:
        print("Need at least 2 methods with per-instance details for significance testing.")
        return

    common_ids = None
    for md in method_details.values():
        ids = set(md.keys())
        common_ids = ids if common_ids is None else common_ids & ids

    if not common_ids:
        print("No common instances across methods.")
        return

    common_ids = sorted(common_ids)
    n = len(common_ids)

    print(f"\nPaired bootstrap significance (K={target_k}, n={n}, B={n_bootstrap}):")
    print(f"{'Method A':<25s} vs {'Method B':<25s} {'Diff':>8s} {'p-value':>10s} {'Sig?':>6s}")
    print("-" * 80)

    rng = np.random.RandomState(seed)
    sig_results = []

    for m1, m2 in combinations(method_details.keys(), 2):
        arr1 = np.array([method_details[m1][idx] for idx in common_ids])
        arr2 = np.array([method_details[m2][idx] for idx in common_ids])
        diff = arr1 - arr2
        observed_diff = diff.mean()

        boot_diffs = []
        for _ in range(n_bootstrap):
            inds = rng.randint(0, n, size=n)
            boot_diffs.append(diff[inds].mean())
        boot_diffs = np.array(boot_diffs)

        p_value = (boot_diffs <= 0).mean() if observed_diff > 0 else (boot_diffs >= 0).mean()
        sig = "*" if p_value < 0.05 else ""

        print(f"{m1:<25s} vs {m2:<25s} {observed_diff:>+8.4f} {p_value:>10.4f} {sig:>6s}")
        sig_results.append({
            "method_a": m1, "method_b": m2,
            "diff_a_minus_b": float(observed_diff),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
        })

    sig_path = os.path.join(output_dir, f"significance_k{target_k}.json")
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2)
    print(f"\nSignificance results saved to {sig_path}")


def print_summary_table(summary):
    """Print a formatted summary table."""
    methods = summary.get("methods", {})
    knockout_sizes = summary.get("knockout_sizes", [])

    print(f"\n{'='*100}")
    print(f"COMPARISON ABLATION SUMMARY")
    print(f"{'='*100}")
    print(f"Instances: {summary.get('num_instances', 'N/A')}")
    print(f"Model: {summary.get('model', 'N/A')}")

    header = f"{'Method':<25s}"
    for K in knockout_sizes:
        header += f"  K={K:>3d}"
    header += f"  {'Drop@16':>8s}"
    print(f"\n{header}")
    print("-" * len(header))

    for method_name in DISPLAY_METHODS:
        if method_name not in methods:
            continue
        info = methods[method_name]
        row = f"{method_name:<25s}"
        for K in knockout_sizes:
            acc = info.get("accuracy_curve", {}).get(str(K), None)
            row += f"  {acc:>6.3f}" if acc is not None else f"  {'N/A':>6s}"
        drop = info.get("drop_at_k16")
        row += f"  {drop:>+8.4f}" if drop is not None else f"  {'N/A':>8s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Plot comparison ablation results")
    parser.add_argument("--summary", default="results/comparison_ablation/comparison_summary.json")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for plots (default: same as summary)")
    parser.add_argument("--target_k", type=int, default=16)
    parser.add_argument("--top_k_overlap", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.summary)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.summary):
        summary = load_summary(args.summary)
        print_summary_table(summary)
        plot_accuracy_curves(summary, args.output_dir)
        plot_per_task_heatmap(summary, args.output_dir, target_k=args.target_k)
        compute_significance(summary, args.output_dir, target_k=args.target_k)
    else:
        print(f"Summary file not found: {args.summary}")
        print("Run the comparison ablation first, or provide a valid path.")

    plot_head_overlap(args.output_dir, top_k=args.top_k_overlap)


if __name__ == "__main__":
    main()
