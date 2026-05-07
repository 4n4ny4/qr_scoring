#!/usr/bin/env python3
"""
Create a focused decline curve comparing two specific head detection methods.
Includes model name and method information prominently.

Usage:
    python scripts/evaluation/plot_custom_decline_filtered.py \\
        --summary_file results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct/comparison_summary.json \\
        --methods QRScore-SEC QRScore-8B-LME-TRAIN \\
        --output_path results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct/sec_vs_lme_decline.png
"""

import argparse
import json
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib not available")
    exit(1)


def load_summary(summary_path):
    """Load comparison_summary.json."""
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def plot_decline_curve_filtered(summary, methods_to_plot, output_path):
    """Plot accuracy decline for selected methods with detailed labels."""
    model_name = summary.get("model_name", "Unknown Model")
    all_methods = summary.get("methods", {})
    
    # Filter to requested methods
    methods = {m: all_methods[m] for m in methods_to_plot if m in all_methods}
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    colors = {
        "QRScore-SEC": "#2196F3",
        "QRScore-8B-LME-TRAIN": "#4CAF50",
        "QRScore-8B-NQ-TRAIN": "#8BC34A",
    }
    
    markers = {
        "QRScore-SEC": "o",
        "QRScore-8B-LME-TRAIN": "s",
        "QRScore-8B-NQ-TRAIN": "D",
    }
    
    linestyles = {
        "QRScore-SEC": "-",
        "QRScore-8B-LME-TRAIN": "-",
        "QRScore-8B-NQ-TRAIN": "--",
    }
    
    for method_name in sorted(methods.keys()):
        method_data = methods[method_name]
        curve = method_data.get("accuracy_curve", {})
        if not curve:
            continue
        
        ks = sorted(int(k) for k in curve.keys())
        accs = [curve[str(k)] for k in ks]
        
        # Determine styling
        color = colors.get(method_name, "#9C27B0")
        marker = markers.get(method_name, "o")
        linestyle = linestyles.get(method_name, "-")
        
        ax.plot(
            ks, accs,
            label=method_name,
            marker=marker,
            markersize=12,
            linewidth=4,
            color=color,
            linestyle=linestyle,
            alpha=0.85,
            markeredgewidth=1.5,
            markeredgecolor="white"
        )
        
        # Add value labels on key points
        for k, acc in zip(ks, accs):
            if k in [0, 8, 16, 32, max(ks)]:
                y_offset = 0.05 if method_name == "QRScore-SEC" else -0.05
                ax.text(k, acc + y_offset, f"{acc:.0%}", 
                       ha="center", va="bottom" if y_offset > 0 else "top",
                       fontsize=10, color=color, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                alpha=0.7, edgecolor=color, linewidth=1))
    
    # Enhanced title with model and data info
    model_display = model_name.split("/")[-1] if "/" in model_name else model_name
    title = (
        f"Head Knockout Accuracy Decline\n"
        f"Model: {model_display} | "
        f"{summary.get('num_instances', 'N/A')} instances | "
        f"8 SEC extraction tasks"
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    # Labels and formatting
    ax.set_xlabel("Number of Attention Heads Knocked Out (K)", 
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Answer Accuracy", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    
    ks_all = sorted(int(k) for ks_list in [sorted(int(k) for k in m["accuracy_curve"].keys()) 
                                           for m in methods.values()] for k in ks_list)
    ax.set_xlim(-3, max(ks_all) + 5)
    
    # Grid with custom styling
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Enhanced legend with detailed stats
    legend_lines = []
    legend_labels = []
    for method_name in sorted(methods.keys()):
        method_data = methods[method_name]
        baseline = method_data.get("baseline_accuracy", 0)
        drop_at_16 = method_data.get("drop_at_k16", 0)
        
        # Find line object
        for line in ax.get_lines():
            if line.get_label() == method_name:
                legend_lines.append(line)
                break
        
        # Create detailed label
        label_text = (
            f"{method_name}\n"
            f"  Baseline (K=0): {baseline:.1%}\n"
            f"  Drop at K=16: {drop_at_16:.1%}"
        )
        legend_labels.append(label_text)
    
    ax.legend(
        legend_lines,
        legend_labels,
        fontsize=11,
        loc="lower left",
        framealpha=0.96,
        title="Head Detection Methods",
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Interpretation text box
    info_text = (
        "Steeper decline = more accurate head detection\n"
        "QRScore-SEC: in-domain (SEC training data)\n"
        "QRScore-8B-LME-TRAIN: cross-domain (general text)"
    )
    ax.text(0.98, 0.08, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment="bottom", horizontalalignment="right",
           bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", 
                    alpha=0.9, edgecolor="gray", linewidth=1.5),
           family="monospace")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot filtered accuracy decline comparison"
    )
    parser.add_argument(
        "--summary_file",
        default="results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct/comparison_summary.json",
        help="Path to comparison_summary.json"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["QRScore-SEC", "QRScore-8B-LME-TRAIN"],
        help="Methods to plot"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output PNG path"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.summary_file):
        print(f"ERROR: {args.summary_file} not found")
        return
    
    summary = load_summary(args.summary_file)
    
    output_path = args.output_path
    if output_path is None:
        output_dir = os.path.dirname(args.summary_file)
        output_path = os.path.join(output_dir, "sec_vs_lme_decline.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_decline_curve_filtered(summary, args.methods, output_path)
    print(f"\nModel: {summary.get('model_name')}")
    print(f"Methods plotted: {', '.join(args.methods)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
