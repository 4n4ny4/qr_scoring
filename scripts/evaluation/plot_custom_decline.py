#!/usr/bin/env python3
"""
Create a publication-ready decline curve showing accuracy drop vs knockout heads.
Includes model name and method labels directly on the plot.

Usage:
    python scripts/evaluation/plot_custom_decline.py \\
        --summary_file results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct_lme_transfer/comparison_summary.json \\
        --output_path results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct_lme_transfer/decline_curve.png
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


def plot_decline_curve(summary, output_path):
    """Plot accuracy decline with model and method labels."""
    model_name = summary.get("model_name", "Unknown Model")
    methods = summary.get("methods", {})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        "QRScore-SEC": "#2196F3",
        "QRScore-8B-LME-TRAIN": "#4CAF50",
        "QRScore-8B-NQ-TRAIN": "#8BC34A",
        "Transfer": "#FF9800",
    }
    
    markers = {
        "QRScore-SEC": "o",
        "QRScore-8B-LME-TRAIN": "s",
        "QRScore-8B-NQ-TRAIN": "D",
    }
    
    for method_name, method_data in sorted(methods.items()):
        curve = method_data.get("accuracy_curve", {})
        if not curve:
            continue
        
        ks = sorted(int(k) for k in curve.keys())
        accs = [curve[str(k)] for k in ks]
        
        # Determine color and marker
        color = colors.get(method_name, "#9C27B0")
        marker = markers.get(method_name, "o")
        
        ax.plot(
            ks, accs,
            label=method_name,
            marker=marker,
            markersize=10,
            linewidth=3,
            color=color,
            alpha=0.8
        )
        
        # Add value labels on key points
        for k, acc in zip(ks, accs):
            if k in [0, 16, max(ks)]:
                ax.text(k, acc + 0.03, f"{acc:.1%}", ha="center", fontsize=9, 
                       color=color, fontweight="bold")
    
    # Title with model info
    title = f"Head Knockout Accuracy Decline\nModel: {model_name}"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    # Labels and formatting
    ax.set_xlabel("Number of Heads Knocked Out (K)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Answer Accuracy", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-3, max(ks) + 5)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Legend with method details
    legend_labels = []
    for method_name in sorted(methods.keys()):
        method_data = methods[method_name]
        baseline = method_data.get("baseline_accuracy", 0)
        drop_at_16 = method_data.get("drop_at_k16", 0)
        legend_labels.append(
            f"{method_name}\n(Base: {baseline:.1%}, Drop@K=16: {drop_at_16:.1%})"
        )
    
    # Clear existing legend and create new one with annotations
    ax.legend(
        [ax.get_lines()[i] for i in range(len(methods))],
        legend_labels,
        fontsize=11,
        loc="lower left",
        framealpha=0.95,
        title="Head Detection Methods",
        title_fontsize=12
    )
    
    # Add info box
    info_text = f"Dataset: SEC extraction tasks\nInstances: {summary.get('num_instances', 'N/A')}"
    ax.text(0.98, 0.05, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment="bottom", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved decline curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy decline with model and method labels"
    )
    parser.add_argument(
        "--summary_file",
        default="results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct/comparison_summary.json",
        help="Path to comparison_summary.json"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output PNG path (default: same dir as summary, named decline_curve.png)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.summary_file):
        print(f"ERROR: {args.summary_file} not found")
        return
    
    summary = load_summary(args.summary_file)
    
    output_path = args.output_path
    if output_path is None:
        output_dir = os.path.dirname(args.summary_file)
        output_path = os.path.join(output_dir, "decline_curve.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_decline_curve(summary, output_path)
    print(f"\nModel: {summary.get('model_name')}")
    print(f"Methods included: {', '.join(summary.get('methods', {}).keys())}")


if __name__ == "__main__":
    main()
