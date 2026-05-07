#!/usr/bin/env python3
"""Plot the target-task sensitivity heatmap (Figure 1).

Three-panel heatmap: target task (y) vs K (x), one panel per model
(Llama, Qwen, Mistral). Each cell is::

    sensitivity[model][t][K] = mean over all 8 sources s of drop[s][t][K]

i.e. the column mean of the source x target drop matrix produced by
``run_ablation.py``. This is the same metric reported in
``compute_target_sensitivity.py`` (we reuse its loader and defaults so
Llama is pulled from ``origin/main`` automatically and provenance stays
consistent across the report and the figure).

Output: ``results/target_sensitivity/target_sensitivity_3models.png``.

Usage:
    python scripts/evaluation/plot_target_sensitivity.py
    python scripts/evaluation/plot_target_sensitivity.py --output_path mypath.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse loader + canonical defaults from the metric script so the figure
# never drifts from the report.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_target_sensitivity import (  # type: ignore  # noqa: E402
    DEFAULT_INPUTS,
    SHORT_LABELS,
    compute_target_metrics,
    load_matrix,
    parse_input_spec,
    short_label,
    validate_matrix_shape,
)

DEFAULT_OUTPUT_PATH = "results/target_sensitivity/target_sensitivity_3models.png"
DEFAULT_KS = (8, 16, 32, 48, 64, 96, 128)

TASK_LABELS = {
    "ceo_lastname":          "CEO last name",
    "employees_count_total": "Employee count",
    "headquarters_city":     "HQ city",
    "headquarters_state":    "HQ state",
    "holder_record_amount":  "Holder record amt",
    "incorporation_state":   "Incorp. state",
    "incorporation_year":    "Incorp. year",
    "registrant_name":       "Registrant name",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--inputs", nargs="*", default=None,
        help='List of "label=path[@gitref]" entries. Defaults to the same '
             'Llama (origin/main) / Qwen / Mistral matrices used by '
             'compute_target_sensitivity.py.',
    )
    p.add_argument(
        "--ks", default=None,
        help="Comma-separated K values to display. Default: 8,16,32,48,64,96,128.",
    )
    p.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--annotate", action="store_true",
                   help="Write the numeric drop value inside each cell.")
    p.add_argument("--cmap", default="YlOrRd",
                   help="matplotlib colormap (default: YlOrRd).")
    p.add_argument("--repo_root", default=None,
                   help="Repo root for resolving paths and running git show.")
    return p.parse_args()


def resolve_inputs(cli_inputs) -> List[Tuple[str, str]]:
    if cli_inputs:
        out = []
        for entry in cli_inputs:
            if "=" not in entry:
                sys.exit(f"--inputs entry missing 'label=': {entry!r}")
            label, spec = entry.split("=", 1)
            out.append((label, spec))
        return out
    return DEFAULT_INPUTS


def load_all(inputs: List[Tuple[str, str]], repo_root: str
             ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    matrices: Dict[str, dict] = {}
    metrics: Dict[str, dict] = {}
    for label, spec in inputs:
        m = load_matrix(spec, repo_root)
        validate_matrix_shape(label, m)
        matrices[label] = m
        metrics[label] = compute_target_metrics(m)
        path, ref = parse_input_spec(spec)
        provenance = f"{path}" + (f" @ {ref}" if ref else " (working tree)")
        print(f"  {label:<28} <- {provenance}")
    return matrices, metrics


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))

    inputs = resolve_inputs(args.inputs)
    if args.ks:
        ks = [int(x) for x in args.ks.split(",")]
    else:
        ks = list(DEFAULT_KS)

    print("Loading transfer matrices:")
    matrices, metrics = load_all(inputs, repo_root)

    # Verify task lists agree across models
    task_lists = [sorted(m["targets"]) for m in matrices.values()]
    if any(tl != task_lists[0] for tl in task_lists):
        sys.exit(f"Target task lists differ across models: {task_lists}")
    all_targets = task_lists[0]

    # Verify every K we want is present in every matrix
    for label, m in matrices.items():
        missing = [k for k in ks if k not in m["knockout_sizes"]]
        if missing:
            sys.exit(f"Matrix for {label} is missing K values {missing}; "
                     f"available: {m['knockout_sizes']}")

    models = list(matrices.keys())

    # Aggregate sensitivity per task (mean across models and K) for ordering
    agg: Dict[str, float] = {}
    for t in all_targets:
        vals = []
        for label in models:
            for k in ks:
                v = metrics[label][t][k]["sensitivity"]
                if v is not None:
                    vals.append(v)
        agg[t] = float(np.mean(vals)) if vals else float("nan")
    task_order = sorted(all_targets, key=lambda t: -agg[t])

    # Build per-model (n_tasks, n_K) matrices
    grids: Dict[str, np.ndarray] = {}
    for label in models:
        M = np.zeros((len(task_order), len(ks)))
        for i, t in enumerate(task_order):
            for j, k in enumerate(ks):
                M[i, j] = metrics[label][t][k]["sensitivity"]
        grids[label] = M

    # Shared color scale clamped to [0, max(1, observed)]
    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    vmin = 0.0
    vmax = max(1.0, float(np.nanmax(all_vals)))

    # Plot.
    # Sizing aims for ~0.7 inch per cell horizontally so 2-decimal annotations
    # comfortably fit at fontsize ~9 with no neighbour collision.
    n_panels = len(models)
    n_tasks = len(task_order)
    n_ks = len(ks)
    panel_w = max(4.0, 0.7 * n_ks + 1.6)        # inches per panel
    fig_h = max(4.6, 0.55 * n_tasks + 2.2)      # leaves room for 2-line titles
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_w * n_panels + 1.6, fig_h),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    annot_fontsize = 9 if args.annotate else 0
    tick_fontsize = 9
    label_fontsize = 10

    im = None
    for ax, label in zip(axes, models):
        M = grids[label]
        im = ax.imshow(M, aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_ks))
        ax.set_xticklabels([str(k) for k in ks], fontsize=tick_fontsize)
        ax.set_xlabel("K (heads ablated)", fontsize=label_fontsize, labelpad=6)
        title_short = SHORT_LABELS.get(label, short_label(label))
        # Two-line title: bold short name, faint full HF name underneath
        ax.set_title(title_short, fontsize=12, fontweight="bold", pad=18)
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color="#555555")
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([TASK_LABELS.get(t, t) for t in task_order],
                           fontsize=tick_fontsize)
        # Thin gridlines to visually separate cells
        ax.set_xticks(np.arange(-0.5, n_ks, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_tasks, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        if args.annotate:
            for i in range(n_tasks):
                for j in range(n_ks):
                    val = M[i, j]
                    # White text on darker reds; threshold tuned to YlOrRd/Reds.
                    txt_color = "white" if val > 0.50 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=annot_fontsize, color=txt_color)

    axes[0].set_ylabel("Target task (sorted by mean sensitivity)",
                       fontsize=label_fontsize, labelpad=8)

    # Reserve clear space on the right for the colorbar (label sits *above*
    # the bar so it can't collide with the rightmost panel's K=128 tick).
    fig.subplots_adjust(left=0.10, right=0.88, wspace=0.18,
                        top=0.84, bottom=0.14)
    cbar_ax = fig.add_axes([0.905, 0.14, 0.015, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar_ax.set_title("Target\nsensitivity", fontsize=9, pad=10, loc="left")

    fig.suptitle(
        "Target-task sensitivity under cross-task ablation\n"
        "(mean accuracy drop across all 8 source-task ablations)",
        fontsize=12, y=0.97,
    )

    out_path = args.output_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print("\nTasks (top to bottom, sorted by aggregate sensitivity):")
    for t in task_order:
        print(f"  {TASK_LABELS.get(t, t):<22}  agg = {agg[t]:.3f}")


if __name__ == "__main__":
    main()
