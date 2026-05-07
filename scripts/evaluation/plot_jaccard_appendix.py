#!/usr/bin/env python3
"""Per-model 8x8 top-K=16 Jaccard heatmap (appendix figure).

For each model in ``DEFAULT_HEAD_INPUTS`` we load the per-task top-K head
rankings produced by ``scripts/detection/detect_qrhead.py`` (the
``results/detection*/topk/long_context_<task>_top<K>.json`` files), then
compute the 8x8 Jaccard overlap matrix at top-K and render one heatmap
panel per model.

Heads are stored as ``[layer, head]`` pairs. Two tasks' top-K head sets
share a Jaccard score ``|A & B| / |A | B|`` in [0, 1].

The figure pairs the model heatmaps with a single-row "highlight" table
that shows the four task pairs the LaTeX (``results.tex``) discusses
specifically:

    headquarters_city  vs headquarters_state
    incorporation_state vs incorporation_year

so a reviewer can verify the "75-95%" claim directly.

Outputs:
    results/appendix/jaccard_topk{K}_heatmap.png
    results/appendix/jaccard_topk{K}_summary.md   (per-model & cross-model
                                                   pair tables, in markdown)

Provenance
----------
Inputs use the same ``path[@gitref]`` syntax as
``compute_target_sensitivity.py``. Defaults pull Llama from ``origin/main``
(the canonical SEC detection output) and Qwen / Mistral from the working
tree (these directories were produced by the Mistral pipeline run).

Usage:
    python scripts/evaluation/plot_jaccard_appendix.py
    python scripts/evaluation/plot_jaccard_appendix.py --top_k 32
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse short-label conventions from the metric script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_target_sensitivity import (  # type: ignore  # noqa: E402
    SHORT_LABELS,
    short_label,
)

# Each entry: (label, "template1[|template2|...][@gitref]")
# Templates may use {task} and {k} placeholders. The loader tries each
# template in order and uses the first one that exists. This makes the
# script robust to branches where some tasks ship only the full
# ``_heads.json`` ranking and others only ship the pre-truncated
# ``_top{K}.json`` snapshot. The ``@gitref`` suffix applies to *all*
# templates in the entry.
DEFAULT_HEAD_INPUTS: List[Tuple[str, str]] = [
    ("Llama-3.1-8B-Instruct",
     "results/detection/long_context_{task}_heads.json|"
     "results/detection/topk/long_context_{task}_top{k}.json@origin/main"),
    ("Qwen2.5-7B-Instruct",
     "results/detection_qwen/long_context_{task}_heads.json|"
     "results/detection_qwen/topk/long_context_{task}_top{k}.json"),
    ("Mistral-7B-Instruct-v0.3",
     "results/detection_mistral/long_context_{task}_heads.json|"
     "results/detection_mistral/topk/long_context_{task}_top{k}.json"),
]

DEFAULT_TASKS = [
    "ceo_lastname",
    "employees_count_total",
    "headquarters_city",
    "headquarters_state",
    "holder_record_amount",
    "incorporation_state",
    "incorporation_year",
    "registrant_name",
]

# Pairs called out in results.tex (Section 4.4 / "high overlap pairs").
HIGHLIGHTED_PAIRS: List[Tuple[str, str]] = [
    ("headquarters_city", "headquarters_state"),
    ("incorporation_state", "incorporation_year"),
]

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

DEFAULT_OUTPUT_DIR = "results/appendix"


def parse_input_spec(spec: str) -> Tuple[List[str], Optional[str]]:
    """Return (template_list, gitref|None) from ``"t1|t2|...[@ref]"``."""
    if "@" in spec:
        body, ref = spec.split("@", 1)
    else:
        body, ref = spec, None
    return body.split("|"), ref


def _read_blob_optional(path: str, ref: Optional[str],
                        repo_root: str) -> Optional[bytes]:
    """Return bytes from disk or git ref, or None if missing."""
    if ref:
        try:
            return subprocess.check_output(
                ["git", "show", f"{ref}:{path}"],
                cwd=repo_root,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            return None
    abs_path = path if os.path.isabs(path) else os.path.join(repo_root, path)
    if not os.path.exists(abs_path):
        return None
    with open(abs_path, "rb") as fh:
        return fh.read()


def _read_blob_first_match(templates: List[str], task: str, k: int,
                           ref: Optional[str], repo_root: str
                           ) -> Tuple[bytes, str]:
    """Try each template until one resolves; return (bytes, resolved_path).

    Raises FileNotFoundError listing every path that was attempted.
    """
    tried: List[str] = []
    for tmpl in templates:
        path = tmpl.format(task=task, k=k)
        tried.append((f"{ref}:{path}" if ref else path))
        blob = _read_blob_optional(path, ref, repo_root)
        if blob is not None:
            return blob, path
    raise FileNotFoundError(
        f"None of the candidate paths exist for task={task!r}, K={k}: "
        + ", ".join(tried)
    )


def _parse_head_id(item) -> Tuple[int, int]:
    head_id = item[0] if isinstance(item, (list, tuple)) else item
    if isinstance(head_id, str):
        l, h = head_id.split("-")
        return int(l), int(h)
    if isinstance(head_id, (list, tuple)):
        return int(head_id[0]), int(head_id[1])
    raise ValueError(f"Unrecognised head entry shape: {item!r}")


def load_head_set(template_spec: str, task: str, k: int, repo_root: str
                  ) -> Tuple[set, str]:
    """Return (top-K head set, resolved path) for ``task``.

    Files may be either the full sorted ranking (``_heads.json``) or the
    pre-truncated ``_top{K}.json`` snapshot. We always slice the first K
    entries — this is a no-op for the snapshot case and a top-K cut for
    the full-ranking case.
    """
    templates, ref = parse_input_spec(template_spec)
    blob, resolved = _read_blob_first_match(templates, task, k, ref, repo_root)
    raw = json.loads(blob)
    if isinstance(raw, dict):
        for key in ("heads", "top_heads", "head_set"):
            if key in raw:
                raw = raw[key]
                break
    if len(raw) < k:
        raise ValueError(
            f"{resolved} has only {len(raw)} ranked heads (< K={k}). "
            "Did you point at a pre-truncated _top<k>.json file with a "
            "smaller K than requested?"
        )
    return {_parse_head_id(item) for item in raw[:k]}, resolved


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compute_matrix(template_spec: str, tasks: List[str], k: int,
                   repo_root: str) -> Tuple[np.ndarray, Dict[str, set],
                                            Dict[str, str]]:
    head_sets: Dict[str, set] = {}
    resolved: Dict[str, str] = {}
    for t in tasks:
        hs, path = load_head_set(template_spec, t, k, repo_root)
        head_sets[t] = hs
        resolved[t] = path
    n = len(tasks)
    M = np.zeros((n, n))
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            M[i, j] = jaccard(head_sets[a], head_sets[b])
    return M, head_sets, resolved


def md_table(headers: List[str], rows: List[List[str]],
             aligns: Optional[List[str]] = None) -> str:
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    sep = {"l": ":---", "c": ":---:", "r": "---:"}
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(sep[a] for a in aligns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="*", default=None,
                   help='List of "label=template_spec[@gitref]" entries with '
                        '{task} and {k} placeholders.')
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cmap", default="YlGnBu")
    p.add_argument("--repo_root", default=None)
    args = p.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    tasks = args.tasks or DEFAULT_TASKS

    inputs: List[Tuple[str, str]] = []
    if args.inputs:
        for entry in args.inputs:
            if "=" not in entry:
                sys.exit(f"--inputs entry missing 'label=': {entry!r}")
            label, spec = entry.split("=", 1)
            inputs.append((label, spec))
    else:
        inputs = DEFAULT_HEAD_INPUTS

    print(f"Loading top-{args.top_k} head sets for {len(tasks)} tasks "
          f"across {len(inputs)} models")
    print("=" * 78)
    matrices: Dict[str, np.ndarray] = {}
    head_sets_by_model: Dict[str, Dict[str, set]] = {}
    resolved_paths: Dict[str, Dict[str, str]] = {}
    for label, template_spec in inputs:
        templates, ref = parse_input_spec(template_spec)
        ref_str = f" @ {ref}" if ref else " (working tree)"
        print(f"  {label:<28} candidates ({len(templates)}):" + ref_str)
        for t in templates:
            print(f"      {t}")
        M, hs, paths = compute_matrix(template_spec, tasks, args.top_k, repo_root)
        matrices[label] = M
        head_sets_by_model[label] = hs
        resolved_paths[label] = paths
        # Show which file actually got used per task (helps verify provenance)
        unique_used = sorted(set(paths.values()))
        for u in unique_used:
            tasks_using = [t for t, p in paths.items() if p == u]
            print(f"    used: {u}  ({len(tasks_using)}/{len(tasks)} tasks)")

    models = list(matrices.keys())

    # ── Plot ────────────────────────────────────────────────────────────────
    n_panels = len(models)
    panel_w = max(4.5, 0.55 * len(tasks) + 2.0)
    fig_h = max(4.6, 0.55 * len(tasks) + 2.4)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(panel_w * n_panels + 1.6, fig_h), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    im = None
    for ax, label in zip(axes, models):
        M = matrices[label]
        im = ax.imshow(M, cmap=args.cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8)
        title_short = SHORT_LABELS.get(label, short_label(label))
        ax.set_title(title_short, fontsize=12, fontweight="bold", pad=18)
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color="#555555")
        ax.set_xticks(np.arange(-0.5, len(tasks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        # Annotate every cell with the Jaccard value
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                val = M[i, j]
                txt_color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

    axes[0].set_ylabel("Task A", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Task B", fontsize=10, labelpad=6)

    fig.subplots_adjust(left=0.08, right=0.88, wspace=0.25,
                        top=0.84, bottom=0.22)
    cbar_ax = fig.add_axes([0.905, 0.22, 0.014, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar_ax.set_title("Jaccard\noverlap", fontsize=9, pad=10, loc="left")

    fig.suptitle(
        f"Per-task top-{args.top_k} head Jaccard overlap "
        f"(per model, computed on the SEC detection set)",
        fontsize=12, y=0.97,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    fig_path = os.path.join(args.output_dir,
                            f"jaccard_topk{args.top_k}_heatmap.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")

    # ── Markdown summary report ────────────────────────────────────────────
    out: List[str] = []
    out.append(f"# Per-task top-{args.top_k} Jaccard overlap")
    out.append("")
    out.append("Top-K detected heads per task were loaded from each model's "
               "QRScore-SEC detection output (see *Provenance* below).")
    out.append("")
    out.append("## Provenance")
    out.append("")
    out.append("Per-task head files actually used for each model:")
    out.append("")
    rows = []
    for label, spec in inputs:
        _templates, ref = parse_input_spec(spec)
        ref_str = f"`{ref}`" if ref else "_working tree_"
        unique_used = sorted(set(resolved_paths[label].values()))
        for path in unique_used:
            n_tasks_using = sum(1 for p in resolved_paths[label].values() if p == path)
            rows.append([label, f"`{path}`", ref_str,
                         f"{n_tasks_using}/{len(tasks)} tasks"])
    out.append(md_table(
        ["Model", "Resolved file", "Git ref", "Coverage"], rows,
        aligns=["l", "l", "l", "l"],
    ))
    out.append("")
    out.append(f"Tasks ({len(tasks)}): " + ", ".join(f"`{t}`" for t in tasks))
    out.append("")

    # Per-model average pairwise off-diagonal Jaccard (skipping i==j == 1.0)
    out.append(f"## Per-model average off-diagonal Jaccard (top-{args.top_k})")
    out.append("")
    avg_rows = []
    for label in models:
        M = matrices[label]
        n = M.shape[0]
        off = [M[i, j] for i in range(n) for j in range(n) if i != j]
        avg_rows.append([
            label,
            f"{np.mean(off):.3f}",
            f"{np.min(off):.3f}",
            f"{np.max(off):.3f}",
        ])
    out.append(md_table(
        ["Model", "Mean", "Min", "Max"], avg_rows,
        aligns=["l", "r", "r", "r"],
    ))
    out.append("")
    out.append("> _The LaTeX states the average top-K Jaccard overlap is "
               "between 50% and 90%. Compare the **Mean** column above to "
               "verify._")
    out.append("")

    # Highlighted pair table (the 75-95% claim in results.tex)
    out.append(f"## Highlighted task pairs (top-{args.top_k})")
    out.append("")
    out.append("These are the pairs explicitly discussed in `results.tex` "
               "Section 4.4 as having 75-95% top-K overlap. The values below "
               "let a reviewer verify that claim directly.")
    out.append("")
    headers = ["Task A", "Task B"] + [SHORT_LABELS.get(m, short_label(m))
                                       for m in models]
    pair_rows = []
    for a, b in HIGHLIGHTED_PAIRS:
        if a not in tasks or b not in tasks:
            continue
        ia, ib = tasks.index(a), tasks.index(b)
        pair_rows.append(
            [f"`{a}`", f"`{b}`"] +
            [f"{matrices[m][ia, ib]:.3f}" for m in models]
        )
    out.append(md_table(headers, pair_rows,
                        aligns=["l", "l"] + ["r"] * len(models)))
    out.append("")

    # Full per-model matrices (for completeness)
    for label in models:
        out.append(f"## Full matrix — {label}")
        out.append("")
        M = matrices[label]
        headers = ["Task"] + tasks
        rows = []
        for i, a in enumerate(tasks):
            rows.append([a] + [f"{M[i, j]:.3f}" for j in range(len(tasks))])
        out.append(md_table(headers, rows))
        out.append("")

    md_path = os.path.join(args.output_dir,
                           f"jaccard_topk{args.top_k}_summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Saved summary: {md_path}")

    # Echo the headlines so the user sees them without opening a file.
    print()
    print("Per-model average off-diagonal Jaccard "
          f"(top-{args.top_k}, target: 50-90%):")
    for row in avg_rows:
        print(f"  {row[0]:<28} mean={row[1]} min={row[2]} max={row[3]}")
    print(f"\nHighlighted pairs (target: 75-95%):")
    for a, b in HIGHLIGHTED_PAIRS:
        if a not in tasks or b not in tasks:
            continue
        ia, ib = tasks.index(a), tasks.index(b)
        cells = "  ".join(
            f"{SHORT_LABELS.get(m, short_label(m))}={matrices[m][ia, ib]:.3f}"
            for m in models
        )
        print(f"  {a} ↔ {b}: {cells}")


if __name__ == "__main__":
    main()
