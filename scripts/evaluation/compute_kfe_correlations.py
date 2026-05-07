#!/usr/bin/env python3
"""K-fixed-effects (K-FE) cross-model correlations.

Replicates the analysis behind ``Table~\\ref{tab:kfe}`` in ``results.tex``::

    "computed by residualizing each measure against within-K task means
     before correlation."

For every model we build two long-form vectors over (task, K) cells:

    sensitivity[model][t][K]  (target-centric, column mean of drop matrix)
    efficacy[model][s][K]     (source-centric — see four candidate
                               definitions below)

We then K-residualize each vector::

    residual[i] = value[i] − mean over tasks at the same K of value

and compute the Pearson correlation between two models' residual vectors;
the K-FE R^2 is the squared Pearson r.

Because the original definition of "efficacy" wasn't preserved in code
anywhere in the repo, we evaluate four plausible source-centric measures
and report all of them. The one closest to the published values
``Llama-Qwen 0.02 / Llama-Mistral 0.04 / Qwen-Mistral 0.15`` is flagged
as the *most likely* definition the prior analysis used.

Definitions of efficacy[s][K] tested
------------------------------------
* ``on_target_drop``     drop[s][s][K]                          (diagonal)
* ``row_mean_drop``      mean over t of drop[s][t][K]           (full row)
* ``off_target_mean``    mean over t != s of drop[s][t][K]      (row no diag)
* ``specificity_index``  on_target_drop − off_target_mean

Outputs (``--output_dir``):

    kfe_table.csv           Long-form K-FE R^2 table.
    kfe_report.md           Human-readable Markdown report with comparison
                            against published values and a winning-definition
                            flag.

Usage:
    python scripts/evaluation/compute_kfe_correlations.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_target_sensitivity import (  # type: ignore  # noqa: E402
    DEFAULT_INPUTS,
    SHORT_LABELS,
    load_matrix,
    parse_input_spec,
    short_label,
    validate_matrix_shape,
)

DEFAULT_OUTPUT_DIR = "results/kfe_correlations"
DEFAULT_KS = (8, 16, 32, 48, 64, 96, 128)

# Published values from results.tex Table 1 (tab:kfe). The script reports
# absolute deviation from these and picks the closest efficacy definition.
PUBLISHED_R2 = {
    "sensitivity": {
        ("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"): 0.18,
        ("Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"): 0.47,
        ("Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"): 0.59,
    },
    "efficacy": {
        ("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"): 0.02,
        ("Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"): 0.04,
        ("Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"): 0.15,
    },
}


# ─── Vector builders ────────────────────────────────────────────────────────
def build_sensitivity_vector(matrix: dict, tasks: List[str], ks: List[int]
                             ) -> np.ndarray:
    """Return shape (|tasks|, |ks|): column mean of drop[s][t][K] over s."""
    sources = matrix["sources"]
    arr = np.zeros((len(tasks), len(ks)))
    for i, t in enumerate(tasks):
        for j, k in enumerate(ks):
            arr[i, j] = float(np.mean([
                matrix["results"][s][t]["by_k"][str(k)]["drop_from_k0"]
                for s in sources
            ]))
    return arr


def build_efficacy_vector(matrix: dict, tasks: List[str], ks: List[int],
                          definition: str) -> np.ndarray:
    """Return shape (|tasks|, |ks|): a source-centric statistic per (s, K)."""
    targets = matrix["targets"]
    arr = np.zeros((len(tasks), len(ks)))
    for i, s in enumerate(tasks):
        for j, k in enumerate(ks):
            cells = {
                t: matrix["results"][s][t]["by_k"][str(k)]["drop_from_k0"]
                for t in targets
            }
            on = cells[s]
            offs = [v for t, v in cells.items() if t != s]
            off_mean = float(np.mean(offs)) if offs else 0.0
            row_mean = float(np.mean(list(cells.values())))
            if definition == "on_target_drop":
                arr[i, j] = on
            elif definition == "row_mean_drop":
                arr[i, j] = row_mean
            elif definition == "off_target_mean":
                arr[i, j] = off_mean
            elif definition == "specificity_index":
                arr[i, j] = on - off_mean
            else:
                raise ValueError(f"Unknown efficacy definition: {definition!r}")
    return arr


def k_residualize(arr: np.ndarray) -> np.ndarray:
    """Subtract the within-K (column) mean from each entry."""
    col_means = arr.mean(axis=0, keepdims=True)
    return arr - col_means


def pearson_r2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom == 0:
        return float("nan")
    r = float((a_c * b_c).sum() / denom)
    return r * r


# ─── Reporting ──────────────────────────────────────────────────────────────
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


def fmt(v: Optional[float], prec: int = 3) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:.{prec}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="*", default=None,
                   help='List of "label=path[@gitref]" entries; defaults to '
                        'the canonical Llama (origin/main) / Qwen / Mistral '
                        'matrices from compute_target_sensitivity.py.')
    p.add_argument("--ks", default=None,
                   help="Comma-separated K values to include (default: "
                        "8,16,32,48,64,96,128). K=0 is always excluded "
                        "because every entry is zero by construction.")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--repo_root", default=None)
    args = p.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))

    if args.inputs:
        inputs: List[Tuple[str, str]] = []
        for entry in args.inputs:
            if "=" not in entry:
                sys.exit(f"--inputs entry missing 'label=': {entry!r}")
            label, spec = entry.split("=", 1)
            inputs.append((label, spec))
    else:
        inputs = DEFAULT_INPUTS

    if args.ks:
        ks = [int(x) for x in args.ks.split(",") if int(x) != 0]
    else:
        ks = list(DEFAULT_KS)

    print("Loading transfer matrices:")
    matrices: Dict[str, dict] = {}
    for label, spec in inputs:
        m = load_matrix(spec, repo_root)
        validate_matrix_shape(label, m)
        matrices[label] = m
        path, ref = parse_input_spec(spec)
        prov = f"{path}" + (f" @ {ref}" if ref else " (working tree)")
        print(f"  {label:<28} <- {prov}")

    # Assert task lists match
    task_lists = [sorted(m["targets"]) for m in matrices.values()]
    if any(tl != task_lists[0] for tl in task_lists):
        sys.exit(f"Target task lists differ across models: {task_lists}")
    tasks = task_lists[0]
    models = list(matrices.keys())

    # ── Build all vectors ──────────────────────────────────────────────────
    sens_vecs = {m: build_sensitivity_vector(matrices[m], tasks, ks)
                 for m in models}
    eff_defs = ["on_target_drop", "row_mean_drop", "off_target_mean",
                "specificity_index"]
    eff_vecs = {
        d: {m: build_efficacy_vector(matrices[m], tasks, ks, d)
            for m in models}
        for d in eff_defs
    }

    # ── Compute K-FE R^2 for every model pair ──────────────────────────────
    pairs = list(combinations(models, 2))
    sens_r2: Dict[Tuple[str, str], float] = {}
    for a, b in pairs:
        sens_r2[(a, b)] = pearson_r2(k_residualize(sens_vecs[a]),
                                     k_residualize(sens_vecs[b]))
    eff_r2: Dict[str, Dict[Tuple[str, str], float]] = {d: {} for d in eff_defs}
    for d in eff_defs:
        for a, b in pairs:
            eff_r2[d][(a, b)] = pearson_r2(k_residualize(eff_vecs[d][a]),
                                           k_residualize(eff_vecs[d][b]))

    # ── Decide which efficacy definition best matches published values ────
    def total_dev(r2_dict: Dict[Tuple[str, str], float],
                  published: Dict[Tuple[str, str], float]) -> float:
        dev = 0.0
        for pair, ref in published.items():
            if pair not in r2_dict:
                # Try reversed key (sort-order independence)
                rev = (pair[1], pair[0])
                if rev in r2_dict:
                    pair = rev
                else:
                    return float("inf")
            dev += abs(r2_dict[pair] - ref)
        return dev

    deviations = {d: total_dev(eff_r2[d], PUBLISHED_R2["efficacy"])
                  for d in eff_defs}
    best_eff_def = min(deviations, key=deviations.get)
    sens_dev = total_dev(sens_r2, PUBLISHED_R2["sensitivity"])

    # ── Write CSV ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "kfe_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", "definition", "model_a", "model_b",
                    "kfe_r2_derived", "kfe_r2_published", "abs_dev"])
        for (a, b), v in sens_r2.items():
            ref = PUBLISHED_R2["sensitivity"].get((a, b)) or \
                  PUBLISHED_R2["sensitivity"].get((b, a))
            w.writerow(["sensitivity", "column_mean_drop", a, b,
                        f"{v:.6f}", fmt(ref),
                        f"{abs(v - ref):.6f}" if ref is not None else ""])
        for d in eff_defs:
            for (a, b), v in eff_r2[d].items():
                ref = PUBLISHED_R2["efficacy"].get((a, b)) or \
                      PUBLISHED_R2["efficacy"].get((b, a))
                w.writerow(["efficacy", d, a, b,
                            f"{v:.6f}", fmt(ref),
                            f"{abs(v - ref):.6f}" if ref is not None else ""])

    # ── Write Markdown report ──────────────────────────────────────────────
    out: List[str] = []
    out.append("# K-FE cross-model correlation report")
    out.append("")
    out.append("## Method")
    out.append("")
    out.append("Replicates `Table 1` of `results.tex`. For every model we "
               "build a `(|tasks|, |K|) = (8, 7)` matrix, K-residualize "
               "(subtract the within-K mean across tasks from each entry), "
               "flatten to a 56-element vector, and report the squared "
               "Pearson correlation between two models' residuals.")
    out.append("")
    out.append("Two measures are computed:")
    out.append("")
    out.append("- **Sensitivity** (target-centric, unambiguous): "
               "`sensitivity[t][K] = mean over sources s of drop[s][t][K]`")
    out.append("- **Efficacy** (source-centric, ambiguous): four candidate "
               "definitions are computed because the original analysis "
               "code wasn't preserved in this repo:")
    out.append("    - `on_target_drop`    = `drop[s][s][K]`")
    out.append("    - `row_mean_drop`     = mean over `t` of `drop[s][t][K]`")
    out.append("    - `off_target_mean`   = mean over `t ≠ s` of `drop[s][t][K]`")
    out.append("    - `specificity_index` = `on_target_drop − off_target_mean`")
    out.append("")
    out.append("Each derived value is compared to the value printed in "
               "`results.tex` Table 1, and the efficacy definition with the "
               "smallest total absolute deviation is flagged as the best fit.")
    out.append("")
    out.append(f"K values used: {ks}  (K=0 always excluded — its drop is 0 "
               "by construction).")
    out.append("")
    out.append("## Sensitivity K-FE R²")
    out.append("")
    headers = ["Model A", "Model B", "Derived", "Published (`results.tex`)",
               "|Δ|"]
    rows = []
    for (a, b), v in sens_r2.items():
        ref = PUBLISHED_R2["sensitivity"].get((a, b))
        rows.append([SHORT_LABELS.get(a, short_label(a)),
                     SHORT_LABELS.get(b, short_label(b)),
                     f"**{v:.3f}**",
                     fmt(ref),
                     f"{abs(v - ref):.3f}" if ref is not None else "—"])
    out.append(md_table(headers, rows, aligns=["l", "l", "r", "r", "r"]))
    out.append("")
    out.append(f"Total |Δ| (sensitivity): **{sens_dev:.3f}**")
    out.append("")

    out.append("## Efficacy K-FE R² — four candidate definitions")
    out.append("")
    out.append("Smaller total |Δ| is closer to the published Table 1.")
    out.append("")
    summary_rows = []
    for d in eff_defs:
        marker = " ← **best fit**" if d == best_eff_def else ""
        summary_rows.append([f"`{d}`{marker}", f"{deviations[d]:.3f}"])
    out.append(md_table(["Definition", "Total |Δ| vs published"],
                        summary_rows, aligns=["l", "r"]))
    out.append("")
    for d in eff_defs:
        title = f"### efficacy = `{d}`"
        if d == best_eff_def:
            title += "  — **best fit**"
        out.append(title)
        out.append("")
        rows = []
        for (a, b), v in eff_r2[d].items():
            ref = PUBLISHED_R2["efficacy"].get((a, b))
            rows.append([SHORT_LABELS.get(a, short_label(a)),
                         SHORT_LABELS.get(b, short_label(b)),
                         f"**{v:.3f}**",
                         fmt(ref),
                         f"{abs(v - ref):.3f}" if ref is not None else "—"])
        out.append(md_table(headers, rows,
                            aligns=["l", "l", "r", "r", "r"]))
        out.append("")

    out.append("## Verdict")
    out.append("")
    if best_eff_def is not None:
        out.append(f"- Best-matching efficacy definition: **`{best_eff_def}`** "
                   f"(total |Δ| = {deviations[best_eff_def]:.3f}).")
    if sens_dev < 0.05:
        out.append(f"- Sensitivity values reproduce the published table to "
                   f"within total |Δ| = {sens_dev:.3f}. **Confirmed.**")
    else:
        out.append(f"- Sensitivity values **disagree** with the published "
                   f"table by total |Δ| = {sens_dev:.3f}. The numbers in "
                   f"`results.tex` Table 1 should be replaced with the "
                   f"`Derived` column above.")
    if deviations[best_eff_def] < 0.05:
        out.append(f"- Best efficacy fit reproduces the published table to "
                   f"within {deviations[best_eff_def]:.3f}. **Confirmed.**")
    else:
        out.append(f"- Best efficacy fit deviates from the published table "
                   f"by total |Δ| = {deviations[best_eff_def]:.3f}. Either "
                   f"the original definition is none of the four tested, "
                   f"or the Table 1 values themselves should be replaced.")

    md_path = os.path.join(args.output_dir, "kfe_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    # ── Echo headlines ─────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("K-FE R² results vs published Table 1")
    print("=" * 78)
    print(f"\n[sensitivity] derived → published   total |Δ| = {sens_dev:.3f}")
    for (a, b), v in sens_r2.items():
        ref = PUBLISHED_R2["sensitivity"].get((a, b))
        print(f"  {short_label(a):<8} ↔ {short_label(b):<8}: "
              f"derived={v:.3f}  published={fmt(ref)}  |Δ|={abs(v - ref):.3f}")
    print(f"\n[efficacy] best definition: {best_eff_def}  "
          f"(total |Δ| = {deviations[best_eff_def]:.3f})")
    for d in eff_defs:
        print(f"\n  efficacy = {d}  (|Δ|={deviations[d]:.3f})")
        for (a, b), v in eff_r2[d].items():
            ref = PUBLISHED_R2["efficacy"].get((a, b))
            print(f"    {short_label(a):<8} ↔ {short_label(b):<8}: "
                  f"derived={v:.3f}  published={fmt(ref)}  "
                  f"|Δ|={abs(v - ref):.3f}")
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
