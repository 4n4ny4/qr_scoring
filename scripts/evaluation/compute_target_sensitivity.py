#!/usr/bin/env python3
"""Target-task sensitivity from cross_task_transfer_matrix.json files.

Distinct from the existing source-centric specificity metrics (see
``run_ablation.compute_specificity_metrics`` and ``FINDINGS.md``). Both views
are valid; this script implements the *target* view explicitly.

Definitions (all per (model, target_task t, knockout K)):
    sensitivity[t][K]      = mean over sources s of drop[s][t][K]
                             (i.e. average accuracy loss on task t when
                             ablating heads detected for *any* source task,
                             including s==t).
    on_target_drop[t][K]   = drop[t][t][K]
                             (loss on t when ablating t's own heads).
    off_target_mean[t][K]  = mean over s != t of drop[s][t][K]
                             (mean collateral loss on t from ablating other
                             tasks' heads).
    baseline_acc[t]        = accuracy[s=t][t][K=0]   (must be K-invariant
                             across all s; the script asserts this).

The matrix files this script reads are produced by
``scripts/evaluation/run_ablation.py`` (search for ``cross_task_transfer_matrix``).

Each cell of ``cross_task_transfer_matrix.json`` looks like::

    results[source][target]["by_k"][str(K)] = {
        "accuracy": <float>,
        "drop_from_k0": <float>,        # accuracy[K=0] - accuracy[K]
    }

Provenance
----------
Inputs may live on disk *or* on a different git ref. Specify either as a
plain path, or as ``path@<gitref>``; in the latter case the matrix is
fetched with ``git show <gitref>:<path>``. The default config below uses
``origin/main`` for Llama because that branch is the canonical source for
the published Llama numbers (it was confirmed identical to the matrices on
``ananya/cross_ablations`` and ``ananya/gamma_olmo``).

Sanity check
------------
``--validate`` (on by default) re-derives the *source*-centric
``specificity_index`` from the matrix and compares it cell-for-cell to any
``cross_task_specificity_metrics.json`` file sitting next to each input
matrix. Mismatches abort the script with an explicit diff.

Usage
-----
Run with no arguments to produce the default Llama / Qwen / Mistral report::

    python scripts/evaluation/compute_target_sensitivity.py

Or override the inputs::

    python scripts/evaluation/compute_target_sensitivity.py \
        --inputs \
            "Llama-3.1-8B-Instruct=results/comparison_ablation/cross_task_transfer_matrix.json@origin/main" \
            "Qwen2.5-7B-Instruct=results/runs/qwen_true_detect_2026-04-07/cross_task_transfer_matrix.json" \
            "Mistral-7B-Instruct-v0.3=results/comparison_ablation_mistral/cross_task_transfer_matrix.json" \
        --output_dir results/target_sensitivity
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ─── Default models / paths ──────────────────────────────────────────────────
# Each entry is (label, "<path>[@<gitref>]"). Paths are resolved relative to
# the repo root. ``@<gitref>`` triggers ``git show <gitref>:<path>``.
DEFAULT_INPUTS: List[Tuple[str, str]] = [
    ("Llama-3.1-8B-Instruct",
     "results/comparison_ablation/cross_task_transfer_matrix.json@origin/main"),
    ("Qwen2.5-7B-Instruct",
     "results/runs/qwen_true_detect_2026-04-07/cross_task_transfer_matrix.json"),
    ("Mistral-7B-Instruct-v0.3",
     "results/comparison_ablation_mistral/cross_task_transfer_matrix.json"),
]

# Map full model labels to short column headers used in report tables.
SHORT_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama",
    "Qwen2.5-7B-Instruct": "Qwen",
    "Mistral-7B-Instruct-v0.3": "Mistral",
}

DEFAULT_OUTPUT_DIR = "results/target_sensitivity"
EARLY_KS = (8, 16)
EPS = 1e-9


def short_label(model: str) -> str:
    if model in SHORT_LABELS:
        return SHORT_LABELS[model]
    # Heuristic fallback: take everything before the first hyphen
    return model.split("-")[0]


# ─── Loading ─────────────────────────────────────────────────────────────────
def parse_input_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split ``path[@gitref]`` into (path, gitref|None)."""
    if "@" in spec:
        path, ref = spec.split("@", 1)
        return path, ref
    return spec, None


def load_matrix(spec: str, repo_root: str) -> dict:
    path, ref = parse_input_spec(spec)
    if ref:
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{ref}:{path}"],
                cwd=repo_root,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            raise FileNotFoundError(
                f"git show {ref}:{path} failed: {exc.stderr.decode().strip()}"
            ) from exc
        return json.loads(blob)
    abs_path = path if os.path.isabs(path) else os.path.join(repo_root, path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Matrix not found: {abs_path}")
    with open(abs_path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_blob(path: str, ref: Optional[str], repo_root: str) -> Optional[bytes]:
    """Return file bytes from disk or git ref; None if missing."""
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


def load_sibling_specificity(spec: str, repo_root: str) -> Optional[dict]:
    """Load ``cross_task_specificity_metrics.json`` next to the matrix, if any.

    Falls back to the CSV form (``specificity_table.csv``) found on
    ``origin/main`` for Llama. Returns a dict with the same shape as the JSON
    file (so the rest of the validator is format-agnostic).
    """
    path, ref = parse_input_spec(spec)
    base_dir = os.path.dirname(path)

    json_path = os.path.join(base_dir, "cross_task_specificity_metrics.json")
    blob = _read_blob(json_path, ref, repo_root)
    if blob is not None:
        return json.loads(blob)

    csv_path = os.path.join(base_dir, "specificity_table.csv")
    blob = _read_blob(csv_path, ref, repo_root)
    if blob is None:
        return None
    # The CSV has no explicit summary_k. The Llama numbers on origin/main
    # were generated at K=16 (the same default used by run_ablation.py).
    rows = list(csv.reader(blob.decode().splitlines()))
    if not rows or rows[0][0].strip().lower() != "source task":
        return None
    sources: dict = {}
    for r in rows[1:]:
        if not r:
            continue
        src, on, off, spec_idx, surg = r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4])
        sources[src] = {
            "on_target_drop": on,
            "off_target_mean_drop": off,
            "specificity_index": spec_idx,
            "surgicality_ratio": surg,
        }
    return {"summary_k": 16, "sources": sources, "_source_format": "csv"}


# ─── Matrix sanity ───────────────────────────────────────────────────────────
def validate_matrix_shape(label: str, m: dict) -> None:
    sources, targets, ks = m["sources"], m["targets"], m["knockout_sizes"]
    if sorted(sources) != sorted(targets):
        raise ValueError(f"[{label}] sources != targets ({sources} vs {targets})")
    for s in sources:
        if s not in m["results"]:
            raise ValueError(f"[{label}] missing source {s}")
        for t in targets:
            if t not in m["results"][s]:
                raise ValueError(f"[{label}] missing target {t} for source {s}")
            for k in ks:
                cell = m["results"][s][t]["by_k"].get(str(k))
                if cell is None or "accuracy" not in cell or "drop_from_k0" not in cell:
                    raise ValueError(f"[{label}] bad cell {s}->{t} K={k}: {cell}")
    # Baseline (K=0) for target t should be the same regardless of source
    # (the model is unablated). Asserting catches accidental schema drift.
    for t in targets:
        baselines = {round(m["results"][s][t]["by_k"]["0"]["accuracy"], 6)
                     for s in sources}
        if len(baselines) != 1:
            raise ValueError(
                f"[{label}] target {t} has inconsistent K=0 accuracy across "
                f"sources: {baselines}"
            )


# ─── Core metrics ────────────────────────────────────────────────────────────
def compute_target_metrics(matrix: dict) -> dict:
    """Return {target: {K: {sensitivity, on_target, off_target_mean, mean_acc, baseline_acc}}}."""
    sources = matrix["sources"]
    targets = matrix["targets"]
    ks = matrix["knockout_sizes"]
    out: dict = {t: {} for t in targets}
    for t in targets:
        baseline = matrix["results"][sources[0]][t]["by_k"]["0"]["accuracy"]
        for k in ks:
            drops, accs = [], []
            on, offs = None, []
            for s in sources:
                cell = matrix["results"][s][t]["by_k"][str(k)]
                drops.append(cell["drop_from_k0"])
                accs.append(cell["accuracy"])
                if s == t:
                    on = cell["drop_from_k0"]
                else:
                    offs.append(cell["drop_from_k0"])
            out[t][int(k)] = {
                "sensitivity": sum(drops) / len(drops),
                "on_target_drop": on,
                "off_target_mean_drop": (sum(offs) / len(offs)) if offs else None,
                "mean_accuracy": sum(accs) / len(accs),
                "baseline_accuracy": baseline,
            }
    return out


def compute_source_specificity(matrix: dict) -> dict:
    """Re-derive source-centric specificity (mirrors run_ablation.compute_specificity_metrics)."""
    sources = matrix["sources"]
    targets = matrix["targets"]
    ks = matrix["knockout_sizes"]
    out: dict = {s: {} for s in sources}
    for s in sources:
        for k in ks:
            on = None
            offs = []
            for t in targets:
                drop = matrix["results"][s][t]["by_k"][str(k)]["drop_from_k0"]
                if s == t:
                    on = drop
                else:
                    offs.append(drop)
            off_mean = sum(offs) / len(offs) if offs else 0.0
            out[s][int(k)] = {
                "on_target_drop": on,
                "off_target_mean_drop": off_mean,
                "specificity_index": (None if on is None else on - off_mean),
                "surgicality_ratio": (None if on is None else on / max(off_mean, EPS)),
            }
    return out


# ─── Cross-validation against existing artifacts ─────────────────────────────
def cross_validate(label: str, matrix: dict, sibling: Optional[dict],
                   atol: Optional[float] = None) -> List[str]:
    """Return list of diff strings; empty list means everything matches."""
    diffs: List[str] = []
    if sibling is None:
        return diffs
    summary_k = sibling.get("summary_k")
    if summary_k is None:
        return [f"[{label}] sibling spec file has no 'summary_k' key"]
    if atol is None:
        # CSV files only have 4 decimal places; allow a half-unit tolerance.
        atol = 5e-4 if sibling.get("_source_format") == "csv" else 1e-6
    derived = compute_source_specificity(matrix)
    for src, ref_vals in sibling["sources"].items():
        if src not in derived or summary_k not in derived[src]:
            diffs.append(f"[{label}] missing src/K {src}/{summary_k} in derived")
            continue
        ours = derived[src][summary_k]
        for key in ("on_target_drop", "off_target_mean_drop",
                    "specificity_index", "surgicality_ratio"):
            ref = ref_vals.get(key)
            mine = ours.get(key)
            if ref is None or mine is None:
                continue
            if abs(ref - mine) > atol:
                diffs.append(
                    f"[{label}] {src} K={summary_k} {key}: "
                    f"existing={ref:.6f} derived={mine:.6f} delta={ref - mine:+.2e}"
                )
    return diffs


# ─── Reporting ───────────────────────────────────────────────────────────────
def fmt(val: Optional[float], width: int = 8, prec: int = 3) -> str:
    if val is None:
        return f"{'--':>{width}}"
    return f"{val:>{width}.{prec}f}"


def md_cell(val: Optional[float], prec: int = 3) -> str:
    return "—" if val is None else f"{val:.{prec}f}"


def md_table(headers: List[str], rows: List[List[str]],
             aligns: Optional[List[str]] = None) -> str:
    """Render a GitHub-flavored markdown table.

    `aligns[i]` ∈ {"l","c","r"}; defaults to left for the first column and
    right for the rest.
    """
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    sep_map = {"l": ":---", "c": ":---:", "r": "---:"}
    sep = [sep_map[a] for a in aligns]
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_long_csv(path: str, model_metrics: Dict[str, dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "target_task", "K",
            "sensitivity_mean_drop",
            "on_target_drop",
            "off_target_mean_drop",
            "mean_accuracy",
            "baseline_accuracy",
        ])
        for model, per_target in model_metrics.items():
            for tgt in sorted(per_target):
                for k in sorted(per_target[tgt]):
                    cell = per_target[tgt][k]
                    w.writerow([
                        model, tgt, k,
                        cell["sensitivity"],
                        cell["on_target_drop"],
                        cell["off_target_mean_drop"],
                        cell["mean_accuracy"],
                        cell["baseline_accuracy"],
                    ])


def write_nested_json(path: str, model_metrics: Dict[str, dict]) -> None:
    serialisable = {
        m: {t: {str(k): v for k, v in ks.items()} for t, ks in per_target.items()}
        for m, per_target in model_metrics.items()
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)


def _bold_first(rows: List[List[str]]) -> List[List[str]]:
    """Bold every cell of the first row (used to highlight the top entry)."""
    if not rows:
        return rows
    rows[0] = [f"**{c}**" for c in rows[0]]
    return rows


def emit_per_k_table(out, k: int, models: List[str],
                     model_metrics: Dict[str, dict], task_list: List[str]) -> None:
    out.append("")
    out.append(f"### K = {k}")
    out.append("")
    rows_data = []
    for t in task_list:
        vals = []
        for m in models:
            cell = model_metrics[m].get(t, {}).get(k)
            vals.append(cell["sensitivity"] if cell else None)
        non_none = [v for v in vals if v is not None]
        mn = sum(non_none) / len(non_none) if non_none else None
        rows_data.append((t, vals, mn))
    rows_data.sort(key=lambda r: -(r[2] if r[2] is not None else 0))
    headers = ["Rank", "Target task"] + [short_label(m) for m in models] + ["Mean"]
    rows = [[str(i + 1), t] + [md_cell(v) for v in vals] + [md_cell(mn)]
            for i, (t, vals, mn) in enumerate(rows_data)]
    out.append(md_table(headers, _bold_first(rows)))


def emit_aggregate_table(out, title: str, ks: List[int], models: List[str],
                         model_metrics: Dict[str, dict], task_list: List[str]) -> None:
    out.append("")
    out.append(f"## {title}")
    out.append("")
    out.append(f"_Mean over K in {{{', '.join(str(k) for k in ks)}}}._")
    out.append("")
    rows_data = []
    for t in task_list:
        vals = []
        for m in models:
            per_k = model_metrics[m].get(t, {})
            kvals = [per_k[k]["sensitivity"] for k in ks if k in per_k]
            vals.append(sum(kvals) / len(kvals) if kvals else None)
        non_none = [v for v in vals if v is not None]
        mn = sum(non_none) / len(non_none) if non_none else None
        rows_data.append((t, vals, mn))
    rows_data.sort(key=lambda r: -(r[2] if r[2] is not None else 0))
    headers = ["Rank", "Target task"] + [short_label(m) for m in models] + ["Mean"]
    rows = [[str(i + 1), t] + [md_cell(v) for v in vals] + [md_cell(mn)]
            for i, (t, vals, mn) in enumerate(rows_data)]
    out.append(md_table(headers, _bold_first(rows)))


def emit_rank_tables(out, ks: List[int], models: List[str],
                     model_metrics: Dict[str, dict], task_list: List[str]) -> None:
    out.append("")
    out.append("## Sensitivity rank per model")
    out.append("")
    out.append("_Rank 1 = most sensitive target task for that (model, K). "
               "Each column is independent — rows are not sorted, just listed alphabetically._")
    for k in ks:
        out.append("")
        out.append(f"### K = {k}")
        out.append("")
        rank_per_model = {}
        for m in models:
            scored = [(t, model_metrics[m].get(t, {}).get(k, {}).get("sensitivity") or 0.0)
                      for t in task_list]
            scored.sort(key=lambda x: -x[1])
            rank_per_model[m] = {t: i + 1 for i, (t, _) in enumerate(scored)}
        headers = ["Target task"] + [short_label(m) for m in models]
        rows = [[t] + [str(rank_per_model[m][t]) for m in models] for t in task_list]
        out.append(md_table(headers, rows))


def emit_topk_consistency(out, ks: List[int], models: List[str],
                          model_metrics: Dict[str, dict], task_list: List[str]) -> None:
    cells = len(ks) * len(models)
    out.append("")
    out.append("## Consistency across (K, model) cells")
    out.append("")
    out.append(f"_How often each task is in the top-N most sensitive across all "
               f"{len(ks)} K-values × {len(models)} models = **{cells}** cells._")
    out.append("")
    top2: defaultdict = defaultdict(int)
    top4: defaultdict = defaultdict(int)
    for k in ks:
        for m in models:
            scored = [(t, model_metrics[m].get(t, {}).get(k, {}).get("sensitivity") or 0.0)
                      for t in task_list]
            scored.sort(key=lambda x: -x[1])
            for i, (t, _) in enumerate(scored):
                if i < 2:
                    top2[t] += 1
                if i < 4:
                    top4[t] += 1
    headers = ["Target task", f"Top-2 / {cells}", f"Top-4 / {cells}"]
    rows = [[t, str(top2[t]), str(top4[t])]
            for t in sorted(task_list, key=lambda x: -top4[x])]
    out.append(md_table(headers, _bold_first(rows)))


def build_worked_example(matrices: Dict[str, dict], models: List[str]
                         ) -> Optional[Tuple[str, str, int, List[Tuple[str, float]], dict]]:
    """Pick a concrete (model, target, K) example, prefer Mistral if available.

    Returns (model_label, target_task, K, [(source, drop), ...], metrics_dict)
    or None if no matrix is loaded.
    """
    if not models:
        return None
    preferred = next((m for m in models if "mistral" in m.lower()), models[0])
    matrix = matrices[preferred]
    if "employees_count_total" in matrix["targets"]:
        target = "employees_count_total"
    else:
        target = matrix["targets"][0]
    ks = [k for k in matrix["knockout_sizes"] if k != 0]
    k = 16 if 16 in ks else (ks[0] if ks else 0)
    rows = [(s, matrix["results"][s][target]["by_k"][str(k)]["drop_from_k0"])
            for s in matrix["sources"]]
    drops = [d for _, d in rows]
    on = next(d for s, d in rows if s == target)
    offs = [d for s, d in rows if s != target]
    metrics = {
        "sensitivity": sum(drops) / len(drops),
        "on_target_drop": on,
        "off_target_mean": sum(offs) / len(offs) if offs else None,
    }
    return preferred, target, k, rows, metrics


def emit_provenance_table(out, inputs: List[Tuple[str, str]],
                          siblings: Dict[str, Optional[dict]]) -> None:
    out.append("## Data provenance & validation")
    out.append("")
    headers = ["Model", "Matrix path", "Git ref", "Validated against", "Status"]
    rows = []
    for label, spec in inputs:
        path, ref = parse_input_spec(spec)
        sib = siblings.get(label)
        if sib is None:
            validated = "—"
            status = "no sibling spec file"
        else:
            fmt_src = sib.get("_source_format", "json")
            fname = "specificity_table.csv" if fmt_src == "csv" else "cross_task_specificity_metrics.json"
            validated = f"`{fname}` (K={sib.get('summary_k')})"
            status = "OK"
        rows.append([label, f"`{path}`", f"`{ref}`" if ref else "_working tree_",
                     validated, status])
    out.append(md_table(headers, rows,
                        aligns=["l", "l", "l", "l", "l"]))


# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inputs", nargs="*", default=None,
        help='List of "label=path[@gitref]" entries. Defaults to the canonical '
             'Llama (origin/main) / Qwen / Mistral matrices.',
    )
    parser.add_argument(
        "--ks", default=None,
        help="Comma-separated K values to include. Default: every K>0 present "
             "in every matrix.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip cross-validation against sibling specificity files.")
    parser.add_argument("--repo_root", default=None,
                        help="Repo root for resolving paths and running git show.")
    args = parser.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))

    if args.inputs:
        inputs: List[Tuple[str, str]] = []
        for entry in args.inputs:
            if "=" not in entry:
                parser.error(f"--inputs entry missing 'label=': {entry!r}")
            label, spec = entry.split("=", 1)
            inputs.append((label, spec))
    else:
        inputs = DEFAULT_INPUTS

    print("=" * 78)
    print("Loading transfer matrices")
    print("=" * 78)
    matrices: Dict[str, dict] = {}
    siblings: Dict[str, Optional[dict]] = {}
    for label, spec in inputs:
        m = load_matrix(spec, repo_root)
        validate_matrix_shape(label, m)
        matrices[label] = m
        siblings[label] = load_sibling_specificity(spec, repo_root)
        path, ref = parse_input_spec(spec)
        provenance = f"{path}" + (f" @ {ref}" if ref else " (working tree)")
        print(f"  {label:<28} <- {provenance}")
        print(f"      sources={len(m['sources'])} targets={len(m['targets'])} ks={m['knockout_sizes']}")

    if not args.no_validate:
        print("\n" + "=" * 78)
        print("Cross-validating derived metrics against existing specificity files")
        print("=" * 78)
        all_diffs = []
        for label in matrices:
            sib = siblings[label]
            if sib is None:
                print(f"  {label}: no sibling cross_task_specificity_metrics.json — skipped")
                continue
            diffs = cross_validate(label, matrices[label], sib)
            if diffs:
                print(f"  {label}: MISMATCH ({len(diffs)} diffs)")
                for d in diffs[:10]:
                    print(f"    {d}")
                all_diffs.extend(diffs)
            else:
                print(f"  {label}: OK (matches existing summary_k={sib.get('summary_k')})")
        if all_diffs:
            sys.exit("\nAborting: derived metrics disagree with existing specificity files.")

    # Compute target-centric sensitivity per model
    model_metrics: Dict[str, dict] = {
        label: compute_target_metrics(m) for label, m in matrices.items()
    }

    # Build common axes
    task_lists = [sorted(m["targets"]) for m in matrices.values()]
    if any(tl != task_lists[0] for tl in task_lists):
        sys.exit(f"Target task lists differ across models: {task_lists}")
    task_list = task_lists[0]

    if args.ks:
        ks = [int(x) for x in args.ks.split(",")]
    else:
        ks_sets = [set(m["knockout_sizes"]) for m in matrices.values()]
        common = sorted(set.intersection(*ks_sets) - {0})
        ks = common
    models = list(matrices.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    # Persist machine-readable artifacts
    long_csv = os.path.join(args.output_dir, "target_sensitivity_long.csv")
    write_long_csv(long_csv, model_metrics)
    nested_json = os.path.join(args.output_dir, "target_sensitivity.json")
    write_nested_json(nested_json, model_metrics)

    # Build report
    out: List[str] = []
    out.append("# Target-Task Sensitivity Report")
    out.append("")
    out.append("## Definitions")
    out.append("")
    out.append("### The cross-task transfer experiment in plain English")
    out.append("")
    out.append("For every model, the QRScore detection step produced a *ranked list of attention "
               "heads* for **each of the 8 SEC tasks** (e.g. `ceo_lastname`, `headquarters_city`, …). "
               "We then run a cross-task ablation:")
    out.append("")
    out.append("> *For every (source task, target task) pair, take the top-K heads detected for "
               "the source task, zero them out (\"ablate\" them), and measure how much accuracy "
               "drops on the target task's test set.*")
    out.append("")
    out.append("That gives an `8 × 8 × |K|` cube of accuracies, stored in "
               "`cross_task_transfer_matrix.json`.")
    out.append("")
    out.append("### Variables used in the formulas")
    out.append("")
    out.append("| Symbol | Type | Meaning | Example value |")
    out.append("| :--- | :--- | :--- | :--- |")
    out.append("| `s` | task name | **Source task** — the task whose top-K detected heads we ablate. | `ceo_lastname` |")
    out.append("| `t` | task name | **Target task** — the task whose accuracy we *measure* after the ablation. | `employees_count_total` |")
    out.append("| `K` | int ≥ 0 | **Knockout size** — how many top-ranked heads we ablate. `K=0` = unablated baseline. | `16` |")
    out.append("| `accuracy[s][t][K]` | float in [0,1] | Accuracy on target `t`'s test set when source `s`'s top-K heads are ablated. | `0.792` |")
    out.append("| `drop[s][t][K]` | float in [-1,1] | Accuracy lost vs the unablated baseline on `t`. Positive = the ablation hurt `t`. | `0.125` |")
    out.append("")
    out.append("> When `s == t` the model is being asked to do task `t` with `t`'s own detected "
               "heads removed — this is the **diagonal** of the source × target heatmaps.")
    out.append("> When `s != t` the model is doing task `t` with some *other* task's heads "
               "removed — collateral damage, **off-diagonal**.")
    out.append("")
    out.append("### Metrics computed in this report")
    out.append("")
    out.append("| Metric | Formula | What it asks |")
    out.append("| :--- | :--- | :--- |")
    out.append("| `drop[s][t][K]` | `accuracy[s][t][K=0] − accuracy[s][t][K]` | How much does ablating `s`'s top-K heads hurt task `t`? |")
    out.append("| **`sensitivity[t][K]`** | mean over **all sources `s`** of `drop[s][t][K]` (8 sources, **including** `s = t`) | **How fragile is target task `t` overall?** Averages the whole `t`-th column of the 8×8 drop heatmap. |")
    out.append("| `on_target_drop[t][K]` | `drop[t][t][K]`  *(single diagonal cell)* | How much does task `t` suffer when its **own** heads are removed? |")
    out.append("| `off_target_mean[t][K]` | mean over `s ≠ t` of `drop[s][t][K]` (7 sources) | How much does task `t` suffer purely from **other tasks'** ablations (collateral)? |")
    out.append("")
    out.append("### Worked example (real numbers, recomputed at runtime)")
    out.append("")
    example = build_worked_example(matrices, models)
    if example is not None:
        ex_model, ex_target, ex_k, ex_rows, ex_metrics = example
        out.append(f"Take **`t = {ex_target}`**, **`K = {ex_k}`**, **model = {ex_model}**. "
                   "The 8 cells of that drop *column* in the cross-task matrix are:")
        out.append("")
        out.append("| Source `s` | `drop[s][t][K]` | Note |")
        out.append("| :--- | ---: | :--- |")
        for src, drop_val in ex_rows:
            note = "← `on_target_drop` (s == t, diagonal)" if src == ex_target else ""
            out.append(f"| `{src}` | {drop_val:.4f} | {note} |")
        out.append("")
        out.append("So for that (model, target, K):")
        out.append("")
        out.append(f"- `sensitivity      = mean of all 8`              = **{ex_metrics['sensitivity']:.4f}**")
        out.append(f"- `on_target_drop   = drop[t][t][K]`              = **{ex_metrics['on_target_drop']:.4f}**")
        out.append(f"- `off_target_mean  = mean of the other 7 (s ≠ t)` = **{ex_metrics['off_target_mean']:.4f}**")
        out.append("")
    out.append("> **`sensitivity` is the column average of the cross-task drop heatmap; "
               "`on_target_drop` is its diagonal; `off_target_mean` is the column average "
               "with the diagonal cell removed.**")
    out.append("")
    out.append("### Why this is *target-centric* (and how it differs from the existing source-centric `specificity_index`)")
    out.append("")
    out.append("The repo's `cross_task_specificity_metrics.json` and `FINDINGS.md` use a "
               "**source-centric** view (one row of the 8×8 heatmap per source `s`):")
    out.append("")
    out.append("```text")
    out.append("source-centric on_target_drop[s]  = drop[s][s][K]                        (same diagonal)")
    out.append("source-centric off_target_mean[s] = mean over t != s of drop[s][t][K]    (row mean, no diagonal)")
    out.append("specificity_index[s]              = on_target - off_target               (row metric)")
    out.append("```")
    out.append("")
    out.append("This report uses a **target-centric** view (one *column* of the same "
               "heatmap per target `t`):")
    out.append("")
    out.append("```text")
    out.append("target-centric on_target_drop[t]  = drop[t][t][K]                        (same diagonal)")
    out.append("target-centric off_target_mean[t] = mean over s != t of drop[s][t][K]    (column mean, no diagonal)")
    out.append("sensitivity[t]                    = mean over ALL s of drop[s][t][K]     (column mean, with diagonal)")
    out.append("```")
    out.append("")
    out.append("Both views read from the **same 8 × 8 matrix**; they just slice it along "
               "different axes. The diagonal `drop[t][t][K]` is identical between the two views; "
               "everything else is different.")
    out.append("")
    out.append("All numbers below are derived directly from `cross_task_transfer_matrix.json` "
               "and cross-validated against the existing source-centric "
               "`cross_task_specificity_metrics.json` / `specificity_table.csv` artifacts "
               "(see *Data provenance & validation* below).")
    out.append("")

    out.append("## Setup")
    out.append("")
    model_rows = [[short_label(m), f"`{m}`"] for m in models]
    out.append("**Models** (short label → full HuggingFace name):")
    out.append("")
    out.append(md_table(["Short", "Model"], model_rows, aligns=["l", "l"]))
    out.append("")
    setup_rows = [
        ["Tasks", f"{len(task_list)} — " + ", ".join(f"`{t}`" for t in task_list)],
        ["K values", ", ".join(str(k) for k in ks)],
    ]
    out.append(md_table(["Field", "Value"], setup_rows, aligns=["l", "l"]))
    out.append("")

    emit_provenance_table(out, inputs, siblings)
    out.append("")

    out.append("## Per-K target sensitivity")
    out.append("")
    out.append("_Each row = one target task. Cells = mean drop in accuracy on that "
               "target when each model's top-K heads (detected for any of the 8 "
               "source tasks) are ablated. Tasks sorted by `Mean` column; bold row "
               "= most-fragile task at this K._")
    for k in ks:
        emit_per_k_table(out, k, models, model_metrics, task_list)

    emit_aggregate_table(out, "Aggregate sensitivity across all K", ks,
                         models, model_metrics, task_list)
    early_present = [k for k in EARLY_KS if k in ks]
    if early_present:
        emit_aggregate_table(out, "Early sensitivity (low-K collapse)", early_present,
                             models, model_metrics, task_list)
    emit_rank_tables(out, ks, models, model_metrics, task_list)
    emit_topk_consistency(out, ks, models, model_metrics, task_list)

    report_md = os.path.join(args.output_dir, "target_sensitivity_report.md")
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    print("\n" + "=" * 78)
    print("Wrote:")
    print(f"  {long_csv}")
    print(f"  {nested_json}")
    print(f"  {report_md}")
    print("=" * 78)
    print("\nOpen the .md file in any markdown viewer (Cursor / GitHub / VS Code preview).")


if __name__ == "__main__":
    main()
