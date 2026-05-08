"""Summarize partial QRHead circuit experiments.

This script combines the query-edge masking, query-position head-subset
patching, and query-position counterfactual patching outputs into compact CSVs
and figures for the workshop paper.
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze partial QRHead circuit results.")
    parser.add_argument("--model_slug", default="meta-llama__Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--results_root",
        default=str(PROJECT_DIR / "results" / "mech_experiments"),
    )
    parser.add_argument("--edge_dir", default=None)
    parser.add_argument("--head_subset_dir", default=None)
    parser.add_argument("--counterfactual_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--damage_threshold", type=float, default=0.5)
    return parser.parse_args()


def ffloat(value):
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows, group_keys, value_key):
    grouped = defaultdict(list)
    for row in rows:
        value = ffloat(row.get(value_key))
        if math.isfinite(value):
            grouped[tuple(row.get(key, "") for key in group_keys)].append(value)
    out = []
    for group, values in sorted(grouped.items()):
        record = dict(zip(group_keys, group))
        record.update({
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
            "value_key": value_key,
        })
        out.append(record)
    return out


def label_condition(condition: str) -> str:
    return {
        "edge_gold": "QR query->gold",
        "edge_distractor": "QR query->distractor",
        "edge_random": "QR query->random",
        "edge_nonqr_gold": "non-QR query->gold",
        "edge_bottom_gold": "bottom query->gold",
        "patch_qr": "QR patch",
        "patch_random": "random patch",
        "patch_bottom": "bottom patch",
        "patch_same_layer": "same-layer patch",
    }.get(condition, condition)


def plot_edge(output_dir: Path, rows):
    if not rows:
        return
    summary = summarize_group(rows, ["condition"], "fraction_of_full_ablation_damage")
    order = ["edge_gold", "edge_distractor", "edge_random", "edge_nonqr_gold", "edge_bottom_gold"]
    by = {row["condition"]: row for row in summary}
    labels = [condition for condition in order if condition in by]
    if not labels:
        return
    values = [by[label]["median"] for label in labels]
    plt.figure(figsize=(5.2, 3.0))
    plt.bar([label_condition(label) for label in labels], values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    plt.ylabel("Median fraction of full QR-ablation damage")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "partial_circuit_edge_masking.png", dpi=220)
    plt.close()


def plot_head_subset(output_dir: Path, rows):
    rows = [row for row in rows if row.get("subset_mode") == "cumulative"]
    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        value = ffloat(row.get("sum_logprob_rescue"))
        size = int(row.get("subset_size", 0) or 0)
        subset = row.get("subset_type", "")
        if size and math.isfinite(value):
            grouped[(subset, size)].append(value)
    if not grouped:
        return
    plt.figure(figsize=(5.2, 3.0))
    for subset in sorted({subset for subset, _size in grouped}):
        points = []
        for size in sorted(size for sub, size in grouped if sub == subset):
            points.append((size, statistics.median(grouped[(subset, size)])))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        linewidth = 2.2 if subset == "qr" else 1.0
        alpha = 1.0 if subset == "qr" else 0.6
        plt.plot(xs, ys, marker="o", label=subset, linewidth=linewidth, alpha=alpha)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Patched head subset size")
    plt.ylabel("Median rescue ratio")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "partial_circuit_head_subset.png", dpi=220)
    plt.close()


def plot_counterfactual(output_dir: Path, rows):
    if not rows:
        return
    patch_rows = [row for row in rows if row.get("condition", "").startswith("patch_")]
    summary = summarize_group(patch_rows, ["condition"], "margin_shift_vs_cf_qr_ablate")
    order = ["patch_qr", "patch_random", "patch_bottom", "patch_same_layer"]
    by = {row["condition"]: row for row in summary}
    labels = [condition for condition in order if condition in by]
    if not labels:
        return
    values = [by[label]["median"] for label in labels]
    plt.figure(figsize=(5.0, 3.0))
    plt.bar([label_condition(label) for label in labels], values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Median counterfactual margin shift")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "partial_circuit_counterfactual.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    model_dir = results_root / args.model_slug
    edge_dir = Path(args.edge_dir) if args.edge_dir else model_dir / "query_edge_masking"
    head_subset_dir = (
        Path(args.head_subset_dir)
        if args.head_subset_dir
        else model_dir / "query_head_subset_patching"
    )
    counterfactual_dir = (
        Path(args.counterfactual_dir)
        if args.counterfactual_dir
        else model_dir / "query_counterfactual_patching"
    )
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "partial_circuit_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_rows_raw = read_csv(edge_dir / "edge_masking_damage.csv")
    edge_rows = [
        row for row in edge_rows_raw
        if ffloat(row.get("full_ablation_damage")) > args.damage_threshold
    ]
    head_rows_raw = read_csv(head_subset_dir / "head_subset_rescue.csv")
    head_rows = [
        row for row in head_rows_raw
        if ffloat(row.get("full_ablation_damage")) > args.damage_threshold
    ]
    cf_rows = read_csv(counterfactual_dir / "counterfactual_patching_effects.csv")

    summary_rows = []
    for row in summarize_group(edge_rows, ["task", "condition"], "fraction_of_full_ablation_damage"):
        row["experiment"] = "query_edge_masking"
        summary_rows.append(row)
    for row in summarize_group(head_rows, ["task", "subset_mode", "subset_type", "condition", "subset_size"], "sum_logprob_rescue"):
        row["experiment"] = "query_head_subset_patching"
        summary_rows.append(row)
    for row in summarize_group(cf_rows, ["task", "condition"], "margin_shift_vs_cf_qr_ablate"):
        row["experiment"] = "query_counterfactual_patching"
        summary_rows.append(row)

    write_csv(output_dir / "partial_circuit_summary.csv", summary_rows)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_slug": args.model_slug,
            "damage_threshold": args.damage_threshold,
            "edge_dir": str(edge_dir),
            "head_subset_dir": str(head_subset_dir),
            "counterfactual_dir": str(counterfactual_dir),
            "n_edge_rows_raw": len(edge_rows_raw),
            "n_edge_rows_filtered": len(edge_rows),
            "n_head_subset_rows_raw": len(head_rows_raw),
            "n_head_subset_rows_filtered": len(head_rows),
            "n_counterfactual_rows": len(cf_rows),
        }, f, indent=2)

    plot_edge(output_dir, edge_rows)
    plot_head_subset(output_dir, head_rows)
    plot_counterfactual(output_dir, cf_rows)
    print(f"Wrote partial-circuit summary to {output_dir}")


if __name__ == "__main__":
    main()
