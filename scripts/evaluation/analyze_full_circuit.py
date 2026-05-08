"""Summarize full-circuit QRHead experiment artifacts."""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import PROJECT_DIR, write_csv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze full QRHead circuit search outputs.")
    parser.add_argument("--model_slug", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Defaults to results/mech_experiments/<model_slug>.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def best_rows(rows: Sequence[dict], value_key: str, limit: int = 12) -> List[dict]:
    sortable = []
    for row in rows:
        value = as_float(row.get(value_key))
        if value is not None:
            sortable.append((value, row))
    return [row for _value, row in sorted(sortable, key=lambda item: item[0], reverse=True)[:limit]]


def aggregate_validation(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = as_float(row.get("necessity_fraction"))
        if value is None:
            continue
        grouped[(row.get("task", ""), row.get("condition", ""))].append(value)
    out = []
    for (task, condition), values in sorted(grouped.items()):
        out.append({
            "section": "final_validation",
            "task": task,
            "condition": condition,
            "n": len(values),
            "metric": "necessity_fraction",
            "mean": sum(values) / len(values),
            "median": statistics.median(values),
        })
    return out


def plot_summary(output_dir: Path, summary_rows: Sequence[dict]) -> None:
    rows = [
        row for row in summary_rows
        if row.get("metric") in {"median_margin_recovery", "median_margin_damage_fraction", "necessity_fraction"}
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["value"]), reverse=True)[:24]
    labels = [f"{row['section']}:{row.get('task', '')}:{row.get('condition', '')}" for row in rows]
    values = [float(row["value"]) for row in rows]
    colors = []
    for row in rows:
        if row["section"] == "component":
            colors.append("#2f6da8")
        elif row["section"] == "path":
            colors.append("#7a9f35")
        elif row["section"] == "final_validation":
            colors.append("#b26a2c")
        else:
            colors.append("#8b8f97")
    plt.figure(figsize=(9.0, max(3.2, 0.30 * len(rows))))
    plt.barh(range(len(rows)), values, color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.yticks(range(len(rows)), labels, fontsize=6.5)
    plt.xlabel("Median fraction / recovery")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "full_circuit_summary.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir) if args.root_dir else (
        PROJECT_DIR / "results" / "mech_experiments" / args.model_slug
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residual_rows = read_csv(root_dir / "residual_circuit_patching" / "residual_patch_summary.csv")
    component_rows = read_csv(root_dir / "component_circuit_patching" / "component_patch_summary.csv")
    path_rows = read_csv(root_dir / "path_circuit_patching" / "path_patch_edge_summary.csv")
    validation_rows = read_csv(root_dir / "final_circuit_validation" / "final_validation.csv")
    if not validation_rows:
        validation_rows = read_csv(root_dir / "component_circuit_patching" / "final_validation.csv")

    summary = []
    for row in best_rows(residual_rows, "median_margin_recovery"):
        summary.append({
            "section": "residual",
            "task": row.get("task", ""),
            "condition": f"{row.get('position_group', '')}_L{row.get('layer', '')}",
            "metric": "median_margin_recovery",
            "value": row.get("median_margin_recovery", ""),
            "n": row.get("n", ""),
        })
    for row in best_rows(component_rows, "median_margin_recovery"):
        summary.append({
            "section": "component",
            "task": row.get("task", ""),
            "condition": row.get("condition", ""),
            "component_type": row.get("component_type", ""),
            "metric": "median_margin_recovery",
            "value": row.get("median_margin_recovery", ""),
            "n": row.get("n", ""),
        })
    for row in best_rows(path_rows, "median_margin_damage_fraction"):
        summary.append({
            "section": "path",
            "task": row.get("task", ""),
            "condition": (
                f"{row.get('head', '')}:{row.get('target_position_group', '')}"
                f"->{row.get('source_position_group', '')}:{row.get('condition', '')}"
            ),
            "metric": "median_margin_damage_fraction",
            "value": row.get("median_margin_damage_fraction", ""),
            "n": row.get("n", ""),
        })
    for row in aggregate_validation(validation_rows):
        summary.append({
            "section": "final_validation",
            "task": row["task"],
            "condition": row["condition"],
            "metric": row["metric"],
            "value": row["median"],
            "n": row["n"],
        })

    write_csv(output_dir / "full_circuit_summary.csv", summary)
    plot_summary(output_dir, summary)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_slug": args.model_slug,
            "root_dir": str(root_dir),
            "residual_rows": len(residual_rows),
            "component_rows": len(component_rows),
            "path_rows": len(path_rows),
            "validation_rows": len(validation_rows),
            "summary_rows": len(summary),
        }, f, indent=2)
    print(f"Wrote full-circuit summary to {output_dir}")


if __name__ == "__main__":
    main()
