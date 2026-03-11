"""
Compute bootstrap 95% confidence intervals for all ablation results.

Reads *_results.json files (which contain per-instance correct/incorrect details)
and produces:
  - confidence_intervals.json   (method × K → accuracy, CI lower, CI upper)
  - confidence_intervals.csv    (flat table for paper inclusion)

Also computes CIs for:
  - Per-task accuracy at each K
  - Specificity index (from cross_task_transfer_matrix.json)

Usage:
  python scripts/evaluation/compute_confidence_intervals.py
  python scripts/evaluation/compute_confidence_intervals.py \
    --results_dir results/comparison_ablation \
    --n_bootstrap 10000
"""

import argparse
import json
import os
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def bootstrap_accuracy(correct_flags, n_bootstrap=10000, ci=0.95, rng=None):
    """Bootstrap CI for accuracy from a list of 0/1 flags."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = arr.mean()
    boots = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return float(point), lo, hi


def bootstrap_drop(correct_k0, correct_k, n_bootstrap=10000, ci=0.95, rng=None):
    """Bootstrap CI for accuracy drop = acc(K=0) - acc(K), paired by instance."""
    if rng is None:
        rng = np.random.default_rng(42)
    a0 = np.array(correct_k0, dtype=float)
    ak = np.array(correct_k, dtype=float)
    n = len(a0)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = a0.mean() - ak.mean()
    indices = np.arange(n)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(indices, size=n, replace=True)
        boots.append(a0[idx].mean() - ak[idx].mean())
    boots = np.array(boots)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return float(point), lo, hi


def load_results(results_dir):
    """Load all *_results.json files, return dict of method -> data."""
    methods = {}
    for path in sorted(Path(results_dir).glob("*_results.json")):
        data = json.loads(path.read_text())
        name = data.get("method", path.stem.replace("_results", ""))
        if "details" in data:
            methods[name] = data
    return methods


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals")
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_DIR, "results", "comparison_ablation"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    methods = load_results(args.results_dir)
    if not methods:
        print("ERROR: No *_results.json files with details found.")
        sys.exit(1)

    print(f"Found {len(methods)} methods: {list(methods.keys())}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")

    # ── 1. Overall accuracy CIs ──
    overall_ci = {}
    for method_name, data in methods.items():
        details = data["details"]
        ks = sorted(details.keys(), key=lambda x: int(x))
        method_ci = {}
        k0_flags = [d["correct"] for d in details.get("0", [])]
        for k_str in ks:
            flags = [d["correct"] for d in details[k_str]]
            acc, lo, hi = bootstrap_accuracy(flags, args.n_bootstrap, rng=rng)
            drop, drop_lo, drop_hi = bootstrap_drop(k0_flags, flags, args.n_bootstrap, rng=rng)
            method_ci[k_str] = {
                "accuracy": acc,
                "accuracy_ci_lo": lo,
                "accuracy_ci_hi": hi,
                "drop": drop,
                "drop_ci_lo": drop_lo,
                "drop_ci_hi": drop_hi,
                "n": len(flags),
            }
        overall_ci[method_name] = method_ci
        print(f"\n  {method_name}:")
        for k_str in ks:
            c = method_ci[k_str]
            print(f"    K={k_str:>3s}: acc={c['accuracy']:.3f} [{c['accuracy_ci_lo']:.3f}, {c['accuracy_ci_hi']:.3f}]  "
                  f"drop={c['drop']:.3f} [{c['drop_ci_lo']:.3f}, {c['drop_ci_hi']:.3f}]")

    # ── 2. Per-task accuracy CIs ──
    per_task_ci = {}
    for method_name, data in methods.items():
        details = data["details"]
        ks = sorted(details.keys(), key=lambda x: int(x))
        task_ci = {}
        for k_str in ks:
            by_task = {}
            for d in details[k_str]:
                by_task.setdefault(d["task"], []).append(d["correct"])
            for task, flags in sorted(by_task.items()):
                if task not in task_ci:
                    task_ci[task] = {}
                acc, lo, hi = bootstrap_accuracy(flags, args.n_bootstrap, rng=rng)
                task_ci.setdefault(task, {})[k_str] = {
                    "accuracy": acc,
                    "accuracy_ci_lo": lo,
                    "accuracy_ci_hi": hi,
                    "n": len(flags),
                }
        per_task_ci[method_name] = task_ci

    # ── 3. Specificity CIs (if transfer matrix exists with per-instance details) ──
    transfer_path = os.path.join(args.results_dir, "cross_task_transfer_matrix.json")
    specificity_ci = {}
    if os.path.exists(transfer_path):
        transfer = json.loads(Path(transfer_path).read_text())
        sources = transfer["sources"]
        targets = transfer["targets"]
        # Transfer matrix has by_k with accuracy/drop but NOT per-instance details.
        # We can still compute CIs on the specificity index by bootstrapping over
        # the 7 off-target drop values (treated as the sample).
        for summary_k in ["16"]:
            specificity_ci[summary_k] = {}
            for source in sources:
                on_drop = transfer["results"][source][source]["by_k"].get(summary_k, {}).get("drop_from_k0")
                off_drops = []
                for target in targets:
                    if target == source:
                        continue
                    drop = transfer["results"][source][target]["by_k"].get(summary_k, {}).get("drop_from_k0")
                    if drop is not None:
                        off_drops.append(drop)
                if on_drop is not None and off_drops:
                    off_arr = np.array(off_drops)
                    off_mean = float(off_arr.mean())
                    spec_point = on_drop - off_mean
                    # Bootstrap over the 7 off-target values
                    boots = []
                    for _ in range(args.n_bootstrap):
                        b = rng.choice(off_arr, size=len(off_arr), replace=True)
                        boots.append(on_drop - b.mean())
                    boots = np.array(boots)
                    spec_lo = float(np.percentile(boots, 2.5))
                    spec_hi = float(np.percentile(boots, 97.5))
                    specificity_ci[summary_k][source] = {
                        "on_target_drop": on_drop,
                        "off_target_mean_drop": off_mean,
                        "specificity_index": spec_point,
                        "specificity_ci_lo": spec_lo,
                        "specificity_ci_hi": spec_hi,
                        "n_off_targets": len(off_drops),
                    }
        print(f"\n  Specificity CIs (K=16):")
        for src, vals in specificity_ci.get("16", {}).items():
            print(f"    {src}: spec={vals['specificity_index']:.3f} "
                  f"[{vals['specificity_ci_lo']:.3f}, {vals['specificity_ci_hi']:.3f}]")

    # ── Write JSON ──
    output = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "ci_level": 0.95,
        "overall": overall_ci,
        "per_task": per_task_ci,
        "specificity": specificity_ci,
    }
    json_path = os.path.join(args.output_dir, "confidence_intervals.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Write CSV (flat table for paper) ──
    csv_path = os.path.join(args.output_dir, "confidence_intervals.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "K", "accuracy", "ci_lo", "ci_hi", "drop", "drop_ci_lo", "drop_ci_hi", "n"])
        for method_name in sorted(overall_ci.keys()):
            for k_str in sorted(overall_ci[method_name].keys(), key=lambda x: int(x)):
                c = overall_ci[method_name][k_str]
                w.writerow([
                    method_name, k_str,
                    f"{c['accuracy']:.4f}", f"{c['accuracy_ci_lo']:.4f}", f"{c['accuracy_ci_hi']:.4f}",
                    f"{c['drop']:.4f}", f"{c['drop_ci_lo']:.4f}", f"{c['drop_ci_hi']:.4f}",
                    c["n"],
                ])
    print(f"Saved: {csv_path}")

    # ── Write per-task CSV ──
    per_task_csv_path = os.path.join(args.output_dir, "per_task_confidence_intervals.csv")
    with open(per_task_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "K", "accuracy", "ci_lo", "ci_hi", "n"])
        for method_name in sorted(per_task_ci.keys()):
            for task in sorted(per_task_ci[method_name].keys()):
                for k_str in sorted(per_task_ci[method_name][task].keys(), key=lambda x: int(x)):
                    c = per_task_ci[method_name][task][k_str]
                    w.writerow([
                        method_name, task, k_str,
                        f"{c['accuracy']:.4f}", f"{c['accuracy_ci_lo']:.4f}", f"{c['accuracy_ci_hi']:.4f}",
                        c["n"],
                    ])
    print(f"Saved: {per_task_csv_path}")

    # ── Write specificity CSV ──
    if specificity_ci:
        spec_csv_path = os.path.join(args.output_dir, "specificity_confidence_intervals.csv")
        with open(spec_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["summary_k", "source_task", "on_target_drop", "off_target_mean",
                         "specificity", "spec_ci_lo", "spec_ci_hi"])
            for sk in sorted(specificity_ci.keys()):
                for src in sorted(specificity_ci[sk].keys()):
                    v = specificity_ci[sk][src]
                    w.writerow([
                        sk, src,
                        f"{v['on_target_drop']:.4f}", f"{v['off_target_mean_drop']:.4f}",
                        f"{v['specificity_index']:.4f}",
                        f"{v['specificity_ci_lo']:.4f}", f"{v['specificity_ci_hi']:.4f}",
                    ])
        print(f"Saved: {spec_csv_path}")


if __name__ == "__main__":
    main()
