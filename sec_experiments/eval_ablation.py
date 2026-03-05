"""
Evaluate ablation experiments: compute Recall@1, Recall@3 with bootstrap CIs.

Reads retrieval score files from results/ablation/ and test data from
data/detection_input/, produces a summary table.
"""

import argparse
import json
import os
import numpy as np

CONDITIONS = ["full_head", "qr_head_top16", "random_head"]
METRICS_K = [1, 3]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


def compute_recall_at_k(samples, scores, k):
    """Compute Recall@k: fraction of instances where gold doc is in top-k."""
    hits = 0
    total = 0
    per_instance = []

    for sample in samples:
        qid = str(sample["idx"])
        gt_docs = [str(d) for d in sample.get("gt_docs", [])]

        if qid not in scores:
            continue

        doc_scores = scores[qid]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = [doc_id for doc_id, _ in sorted_docs[:k]]

        hit = int(any(g in top_k_docs for g in gt_docs))
        per_instance.append(hit)
        hits += hit
        total += 1

    recall = hits / total if total > 0 else 0.0
    return recall, per_instance


def bootstrap_ci(per_instance_scores, n_bootstrap=BOOTSTRAP_N, ci_level=CI_LEVEL, seed=BOOTSTRAP_SEED):
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.RandomState(seed)
    scores = np.array(per_instance_scores)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(sample.mean())

    alpha = (1 - ci_level) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return lo, hi


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """
    Paired bootstrap test: is the mean of scores_a significantly different from scores_b?
    Returns p-value (two-sided).
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = a.mean() - b.mean()
    n = len(a)

    count = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diff = a[idx].mean() - b[idx].mean()
        if abs(boot_diff) >= abs(observed_diff):
            count += 1

    return count / n_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    results = {}
    all_per_instance = {cond: {k: [] for k in METRICS_K} for cond in CONDITIONS}

    print(f"\n{'='*80}")
    print(f"ABLATION EVALUATION")
    print(f"{'='*80}")

    for task in args.tasks:
        test_file = os.path.join(args.data_dir, f"{task}_test.json")
        if not os.path.exists(test_file):
            print(f"WARNING: {test_file} not found, skipping {task}")
            continue

        with open(test_file) as f:
            test_data = json.load(f)

        results[task] = {}

        for condition in CONDITIONS:
            score_file = os.path.join(args.ablation_dir, f"{task}_{condition}.json")
            if not os.path.exists(score_file):
                print(f"WARNING: {score_file} not found, skipping {task}/{condition}")
                continue

            with open(score_file) as f:
                scores = json.load(f)

            results[task][condition] = {}
            for k in METRICS_K:
                recall, per_inst = compute_recall_at_k(test_data, scores, k)
                ci_lo, ci_hi = bootstrap_ci(per_inst)
                results[task][condition][f"recall@{k}"] = {
                    "value": recall,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n": len(per_inst),
                }
                all_per_instance[condition][k].extend(per_inst)

    print(f"\n{'='*80}")
    print(f"PER-TASK RESULTS")
    print(f"{'='*80}")

    header = f"{'Task':>25s}"
    for cond in CONDITIONS:
        header += f"  {cond:>20s}"
    print(f"\n--- Recall@1 ---")
    print(header)
    for task in args.tasks:
        if task not in results:
            continue
        row = f"{task:>25s}"
        for cond in CONDITIONS:
            if cond in results[task] and "recall@1" in results[task][cond]:
                r = results[task][cond]["recall@1"]
                row += f"  {r['value']:6.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            else:
                row += f"  {'N/A':>20s}"
        print(row)

    print(f"\n--- Recall@3 ---")
    print(header)
    for task in args.tasks:
        if task not in results:
            continue
        row = f"{task:>25s}"
        for cond in CONDITIONS:
            if cond in results[task] and "recall@3" in results[task][cond]:
                r = results[task][cond]["recall@3"]
                row += f"  {r['value']:6.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            else:
                row += f"  {'N/A':>20s}"
        print(row)

    print(f"\n{'='*80}")
    print(f"POOLED RESULTS (all tasks combined)")
    print(f"{'='*80}")
    pooled = {}
    for cond in CONDITIONS:
        pooled[cond] = {}
        for k in METRICS_K:
            scores = all_per_instance[cond][k]
            if scores:
                recall = np.mean(scores)
                ci_lo, ci_hi = bootstrap_ci(scores)
                pooled[cond][f"recall@{k}"] = {
                    "value": float(recall),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "n": len(scores),
                }
                print(f"  {cond:>20s}  Recall@{k}: {recall:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  (n={len(scores)})")

    print(f"\n{'='*80}")
    print(f"SIGNIFICANCE TESTS (paired bootstrap, QR top-16 vs others)")
    print(f"{'='*80}")
    for k in METRICS_K:
        qr_scores = all_per_instance["qr_head_top16"][k]
        for other_cond in ["full_head", "random_head"]:
            other_scores = all_per_instance[other_cond][k]
            if qr_scores and other_scores and len(qr_scores) == len(other_scores):
                p_val = paired_bootstrap_test(qr_scores, other_scores)
                diff = np.mean(qr_scores) - np.mean(other_scores)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                print(f"  Recall@{k}: qr_head_top16 vs {other_cond}: diff={diff:+.3f}, p={p_val:.4f} {sig}")

    summary = {
        "per_task": results,
        "pooled": pooled,
        "conditions": CONDITIONS,
        "metrics": [f"recall@{k}" for k in METRICS_K],
    }
    summary_path = os.path.join(args.ablation_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
