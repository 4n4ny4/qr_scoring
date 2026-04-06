#!/usr/bin/env python3
"""Compute Jaccard similarity between head rankings from the three methods
(SEC, LME-TRAIN, NQ-TRAIN) at various top-K thresholds.

Outputs results to results/comparison_ablation/cross_method_head_overlap.json
and prints a summary table.
"""

import json
import os
import argparse

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

K_VALUES = [8, 16, 32, 48, 64, 96, 128]


def load_heads(path):
    """Load ranked heads from a JSON file, return list of (layer, head) tuples."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    heads = []
    for row in data:
        head_str = row[0] if isinstance(row, list) else row.get("head")
        layer, head = map(int, head_str.split("-"))
        heads.append((layer, head))
    return heads


def jaccard(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def expected_random_jaccard(k, n):
    """Expected Jaccard for two random subsets of size k drawn from n items."""
    return k / (2 * n - k)


def main():
    parser = argparse.ArgumentParser(description="Cross-method head overlap (Jaccard)")
    parser.add_argument(
        "--external_rankings_dir",
        default=os.path.join(PROJECT_DIR, "Llama-3.1-8B-Instruct"),
        help="Directory containing lme_TRAIN.json and nq_TRAIN.json",
    )
    parser.add_argument(
        "--num_total_heads",
        type=int,
        default=1024,
        help="Total number of attention heads in the target model",
    )
    parser.add_argument(
        "--results_dir",
        default=os.path.join(PROJECT_DIR, "results", "comparison_ablation"),
    )
    args = parser.parse_args()

    ranking_paths = {
        "SEC": os.path.join(PROJECT_DIR, "results", "detection", "long_context_combined_heads.json"),
        "LME": os.path.join(args.external_rankings_dir, "lme_TRAIN.json"),
        "NQ": os.path.join(args.external_rankings_dir, "nq_TRAIN.json"),
    }

    # Load all rankings
    rankings = {}
    for name, path in ranking_paths.items():
        rankings[name] = load_heads(path)
        print(f"Loaded {name}: {len(rankings[name])} heads from {path}")

    pairs = [("SEC", "LME"), ("SEC", "NQ"), ("LME", "NQ")]

    results = {"k_values": K_VALUES, "pairs": {}, "random_expected": {}}

    # Compute expected random Jaccard
    for k in K_VALUES:
        results["random_expected"][str(k)] = round(expected_random_jaccard(k, args.num_total_heads), 4)

    # Compute Jaccard for each pair at each K
    for a, b in pairs:
        pair_key = f"{a}-{b}"
        results["pairs"][pair_key] = {}
        for k in K_VALUES:
            set_a = set(rankings[a][:k])
            set_b = set(rankings[b][:k])
            j = jaccard(set_a, set_b)
            overlap_count = len(set_a & set_b)
            results["pairs"][pair_key][str(k)] = {
                "jaccard": round(j, 4),
                "overlap_count": overlap_count,
                "union_size": len(set_a | set_b),
            }

    # Print summary table
    print(f"\n{'K':>5}", end="")
    for a, b in pairs:
        print(f"  {a}-{b:>5}", end="")
    print("  Random(expected)")

    for k in K_VALUES:
        print(f"{k:>5}", end="")
        for a, b in pairs:
            pair_key = f"{a}-{b}"
            j = results["pairs"][pair_key][str(k)]["jaccard"]
            print(f"  {j:>10.4f}", end="")
        rj = results["random_expected"][str(k)]
        print(f"  {rj:>10.4f}")

    # Print overlap counts
    print(f"\nOverlap counts (|intersection|):")
    print(f"{'K':>5}", end="")
    for a, b in pairs:
        print(f"  {a}-{b:>5}", end="")
    print()
    for k in K_VALUES:
        print(f"{k:>5}", end="")
        for a, b in pairs:
            pair_key = f"{a}-{b}"
            c = results["pairs"][pair_key][str(k)]["overlap_count"]
            print(f"  {c:>10}", end="")
        print()

    # Save results
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_method_head_overlap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
