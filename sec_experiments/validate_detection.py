"""
Validate QRHead detection results.

Checks:
  1. Are top heads in middle layers (8-17) rather than early layers (0-2)?
  2. How much overlap with the paper's pre-computed heads?
  3. Are QRScores positive for top heads?

Usage:
  python sec_experiments/validate_detection.py
  python sec_experiments/validate_detection.py --heads_file results/detection/long_context_combined_heads.json
"""

import argparse
import json
import os
import sys
import yaml


def load_heads_from_json(path):
    """Load ranked heads from detection JSON. Returns list of (layer, head, score)."""
    with open(path) as f:
        data = json.load(f)
    result = []
    for head_str, score in data:
        layer, head = map(int, head_str.split("-"))
        result.append((layer, head, score))
    return result


def load_heads_from_yaml(path):
    """Load heads from paper's YAML config. Returns set of (layer, head)."""
    with open(path) as f:
        config = yaml.safe_load(f)
    heads = set()
    for h in config["attn_head_set"].split(","):
        layer, head = map(int, h.strip().split("-"))
        heads.add((layer, head))
    return heads


def analyze_heads(ranked_heads, top_k=16, label=""):
    """Print analysis of head ranking."""
    top = ranked_heads[:top_k]
    print(f"\n{'='*60}")
    print(f"Analysis: {label} (top-{top_k})")
    print(f"{'='*60}")

    layers = [l for l, h, s in top]
    layer_range = f"{min(layers)}-{max(layers)}" if layers else "N/A"
    print(f"  Layer range: {layer_range}")
    print(f"  Mean layer:  {sum(layers)/len(layers):.1f}")

    early = sum(1 for l in layers if l <= 3)
    middle = sum(1 for l in layers if 8 <= l <= 17)
    late = sum(1 for l in layers if l >= 25)
    print(f"  Early (0-3):  {early}/{top_k}")
    print(f"  Middle (8-17): {middle}/{top_k}")
    print(f"  Late (25+):   {late}/{top_k}")

    positive = sum(1 for _, _, s in top if s > 0)
    print(f"  Positive scores: {positive}/{top_k}")
    print(f"  Score range: {top[-1][2]:.6f} to {top[0][2]:.6f}")

    print(f"\n  Top-{top_k} heads:")
    for i, (l, h, s) in enumerate(top):
        print(f"    {i+1:2d}. layer {l:2d} head {h:2d}  score={s:.6f}")

    in_middle = middle >= top_k * 0.4
    has_positive = positive >= top_k * 0.5

    if in_middle and has_positive:
        verdict = "GOOD - heads in expected middle layers with positive scores"
    elif in_middle:
        verdict = "PARTIAL - heads in middle layers but scores still negative"
    elif has_positive:
        verdict = "PARTIAL - positive scores but heads not in expected layers"
    else:
        verdict = "BROKEN - heads in early layers with negative scores (context too short?)"

    print(f"\n  Verdict: {verdict}")
    return in_middle, has_positive


def compare_with_paper(ranked_heads, paper_heads, top_k=16, label=""):
    """Compare detected heads with paper's pre-computed heads."""
    detected_set = {(l, h) for l, h, s in ranked_heads[:top_k]}
    overlap = detected_set & paper_heads
    jaccard = len(overlap) / len(detected_set | paper_heads) if (detected_set | paper_heads) else 0

    print(f"\n  Comparison with {label}:")
    print(f"    Overlap:  {len(overlap)}/{top_k} heads match")
    print(f"    Jaccard:  {jaccard:.3f}")
    if overlap:
        print(f"    Matching: {sorted(overlap)}")
    return len(overlap), jaccard


def main():
    parser = argparse.ArgumentParser(description="Validate QRHead detection results")
    parser.add_argument("--heads_file", default=None,
                        help="Path to detection heads JSON (default: auto-detect)")
    parser.add_argument("--top_k", type=int, default=16)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, "results", "detection")
    configs_dir = os.path.join(project_dir, "src", "qrretriever", "configs")

    lme_path = os.path.join(configs_dir, "Llama-3.1-8B-Instruct_qr_head_LME.yaml")
    nq_path = os.path.join(configs_dir, "Llama-3.1-8B-Instruct_qr_head_NQ.yaml")

    paper_lme = load_heads_from_yaml(lme_path) if os.path.exists(lme_path) else set()
    paper_nq = load_heads_from_yaml(nq_path) if os.path.exists(nq_path) else set()

    heads_files = []
    if args.heads_file:
        heads_files.append(("custom", args.heads_file))
    else:
        lc_combined = os.path.join(results_dir, "long_context_combined_heads.json")
        if os.path.exists(lc_combined):
            heads_files.append(("Long-context combined", lc_combined))

        for task in ["registrant_name", "headquarters_city", "ceo_lastname"]:
            lc_task = os.path.join(results_dir, f"long_context_{task}_heads.json")
            if os.path.exists(lc_task):
                heads_files.append((f"Long-context {task}", lc_task))

        old_combined = os.path.join(results_dir, "niah_combined_heads.json")
        if os.path.exists(old_combined):
            heads_files.append(("Old NIAH combined (baseline)", old_combined))

        for task in ["registrant_name"]:
            old_task = os.path.join(results_dir, f"{task}_heads.json")
            if os.path.exists(old_task):
                heads_files.append((f"Old short-context {task} (baseline)", old_task))

    if not heads_files:
        print("No detection results found. Run detection first:")
        print("  bash sec_experiments/run_long_context_detection.sh")
        sys.exit(1)

    for label, path in heads_files:
        ranked = load_heads_from_json(path)
        analyze_heads(ranked, top_k=args.top_k, label=label)

        if paper_lme:
            compare_with_paper(ranked, paper_lme, top_k=args.top_k, label="Paper LME heads")
        if paper_nq:
            compare_with_paper(ranked, paper_nq, top_k=args.top_k, label="Paper NQ heads")

    if paper_lme:
        print(f"\n{'='*60}")
        print("Paper LME heads (reference):")
        for l, h in sorted(paper_lme):
            print(f"  layer {l:2d} head {h:2d}")
    if paper_nq:
        print(f"\nPaper NQ heads (reference):")
        for l, h in sorted(paper_nq):
            print(f"  layer {l:2d} head {h:2d}")


if __name__ == "__main__":
    main()
