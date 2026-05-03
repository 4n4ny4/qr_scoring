#!/usr/bin/env python3
"""Build Jaccard similarity heatmaps for top-k head overlap across SEC tasks.

Uses per-task QRScore rankings for meta-llama/Llama-3.1-8B-Instruct (1024 heads),
same logic as ``run_ablation.load_method_rankings`` when cross-task transfer is enabled.

Writes:
  - cross_task_head_similarity_topk.json
  - head_similarity_heatmaps.png

No GPU or model load required.

Example:
  python scripts/evaluation/plot_task_head_jaccard.py \\
      --output_dir results/comparison_ablation
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_ablation import (  # noqa: E402
    DEFAULT_EXPORT_TOP_K,
    MODEL_NAME,
    PROJECT_DIR,
    TASKS,
    ensure_full_ranking,
    load_ranked_heads_json,
    resolve_topk_export_from_manifest,
    save_head_similarity,
)

from plot_transfer import plot_head_similarity  # noqa: E402


NUM_LAYERS = 32
NUM_HEADS_PER_LAYER = 32


def load_per_task_rankings(detection_dir, tasks):
    """Return task -> full head ranking (layer, head), same resolution as ablation."""
    rankings = {}
    for i, task in enumerate(tasks):
        candidate_paths = [
            os.path.join(detection_dir, f"long_context_{task}_heads.json"),
            os.path.join(detection_dir, f"{task}_heads.json"),
        ]
        task_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if task_path is not None:
            task_heads = load_ranked_heads_json(task_path)
        else:
            manifest_candidates = [
                os.path.join(detection_dir, "topk", f"long_context_{task}_heads_manifest.json"),
                os.path.join(detection_dir, "topk", f"{task}_heads_manifest.json"),
            ]
            manifest_path = next((p for p in manifest_candidates if os.path.exists(p)), None)
            if manifest_path is None:
                raise FileNotFoundError(
                    f"Missing per-task ranking for `{task}`. Tried:\n  "
                    + "\n  ".join(candidate_paths + manifest_candidates)
                )
            export_path, recovered_k = resolve_topk_export_from_manifest(manifest_path)
            if export_path is None:
                raise FileNotFoundError(
                    f"Manifest found for `{task}` but could not resolve export: {manifest_path}"
                )
            task_heads = load_ranked_heads_json(export_path)
            print(
                f"  {task}: using top-{recovered_k} fallback from {export_path}"
            )
        rankings[task] = ensure_full_ranking(
            task_heads, NUM_LAYERS, NUM_HEADS_PER_LAYER, seed=100 + i
        )
        if task_path is not None:
            print(f"  {task}: loaded {task_path}")
    return rankings


def main():
    parser = argparse.ArgumentParser(description="Plot Jaccard head overlap across SEC tasks.")
    parser.add_argument(
        "--detection_dir",
        default=os.path.join(PROJECT_DIR, "results", "detection"),
        help="Directory with long_context_<task>_heads.json and/or topk manifests",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_DIR, "results", "comparison_ablation"),
        help="Where to write JSON and PNG",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=DEFAULT_EXPORT_TOP_K,
        help="Top-k cutoffs for Jaccard matrices (default: 8 16 ... 128)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading per-task SEC detection rankings (Llama 3.1 8B)...")
    rankings = load_per_task_rankings(args.detection_dir, TASKS)

    json_path = os.path.join(args.output_dir, "cross_task_head_similarity_topk.json")
    save_head_similarity(rankings, args.top_k, json_path, model_name=MODEL_NAME)
    print(f"Wrote {json_path}")

    with open(json_path, encoding="utf-8") as f:
        sim_data = json.load(f)
    plot_head_similarity(sim_data, args.output_dir)
    print(f"Done. Heatmaps in {args.output_dir}")


if __name__ == "__main__":
    main()
