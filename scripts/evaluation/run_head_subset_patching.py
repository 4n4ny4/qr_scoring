"""Head-subset QRHead activation patching.

This experiment asks whether the top-K QRHead rescue is concentrated in a
small subset of heads. It ablates the full top-K QRHead set everywhere, then
patches clean activations back at a selected position set for either individual
QRHeads or cumulative prefixes of the QRScore ranking.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    HeadPatchController,
    PROJECT_DIR,
    bottom_control_heads,
    clear_device_cache,
    default_ablation_results_path,
    default_ranking_path,
    get_decoder_layers,
    get_num_heads,
    intervention_positions_for_mode,
    load_clean_to_ablated_failure_ids,
    load_model_and_tokenizer,
    load_ranked_heads_json,
    load_task_instances,
    matched_layer_random_control_heads,
    random_control_heads,
    safe_div,
    same_layer_control_heads,
    score_gold_answer,
    select_instances,
    write_csv,
    write_jsonl,
    encode_prompt_and_answer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run QRHead subset patching.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--ranking_name", default="long_context_combined")
    parser.add_argument("--ranking_path", default=None)
    parser.add_argument("--ablation_results_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_examples_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subset_mode", choices=["individual", "cumulative", "both"], default="both")
    parser.add_argument("--patch_position_mode", choices=["answer", "query"], default="query")
    parser.add_argument("--num_random_controls", type=int, default=5)
    parser.add_argument("--num_matched_layer_random_controls", type=int, default=0)
    parser.add_argument("--cumulative_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument(
        "--use_all_examples",
        action="store_true",
        help="Do not filter to examples that clean generation solved and K-ablation missed.",
    )
    return parser.parse_args()


def head_name(head: Tuple[int, int]) -> str:
    return f"{head[0]}-{head[1]}"


def default_cumulative_sizes(k: int) -> List[int]:
    sizes = []
    value = 1
    while value < k:
        sizes.append(value)
        value *= 2
    sizes.append(k)
    return sorted(dict.fromkeys(size for size in sizes if 1 <= size <= k))


def summarize(rescue_rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rescue_rows:
        value = row.get("sum_logprob_rescue")
        if value is None or math.isnan(float(value)):
            continue
        grouped[(row["task"], row["subset_type"], row["condition"], row["subset_size"])].append(float(value))
    out = []
    for (task, subset_type, condition, subset_size), values in sorted(grouped.items()):
        out.append({
            "task": task,
            "subset_type": subset_type,
            "condition": condition,
            "subset_size": subset_size,
            "n": len(values),
            "mean_sum_logprob_rescue": sum(values) / len(values),
            "median_sum_logprob_rescue": statistics.median(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        })
    return out


def plot_cumulative(output_dir: Path, rescue_rows: Sequence[dict]) -> None:
    grouped = defaultdict(list)
    for row in rescue_rows:
        if row.get("subset_mode") != "cumulative":
            continue
        value = row.get("sum_logprob_rescue")
        if value is None or math.isnan(float(value)):
            continue
        label = row["subset_type"]
        grouped[(label, int(row["subset_size"]))].append(float(value))
    if not grouped:
        return

    plt.figure(figsize=(5.4, 3.2))
    labels = sorted({label for label, _size in grouped})
    for label in labels:
        points = []
        for size in sorted(size for lab, size in grouped if lab == label):
            points.append((size, statistics.median(grouped[(label, size)])))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        linewidth = 2.2 if label == "qr" else 1.2
        alpha = 1.0 if label == "qr" else 0.65
        plt.plot(xs, ys, marker="o", label=label, linewidth=linewidth, alpha=alpha)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Patched head subset size")
    plt.ylabel("Median rescue ratio")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "head_subset_cumulative_rescue.png", dpi=200)
    plt.close()


def condition_row(inst, condition, subset_type, subset_mode, subset_size, heads, positions, sum_lp, mean_lp, toks):
    return {
        "idx": inst["idx"],
        "task": inst["task"],
        "condition": condition,
        "subset_type": subset_type,
        "subset_mode": subset_mode,
        "subset_size": subset_size,
        "heads": [head_name(head) for head in heads],
        "position_count": len(positions),
        "positions": positions,
        "gold": inst["needle_value"],
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "num_answer_tokens": len(toks),
        "token_logprobs": toks,
    }


def rescue_row(inst, condition, subset_type, subset_mode, subset_size, heads, clean_sum, ablated_sum, patch_sum):
    return {
        "idx": inst["idx"],
        "task": inst["task"],
        "condition": condition,
        "subset_type": subset_type,
        "subset_mode": subset_mode,
        "subset_size": subset_size,
        "heads": [head_name(head) for head in heads],
        "gold": inst["needle_value"],
        "clean_sum_logprob": clean_sum,
        "ablated_sum_logprob": ablated_sum,
        "patched_sum_logprob": patch_sum,
        "full_ablation_damage": clean_sum - ablated_sum,
        "patched_recovery": patch_sum - ablated_sum,
        "sum_logprob_rescue": safe_div(patch_sum - ablated_sum, clean_sum - ablated_sum),
    }


def build_cumulative_conditions(
    *,
    qr_heads: Sequence[Tuple[int, int]],
    ranking: Sequence[Tuple[int, int]],
    num_layers: int,
    num_heads: int,
    sizes: Sequence[int],
    seed: int,
    num_random_controls: int,
    num_matched_layer_random_controls: int,
):
    bottom_heads = bottom_control_heads(ranking=ranking, qr_heads=qr_heads, k=len(qr_heads))
    same_layer_heads = same_layer_control_heads(qr_heads=qr_heads, num_heads=num_heads)
    random_sets = [
        random_control_heads(
            qr_heads=qr_heads,
            num_layers=num_layers,
            num_heads=num_heads,
            seed=seed + i,
        )
        for i in range(num_random_controls)
    ]
    matched_random_sets = [
        matched_layer_random_control_heads(
            qr_heads=qr_heads,
            num_heads=num_heads,
            seed=seed + 10_000 + i,
        )
        for i in range(num_matched_layer_random_controls)
    ]

    conditions = []
    for size in sizes:
        conditions.append((f"cumulative_qr_top{size}", "qr", qr_heads[:size]))
        conditions.append((f"cumulative_bottom_top{size}", "bottom", bottom_heads[:size]))
        conditions.append((f"cumulative_same_layer_top{size}", "same_layer", same_layer_heads[:size]))
        for i, heads in enumerate(random_sets):
            conditions.append((f"cumulative_random{i:02d}_top{size}", "random", heads[:size]))
        for i, heads in enumerate(matched_random_sets):
            conditions.append((f"cumulative_matched_random{i:02d}_top{size}", "matched_layer_random", heads[:size]))
    return conditions


def main():
    args = parse_args()
    model_spec, tokenizer, model = load_model_and_tokenizer(args, for_detection=False)

    ranking_path = Path(args.ranking_path) if args.ranking_path else default_ranking_path(
        PROJECT_DIR, model_spec, args.ranking_name
    )
    ranking = load_ranked_heads_json(ranking_path)
    qr_heads = ranking[: args.k]
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    num_heads = get_num_heads(model)
    cumulative_sizes = args.cumulative_sizes or default_cumulative_sizes(args.k)
    cumulative_sizes = [size for size in cumulative_sizes if 1 <= size <= args.k]
    if not cumulative_sizes:
        raise ValueError("No valid cumulative subset sizes.")

    ablation_results_path = (
        Path(args.ablation_results_path)
        if args.ablation_results_path
        else default_ablation_results_path(PROJECT_DIR, model_spec)
    )
    failure_ids = None if args.use_all_examples else load_clean_to_ablated_failure_ids(
        ablation_results_path, args.k
    )
    instances = load_task_instances(PROJECT_DIR, args.tasks)
    selected = select_instances(
        instances,
        failure_ids=failure_ids,
        max_examples_per_task=args.max_examples_per_task,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No examples selected. Try --use_all_examples.")

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_DIR / "results" / "mech_experiments" / model_spec.model_slug / "head_subset_patching"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cumulative_conditions = build_cumulative_conditions(
        qr_heads=qr_heads,
        ranking=ranking,
        num_layers=num_layers,
        num_heads=num_heads,
        sizes=cumulative_sizes,
        seed=args.seed,
        num_random_controls=args.num_random_controls,
        num_matched_layer_random_controls=args.num_matched_layer_random_controls,
    )

    metadata = {
        "model": model_spec.as_metadata(),
        "tasks": args.tasks,
        "k": args.k,
        "ranking_path": str(ranking_path),
        "ablation_results_path": str(ablation_results_path),
        "num_selected_examples": len(selected),
        "filtered_to_clean_correct_ablated_wrong": failure_ids is not None,
        "patch_position_mode": args.patch_position_mode,
        "subset_mode": args.subset_mode,
        "cumulative_sizes": cumulative_sizes,
        "num_random_controls": args.num_random_controls,
        "num_matched_layer_random_controls": args.num_matched_layer_random_controls,
        "qr_heads": [head_name(head) for head in qr_heads],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    controller = HeadPatchController(model)
    controller.install()
    condition_rows = []
    rescue_rows = []

    try:
        for n, inst in enumerate(selected, start=1):
            encoded = encode_prompt_and_answer(
                tokenizer,
                inst,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            patch_positions = intervention_positions_for_mode(
                encoded,
                inst,
                args.patch_position_mode,
            )
            if not patch_positions:
                print(f"[skip] no {args.patch_position_mode} positions for {inst['idx']}")
                continue

            controller.set_cache_mode(patch_positions)
            clean_sum, clean_mean, clean_toks = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_row(
                    inst, "clean", "clean", "baseline", 0, [], patch_positions,
                    clean_sum, clean_mean, clean_toks,
                )
            )

            controller.set_ablate_mode(qr_heads)
            ablated_sum, ablated_mean, ablated_toks = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_row(
                    inst, "full_qr_ablation", "qr", "ablation", args.k, qr_heads,
                    patch_positions, ablated_sum, ablated_mean, ablated_toks,
                )
            )

            if args.subset_mode in {"individual", "both"}:
                for rank, head in enumerate(qr_heads):
                    condition = f"individual_qr_rank{rank:02d}_{head_name(head)}"
                    controller.set_patch_mode(qr_heads, [head], patch_positions)
                    patch_sum, patch_mean, patch_toks = score_gold_answer(model, encoded)
                    condition_rows.append(
                        condition_row(
                            inst, condition, "qr", "individual", 1, [head],
                            patch_positions, patch_sum, patch_mean, patch_toks,
                        )
                    )
                    row = rescue_row(
                        inst, condition, "qr", "individual", 1, [head],
                        clean_sum, ablated_sum, patch_sum,
                    )
                    row["head_rank"] = rank
                    row["head"] = head_name(head)
                    rescue_rows.append(row)

            if args.subset_mode in {"cumulative", "both"}:
                for condition, subset_type, heads in cumulative_conditions:
                    controller.set_patch_mode(qr_heads, heads, patch_positions)
                    patch_sum, patch_mean, patch_toks = score_gold_answer(model, encoded)
                    condition_rows.append(
                        condition_row(
                            inst, condition, subset_type, "cumulative", len(heads), heads,
                            patch_positions, patch_sum, patch_mean, patch_toks,
                        )
                    )
                    rescue_rows.append(
                        rescue_row(
                            inst, condition, subset_type, "cumulative", len(heads), heads,
                            clean_sum, ablated_sum, patch_sum,
                        )
                    )

            controller.reset()
            clear_device_cache(next(model.parameters()).device)
            print(f"[{n}/{len(selected)}] {inst['task']} {inst['idx']}")
    finally:
        controller.remove()
        controller.reset()

    write_jsonl(output_dir / "head_subset_conditions.jsonl", condition_rows)
    write_csv(
        output_dir / "head_subset_conditions.csv",
        [
            {key: value for key, value in row.items() if key != "token_logprobs"}
            for row in condition_rows
        ],
    )
    write_jsonl(output_dir / "head_subset_rescue.jsonl", rescue_rows)
    write_csv(output_dir / "head_subset_rescue.csv", rescue_rows)
    summary = summarize(rescue_rows)
    write_csv(output_dir / "head_subset_summary.csv", summary)
    with open(output_dir / "head_subset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_cumulative(output_dir, rescue_rows)

    print(f"Wrote head-subset patching results to {output_dir}")


if __name__ == "__main__":
    main()
