"""Position-specific QRHead activation patching.

This is a localization control for the activation-patching rescue result. It
ablates the same top-K QRHeads everywhere, then restores clean QRHead outputs at
different token-position sets and measures teacher-forced gold-answer logprob.

The intended comparison is answer-prediction positions versus query, early
prompt, and random prompt positions with the same number of patched positions.
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
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    HeadPatchController,
    PROJECT_DIR,
    default_ablation_results_path,
    default_ranking_path,
    encode_prompt_and_answer,
    get_decoder_layers,
    get_num_heads,
    load_clean_to_ablated_failure_ids,
    load_model_and_tokenizer,
    load_ranked_heads_json,
    load_task_instances,
    safe_div,
    select_instances,
    score_gold_answer,
    write_csv,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run position-specific QRHead activation patching."
    )
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
    parser.add_argument("--num_random_position_controls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--use_all_examples",
        action="store_true",
        help="Do not filter to examples that clean generation solved and K-ablation missed.",
    )
    parser.add_argument(
        "--answer_prefix",
        default="",
        help="Optional prefix prepended to the gold answer before tokenization.",
    )
    return parser.parse_args()


def nonempty_prompt_positions(encoded) -> List[int]:
    positions = []
    for pos, (start, end) in enumerate(encoded.prompt_offsets):
        if 0 <= pos < encoded.prompt_len and end > start:
            positions.append(pos)
    return positions


def positions_for_substring(encoded, text: str) -> List[int]:
    if not text:
        return []
    start = encoded.rendered_prompt.find(text)
    if start < 0:
        return []
    end = start + len(text)
    positions = []
    for pos, (tok_start, tok_end) in enumerate(encoded.prompt_offsets):
        if tok_end <= tok_start:
            continue
        if tok_start < end and tok_end > start:
            positions.append(pos)
    return positions


def take_tail(positions: Sequence[int], n_positions: int) -> List[int]:
    if n_positions <= 0:
        return []
    if len(positions) <= n_positions:
        return list(positions)
    return list(positions[-n_positions:])


def take_head(positions: Sequence[int], n_positions: int) -> List[int]:
    if n_positions <= 0:
        return []
    return list(positions[:n_positions])


def sample_random_positions(
    *,
    candidates: Sequence[int],
    n_positions: int,
    seed: int,
) -> List[int]:
    if n_positions <= 0:
        return []
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) <= n_positions:
        return sorted(candidates)
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n_positions))


def build_position_sets(encoded, inst: dict, *, n_random: int, seed: int) -> Dict[str, List[int]]:
    n_positions = len(encoded.prediction_positions)
    prompt_positions = nonempty_prompt_positions(encoded)
    answer_prompt_pos = {pos for pos in encoded.prediction_positions if pos < encoded.prompt_len}
    prompt_candidates = [pos for pos in prompt_positions if pos not in answer_prompt_pos]

    query_positions = positions_for_substring(encoded, inst.get("question", ""))
    if not query_positions:
        query_marker = "Question:"
        marker_positions = positions_for_substring(encoded, query_marker)
        if marker_positions:
            start = marker_positions[-1] + 1
            query_positions = [pos for pos in prompt_positions if start <= pos < encoded.prompt_len]

    position_sets = {
        "answer": list(encoded.prediction_positions),
        "query": take_tail(query_positions, n_positions),
        "early_prompt": take_head(prompt_candidates, n_positions),
    }
    for i in range(n_random):
        position_sets[f"random_prompt_{i:02d}"] = sample_random_positions(
            candidates=prompt_candidates,
            n_positions=n_positions,
            seed=seed + i,
        )
    return {name: positions for name, positions in position_sets.items() if positions}


def condition_record(inst, condition, position_count, positions, sum_logprob, mean_logprob, token_logprobs):
    return {
        "idx": inst["idx"],
        "task": inst["task"],
        "condition": condition,
        "gold": inst["needle_value"],
        "position_count": position_count,
        "positions": positions,
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "num_answer_tokens": len(token_logprobs),
        "token_logprobs": token_logprobs,
    }


def summarize(rescue_rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rescue_rows:
        value = row.get("sum_logprob_rescue")
        if value is None or math.isnan(float(value)):
            continue
        grouped[(row["task"], row["condition"])].append(float(value))
    out = []
    for (task, condition), values in sorted(grouped.items()):
        out.append({
            "task": task,
            "condition": condition,
            "n": len(values),
            "mean_sum_logprob_rescue": sum(values) / len(values),
            "median_sum_logprob_rescue": statistics.median(values),
            "positive_fraction": sum(v > 0 for v in values) / len(values),
        })
    return out


def plot_outputs(output_dir: Path, rescue_rows: Sequence[dict]) -> None:
    grouped = defaultdict(list)
    for row in rescue_rows:
        value = row.get("sum_logprob_rescue")
        if value is None or math.isnan(float(value)):
            continue
        grouped[row["condition"]].append(float(value))
    if not grouped:
        return
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    plt.figure(figsize=(6.0, 3.2))
    plt.bar(labels, values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("Recovered fraction of lost logprob")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "position_specific_rescue_bar.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    model_spec, tokenizer, model = load_model_and_tokenizer(args, for_detection=False)

    ranking_path = Path(args.ranking_path) if args.ranking_path else default_ranking_path(
        PROJECT_DIR, model_spec, args.ranking_name
    )
    ranking = load_ranked_heads_json(ranking_path)
    qr_heads = ranking[: args.k]

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
        raise RuntimeError(
            "No examples selected. Try --use_all_examples or lower filtering requirements."
        )

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_DIR / "results" / "mech_experiments" / model_spec.model_slug / "position_specific_patching"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model_spec.as_metadata(),
        "tasks": args.tasks,
        "k": args.k,
        "ranking_path": str(ranking_path),
        "ablation_results_path": str(ablation_results_path),
        "num_selected_examples": len(selected),
        "filtered_to_clean_correct_ablated_wrong": failure_ids is not None,
        "num_random_position_controls": args.num_random_position_controls,
        "position_policy": {
            "answer": "gold-answer prediction positions",
            "query": "last N question-token positions, where N is answer-token count",
            "early_prompt": "first N non-special prompt-token positions",
            "random_prompt": "N random non-special prompt-token positions excluding answer-prediction prompt token",
        },
        "qr_heads": [f"{layer}-{head}" for layer, head in qr_heads],
        "num_layers": len(get_decoder_layers(model)),
        "num_heads": get_num_heads(model),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    controller = HeadPatchController(model)
    controller.install()
    condition_rows = []
    rescue_rows = []

    try:
        for inst_idx, inst in enumerate(selected):
            encoded = encode_prompt_and_answer(
                tokenizer,
                inst,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )

            position_sets = build_position_sets(
                encoded,
                inst,
                n_random=args.num_random_position_controls,
                seed=args.seed + inst_idx * 100,
            )

            controller.set_cache_mode(encoded.prediction_positions)
            clean_sum, clean_mean, clean_token = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_record(
                    inst,
                    "clean",
                    len(encoded.prediction_positions),
                    encoded.prediction_positions,
                    clean_sum,
                    clean_mean,
                    clean_token,
                )
            )

            controller.set_ablate_mode(qr_heads)
            ablated_sum, ablated_mean, ablated_token = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_record(
                    inst,
                    "qr_ablate",
                    0,
                    [],
                    ablated_sum,
                    ablated_mean,
                    ablated_token,
                )
            )
            denom_sum = clean_sum - ablated_sum
            denom_mean = clean_mean - ablated_mean

            for condition, positions in position_sets.items():
                controller.set_cache_mode(positions)
                cache_sum, cache_mean, cache_token = score_gold_answer(model, encoded)
                condition_rows.append(
                    condition_record(
                        inst,
                        f"cache_{condition}",
                        len(positions),
                        positions,
                        cache_sum,
                        cache_mean,
                        cache_token,
                    )
                )

                controller.set_patch_mode(qr_heads, qr_heads, positions)
                patched_sum, patched_mean, patched_token = score_gold_answer(model, encoded)
                patch_condition = f"patch_{condition}"
                condition_rows.append(
                    condition_record(
                        inst,
                        patch_condition,
                        len(positions),
                        positions,
                        patched_sum,
                        patched_mean,
                        patched_token,
                    )
                )
                rescue_rows.append({
                    "idx": inst["idx"],
                    "task": inst["task"],
                    "condition": patch_condition,
                    "gold": inst["needle_value"],
                    "position_count": len(positions),
                    "positions": positions,
                    "clean_sum_logprob": clean_sum,
                    "ablated_sum_logprob": ablated_sum,
                    "patched_sum_logprob": patched_sum,
                    "sum_logprob_rescue": safe_div(patched_sum - ablated_sum, denom_sum),
                    "clean_mean_logprob": clean_mean,
                    "ablated_mean_logprob": ablated_mean,
                    "patched_mean_logprob": patched_mean,
                    "mean_logprob_rescue": safe_div(patched_mean - ablated_mean, denom_mean),
                    "cache_clean_sum_logprob": cache_sum,
                    "num_answer_tokens": len(patched_token),
                })
    finally:
        controller.remove()
        controller.reset()

    write_jsonl(output_dir / "position_specific_conditions.jsonl", condition_rows)
    write_csv(output_dir / "position_specific_conditions.csv", condition_rows)
    write_jsonl(output_dir / "position_specific_rescue.jsonl", rescue_rows)
    write_csv(output_dir / "position_specific_rescue.csv", rescue_rows)
    summary = summarize(rescue_rows)
    write_csv(output_dir / "position_specific_summary.csv", summary)
    with open(output_dir / "position_specific_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_outputs(output_dir, rescue_rows)

    print(f"Wrote position-specific patching results to {output_dir}")
    print(f"Selected {len(selected)} examples")


if __name__ == "__main__":
    main()
