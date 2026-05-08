"""Activation-patching rescue experiment for QRHeads.

This script tests whether clean QRHead activations restore gold-answer
likelihood after QRHead ablation. It uses teacher-forced scoring by default:
for each selected SEC NIAH example, the model sees the prompt plus the gold
answer, and we score the gold answer tokens under clean, ablated, and patched
conditions.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (
    DEFAULT_TASKS,
    HeadPatchController,
    PROJECT_DIR,
    bottom_control_heads,
    clear_device_cache,
    default_ablation_results_path,
    default_ranking_path,
    encode_prompt_and_answer,
    get_decoder_layers,
    get_num_heads,
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
    summarize_by_task_and_condition,
    write_csv,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run teacher-forced QRHead activation-patching rescue."
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--num_random_controls",
        type=int,
        default=1,
        help="Number of full-population random non-QR head-set controls.",
    )
    parser.add_argument(
        "--num_matched_layer_random_controls",
        type=int,
        default=0,
        help="Number of random non-QR controls matched to QRHead layer counts.",
    )
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


def condition_record(inst, condition, sum_logprob, mean_logprob, token_logprobs):
    return {
        "idx": inst["idx"],
        "task": inst["task"],
        "condition": condition,
        "gold": inst["needle_value"],
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "num_answer_tokens": len(token_logprobs),
        "token_logprobs": token_logprobs,
    }


def plot_outputs(output_dir: Path, condition_rows, rescue_rows):
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = ["clean", "qr_ablate", "patch_qr"]
    values = {condition: [] for condition in wanted}
    for row in condition_rows:
        if row["condition"] in values:
            values[row["condition"]].append(row["mean_logprob"])
    labels = [label for label in wanted if values[label]]
    if labels:
        plt.figure(figsize=(5.0, 3.2))
        data = [values[label] for label in labels]
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Gold answer mean logprob")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "activation_patching_logprob_boxplot.png", dpi=200)
        plt.close()

    by_condition = {}
    for row in rescue_rows:
        value = row.get("sum_logprob_rescue")
        if value is None or math.isnan(float(value)):
            continue
        by_condition.setdefault(row["condition"], []).append(float(value))
    labels = sorted(by_condition)
    if labels:
        means = [sum(by_condition[label]) / len(by_condition[label]) for label in labels]
        plt.figure(figsize=(5.0, 3.2))
        plt.bar(labels, means)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        plt.ylabel("Recovered fraction of lost logprob")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "activation_patching_rescue_bar.png", dpi=200)
        plt.close()


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
    controls = {
        "patch_qr": qr_heads,
        "patch_bottom": bottom_control_heads(
            ranking=ranking,
            qr_heads=qr_heads,
            k=args.k,
        ),
        "patch_same_layer": same_layer_control_heads(
            qr_heads=qr_heads,
            num_heads=num_heads,
        ),
    }
    if args.num_random_controls == 1:
        controls["patch_random"] = random_control_heads(
            qr_heads=qr_heads,
            num_layers=num_layers,
            num_heads=num_heads,
            seed=args.seed,
        )
    else:
        for i in range(args.num_random_controls):
            controls[f"patch_random_{i:02d}"] = random_control_heads(
                qr_heads=qr_heads,
                num_layers=num_layers,
                num_heads=num_heads,
                seed=args.seed + i,
            )
    for i in range(args.num_matched_layer_random_controls):
        controls[f"patch_matched_layer_random_{i:02d}"] = matched_layer_random_control_heads(
            qr_heads=qr_heads,
            num_heads=num_heads,
            seed=args.seed + 10_000 + i,
        )

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
        PROJECT_DIR / "results" / "mech_experiments" / model_spec.model_slug / "activation_patching"
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
        "controls": {
            name: [f"{layer}-{head}" for layer, head in heads]
            for name, heads in controls.items()
        },
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

            controller.set_cache_mode(encoded.prediction_positions)
            clean_sum, clean_mean, clean_token = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_record(inst, "clean", clean_sum, clean_mean, clean_token)
            )

            controller.set_ablate_mode(qr_heads)
            ablated_sum, ablated_mean, ablated_token = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_record(inst, "qr_ablate", ablated_sum, ablated_mean, ablated_token)
            )

            denom_sum = clean_sum - ablated_sum
            denom_mean = clean_mean - ablated_mean
            for condition, patch_heads in controls.items():
                controller.set_patch_mode(
                    qr_heads,
                    patch_heads,
                    encoded.prediction_positions,
                )
                patched_sum, patched_mean, patched_token = score_gold_answer(model, encoded)
                condition_rows.append(
                    condition_record(inst, condition, patched_sum, patched_mean, patched_token)
                )
                rescue_rows.append({
                    "idx": inst["idx"],
                    "task": inst["task"],
                    "condition": condition,
                    "gold": inst["needle_value"],
                    "clean_sum_logprob": clean_sum,
                    "ablated_sum_logprob": ablated_sum,
                    "patched_sum_logprob": patched_sum,
                    "sum_logprob_rescue": safe_div(patched_sum - ablated_sum, denom_sum),
                    "clean_mean_logprob": clean_mean,
                    "ablated_mean_logprob": ablated_mean,
                    "patched_mean_logprob": patched_mean,
                    "mean_logprob_rescue": safe_div(patched_mean - ablated_mean, denom_mean),
                    "num_answer_tokens": len(patched_token),
                })

            controller.reset()
            clear_device_cache(next(model.parameters()).device)
            print(f"[{n}/{len(selected)}] {inst['task']} {inst['idx']}")
    finally:
        controller.remove()

    write_jsonl(output_dir / "activation_patching_conditions.jsonl", condition_rows)
    write_csv(
        output_dir / "activation_patching_conditions.csv",
        [
            {k: v for k, v in row.items() if k != "token_logprobs"}
            for row in condition_rows
        ],
    )
    write_jsonl(output_dir / "activation_patching_rescue.jsonl", rescue_rows)
    write_csv(output_dir / "activation_patching_rescue.csv", rescue_rows)

    summary = {
        "condition_mean_logprob": summarize_by_task_and_condition(
            condition_rows, "mean_logprob"
        ),
        "rescue_sum_logprob": summarize_by_task_and_condition(
            rescue_rows, "sum_logprob_rescue"
        ),
        "rescue_mean_logprob": summarize_by_task_and_condition(
            rescue_rows, "mean_logprob_rescue"
        ),
    }
    with open(output_dir / "activation_patching_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / "activation_patching_rescue_summary.csv", summary["rescue_sum_logprob"])
    plot_outputs(output_dir, condition_rows, rescue_rows)
    print(f"Wrote activation-patching results to {output_dir}")


if __name__ == "__main__":
    main()
