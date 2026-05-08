"""Build clean/corrupted pairs for QRHead circuit discovery."""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    HeadPatchController,
    PROJECT_DIR,
    answer_token_len,
    default_ablation_results_path,
    default_ranking_path,
    encode_prompt_and_answer,
    load_model_and_tokenizer,
    load_ranked_heads_json,
    load_task_instances,
    make_counterfactual_instance,
    score_contrastive_margin,
    score_gold_answer,
    with_answer,
    write_csv,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build clean/corrupted circuit pairs.")
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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_pairs_per_task", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument("--min_clean_margin", type=float, default=0.0)
    parser.add_argument("--min_corruption_effect", type=float, default=0.5)
    parser.add_argument("--min_ablation_damage", type=float, default=0.5)
    parser.add_argument("--allow_prompt_len_change", action="store_true")
    return parser.parse_args()


def flat_pair_row(row):
    keep = [
        "task",
        "source_idx",
        "alt_idx",
        "source_answer",
        "alt_answer",
        "clean_margin",
        "corrupted_margin",
        "corruption_effect",
        "clean_gold_sum_logprob",
        "clean_alt_sum_logprob",
        "corrupted_gold_sum_logprob",
        "corrupted_alt_sum_logprob",
        "qr_ablated_gold_sum_logprob",
        "qr_ablation_damage",
        "prompt_len",
        "corrupted_prompt_len",
        "answer_token_len",
    ]
    return {key: row.get(key) for key in keep}


def main():
    args = parse_args()
    model_spec, tokenizer, model = load_model_and_tokenizer(args, for_detection=False)
    ranking_path = Path(args.ranking_path) if args.ranking_path else default_ranking_path(
        PROJECT_DIR, model_spec, args.ranking_name
    )
    qr_heads = load_ranked_heads_json(ranking_path)[: args.k]
    instances = load_task_instances(PROJECT_DIR, args.tasks)
    by_task = defaultdict(list)
    for inst in instances:
        by_task[inst["task"]].append(inst)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    controller = HeadPatchController(model)
    controller.install()
    pairs = []

    try:
        for task in args.tasks:
            bases = list(by_task.get(task, []))
            alts = list(by_task.get(task, []))
            rng.shuffle(bases)
            rng.shuffle(alts)
            task_count = 0
            for base in bases:
                if task_count >= args.max_pairs_per_task:
                    break
                source_answer = str(base.get("needle_value", ""))
                if not source_answer:
                    continue
                source_len = answer_token_len(tokenizer, source_answer, args.answer_prefix)
                candidates = []
                for alt in alts:
                    alt_answer = str(alt.get("needle_value", ""))
                    if alt.get("idx") == base.get("idx") or alt_answer == source_answer:
                        continue
                    if answer_token_len(tokenizer, alt_answer, args.answer_prefix) != source_len:
                        continue
                    candidates.append(alt)
                rng.shuffle(candidates)
                for alt in candidates:
                    alt_answer = str(alt.get("needle_value", ""))
                    corrupted = make_counterfactual_instance(base, alt)
                    if corrupted is None:
                        continue
                    try:
                        clean = score_contrastive_margin(
                            model,
                            tokenizer,
                            base,
                            source_answer,
                            alt_answer,
                            max_context_tokens=args.max_context_tokens,
                            answer_prefix=args.answer_prefix,
                        )
                        corrupted_score = score_contrastive_margin(
                            model,
                            tokenizer,
                            corrupted,
                            source_answer,
                            alt_answer,
                            max_context_tokens=args.max_context_tokens,
                            answer_prefix=args.answer_prefix,
                        )
                        clean_gold_encoded = encode_prompt_and_answer(
                            tokenizer,
                            with_answer(base, source_answer),
                            max_context_tokens=args.max_context_tokens,
                            answer_prefix=args.answer_prefix,
                        )
                        corrupted_gold_encoded = encode_prompt_and_answer(
                            tokenizer,
                            with_answer(corrupted, source_answer),
                            max_context_tokens=args.max_context_tokens,
                            answer_prefix=args.answer_prefix,
                        )
                    except Exception:
                        continue
                    if (
                        not args.allow_prompt_len_change
                        and clean_gold_encoded.prompt_len != corrupted_gold_encoded.prompt_len
                    ):
                        continue
                    controller.set_ablate_mode(qr_heads)
                    ablated_gold_sum, _, _ = score_gold_answer(model, clean_gold_encoded)
                    controller.reset()
                    ablation_damage = clean["positive_sum_logprob"] - ablated_gold_sum
                    corruption_effect = clean["margin"] - corrupted_score["margin"]
                    if clean["margin"] < args.min_clean_margin:
                        continue
                    if corruption_effect < args.min_corruption_effect:
                        continue
                    if ablation_damage < args.min_ablation_damage:
                        continue
                    row = {
                        "task": task,
                        "source_idx": base["idx"],
                        "alt_idx": alt["idx"],
                        "source_answer": source_answer,
                        "alt_answer": alt_answer,
                        "clean_margin": clean["margin"],
                        "corrupted_margin": corrupted_score["margin"],
                        "corruption_effect": corruption_effect,
                        "clean_gold_sum_logprob": clean["positive_sum_logprob"],
                        "clean_alt_sum_logprob": clean["negative_sum_logprob"],
                        "corrupted_gold_sum_logprob": corrupted_score["positive_sum_logprob"],
                        "corrupted_alt_sum_logprob": corrupted_score["negative_sum_logprob"],
                        "qr_ablated_gold_sum_logprob": ablated_gold_sum,
                        "qr_ablation_damage": ablation_damage,
                        "prompt_len": clean_gold_encoded.prompt_len,
                        "corrupted_prompt_len": corrupted_gold_encoded.prompt_len,
                        "answer_token_len": source_len,
                        "clean_instance": base,
                        "corrupted_instance": corrupted,
                    }
                    pairs.append(row)
                    task_count += 1
                    print(f"[keep] {task} {task_count}/{args.max_pairs_per_task} {base['idx']} -> {alt['idx']}")
                    break
    finally:
        controller.remove()

    write_jsonl(output_dir / "circuit_pairs.jsonl", pairs)
    write_csv(output_dir / "circuit_pairs.csv", [flat_pair_row(row) for row in pairs])
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_spec.as_metadata(),
            "tasks": args.tasks,
            "k": args.k,
            "ranking_path": str(ranking_path),
            "num_pairs": len(pairs),
            "min_clean_margin": args.min_clean_margin,
            "min_corruption_effect": args.min_corruption_effect,
            "min_ablation_damage": args.min_ablation_damage,
            "allow_prompt_len_change": args.allow_prompt_len_change,
        }, f, indent=2)
    print(f"Wrote {len(pairs)} circuit pairs to {output_dir}")


if __name__ == "__main__":
    main()
