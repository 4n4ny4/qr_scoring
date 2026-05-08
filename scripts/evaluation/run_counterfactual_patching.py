"""Counterfactual answer patching for QRHeads.

This experiment asks whether QRHead activations carry answer-specific evidence,
not merely generic clean-run state. For a SEC extraction instance, we replace the
gold value in the evidence sentence with another plausible value from the same
task, then patch clean activations from the original prompt into the
counterfactual prompt and score both the original and counterfactual answers.

Primary readout: the shift in

    log p(original answer) - log p(counterfactual answer)

under QRHead patching relative to controls.
"""

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    default_ablation_results_path,
    default_ranking_path,
    encode_prompt_and_answer,
    get_decoder_layers,
    get_num_heads,
    load_clean_to_ablated_failure_ids,
    load_model_and_tokenizer,
    load_ranked_heads_json,
    load_task_instances,
    random_control_heads,
    same_layer_control_heads,
    score_gold_answer,
    write_csv,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run counterfactual answer patching for QRHeads."
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
    parser.add_argument("--max_pairs_per_task", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--use_all_examples",
        action="store_true",
        help="Do not filter bases to examples that clean generation solved and K-ablation missed.",
    )
    parser.add_argument(
        "--allow_prompt_len_change",
        action="store_true",
        help="Allow counterfactual replacement to change prompt token length. Default skips these pairs.",
    )
    parser.add_argument(
        "--answer_prefix",
        default="",
        help="Optional prefix prepended to each answer before tokenization.",
    )
    return parser.parse_args()


def answer_surfaces(value: str) -> List[str]:
    value = str(value)
    surfaces = [value]
    digits = re_digits(value)
    if digits and digits == value:
        try:
            surfaces.append(f"{int(value):,}")
        except ValueError:
            pass
    # Preserve order while deduplicating.
    seen = set()
    out = []
    for surface in surfaces:
        if surface and surface not in seen:
            out.append(surface)
            seen.add(surface)
    return out


def re_digits(value: str) -> Optional[str]:
    stripped = value.replace(",", "").strip()
    return stripped if stripped.isdigit() else None


def find_surface(inst: dict) -> Optional[str]:
    sentence = inst.get("needle_sentence") or ""
    context = inst.get("context") or ""
    for surface in answer_surfaces(str(inst.get("needle_value", ""))):
        if surface in sentence or surface in context:
            return surface
    return None


def surface_for_alt(base_surface: str, alt_value: str) -> Optional[str]:
    alt_value = str(alt_value)
    if "," in base_surface:
        digits = re_digits(alt_value)
        if digits is None:
            return None
        return f"{int(digits):,}"
    return alt_value


def replace_first(value: str, old: str, new: str) -> Tuple[str, bool]:
    if old not in value:
        return value, False
    return value.replace(old, new, 1), True


def make_counterfactual_instance(base: dict, alt: dict) -> Optional[dict]:
    source_value = str(base.get("needle_value", ""))
    alt_value = str(alt.get("needle_value", ""))
    if not source_value or not alt_value or source_value == alt_value:
        return None

    source_surface = find_surface(base)
    if source_surface is None:
        return None
    alt_surface = surface_for_alt(source_surface, alt_value)
    if alt_surface is None or alt_surface == source_surface:
        return None

    cf = copy.deepcopy(base)
    sentence = cf.get("needle_sentence") or ""
    context = cf.get("context") or ""

    new_sentence, sentence_changed = replace_first(sentence, source_surface, alt_surface)
    if not sentence_changed:
        return None

    if sentence in context:
        new_context, context_changed = replace_first(context, sentence, new_sentence)
    else:
        new_context, context_changed = replace_first(context, source_surface, alt_surface)
    if not context_changed:
        return None

    cf["context"] = new_context
    cf["needle_sentence"] = new_sentence
    cf["needle_value"] = alt_value
    cf["counterfactual_from_idx"] = base.get("idx")
    cf["counterfactual_alt_idx"] = alt.get("idx")
    cf["counterfactual_source_value"] = source_value
    cf["counterfactual_alt_value"] = alt_value
    cf["counterfactual_source_surface"] = source_surface
    cf["counterfactual_alt_surface"] = alt_surface

    for para in cf.get("paragraphs", []):
        text = para.get("paragraph_text") or ""
        if sentence in text:
            para["paragraph_text"], _ = replace_first(text, sentence, new_sentence)
        elif para.get("idx") == "needle_chunk" and source_surface in text:
            para["paragraph_text"], _ = replace_first(text, source_surface, alt_surface)

    return cf


def with_answer(inst: dict, answer: str) -> dict:
    row = dict(inst)
    row["needle_value"] = str(answer)
    return row


def answer_token_len(tokenizer, answer: str, answer_prefix: str) -> int:
    ids = tokenizer(
        f"{answer_prefix}{answer}",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    return len(ids)


def build_pairs(
    tokenizer,
    instances: Sequence[dict],
    *,
    tasks: Sequence[str],
    failure_ids: Optional[set],
    max_pairs_per_task: int,
    max_context_tokens: int,
    answer_prefix: str,
    allow_prompt_len_change: bool,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for inst in instances:
        if inst.get("task") in tasks and (failure_ids is None or inst["idx"] in failure_ids):
            by_task[inst["task"]].append(inst)

    pairs = []
    for task in tasks:
        bases = list(by_task.get(task, []))
        alts = list(by_task.get(task, []))
        rng.shuffle(bases)
        rng.shuffle(alts)
        task_pairs = 0
        for base in bases:
            if task_pairs >= max_pairs_per_task:
                break
            source_answer = str(base.get("needle_value", ""))
            source_len = answer_token_len(tokenizer, source_answer, answer_prefix)
            candidates = []
            for alt in alts:
                alt_answer = str(alt.get("needle_value", ""))
                if alt.get("idx") == base.get("idx") or alt_answer == source_answer:
                    continue
                if answer_token_len(tokenizer, alt_answer, answer_prefix) != source_len:
                    continue
                candidates.append(alt)
            rng.shuffle(candidates)
            for alt in candidates:
                cf = make_counterfactual_instance(base, alt)
                if cf is None:
                    continue
                try:
                    enc_orig = encode_prompt_and_answer(
                        tokenizer,
                        with_answer(base, source_answer),
                        max_context_tokens=max_context_tokens,
                        answer_prefix=answer_prefix,
                    )
                    enc_cf_orig = encode_prompt_and_answer(
                        tokenizer,
                        with_answer(cf, source_answer),
                        max_context_tokens=max_context_tokens,
                        answer_prefix=answer_prefix,
                    )
                    enc_cf_alt = encode_prompt_and_answer(
                        tokenizer,
                        with_answer(cf, alt["needle_value"]),
                        max_context_tokens=max_context_tokens,
                        answer_prefix=answer_prefix,
                    )
                except Exception:
                    continue
                if len(enc_orig.answer_ids) != len(enc_cf_orig.answer_ids):
                    continue
                if len(enc_orig.answer_ids) != len(enc_cf_alt.answer_ids):
                    continue
                if not allow_prompt_len_change and enc_orig.prompt_len != enc_cf_orig.prompt_len:
                    continue
                pairs.append({
                    "task": task,
                    "base": base,
                    "counterfactual": cf,
                    "source_answer": source_answer,
                    "alt_answer": str(alt["needle_value"]),
                    "source_idx": base["idx"],
                    "alt_idx": alt["idx"],
                    "answer_tokens": len(enc_orig.answer_ids),
                    "prompt_len": enc_orig.prompt_len,
                    "counterfactual_prompt_len": enc_cf_orig.prompt_len,
                })
                task_pairs += 1
                break
    return pairs


def condition_row(pair: dict, condition: str, answer_label: str, answer: str, sum_lp: float, mean_lp: float, toks: List[float]) -> dict:
    return {
        "task": pair["task"],
        "source_idx": pair["source_idx"],
        "alt_idx": pair["alt_idx"],
        "condition": condition,
        "answer_label": answer_label,
        "answer": answer,
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "num_answer_tokens": len(toks),
        "token_logprobs": toks,
    }


def score_two_answers(model, tokenizer, inst: dict, source_answer: str, alt_answer: str, args):
    enc_source = encode_prompt_and_answer(
        tokenizer,
        with_answer(inst, source_answer),
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )
    source_sum, source_mean, source_tok = score_gold_answer(model, enc_source)
    enc_alt = encode_prompt_and_answer(
        tokenizer,
        with_answer(inst, alt_answer),
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )
    alt_sum, alt_mean, alt_tok = score_gold_answer(model, enc_alt)
    return (enc_source, source_sum, source_mean, source_tok), (enc_alt, alt_sum, alt_mean, alt_tok)


def summarize_effects(effect_rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in effect_rows:
        grouped[(row["task"], row["condition"])].append(row)
    out = []
    for (task, condition), rows in sorted(grouped.items()):
        shifts = [float(r["margin_shift_vs_cf_clean"]) for r in rows]
        shifts_ablate = [float(r["margin_shift_vs_cf_qr_ablate"]) for r in rows]
        margins = [float(r["margin"]) for r in rows]
        out.append({
            "task": task,
            "condition": condition,
            "n": len(rows),
            "mean_margin": sum(margins) / len(margins),
            "median_margin": statistics.median(margins),
            "mean_margin_shift_vs_cf_clean": sum(shifts) / len(shifts),
            "median_margin_shift_vs_cf_clean": statistics.median(shifts),
            "mean_margin_shift_vs_cf_qr_ablate": sum(shifts_ablate) / len(shifts_ablate),
            "median_margin_shift_vs_cf_qr_ablate": statistics.median(shifts_ablate),
        })
    return out


def plot_effects(output_dir: Path, effect_rows: Sequence[dict]) -> None:
    grouped = defaultdict(list)
    for row in effect_rows:
        if row["condition"].startswith("patch_"):
            grouped[row["condition"]].append(float(row["margin_shift_vs_cf_qr_ablate"]))
    if not grouped:
        return
    labels = sorted(grouped)
    values = [statistics.median(grouped[label]) for label in labels]
    plt.figure(figsize=(5.0, 3.2))
    plt.bar(labels, values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Median shift in log p(orig) - log p(cf)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "counterfactual_margin_shift_bar.png", dpi=200)
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
        "patch_random": random_control_heads(
            qr_heads=qr_heads,
            num_layers=num_layers,
            num_heads=num_heads,
            seed=args.seed,
        ),
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

    ablation_results_path = (
        Path(args.ablation_results_path)
        if args.ablation_results_path
        else default_ablation_results_path(PROJECT_DIR, model_spec)
    )
    failure_ids = None if args.use_all_examples else load_clean_to_ablated_failure_ids(
        ablation_results_path, args.k
    )
    instances = load_task_instances(PROJECT_DIR, args.tasks)
    pairs = build_pairs(
        tokenizer,
        instances,
        tasks=args.tasks,
        failure_ids=failure_ids,
        max_pairs_per_task=args.max_pairs_per_task,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
        allow_prompt_len_change=args.allow_prompt_len_change,
        seed=args.seed,
    )
    if not pairs:
        raise RuntimeError(
            "No counterfactual pairs selected. Try --use_all_examples, "
            "--allow_prompt_len_change, or fewer tasks."
        )

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_DIR / "results" / "mech_experiments" / model_spec.model_slug / "counterfactual_patching"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model_spec.as_metadata(),
        "tasks": args.tasks,
        "k": args.k,
        "ranking_path": str(ranking_path),
        "ablation_results_path": str(ablation_results_path),
        "num_selected_pairs": len(pairs),
        "filtered_to_clean_correct_ablated_wrong": failure_ids is not None,
        "allow_prompt_len_change": args.allow_prompt_len_change,
        "controls": {
            name: [f"{layer}-{head}" for layer, head in heads]
            for name, heads in controls.items()
        },
        "pairs": [
            {
                key: pair[key]
                for key in [
                    "task",
                    "source_idx",
                    "alt_idx",
                    "source_answer",
                    "alt_answer",
                    "answer_tokens",
                    "prompt_len",
                    "counterfactual_prompt_len",
                ]
            }
            for pair in pairs
        ],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    controller = HeadPatchController(model)
    controller.install()
    condition_rows = []
    effect_rows = []

    try:
        for pair in pairs:
            base = pair["base"]
            cf = pair["counterfactual"]
            source_answer = pair["source_answer"]
            alt_answer = pair["alt_answer"]

            orig_encoded = encode_prompt_and_answer(
                tokenizer,
                with_answer(base, source_answer),
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            controller.set_cache_mode(orig_encoded.prediction_positions)
            orig_sum, orig_mean, orig_tok = score_gold_answer(model, orig_encoded)
            condition_rows.append(
                condition_row(pair, "orig_clean", "original", source_answer, orig_sum, orig_mean, orig_tok)
            )

            controller.reset()
            (cf_orig_enc, cf_orig_sum, cf_orig_mean, cf_orig_tok), (
                cf_alt_enc,
                cf_alt_sum,
                cf_alt_mean,
                cf_alt_tok,
            ) = score_two_answers(model, tokenizer, cf, source_answer, alt_answer, args)
            condition_rows.append(
                condition_row(pair, "cf_clean", "original", source_answer, cf_orig_sum, cf_orig_mean, cf_orig_tok)
            )
            condition_rows.append(
                condition_row(pair, "cf_clean", "counterfactual", alt_answer, cf_alt_sum, cf_alt_mean, cf_alt_tok)
            )
            cf_clean_margin = cf_orig_sum - cf_alt_sum
            effect_rows.append({
                "task": pair["task"],
                "source_idx": pair["source_idx"],
                "alt_idx": pair["alt_idx"],
                "source_answer": source_answer,
                "alt_answer": alt_answer,
                "condition": "cf_clean",
                "orig_answer_sum_logprob": cf_orig_sum,
                "cf_answer_sum_logprob": cf_alt_sum,
                "margin": cf_clean_margin,
                "margin_shift_vs_cf_clean": 0.0,
                "margin_shift_vs_cf_qr_ablate": math.nan,
            })

            controller.set_ablate_mode(qr_heads)
            cf_orig_ab_sum, cf_orig_ab_mean, cf_orig_ab_tok = score_gold_answer(model, cf_orig_enc)
            cf_alt_ab_sum, cf_alt_ab_mean, cf_alt_ab_tok = score_gold_answer(model, cf_alt_enc)
            condition_rows.append(
                condition_row(pair, "cf_qr_ablate", "original", source_answer, cf_orig_ab_sum, cf_orig_ab_mean, cf_orig_ab_tok)
            )
            condition_rows.append(
                condition_row(pair, "cf_qr_ablate", "counterfactual", alt_answer, cf_alt_ab_sum, cf_alt_ab_mean, cf_alt_ab_tok)
            )
            cf_ablate_margin = cf_orig_ab_sum - cf_alt_ab_sum
            effect_rows.append({
                "task": pair["task"],
                "source_idx": pair["source_idx"],
                "alt_idx": pair["alt_idx"],
                "source_answer": source_answer,
                "alt_answer": alt_answer,
                "condition": "cf_qr_ablate",
                "orig_answer_sum_logprob": cf_orig_ab_sum,
                "cf_answer_sum_logprob": cf_alt_ab_sum,
                "margin": cf_ablate_margin,
                "margin_shift_vs_cf_clean": cf_ablate_margin - cf_clean_margin,
                "margin_shift_vs_cf_qr_ablate": 0.0,
            })

            for condition, patch_heads in controls.items():
                controller.set_patch_mode(qr_heads, patch_heads, cf_orig_enc.prediction_positions)
                patch_orig_sum, patch_orig_mean, patch_orig_tok = score_gold_answer(model, cf_orig_enc)
                controller.set_patch_mode(qr_heads, patch_heads, cf_alt_enc.prediction_positions)
                patch_alt_sum, patch_alt_mean, patch_alt_tok = score_gold_answer(model, cf_alt_enc)
                condition_rows.append(
                    condition_row(pair, condition, "original", source_answer, patch_orig_sum, patch_orig_mean, patch_orig_tok)
                )
                condition_rows.append(
                    condition_row(pair, condition, "counterfactual", alt_answer, patch_alt_sum, patch_alt_mean, patch_alt_tok)
                )
                margin = patch_orig_sum - patch_alt_sum
                effect_rows.append({
                    "task": pair["task"],
                    "source_idx": pair["source_idx"],
                    "alt_idx": pair["alt_idx"],
                    "source_answer": source_answer,
                    "alt_answer": alt_answer,
                    "condition": condition,
                    "orig_answer_sum_logprob": patch_orig_sum,
                    "cf_answer_sum_logprob": patch_alt_sum,
                    "margin": margin,
                    "margin_shift_vs_cf_clean": margin - cf_clean_margin,
                    "margin_shift_vs_cf_qr_ablate": margin - cf_ablate_margin,
                })
    finally:
        controller.remove()
        controller.reset()

    write_jsonl(output_dir / "counterfactual_patching_conditions.jsonl", condition_rows)
    write_csv(output_dir / "counterfactual_patching_conditions.csv", condition_rows)
    write_jsonl(output_dir / "counterfactual_patching_effects.jsonl", effect_rows)
    write_csv(output_dir / "counterfactual_patching_effects.csv", effect_rows)
    summary = summarize_effects(effect_rows)
    write_csv(output_dir / "counterfactual_patching_summary.csv", summary)
    with open(output_dir / "counterfactual_patching_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_effects(output_dir, effect_rows)

    print(f"Wrote counterfactual patching results to {output_dir}")
    print(f"Selected {len(pairs)} counterfactual pairs")


if __name__ == "__main__":
    main()
