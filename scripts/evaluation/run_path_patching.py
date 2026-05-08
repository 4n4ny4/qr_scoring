"""Narrow edge tests for candidate QRHead circuit paths.

This script masks head-specific attention edges from a target position group
(query or answer-prediction positions) to evidence spans in clean prompts. It
is intentionally narrower than the broad edge-masking experiment: conditions
are per candidate head and per source/target position group.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    HeadPatchController,
    find_distractor_span,
    find_random_span,
    parse_head_name,
    position_group_positions,
    read_jsonl,
    safe_div,
    score_answer_text,
    write_csv,
    write_jsonl,
)
from run_edge_masking import SdpaEdgeMaskController, load_sdpa_model_and_tokenizer  # noqa: E402


def head_name(head: Tuple[int, int]) -> str:
    return f"{head[0]}-{head[1]}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run narrow QRHead path/edge tests.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--candidate_heads", nargs="+", required=True)
    parser.add_argument("--source_position_groups", nargs="+", default=["gold_value", "gold_sentence"])
    parser.add_argument("--target_position_groups", nargs="+", default=["query", "answer"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def span_from_positions(positions: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not positions:
        return None
    return min(positions), max(positions) + 1


def score_margin(model, tokenizer, inst, positive_answer, negative_answer, args):
    pos_encoded, pos_sum, _pos_mean, _pos_toks = score_answer_text(
        model,
        tokenizer,
        inst,
        positive_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )
    neg_sum = score_answer_text(
        model,
        tokenizer,
        inst,
        negative_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )[1]
    return {
        "positive_encoded": pos_encoded,
        "positive_sum_logprob": pos_sum,
        "negative_sum_logprob": neg_sum,
        "margin": float(pos_sum - neg_sum),
    }


def summarize(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("margin_damage_fraction")
        if value is None or math.isnan(float(value)):
            continue
        grouped[
            (
                row["task"],
                row["head"],
                row["source_position_group"],
                row["target_position_group"],
                row["condition"],
            )
        ].append(float(value))
    out = []
    for (task, head, source_group, target_group, condition), values in sorted(grouped.items()):
        out.append({
            "task": task,
            "head": head,
            "source_position_group": source_group,
            "target_position_group": target_group,
            "condition": condition,
            "n": len(values),
            "mean_margin_damage_fraction": sum(values) / len(values),
            "median_margin_damage_fraction": statistics.median(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        })
    return out


def plot_edges(output_dir: Path, summary_rows: Sequence[dict]) -> None:
    if not summary_rows:
        return
    rows = sorted(
        summary_rows,
        key=lambda row: float(row["median_margin_damage_fraction"]),
        reverse=True,
    )[:24]
    labels = [
        f"{row['task']}:{row['head']}:{row['target_position_group']}->{row['source_position_group']}:{row['condition']}"
        for row in rows
    ]
    values = [float(row["median_margin_damage_fraction"]) for row in rows]
    plt.figure(figsize=(9.2, max(3.2, 0.30 * len(rows))))
    colors = ["#2f6da8" if row["condition"] == "gold" else "#9aa6b2" for row in rows]
    plt.barh(range(len(rows)), values, color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.yticks(range(len(rows)), labels, fontsize=6.5)
    plt.xlabel("Median fraction of clean-corrupted margin removed")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "path_edge_ranking.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    _model_spec, tokenizer, model = load_sdpa_model_and_tokenizer(args)
    pairs = [row for row in read_jsonl(Path(args.pairs_path)) if row.get("task") in set(args.tasks)]
    if not pairs:
        raise RuntimeError("No circuit pairs selected. Check --pairs_path and --tasks.")
    candidate_heads = [parse_head_name(value) for value in args.candidate_heads]
    same_layer_controls = [(layer, (head + 1) % int(model.config.num_attention_heads)) for layer, head in candidate_heads]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    head_controller = HeadPatchController(model)
    edge_controller = SdpaEdgeMaskController(model)
    head_controller.install()
    edge_controller.install()
    rows = []

    try:
        for pair_idx, pair in enumerate(pairs):
            clean_inst = pair["clean_instance"]
            corrupted_inst = pair["corrupted_instance"]
            source_answer = str(pair["source_answer"])
            alt_answer = str(pair["alt_answer"])
            task = pair["task"]

            clean = score_margin(model, tokenizer, clean_inst, source_answer, alt_answer, args)
            corrupted = score_margin(model, tokenizer, corrupted_inst, source_answer, alt_answer, args)
            margin_denom = clean["margin"] - corrupted["margin"]
            if margin_denom <= 1e-8:
                continue
            encoded = clean["positive_encoded"]

            for source_group in args.source_position_groups:
                source_positions = position_group_positions(
                    encoded,
                    clean_inst,
                    source_group,
                    seed=args.seed + pair_idx,
                    length=len(encoded.prediction_positions),
                )
                gold_span = span_from_positions(source_positions)
                if gold_span is None:
                    continue
                span_len = max(1, gold_span[1] - gold_span[0])
                distractor_span = find_distractor_span(encoded, clean_inst, span_len)
                random_span = find_random_span(
                    encoded,
                    length=span_len,
                    exclude=gold_span,
                    seed=args.seed + pair_idx,
                )
                span_conditions = [
                    ("gold", gold_span),
                    ("distractor", distractor_span),
                    ("random", random_span),
                ]
                for target_group in args.target_position_groups:
                    target_positions = position_group_positions(
                        encoded,
                        clean_inst,
                        target_group,
                        seed=args.seed + pair_idx,
                        length=len(encoded.prediction_positions),
                    )
                    if not target_positions:
                        continue
                    for head, control_head in zip(candidate_heads, same_layer_controls):
                        for condition, span in span_conditions:
                            if span is None:
                                continue
                            edge_controller.set_edge_mask([head], target_positions, span)
                            masked = score_margin(model, tokenizer, clean_inst, source_answer, alt_answer, args)
                            edge_controller.reset()
                            rows.append({
                                "pair_idx": pair_idx,
                                "idx": clean_inst.get("idx"),
                                "task": task,
                                "head": head_name(head),
                                "head_type": "candidate",
                                "source_position_group": source_group,
                                "target_position_group": target_group,
                                "condition": condition,
                                "span_start": span[0],
                                "span_end": span[1],
                                "span_len": span[1] - span[0],
                                "target_positions": target_positions,
                                "target_position_count": len(target_positions),
                                "clean_margin": clean["margin"],
                                "corrupted_margin": corrupted["margin"],
                                "masked_margin": masked["margin"],
                                "margin_denom": margin_denom,
                                "margin_damage": clean["margin"] - masked["margin"],
                                "margin_damage_fraction": safe_div(clean["margin"] - masked["margin"], margin_denom),
                            })

                        edge_controller.set_edge_mask([control_head], target_positions, gold_span)
                        control = score_margin(model, tokenizer, clean_inst, source_answer, alt_answer, args)
                        edge_controller.reset()
                        rows.append({
                            "pair_idx": pair_idx,
                            "idx": clean_inst.get("idx"),
                            "task": task,
                            "head": head_name(control_head),
                            "head_type": "same_layer_control",
                            "source_position_group": source_group,
                            "target_position_group": target_group,
                            "condition": "gold",
                            "span_start": gold_span[0],
                            "span_end": gold_span[1],
                            "span_len": gold_span[1] - gold_span[0],
                            "target_positions": target_positions,
                            "target_position_count": len(target_positions),
                            "clean_margin": clean["margin"],
                            "corrupted_margin": corrupted["margin"],
                            "masked_margin": control["margin"],
                            "margin_denom": margin_denom,
                            "margin_damage": clean["margin"] - control["margin"],
                            "margin_damage_fraction": safe_div(clean["margin"] - control["margin"], margin_denom),
                        })
            print(f"[path] pair={pair_idx} task={task}")
    finally:
        head_controller.remove()
        edge_controller.remove()

    summary_rows = summarize(rows)
    write_jsonl(output_dir / "path_patch_edges.jsonl", rows)
    write_csv(output_dir / "path_patch_edges.csv", rows)
    write_csv(output_dir / "path_patch_edge_summary.csv", summary_rows)
    plot_edges(output_dir, summary_rows)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "pairs_path": str(args.pairs_path),
            "tasks": args.tasks,
            "candidate_heads": args.candidate_heads,
            "source_position_groups": args.source_position_groups,
            "target_position_groups": args.target_position_groups,
            "num_rows": len(rows),
        }, f, indent=2)
    print(f"Wrote {len(rows)} path-patching rows to {output_dir}")


if __name__ == "__main__":
    main()
