"""Component-level patching for a QRHead retrieval circuit.

This runner uses clean/corrupted SEC pairs and asks which individual heads,
QRHead prefixes, and downstream MLP outputs can move the corrupted prompt back
toward the clean answer margin.
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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    PROJECT_DIR,
    HeadPatchController,
    bottom_control_heads,
    default_ranking_path,
    get_decoder_layers,
    get_num_heads,
    load_model_and_tokenizer,
    load_ranked_heads_json,
    parse_head_name,
    position_group_positions,
    random_control_heads,
    read_jsonl,
    safe_div,
    same_layer_control_heads,
    score_answer_text,
    write_csv,
    write_jsonl,
)


def head_name(head: Tuple[int, int]) -> str:
    return f"{head[0]}-{head[1]}"


class MlpOutputPatchController:
    """Cache and patch MLP outputs at selected positions."""

    def __init__(self, model):
        self.model = model
        self.layers = get_decoder_layers(model)
        self.handles = []
        self.mode = "none"
        self.layer_idx: Optional[int] = None
        self.positions: List[int] = []
        self.cache: Dict[int, torch.Tensor] = {}

    def install(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            handle = layer.mlp.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset(self) -> None:
        self.mode = "none"
        self.layer_idx = None
        self.positions = []

    def set_cache_mode(self, positions: Sequence[int]) -> None:
        self.cache = {}
        self.mode = "cache"
        self.layer_idx = None
        self.positions = list(positions)

    def set_patch_mode(self, layer_idx: int, positions: Sequence[int]) -> None:
        if layer_idx not in self.cache:
            raise ValueError(f"Missing clean MLP cache for layer {layer_idx}.")
        self.mode = "patch"
        self.layer_idx = int(layer_idx)
        self.positions = list(positions)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.mode == "none":
                return output
            positions = [pos for pos in self.positions if 0 <= pos < hidden.shape[1]]
            if not positions:
                return output
            if self.mode == "cache":
                self.cache[layer_idx] = hidden[:, positions, :].detach().cpu()
                return output
            if self.mode == "patch" and layer_idx == self.layer_idx:
                patched = hidden.clone()
                cached = self.cache[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
                if cached.shape[1] != len(positions):
                    raise ValueError(
                        f"Position count mismatch for layer {layer_idx}: "
                        f"cached {cached.shape[1]} vs patch {len(positions)}."
                    )
                pos_index = torch.tensor(positions, device=hidden.device)
                patched[:, pos_index, :] = cached
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return output

        return hook


def parse_args():
    parser = argparse.ArgumentParser(description="Run component-level circuit patching.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--candidate_heads", nargs="+", default=[])
    parser.add_argument("--include_top_qr", type=int, default=4)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--ranking_name", default="long_context_combined")
    parser.add_argument("--ranking_path", default=None)
    parser.add_argument("--position_group", default="query")
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--num_random_controls", type=int, default=5)
    parser.add_argument("--include_discovered_downstream_nodes", action="store_true")
    parser.add_argument("--run_final_validation", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def score_margin(model, tokenizer, inst, positive_answer, negative_answer, args):
    pos_sum = score_answer_text(
        model,
        tokenizer,
        inst,
        positive_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )[1]
    neg_sum = score_answer_text(
        model,
        tokenizer,
        inst,
        negative_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )[1]
    return float(pos_sum - neg_sum), float(pos_sum), float(neg_sum)


def summarize(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("margin_recovery")
        if value is None or math.isnan(float(value)):
            continue
        grouped[(row["task"], row["component_type"], row["condition"])].append(float(value))
    out = []
    for (task, component_type, condition), values in sorted(grouped.items()):
        out.append({
            "task": task,
            "component_type": component_type,
            "condition": condition,
            "n": len(values),
            "mean_margin_recovery": sum(values) / len(values),
            "median_margin_recovery": statistics.median(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        })
    return out


def plot_nodes(output_dir: Path, summary_rows: Sequence[dict]) -> None:
    rows = [
        row for row in summary_rows
        if row["component_type"] in {"head", "head_set", "mlp"} and not row["condition"].startswith("random")
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["median_margin_recovery"]), reverse=True)[:24]
    labels = [f"{row['task']}:{row['condition']}" for row in rows]
    values = [float(row["median_margin_recovery"]) for row in rows]
    plt.figure(figsize=(8.5, max(3.0, 0.28 * len(rows))))
    plt.barh(range(len(rows)), values, color="#2f6da8")
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.yticks(range(len(rows)), labels, fontsize=7)
    plt.xlabel("Median clean-to-corrupted margin recovery")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "component_node_ranking.png", dpi=220)
    plt.close()


def build_head_conditions(
    *,
    ranking: Sequence[Tuple[int, int]],
    qr_heads: Sequence[Tuple[int, int]],
    candidate_heads: Sequence[Tuple[int, int]],
    include_top_qr: int,
    num_layers: int,
    num_heads: int,
    seed: int,
    num_random_controls: int,
) -> List[dict]:
    conditions = []
    for head in candidate_heads:
        conditions.append({
            "condition": f"candidate_{head_name(head)}",
            "component_type": "head",
            "heads": [head],
        })
    for size in sorted({1, 2, 4, include_top_qr, min(len(qr_heads), 16)}):
        if 1 <= size <= len(qr_heads):
            conditions.append({
                "condition": f"qr_top{size}",
                "component_type": "head_set",
                "heads": list(qr_heads[:size]),
            })
    if candidate_heads:
        conditions.append({
            "condition": "candidate_set",
            "component_type": "head_set",
            "heads": list(candidate_heads),
        })

    bottom = bottom_control_heads(ranking=ranking, qr_heads=qr_heads, k=max(1, min(len(qr_heads), 16)))
    same_layer = same_layer_control_heads(qr_heads=qr_heads, num_heads=num_heads)
    for size in sorted({1, 2, 4, min(len(qr_heads), 16)}):
        conditions.append({
            "condition": f"bottom_top{size}",
            "component_type": "control",
            "heads": bottom[:size],
        })
        conditions.append({
            "condition": f"same_layer_top{size}",
            "component_type": "control",
            "heads": same_layer[:size],
        })
        for idx in range(num_random_controls):
            random_full = random_control_heads(
                qr_heads=qr_heads,
                num_layers=num_layers,
                num_heads=num_heads,
                seed=seed + 1000 * size + idx,
            )
            conditions.append({
                "condition": f"random{idx:02d}_top{size}",
                "component_type": "control",
                "heads": random_full[:size],
            })
    return conditions


def main():
    args = parse_args()
    model_spec, tokenizer, model = load_model_and_tokenizer(args, for_detection=False)
    ranking_path = Path(args.ranking_path) if args.ranking_path else default_ranking_path(
        PROJECT_DIR, model_spec, args.ranking_name
    )
    ranking = load_ranked_heads_json(ranking_path)
    qr_heads = ranking[: args.k]
    candidate_heads = [parse_head_name(value) for value in args.candidate_heads]
    pairs = [row for row in read_jsonl(Path(args.pairs_path)) if row.get("task") in set(args.tasks)]
    if not pairs:
        raise RuntimeError("No circuit pairs selected. Check --pairs_path and --tasks.")

    layers = get_decoder_layers(model)
    selected_layers = args.layers if args.layers is not None else list(range(len(layers)))
    num_layers = len(layers)
    num_heads = get_num_heads(model)
    head_conditions = build_head_conditions(
        ranking=ranking,
        qr_heads=qr_heads,
        candidate_heads=candidate_heads,
        include_top_qr=args.include_top_qr,
        num_layers=num_layers,
        num_heads=num_heads,
        seed=args.seed,
        num_random_controls=args.num_random_controls,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    head_controller = HeadPatchController(model)
    mlp_controller = MlpOutputPatchController(model)
    head_controller.install()
    mlp_controller.install()
    rows = []
    validation_rows = []

    try:
        for pair_idx, pair in enumerate(pairs):
            clean_inst = pair["clean_instance"]
            corrupted_inst = pair["corrupted_instance"]
            task = pair["task"]
            source_answer = str(pair["source_answer"])
            alt_answer = str(pair["alt_answer"])
            clean_margin, clean_pos_sum, clean_neg_sum = score_margin(
                model, tokenizer, clean_inst, source_answer, alt_answer, args
            )
            corrupted_margin, corrupted_pos_sum, corrupted_neg_sum = score_margin(
                model, tokenizer, corrupted_inst, source_answer, alt_answer, args
            )
            denom = clean_margin - corrupted_margin
            if denom <= 1e-8:
                continue

            clean_pos_encoded = score_answer_text(
                model,
                tokenizer,
                clean_inst,
                source_answer,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )[0]
            corrupt_pos_encoded = score_answer_text(
                model,
                tokenizer,
                corrupted_inst,
                source_answer,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )[0]
            clean_positions = position_group_positions(
                clean_pos_encoded,
                clean_inst,
                args.position_group,
                seed=args.seed + pair_idx,
                length=len(clean_pos_encoded.prediction_positions),
            )
            corrupt_positions = position_group_positions(
                corrupt_pos_encoded,
                corrupted_inst,
                args.position_group,
                seed=args.seed + pair_idx,
                length=len(clean_pos_encoded.prediction_positions),
            )
            if not clean_positions or not corrupt_positions:
                continue
            if len(clean_positions) != len(corrupt_positions):
                min_len = min(len(clean_positions), len(corrupt_positions))
                clean_positions = clean_positions[-min_len:]
                corrupt_positions = corrupt_positions[-min_len:]

            head_controller.set_cache_mode(clean_positions)
            score_answer_text(
                model,
                tokenizer,
                clean_inst,
                source_answer,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            for condition in head_conditions:
                heads = condition["heads"]
                head_controller.set_patch_mode([], heads, corrupt_positions)
                patched_pos_sum = score_answer_text(
                    model,
                    tokenizer,
                    corrupted_inst,
                    source_answer,
                    max_context_tokens=args.max_context_tokens,
                    answer_prefix=args.answer_prefix,
                )[1]
                head_controller.set_patch_mode([], heads, corrupt_positions)
                patched_neg_sum = score_answer_text(
                    model,
                    tokenizer,
                    corrupted_inst,
                    alt_answer,
                    max_context_tokens=args.max_context_tokens,
                    answer_prefix=args.answer_prefix,
                )[1]
                patched_margin = patched_pos_sum - patched_neg_sum
                rows.append({
                    "pair_idx": pair_idx,
                    "idx": clean_inst.get("idx"),
                    "task": task,
                    "component_type": condition["component_type"],
                    "condition": condition["condition"],
                    "heads": [head_name(head) for head in heads],
                    "position_group": args.position_group,
                    "position_count": len(corrupt_positions),
                    "clean_margin": clean_margin,
                    "corrupted_margin": corrupted_margin,
                    "patched_margin": patched_margin,
                    "margin_denom": denom,
                    "margin_recovery": safe_div(patched_margin - corrupted_margin, denom),
                    "clean_pos_sum_logprob": clean_pos_sum,
                    "corrupted_pos_sum_logprob": corrupted_pos_sum,
                    "patched_pos_sum_logprob": patched_pos_sum,
                    "patched_neg_sum_logprob": patched_neg_sum,
                })
            head_controller.reset()

            mlp_controller.set_cache_mode(clean_positions)
            score_answer_text(
                model,
                tokenizer,
                clean_inst,
                source_answer,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            for layer_idx in selected_layers:
                mlp_controller.set_patch_mode(layer_idx, corrupt_positions)
                patched_pos_sum = score_answer_text(
                    model,
                    tokenizer,
                    corrupted_inst,
                    source_answer,
                    max_context_tokens=args.max_context_tokens,
                    answer_prefix=args.answer_prefix,
                )[1]
                mlp_controller.set_patch_mode(layer_idx, corrupt_positions)
                patched_neg_sum = score_answer_text(
                    model,
                    tokenizer,
                    corrupted_inst,
                    alt_answer,
                    max_context_tokens=args.max_context_tokens,
                    answer_prefix=args.answer_prefix,
                )[1]
                patched_margin = patched_pos_sum - patched_neg_sum
                rows.append({
                    "pair_idx": pair_idx,
                    "idx": clean_inst.get("idx"),
                    "task": task,
                    "component_type": "mlp",
                    "condition": f"mlp_L{layer_idx}",
                    "layer": layer_idx,
                    "heads": [],
                    "position_group": args.position_group,
                    "position_count": len(corrupt_positions),
                    "clean_margin": clean_margin,
                    "corrupted_margin": corrupted_margin,
                    "patched_margin": patched_margin,
                    "margin_denom": denom,
                    "margin_recovery": safe_div(patched_margin - corrupted_margin, denom),
                    "clean_pos_sum_logprob": clean_pos_sum,
                    "corrupted_pos_sum_logprob": corrupted_pos_sum,
                    "patched_pos_sum_logprob": patched_pos_sum,
                    "patched_neg_sum_logprob": patched_neg_sum,
                })
            mlp_controller.reset()

            if args.run_final_validation:
                validation_sets = {
                    "candidate_set": candidate_heads,
                    "qr_top2": qr_heads[:2],
                    "qr_top4": qr_heads[:4],
                    "qr_top16": qr_heads[:16],
                }
                for condition, heads in validation_sets.items():
                    if not heads:
                        continue
                    head_controller.set_ablate_mode(heads)
                    ablated_margin, ablated_pos_sum, ablated_neg_sum = score_margin(
                        model, tokenizer, clean_inst, source_answer, alt_answer, args
                    )
                    head_controller.reset()
                    validation_rows.append({
                        "pair_idx": pair_idx,
                        "idx": clean_inst.get("idx"),
                        "task": task,
                        "condition": condition,
                        "heads": [head_name(head) for head in heads],
                        "clean_margin": clean_margin,
                        "corrupted_margin": corrupted_margin,
                        "ablated_margin": ablated_margin,
                        "margin_denom": denom,
                        "necessity_fraction": safe_div(clean_margin - ablated_margin, denom),
                        "clean_pos_sum_logprob": clean_pos_sum,
                        "ablated_pos_sum_logprob": ablated_pos_sum,
                        "ablated_neg_sum_logprob": ablated_neg_sum,
                    })
            print(f"[component] pair={pair_idx} task={task}")
    finally:
        head_controller.remove()
        mlp_controller.remove()

    summary_rows = summarize(rows)
    write_jsonl(output_dir / "component_patch_results.jsonl", rows)
    write_csv(output_dir / "component_patch_results.csv", rows)
    write_csv(output_dir / "component_patch_summary.csv", summary_rows)
    if validation_rows:
        write_jsonl(output_dir / "final_validation.jsonl", validation_rows)
        write_csv(output_dir / "final_validation.csv", validation_rows)
    plot_nodes(output_dir, summary_rows)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "pairs_path": str(args.pairs_path),
            "tasks": args.tasks,
            "candidate_heads": args.candidate_heads,
            "include_top_qr": args.include_top_qr,
            "position_group": args.position_group,
            "layers": selected_layers,
            "num_rows": len(rows),
            "num_validation_rows": len(validation_rows),
        }, f, indent=2)
    print(f"Wrote {len(rows)} component patch rows to {output_dir}")


if __name__ == "__main__":
    main()
