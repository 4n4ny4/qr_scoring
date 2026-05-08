"""Gold-evidence edge masking for QRHeads.

This script causally tests whether the damage from QRHead ablation is localized
to attention edges from selected query-side positions to the gold evidence span.
It wraps PyTorch SDPA for teacher-forced scoring and renormalizes attention
after masking the selected edges.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import transformers

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    HeadPatchController,
    bottom_control_heads,
    clear_device_cache,
    default_ablation_results_path,
    default_ranking_path,
    encode_prompt_and_answer,
    find_distractor_span,
    find_gold_span,
    find_random_span,
    get_decoder_layers,
    get_num_heads,
    intervention_positions_for_mode,
    load_clean_to_ablated_failure_ids,
    load_ranked_heads_json,
    load_task_instances,
    resolve_device,
    safe_div,
    same_layer_control_heads,
    score_gold_answer,
    select_instances,
    summarize_by_task_and_condition,
    write_csv,
    write_jsonl,
)
from qrretriever.model_runtime import (  # noqa: E402
    preflight_model_environment,
    resolve_ablation_dir,
    resolve_model_spec,
)


class SdpaEdgeMaskController:
    """Head-specific edge masking by wrapping scaled_dot_product_attention."""

    def __init__(self, model):
        self.model = model
        self.layers = get_decoder_layers(model)
        self.current_layer = None
        self.active = False
        self.heads_by_layer = defaultdict(list)
        self.query_positions = []
        self.key_span = None
        self._original_sdpa = None
        self._original_forwards = []

    def install(self):
        self._original_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._patched_sdpa
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            original = attn.forward

            def wrapped_forward(*args, __orig=original, __layer=layer_idx, **kwargs):
                previous = self.current_layer
                self.current_layer = __layer
                try:
                    return __orig(*args, **kwargs)
                finally:
                    self.current_layer = previous

            self._original_forwards.append((attn, original))
            attn.forward = wrapped_forward

    def remove(self):
        if self._original_sdpa is not None:
            F.scaled_dot_product_attention = self._original_sdpa
            self._original_sdpa = None
        for attn, original in self._original_forwards:
            attn.forward = original
        self._original_forwards = []

    def reset(self):
        self.active = False
        self.heads_by_layer = defaultdict(list)
        self.query_positions = []
        self.key_span = None

    def set_edge_mask(self, heads, query_positions, key_span):
        self.active = True
        self.heads_by_layer = defaultdict(list)
        for layer, head in heads:
            self.heads_by_layer[int(layer)].append(int(head))
        self.query_positions = [int(pos) for pos in query_positions]
        self.key_span = (int(key_span[0]), int(key_span[1]))

    def _patched_sdpa(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        layer_idx = self.current_layer
        if (
            not self.active
            or layer_idx is None
            or layer_idx not in self.heads_by_layer
            or self.key_span is None
        ):
            try:
                return self._original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            except TypeError:
                return self._original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

        return self._manual_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            masked_heads=self.heads_by_layer[layer_idx],
        )

    def _manual_sdpa(
        self,
        query,
        key,
        value,
        *,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        masked_heads,
    ):
        if key.shape[1] != query.shape[1]:
            if enable_gqa or query.shape[1] % key.shape[1] == 0:
                repeats = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
            else:
                raise ValueError(
                    f"Cannot align query heads ({query.shape[1]}) with key heads ({key.shape[1]})."
                )

        scale_factor = (1.0 / math.sqrt(query.shape[-1])) if scale is None else scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        q_len = scores.shape[-2]
        k_len = scores.shape[-1]

        if is_causal:
            causal = torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal.view(1, 1, q_len, k_len), torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attn_mask

        key_start, key_end = self.key_span
        key_start = max(0, min(key_start, k_len))
        key_end = max(key_start, min(key_end, k_len))
        valid_q = [pos for pos in self.query_positions if 0 <= pos < q_len]
        valid_heads = [head for head in masked_heads if 0 <= head < scores.shape[1]]
        if valid_q and valid_heads and key_end > key_start:
            q_index = torch.tensor(valid_q, device=scores.device)
            h_index = torch.tensor(valid_heads, device=scores.device)
            scores[:, h_index[:, None], q_index[None, :], key_start:key_end] = torch.finfo(scores.dtype).min

        probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        if dropout_p:
            probs = torch.dropout(probs, dropout_p, train=True)
        return torch.matmul(probs, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run QRHead gold-evidence edge masking.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--ranking_name", default="long_context_combined")
    parser.add_argument("--ranking_path", default=None)
    parser.add_argument("--ablation_results_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_examples_per_task", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gold_span_mode", choices=["sentence", "value", "value_in_sentence"], default="sentence")
    parser.add_argument(
        "--edge_query_position_mode",
        choices=["answer", "query", "answer_and_query"],
        default="answer",
        help=(
            "Which positions issue the masked attention queries. 'answer' "
            "preserves the original answer-prediction edge-masking behavior; "
            "'query' uses the late question tokens matched to answer length."
        ),
    )
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument("--use_all_examples", action="store_true")
    return parser.parse_args()


def load_sdpa_model_and_tokenizer(args):
    model_spec = resolve_model_spec(
        args.model_name,
        model_slug=args.model_slug,
        tokenizer_name=args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    preflight_model_environment(model_spec)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_spec.tokenizer_name,
        trust_remote_code=model_spec.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = resolve_device(args.device)
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": model_spec.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    if device == "cuda" and not args.load_in_8bit:
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = "auto" if model_spec.model_family in {"olmo", "olmo2"} else torch.float16
    elif model_spec.model_family in {"olmo", "olmo2"}:
        load_kwargs["torch_dtype"] = "auto"
    elif device in {"cuda", "mps"}:
        load_kwargs["torch_dtype"] = torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_spec.model_name,
        **load_kwargs,
    )
    if device != "cuda" and not args.load_in_8bit:
        model = model.to(device)
    model.eval()
    return model_spec, tokenizer, model


def condition_record(inst, condition, span, sum_logprob, mean_logprob, token_logprobs):
    return {
        "idx": inst["idx"],
        "task": inst["task"],
        "condition": condition,
        "span_start": None if span is None else span[0],
        "span_end": None if span is None else span[1],
        "gold": inst["needle_value"],
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "num_answer_tokens": len(token_logprobs),
        "token_logprobs": token_logprobs,
    }


def plot_outputs(output_dir: Path, damage_rows):
    by_condition = {}
    for row in damage_rows:
        value = row.get("fraction_of_full_ablation_damage")
        if value is None or math.isnan(float(value)):
            continue
        by_condition.setdefault(row["condition"], []).append(float(value))
    labels = sorted(by_condition)
    if not labels:
        return
    means = [sum(by_condition[label]) / len(by_condition[label]) for label in labels]
    plt.figure(figsize=(5.4, 3.2))
    plt.bar(labels, means)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("Fraction of full-ablation logprob damage")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "edge_masking_damage_bar.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    model_spec, tokenizer, model = load_sdpa_model_and_tokenizer(args)

    ranking_path = Path(args.ranking_path) if args.ranking_path else default_ranking_path(
        PROJECT_DIR, model_spec, args.ranking_name
    )
    ranking = load_ranked_heads_json(ranking_path)
    qr_heads = ranking[: args.k]
    num_heads = get_num_heads(model)
    nonqr_heads = same_layer_control_heads(qr_heads=qr_heads, num_heads=num_heads)
    bottom_heads = bottom_control_heads(ranking=ranking, qr_heads=qr_heads, k=args.k)

    ablation_results_path = (
        Path(args.ablation_results_path)
        if args.ablation_results_path
        else Path(resolve_ablation_dir(str(PROJECT_DIR), model_spec)) / "QRScore-SEC_results.json"
    )
    if not ablation_results_path.exists():
        ablation_results_path = default_ablation_results_path(PROJECT_DIR, model_spec)
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
        PROJECT_DIR / "results" / "mech_experiments" / model_spec.model_slug / "edge_masking"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model_spec.as_metadata(),
        "tasks": args.tasks,
        "k": args.k,
        "ranking_path": str(ranking_path),
        "ablation_results_path": str(ablation_results_path),
        "num_selected_examples": len(selected),
        "gold_span_mode": args.gold_span_mode,
        "edge_query_position_mode": args.edge_query_position_mode,
        "attn_implementation": args.attn_implementation,
        "qr_heads": [f"{layer}-{head}" for layer, head in qr_heads],
        "nonqr_gold_control_heads": [f"{layer}-{head}" for layer, head in nonqr_heads],
        "bottom_gold_control_heads": [f"{layer}-{head}" for layer, head in bottom_heads],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    head_controller = HeadPatchController(model)
    edge_controller = SdpaEdgeMaskController(model)
    head_controller.install()
    edge_controller.install()
    condition_rows = []
    damage_rows = []

    try:
        for n, inst in enumerate(selected, start=1):
            encoded = encode_prompt_and_answer(
                tokenizer,
                inst,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            gold_span = find_gold_span(encoded, inst, mode=args.gold_span_mode)
            if gold_span is None:
                print(f"[skip] could not locate gold span for {inst['idx']}")
                continue
            edge_query_positions = intervention_positions_for_mode(
                encoded,
                inst,
                args.edge_query_position_mode,
            )
            if not edge_query_positions:
                print(f"[skip] could not locate {args.edge_query_position_mode} positions for {inst['idx']}")
                continue
            span_len = max(1, gold_span[1] - gold_span[0])
            distractor_span = find_distractor_span(encoded, inst, span_len)
            random_span = find_random_span(
                encoded,
                length=span_len,
                exclude=gold_span,
                seed=args.seed + n,
            )

            head_controller.reset()
            edge_controller.reset()
            clean_sum, clean_mean, clean_token = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_record(inst, "clean", None, clean_sum, clean_mean, clean_token)
            )

            edge_controller.reset()
            head_controller.set_ablate_mode(qr_heads)
            full_sum, full_mean, full_token = score_gold_answer(model, encoded)
            condition_rows.append(
                condition_record(inst, "full_qr_ablation", None, full_sum, full_mean, full_token)
            )
            full_damage = clean_sum - full_sum

            edge_conditions = [
                ("edge_gold", qr_heads, gold_span),
                ("edge_distractor", qr_heads, distractor_span),
                ("edge_random", qr_heads, random_span),
                ("edge_nonqr_gold", nonqr_heads, gold_span),
                ("edge_bottom_gold", bottom_heads, gold_span),
            ]
            for condition, heads, span in edge_conditions:
                if span is None:
                    continue
                head_controller.reset()
                edge_controller.set_edge_mask(heads, edge_query_positions, span)
                edge_sum, edge_mean, edge_token = score_gold_answer(model, encoded)
                condition_rows.append(
                    condition_record(inst, condition, span, edge_sum, edge_mean, edge_token)
                )
                damage_rows.append({
                    "idx": inst["idx"],
                    "task": inst["task"],
                    "condition": condition,
                    "gold": inst["needle_value"],
                    "clean_sum_logprob": clean_sum,
                    "full_ablation_sum_logprob": full_sum,
                    "condition_sum_logprob": edge_sum,
                    "full_ablation_damage": full_damage,
                    "condition_damage": clean_sum - edge_sum,
                    "fraction_of_full_ablation_damage": safe_div(clean_sum - edge_sum, full_damage),
                    "span_start": span[0],
                    "span_end": span[1],
                    "span_len": span[1] - span[0],
                    "edge_query_position_mode": args.edge_query_position_mode,
                    "edge_query_positions": edge_query_positions,
                    "edge_query_position_count": len(edge_query_positions),
                })

            head_controller.reset()
            edge_controller.reset()
            clear_device_cache(next(model.parameters()).device)
            print(f"[{n}/{len(selected)}] {inst['task']} {inst['idx']}")
    finally:
        head_controller.remove()
        edge_controller.remove()

    write_jsonl(output_dir / "edge_masking_conditions.jsonl", condition_rows)
    write_csv(
        output_dir / "edge_masking_conditions.csv",
        [
            {k: v for k, v in row.items() if k != "token_logprobs"}
            for row in condition_rows
        ],
    )
    write_jsonl(output_dir / "edge_masking_damage.jsonl", damage_rows)
    write_csv(output_dir / "edge_masking_damage.csv", damage_rows)
    summary = {
        "mean_logprob": summarize_by_task_and_condition(condition_rows, "mean_logprob"),
        "fraction_damage": summarize_by_task_and_condition(
            damage_rows, "fraction_of_full_ablation_damage"
        ),
    }
    with open(output_dir / "edge_masking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / "edge_masking_damage_summary.csv", summary["fraction_damage"])
    plot_outputs(output_dir, damage_rows)
    print(f"Wrote edge-masking results to {output_dir}")


if __name__ == "__main__":
    main()
