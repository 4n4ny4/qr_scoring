"""Residual-stream patching for QRHead circuit discovery.

This script patches clean residual-stream states into corrupted prompts over
layer x position-group sites. It is meant to localize where information about
the counterfactually edited SEC answer re-enters the computation.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qr_scoring_matplotlib")

import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    DEFAULT_TASKS,
    get_decoder_layers,
    load_model_and_tokenizer,
    position_group_positions,
    read_jsonl,
    safe_div,
    score_answer_text,
    write_csv,
    write_jsonl,
)


class ResidualPatchController:
    """Cache and patch residual-stream activations after transformer blocks."""

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
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
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
            raise ValueError(f"Missing clean cache for layer {layer_idx}.")
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
    parser = argparse.ArgumentParser(description="Run residual stream patching.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--position_groups",
        nargs="+",
        default=["gold_value", "gold_sentence", "query", "answer", "random_prompt"],
    )
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def margin_from_scores(pos_sum: float, neg_sum: float) -> float:
    return float(pos_sum - neg_sum)


def score_margin_with_optional_patch(
    *,
    model,
    tokenizer,
    inst: dict,
    positive_answer: str,
    negative_answer: str,
    max_context_tokens: int,
    answer_prefix: str,
) -> dict:
    pos_encoded, pos_sum, _pos_mean, _pos_toks = score_answer_text(
        model,
        tokenizer,
        inst,
        positive_answer,
        max_context_tokens=max_context_tokens,
        answer_prefix=answer_prefix,
    )
    neg_encoded, neg_sum, _neg_mean, _neg_toks = score_answer_text(
        model,
        tokenizer,
        inst,
        negative_answer,
        max_context_tokens=max_context_tokens,
        answer_prefix=answer_prefix,
    )
    return {
        "positive_encoded": pos_encoded,
        "negative_encoded": neg_encoded,
        "positive_sum_logprob": pos_sum,
        "negative_sum_logprob": neg_sum,
        "margin": margin_from_scores(pos_sum, neg_sum),
    }


def median_or_nan(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float("nan") if not clean else statistics.median(clean)


def summarize(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("margin_recovery")
        if value is None or math.isnan(float(value)):
            continue
        grouped[(row["task"], row["position_group"], row["layer"])].append(float(value))
    out = []
    for (task, position_group, layer), values in sorted(grouped.items()):
        out.append({
            "task": task,
            "position_group": position_group,
            "layer": layer,
            "n": len(values),
            "mean_margin_recovery": sum(values) / len(values),
            "median_margin_recovery": statistics.median(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        })
    return out


def plot_heatmap(output_dir: Path, summary_rows: Sequence[dict]) -> None:
    if not summary_rows:
        return
    groups = sorted({row["position_group"] for row in summary_rows})
    layers = sorted({int(row["layer"]) for row in summary_rows})
    data_by_group = defaultdict(dict)
    for row in summary_rows:
        data_by_group[row["position_group"]][int(row["layer"])] = float(row["median_margin_recovery"])

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(7.2, max(1.8, 1.45 * len(groups))),
        sharex=True,
        squeeze=False,
    )
    for ax, group in zip(axes[:, 0], groups):
        values = [[data_by_group[group].get(layer, float("nan")) for layer in layers]]
        im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(1.0, max(
            [0.0] + [v for row in values for v in row if not math.isnan(v)]
        )))
        ax.set_yticks([0])
        ax.set_yticklabels([group])
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=7)
    axes[-1, 0].set_xlabel("Layer")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Median margin recovery")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_patch_heatmap.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    _model_spec, tokenizer, model = load_model_and_tokenizer(args, for_detection=False)
    pairs = [row for row in read_jsonl(Path(args.pairs_path)) if row.get("task") in set(args.tasks)]
    if not pairs:
        raise RuntimeError("No circuit pairs selected. Check --pairs_path and --tasks.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = get_decoder_layers(model)
    selected_layers = args.layers if args.layers is not None else list(range(len(layers)))
    controller = ResidualPatchController(model)
    controller.install()
    rows = []

    try:
        for pair_idx, pair in enumerate(pairs):
            clean_inst = pair["clean_instance"]
            corrupted_inst = pair["corrupted_instance"]
            task = pair["task"]
            source_answer = str(pair["source_answer"])
            alt_answer = str(pair["alt_answer"])

            clean = score_margin_with_optional_patch(
                model=model,
                tokenizer=tokenizer,
                inst=clean_inst,
                positive_answer=source_answer,
                negative_answer=alt_answer,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            corrupted = score_margin_with_optional_patch(
                model=model,
                tokenizer=tokenizer,
                inst=corrupted_inst,
                positive_answer=source_answer,
                negative_answer=alt_answer,
                max_context_tokens=args.max_context_tokens,
                answer_prefix=args.answer_prefix,
            )
            margin_denom = clean["margin"] - corrupted["margin"]
            if margin_denom <= 1e-8:
                continue

            for group in args.position_groups:
                pos_len = len(clean["positive_encoded"].prediction_positions)
                clean_positions = position_group_positions(
                    clean["positive_encoded"],
                    clean_inst,
                    group,
                    seed=args.seed + pair_idx,
                    length=pos_len,
                )
                corrupted_positions = position_group_positions(
                    corrupted["positive_encoded"],
                    corrupted_inst,
                    group,
                    seed=args.seed + pair_idx,
                    length=pos_len,
                )
                if not clean_positions or not corrupted_positions:
                    continue
                if len(clean_positions) != len(corrupted_positions):
                    min_len = min(len(clean_positions), len(corrupted_positions))
                    clean_positions = clean_positions[-min_len:]
                    corrupted_positions = corrupted_positions[-min_len:]

                controller.set_cache_mode(clean_positions)
                score_answer_text(
                    model,
                    tokenizer,
                    clean_inst,
                    source_answer,
                    max_context_tokens=args.max_context_tokens,
                    answer_prefix=args.answer_prefix,
                )
                for layer_idx in selected_layers:
                    controller.set_patch_mode(layer_idx, corrupted_positions)
                    patched_pos = score_answer_text(
                        model,
                        tokenizer,
                        corrupted_inst,
                        source_answer,
                        max_context_tokens=args.max_context_tokens,
                        answer_prefix=args.answer_prefix,
                    )[1]
                    controller.set_patch_mode(layer_idx, corrupted_positions)
                    patched_neg = score_answer_text(
                        model,
                        tokenizer,
                        corrupted_inst,
                        alt_answer,
                        max_context_tokens=args.max_context_tokens,
                        answer_prefix=args.answer_prefix,
                    )[1]
                    patched_margin = margin_from_scores(patched_pos, patched_neg)
                    rows.append({
                        "pair_idx": pair_idx,
                        "idx": clean_inst.get("idx"),
                        "task": task,
                        "source_answer": source_answer,
                        "alt_answer": alt_answer,
                        "position_group": group,
                        "layer": layer_idx,
                        "position_count": len(corrupted_positions),
                        "clean_margin": clean["margin"],
                        "corrupted_margin": corrupted["margin"],
                        "patched_margin": patched_margin,
                        "margin_denom": margin_denom,
                        "margin_recovery": safe_div(patched_margin - corrupted["margin"], margin_denom),
                    })
                controller.reset()
                print(f"[residual] pair={pair_idx} task={task} group={group}")
    finally:
        controller.remove()

    summary_rows = summarize(rows)
    write_jsonl(output_dir / "residual_patch_grid.jsonl", rows)
    write_csv(output_dir / "residual_patch_grid.csv", rows)
    write_csv(output_dir / "residual_patch_summary.csv", summary_rows)
    plot_heatmap(output_dir, summary_rows)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "pairs_path": str(args.pairs_path),
            "tasks": args.tasks,
            "position_groups": args.position_groups,
            "layers": selected_layers,
            "num_rows": len(rows),
        }, f, indent=2)
    print(f"Wrote {len(rows)} residual patch rows to {output_dir}")


if __name__ == "__main__":
    main()
