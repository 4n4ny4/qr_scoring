"""Focused answer-position circuit search.

This script follows the strongest signal from the residual-patching runs:
clean-to-corrupted margin recovery is concentrated at late answer-prediction
positions, not query positions. It scans late-layer attention heads and MLPs,
then validates the discovered nodes with position controls and cumulative
same-range random controls.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qr_scoring_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mech_utils import (  # noqa: E402
    HeadPatchController,
    get_decoder_layers,
    get_num_heads,
    load_model_and_tokenizer,
    position_group_positions,
    read_jsonl,
    safe_div,
    score_answer_text,
    write_csv,
    write_jsonl,
)


DEFAULT_LAYERS = list(range(24, 32))


@dataclass(frozen=True)
class Node:
    node_type: str
    layer: int
    head: Optional[int] = None

    @property
    def node_id(self) -> str:
        if self.node_type == "attn":
            return f"attn:{self.layer}-{self.head}"
        return f"mlp:{self.layer}"


class MultiMlpPatchController:
    """Cache MLP outputs and patch any subset of layers at selected positions."""

    def __init__(self, model):
        self.model = model
        self.layers = get_decoder_layers(model)
        self.handles = []
        self.mode = "none"
        self.positions: List[int] = []
        self.patch_layers = set()
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
        self.positions = []
        self.patch_layers = set()

    def set_cache_mode(self, positions: Sequence[int]) -> None:
        self.cache = {}
        self.mode = "cache"
        self.positions = list(positions)
        self.patch_layers = set()

    def set_patch_mode(self, layers: Sequence[int], positions: Sequence[int]) -> None:
        if any(int(layer) not in self.cache for layer in layers):
            missing = [int(layer) for layer in layers if int(layer) not in self.cache]
            raise ValueError(f"Missing clean MLP cache for layers {missing}.")
        self.mode = "patch"
        self.positions = list(positions)
        self.patch_layers = {int(layer) for layer in layers}

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
            if self.mode == "patch" and layer_idx in self.patch_layers:
                patched = hidden.clone()
                cached = self.cache[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
                if cached.shape[1] != len(positions):
                    raise ValueError(
                        f"Position count mismatch for MLP layer {layer_idx}: "
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
    parser = argparse.ArgumentParser(description="Run focused answer-position circuit search.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=["ceo_lastname"])
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--scan_all_heads", action="store_true")
    parser.add_argument("--scan_mlps", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--top_n_nodes", type=int, default=16)
    parser.add_argument("--num_random_controls", type=int, default=20)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--answer_prefix", default="")
    parser.add_argument("--min_margin_denom", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def node_to_row(node: Node) -> dict:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "layer": node.layer,
        "head": "" if node.head is None else node.head,
    }


def parse_node_id(value: str) -> Node:
    kind, rest = value.split(":", 1)
    if kind == "attn":
        layer, head = rest.split("-", 1)
        return Node("attn", int(layer), int(head))
    if kind == "mlp":
        return Node("mlp", int(rest), None)
    raise ValueError(f"Unsupported node id: {value}")


def score_margin(model, tokenizer, inst, positive_answer: str, negative_answer: str, args) -> dict:
    pos_encoded, pos_sum, _pos_mean, _pos_toks = score_answer_text(
        model,
        tokenizer,
        inst,
        positive_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )
    neg_encoded, neg_sum, _neg_mean, _neg_toks = score_answer_text(
        model,
        tokenizer,
        inst,
        negative_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )
    return {
        "positive_encoded": pos_encoded,
        "negative_encoded": neg_encoded,
        "positive_sum_logprob": pos_sum,
        "negative_sum_logprob": neg_sum,
        "margin": float(pos_sum - neg_sum),
    }


def resolve_positions(encoded, inst, group: str, seed: int) -> List[int]:
    return position_group_positions(
        encoded,
        inst,
        group,
        seed=seed,
        length=len(encoded.prediction_positions),
    )


def align_positions(clean_positions: Sequence[int], corrupted_positions: Sequence[int]) -> Tuple[List[int], List[int]]:
    if not clean_positions or not corrupted_positions:
        return [], []
    if len(clean_positions) == len(corrupted_positions):
        return list(clean_positions), list(corrupted_positions)
    n = min(len(clean_positions), len(corrupted_positions))
    return list(clean_positions)[-n:], list(corrupted_positions)[-n:]


def build_valid_pairs(model, tokenizer, pairs: Sequence[dict], args) -> List[dict]:
    valid = []
    for pair in pairs:
        clean_inst = pair["clean_instance"]
        corrupted_inst = pair["corrupted_instance"]
        source_answer = str(pair["source_answer"])
        alt_answer = str(pair["alt_answer"])
        clean = score_margin(model, tokenizer, clean_inst, source_answer, alt_answer, args)
        corrupted = score_margin(model, tokenizer, corrupted_inst, source_answer, alt_answer, args)
        denom = clean["margin"] - corrupted["margin"]
        if denom <= args.min_margin_denom:
            continue
        valid.append({
            "pair": pair,
            "clean": clean,
            "corrupted": corrupted,
            "denom": denom,
        })
        if args.max_pairs > 0 and len(valid) >= args.max_pairs:
            break
    return valid


def all_nodes(args, num_heads: int) -> List[Node]:
    nodes = []
    if args.scan_all_heads or not args.scan_mlps:
        for layer in args.layers:
            for head in range(num_heads):
                nodes.append(Node("attn", int(layer), int(head)))
    if args.scan_mlps or not args.scan_all_heads:
        for layer in args.layers:
            nodes.append(Node("mlp", int(layer), None))
    return nodes


def split_nodes(nodes: Sequence[Node]) -> Tuple[List[Tuple[int, int]], List[int]]:
    heads = [(node.layer, int(node.head)) for node in nodes if node.node_type == "attn"]
    mlp_layers = [node.layer for node in nodes if node.node_type == "mlp"]
    return heads, mlp_layers


def cache_clean(
    *,
    model,
    tokenizer,
    clean_inst: dict,
    source_answer: str,
    clean_positions: Sequence[int],
    head_controller: HeadPatchController,
    mlp_controller: MultiMlpPatchController,
    args,
) -> None:
    head_controller.set_cache_mode(clean_positions)
    mlp_controller.set_cache_mode(clean_positions)
    score_answer_text(
        model,
        tokenizer,
        clean_inst,
        source_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )


def set_node_patch(
    *,
    nodes: Sequence[Node],
    positions: Sequence[int],
    head_controller: HeadPatchController,
    mlp_controller: MultiMlpPatchController,
) -> None:
    heads, mlp_layers = split_nodes(nodes)
    if heads:
        head_controller.set_patch_mode([], heads, positions)
    else:
        head_controller.reset()
    if mlp_layers:
        mlp_controller.set_patch_mode(mlp_layers, positions)
    else:
        mlp_controller.reset()


def score_patched_margin(
    *,
    model,
    tokenizer,
    corrupted_inst: dict,
    source_answer: str,
    alt_answer: str,
    nodes: Sequence[Node],
    corrupted_positions: Sequence[int],
    head_controller: HeadPatchController,
    mlp_controller: MultiMlpPatchController,
    args,
) -> float:
    set_node_patch(
        nodes=nodes,
        positions=corrupted_positions,
        head_controller=head_controller,
        mlp_controller=mlp_controller,
    )
    pos_sum = score_answer_text(
        model,
        tokenizer,
        corrupted_inst,
        source_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )[1]
    set_node_patch(
        nodes=nodes,
        positions=corrupted_positions,
        head_controller=head_controller,
        mlp_controller=mlp_controller,
    )
    neg_sum = score_answer_text(
        model,
        tokenizer,
        corrupted_inst,
        alt_answer,
        max_context_tokens=args.max_context_tokens,
        answer_prefix=args.answer_prefix,
    )[1]
    return float(pos_sum - neg_sum)


def margin_row(base: dict, condition: str, nodes: Sequence[Node], group: str, patched_margin: float) -> dict:
    return {
        "pair_idx": base["pair_idx"],
        "idx": base["idx"],
        "task": base["task"],
        "condition": condition,
        "position_group": group,
        "nodes": [node.node_id for node in nodes],
        "node_count": len(nodes),
        "clean_margin": base["clean_margin"],
        "corrupted_margin": base["corrupted_margin"],
        "patched_margin": patched_margin,
        "margin_denom": base["margin_denom"],
        "margin_recovery": safe_div(patched_margin - base["corrupted_margin"], base["margin_denom"]),
    }


def summarize(rows: Sequence[dict], group_keys: Sequence[str]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("margin_recovery")
        if value is None or math.isnan(float(value)):
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        grouped[key].append(float(value))
    out = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(group_keys, key))
        row.update({
            "n": len(values),
            "mean_margin_recovery": sum(values) / len(values),
            "median_margin_recovery": statistics.median(values),
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        })
        out.append(row)
    return out


def top_nodes_from_summary(summary_rows: Sequence[dict], top_n: int) -> List[Node]:
    rows = sorted(
        summary_rows,
        key=lambda row: float(row["median_margin_recovery"]),
        reverse=True,
    )
    return [parse_node_id(row["node_id"]) for row in rows[:top_n]]


def sample_random_nodes(
    *,
    universe: Sequence[Node],
    exclude: Sequence[Node],
    k: int,
    seed: int,
) -> List[Node]:
    exclude_ids = {node.node_id for node in exclude}
    candidates = [node for node in universe if node.node_id not in exclude_ids]
    rng = random.Random(seed)
    if len(candidates) < k:
        raise ValueError(f"Need {k} random nodes but only {len(candidates)} are available.")
    return rng.sample(candidates, k)


def run_node_scan(
    *,
    model,
    tokenizer,
    valid_pairs: Sequence[dict],
    nodes: Sequence[Node],
    head_controller: HeadPatchController,
    mlp_controller: MultiMlpPatchController,
    args,
) -> List[dict]:
    rows = []
    for pair_idx, item in enumerate(valid_pairs):
        pair = item["pair"]
        clean_inst = pair["clean_instance"]
        corrupted_inst = pair["corrupted_instance"]
        source_answer = str(pair["source_answer"])
        alt_answer = str(pair["alt_answer"])
        clean_positions, corrupted_positions = align_positions(
            resolve_positions(item["clean"]["positive_encoded"], clean_inst, "answer", args.seed + pair_idx),
            resolve_positions(item["corrupted"]["positive_encoded"], corrupted_inst, "answer", args.seed + pair_idx),
        )
        if not clean_positions:
            continue
        cache_clean(
            model=model,
            tokenizer=tokenizer,
            clean_inst=clean_inst,
            source_answer=source_answer,
            clean_positions=clean_positions,
            head_controller=head_controller,
            mlp_controller=mlp_controller,
            args=args,
        )
        base = {
            "pair_idx": pair_idx,
            "idx": clean_inst.get("idx", ""),
            "task": pair["task"],
            "clean_margin": item["clean"]["margin"],
            "corrupted_margin": item["corrupted"]["margin"],
            "margin_denom": item["denom"],
        }
        for node in nodes:
            patched_margin = score_patched_margin(
                model=model,
                tokenizer=tokenizer,
                corrupted_inst=corrupted_inst,
                source_answer=source_answer,
                alt_answer=alt_answer,
                nodes=[node],
                corrupted_positions=corrupted_positions,
                head_controller=head_controller,
                mlp_controller=mlp_controller,
                args=args,
            )
            row = margin_row(base, node.node_id, [node], "answer", patched_margin)
            row.update(node_to_row(node))
            rows.append(row)
        head_controller.reset()
        mlp_controller.reset()
        print(f"[node-scan] pair={pair_idx + 1}/{len(valid_pairs)} task={pair['task']}")
    return rows


def run_position_controls(
    *,
    model,
    tokenizer,
    valid_pairs: Sequence[dict],
    top_nodes: Sequence[Node],
    head_controller: HeadPatchController,
    mlp_controller: MultiMlpPatchController,
    args,
) -> List[dict]:
    rows = []
    for pair_idx, item in enumerate(valid_pairs):
        pair = item["pair"]
        clean_inst = pair["clean_instance"]
        corrupted_inst = pair["corrupted_instance"]
        source_answer = str(pair["source_answer"])
        alt_answer = str(pair["alt_answer"])
        base = {
            "pair_idx": pair_idx,
            "idx": clean_inst.get("idx", ""),
            "task": pair["task"],
            "clean_margin": item["clean"]["margin"],
            "corrupted_margin": item["corrupted"]["margin"],
            "margin_denom": item["denom"],
        }
        for group in ["answer", "query", "random_prompt"]:
            clean_positions, corrupted_positions = align_positions(
                resolve_positions(item["clean"]["positive_encoded"], clean_inst, group, args.seed + pair_idx),
                resolve_positions(item["corrupted"]["positive_encoded"], corrupted_inst, group, args.seed + pair_idx),
            )
            if not clean_positions:
                continue
            cache_clean(
                model=model,
                tokenizer=tokenizer,
                clean_inst=clean_inst,
                source_answer=source_answer,
                clean_positions=clean_positions,
                head_controller=head_controller,
                mlp_controller=mlp_controller,
                args=args,
            )
            for node in top_nodes:
                patched_margin = score_patched_margin(
                    model=model,
                    tokenizer=tokenizer,
                    corrupted_inst=corrupted_inst,
                    source_answer=source_answer,
                    alt_answer=alt_answer,
                    nodes=[node],
                    corrupted_positions=corrupted_positions,
                    head_controller=head_controller,
                    mlp_controller=mlp_controller,
                    args=args,
                )
                row = margin_row(base, node.node_id, [node], group, patched_margin)
                row.update(node_to_row(node))
                rows.append(row)
            head_controller.reset()
            mlp_controller.reset()
        print(f"[position-controls] pair={pair_idx + 1}/{len(valid_pairs)} task={pair['task']}")
    return rows


def cumulative_sizes(n_nodes: int) -> List[int]:
    return [size for size in [1, 2, 4, 8] if size <= n_nodes]


def run_joint_validation(
    *,
    model,
    tokenizer,
    valid_pairs: Sequence[dict],
    top_nodes: Sequence[Node],
    universe: Sequence[Node],
    head_controller: HeadPatchController,
    mlp_controller: MultiMlpPatchController,
    args,
) -> List[dict]:
    rows = []
    sizes = cumulative_sizes(len(top_nodes))
    random_sets = {
        (size, control_idx): sample_random_nodes(
            universe=universe,
            exclude=top_nodes,
            k=size,
            seed=args.seed + 10_000 * size + control_idx,
        )
        for size in sizes
        for control_idx in range(args.num_random_controls)
    }
    for pair_idx, item in enumerate(valid_pairs):
        pair = item["pair"]
        clean_inst = pair["clean_instance"]
        corrupted_inst = pair["corrupted_instance"]
        source_answer = str(pair["source_answer"])
        alt_answer = str(pair["alt_answer"])
        clean_positions, corrupted_positions = align_positions(
            resolve_positions(item["clean"]["positive_encoded"], clean_inst, "answer", args.seed + pair_idx),
            resolve_positions(item["corrupted"]["positive_encoded"], corrupted_inst, "answer", args.seed + pair_idx),
        )
        if not clean_positions:
            continue
        cache_clean(
            model=model,
            tokenizer=tokenizer,
            clean_inst=clean_inst,
            source_answer=source_answer,
            clean_positions=clean_positions,
            head_controller=head_controller,
            mlp_controller=mlp_controller,
            args=args,
        )
        base = {
            "pair_idx": pair_idx,
            "idx": clean_inst.get("idx", ""),
            "task": pair["task"],
            "clean_margin": item["clean"]["margin"],
            "corrupted_margin": item["corrupted"]["margin"],
            "margin_denom": item["denom"],
        }
        for size in sizes:
            node_set = list(top_nodes[:size])
            patched_margin = score_patched_margin(
                model=model,
                tokenizer=tokenizer,
                corrupted_inst=corrupted_inst,
                source_answer=source_answer,
                alt_answer=alt_answer,
                nodes=node_set,
                corrupted_positions=corrupted_positions,
                head_controller=head_controller,
                mlp_controller=mlp_controller,
                args=args,
            )
            rows.append({
                **margin_row(base, f"top{size}", node_set, "answer", patched_margin),
                "set_type": "discovered",
                "set_size": size,
            })
            for control_idx in range(args.num_random_controls):
                random_nodes = random_sets[(size, control_idx)]
                patched_margin = score_patched_margin(
                    model=model,
                    tokenizer=tokenizer,
                    corrupted_inst=corrupted_inst,
                    source_answer=source_answer,
                    alt_answer=alt_answer,
                    nodes=random_nodes,
                    corrupted_positions=corrupted_positions,
                    head_controller=head_controller,
                    mlp_controller=mlp_controller,
                    args=args,
                )
                rows.append({
                    **margin_row(
                        base,
                        f"random{control_idx:02d}_size{size}",
                        random_nodes,
                        "answer",
                        patched_margin,
                    ),
                    "set_type": "random",
                    "set_size": size,
                })
        head_controller.reset()
        mlp_controller.reset()
        print(f"[joint-validation] pair={pair_idx + 1}/{len(valid_pairs)} task={pair['task']}")
    return rows


def write_top_nodes(output_dir: Path, summary_rows: Sequence[dict], top_n: int) -> List[dict]:
    rows = sorted(
        summary_rows,
        key=lambda row: float(row["median_margin_recovery"]),
        reverse=True,
    )[:top_n]
    out = []
    for rank, row in enumerate(rows, start=1):
        out.append({"rank": rank, **row})
    write_csv(output_dir / "top_nodes.csv", out)
    return out


def median_lookup(rows: Sequence[dict], *, condition: str = None, set_type: str = None, set_size: int = None, group: str = None):
    values = []
    for row in rows:
        if condition is not None and row.get("condition") != condition:
            continue
        if set_type is not None and row.get("set_type") != set_type:
            continue
        if set_size is not None and int(row.get("set_size", -1)) != set_size:
            continue
        if group is not None and row.get("position_group") != group:
            continue
        value = row.get("margin_recovery")
        if value is not None and not math.isnan(float(value)):
            values.append(float(value))
    return None if not values else statistics.median(values)


def plot_summary(
    output_dir: Path,
    node_summary: Sequence[dict],
    position_summary: Sequence[dict],
    joint_summary: Sequence[dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    top_nodes = sorted(
        node_summary,
        key=lambda row: float(row["median_margin_recovery"]),
        reverse=True,
    )[:12]
    axes[0].barh(
        range(len(top_nodes)),
        [float(row["median_margin_recovery"]) for row in top_nodes],
        color="#2f6da8",
    )
    axes[0].set_yticks(range(len(top_nodes)))
    axes[0].set_yticklabels([row["node_id"] for row in top_nodes], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median recovery")
    axes[0].set_title("Individual nodes")

    group_values = defaultdict(list)
    for row in position_summary:
        group_values[row["position_group"]].append(float(row["median_margin_recovery"]))
    groups = ["answer", "query", "random_prompt"]
    axes[1].bar(
        groups,
        [statistics.median(group_values[group]) if group_values[group] else 0.0 for group in groups],
        color=["#2f6da8", "#9aa6b2", "#9aa6b2"],
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticklabels(groups, rotation=20, ha="right")
    axes[1].set_ylabel("Median over top-node controls")
    axes[1].set_title("Position controls")

    discovered = defaultdict(list)
    random_control = defaultdict(list)
    for row in joint_summary:
        value = float(row["median_margin_recovery"])
        if row["set_type"] == "discovered":
            discovered[int(row["set_size"])].append(value)
        elif row["set_type"] == "random":
            random_control[int(row["set_size"])].append(value)
    sizes = sorted(set(discovered) | set(random_control))
    axes[2].plot(
        sizes,
        [statistics.median(discovered[size]) if discovered[size] else 0.0 for size in sizes],
        marker="o",
        label="discovered",
        color="#2f6da8",
    )
    axes[2].plot(
        sizes,
        [statistics.median(random_control[size]) if random_control[size] else 0.0 for size in sizes],
        marker="o",
        label="random",
        color="#9aa6b2",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Patched node count")
    axes[2].set_ylabel("Median recovery")
    axes[2].set_title("Cumulative validation")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "focused_answer_circuit.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.scan_all_heads and not args.scan_mlps:
        args.scan_all_heads = True
        args.scan_mlps = True

    model_spec, tokenizer, model = load_model_and_tokenizer(args, for_detection=False)
    pairs = [row for row in read_jsonl(Path(args.pairs_path)) if row.get("task") in set(args.tasks)]
    if not pairs:
        raise RuntimeError("No circuit pairs selected. Check --pairs_path and --tasks.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_heads = get_num_heads(model)
    universe = all_nodes(args, num_heads)
    valid_pairs = build_valid_pairs(model, tokenizer, pairs, args)
    if not valid_pairs:
        raise RuntimeError("No valid pairs after denominator filtering.")

    head_controller = HeadPatchController(model)
    mlp_controller = MultiMlpPatchController(model)
    head_controller.install()
    mlp_controller.install()
    try:
        node_rows = run_node_scan(
            model=model,
            tokenizer=tokenizer,
            valid_pairs=valid_pairs,
            nodes=universe,
            head_controller=head_controller,
            mlp_controller=mlp_controller,
            args=args,
        )
        node_summary = summarize(node_rows, ["task", "node_id", "node_type", "layer", "head"])
        top_node_rows = write_top_nodes(output_dir, node_summary, args.top_n_nodes)
        top_nodes = [parse_node_id(row["node_id"]) for row in top_node_rows]

        position_rows = run_position_controls(
            model=model,
            tokenizer=tokenizer,
            valid_pairs=valid_pairs,
            top_nodes=top_nodes,
            head_controller=head_controller,
            mlp_controller=mlp_controller,
            args=args,
        )
        position_summary = summarize(
            position_rows,
            ["task", "node_id", "node_type", "layer", "head", "position_group"],
        )

        joint_rows = run_joint_validation(
            model=model,
            tokenizer=tokenizer,
            valid_pairs=valid_pairs,
            top_nodes=top_nodes,
            universe=universe,
            head_controller=head_controller,
            mlp_controller=mlp_controller,
            args=args,
        )
        joint_summary = summarize(joint_rows, ["task", "condition", "set_type", "set_size"])
    finally:
        head_controller.remove()
        mlp_controller.remove()

    write_jsonl(output_dir / "node_scan_results.jsonl", node_rows)
    write_csv(output_dir / "node_scan_results.csv", node_rows)
    write_csv(output_dir / "node_scan_summary.csv", node_summary)
    write_jsonl(output_dir / "position_control_results.jsonl", position_rows)
    write_csv(output_dir / "position_control_results.csv", position_rows)
    write_csv(output_dir / "position_control_summary.csv", position_summary)
    write_jsonl(output_dir / "joint_validation.jsonl", joint_rows)
    write_csv(output_dir / "joint_validation.csv", joint_rows)
    write_csv(output_dir / "joint_validation_summary.csv", joint_summary)
    plot_summary(output_dir, node_summary, position_summary, joint_summary)

    best_node = top_node_rows[0] if top_node_rows else None
    best_joint = None
    discovered_joint = [row for row in joint_summary if row.get("set_type") == "discovered"]
    if discovered_joint:
        best_joint = max(discovered_joint, key=lambda row: float(row["median_margin_recovery"]))
    summary = {
        "model": model_spec.as_metadata(),
        "pairs_path": str(args.pairs_path),
        "tasks": args.tasks,
        "layers": args.layers,
        "num_pairs_loaded": len(pairs),
        "num_valid_pairs": len(valid_pairs),
        "min_margin_denom": args.min_margin_denom,
        "num_nodes_scanned": len(universe),
        "top_n_nodes": args.top_n_nodes,
        "num_random_controls": args.num_random_controls,
        "best_node": best_node,
        "best_joint_set": best_joint,
        "success_thresholds": {
            "strong_circuit_median_recovery": 0.6,
            "useful_subcircuit_median_recovery": 0.2,
            "random_control_max_preferred": 0.1,
        },
    }
    with open(output_dir / "focused_answer_circuit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "num_valid_pairs": len(valid_pairs),
        "best_node": best_node,
        "best_joint_set": best_joint,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
