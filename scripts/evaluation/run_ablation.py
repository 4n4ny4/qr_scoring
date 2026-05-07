"""
Comparison ablation for SEC NIAH tasks using stock Hugging Face causal LMs.

Methods supported:
- QRScore-SEC (combined SEC detection)
- QRScore-8B-LME-TRAIN (same-model LME ranking, when available)
- QRScore-8B-NQ-TRAIN (same-model NQ ranking, when available)
- Transfer-<task> (optional cross-task transfer from per-task SEC rankings)
- Optional method rankings as transfer sources, e.g. QRScore-8B-LME-TRAIN
- Random-seed{42,123,456} (optional)
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import transformers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qrretriever.model_runtime import (
    load_stock_causal_lm,
    load_tokenizer,
    resolve_ablation_dir,
    resolve_detection_dir,
    resolve_model_spec,
)

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 30
DEFAULT_KNOCKOUT_SIZES = [0, 8, 16, 32, 48, 64, 96, 128]
DEFAULT_EXPORT_TOP_K = [8, 16, 32, 48, 64, 96, 128]


def resolve_device(device_arg):
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS is fragile for long-context 7B generation; prefer CPU by default.
        return "cpu"
    return "cpu"


def clear_device_cache(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lightweight head-masking on top of stock transformers causal LMs.
# We register a pre-forward hook on each attention layer's o_proj that zeros
# out the head slices listed in _masked_head_indices before the projection.
# This works for Llama/Qwen-style decoder stacks whose attention module exposes
# an o_proj receiving concatenated attention-head outputs.
# ---------------------------------------------------------------------------

def _make_o_proj_hook(attn_module, head_dim):
    """Return a hook that zeros masked heads in the input to o_proj."""
    def hook(_module, args):
        x = args[0]                          # (bsz, seq_len, num_heads * head_dim)
        indices = attn_module._masked_head_indices
        if indices is None:
            return args
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, -1, head_dim)  # (bsz, seq_len, num_heads, head_dim)
        x[:, :, indices, :] = 0
        return (x.view(bsz, seq_len, -1),) + args[1:]
    return hook


def get_decoder_layers(model):
    """Return the decoder layers for common HF causal LM architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        "Unsupported model structure for head masking. Expected `model.layers` "
        "or `transformer.h` decoder blocks."
    )


def install_head_masking(model):
    """Patch a stock causal LM with set_head_mask / per-layer hooks."""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    layers = get_decoder_layers(model)
    for layer in layers:
        attn = layer.self_attn
        attn._masked_head_indices = None
        attn.o_proj.register_forward_pre_hook(_make_o_proj_hook(attn, head_dim))

    def set_head_mask(masked_heads):
        for layer in layers:
            layer.self_attn._masked_head_indices = None
        if not masked_heads:
            return
        per_layer = defaultdict(list)
        for layer_idx, head_idx in masked_heads:
            per_layer[layer_idx].append(head_idx)
        for layer_idx, head_indices in per_layer.items():
            if layer_idx >= len(layers):
                raise IndexError(
                    f"Head ranking references layer {layer_idx}, but model only has {len(layers)} layers."
                )
            max_head = model.config.num_attention_heads
            invalid = [h for h in head_indices if h >= max_head]
            if invalid:
                raise IndexError(
                    f"Head ranking references head(s) {invalid} in layer {layer_idx}, "
                    f"but model only has {max_head} attention heads."
                )
            layers[layer_idx].self_attn._masked_head_indices = head_indices

    model.set_head_mask = set_head_mask


def normalize_answer(s):
    if s is None or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"[,.;:!?\"']+$", "", s)
    return " ".join(s.split())


def extract_short_answer(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.split("\n")[0].strip()
    text = re.split(r"[.!?]\s", text)[0].strip()
    text = re.sub(r"[.!?]+$", "", text)
    return text


def answers_match(pred, gold):
    if not pred or not gold:
        return False
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p or p in g:
        return True
    return False


def build_prompt(context, question):
    return (
        "Read the following document and answer the question.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely with just the answer value."
    )


def format_messages(context, question):
    return [{"role": "user", "content": build_prompt(context, question)}]


def tokenize_messages(tokenizer, messages, prompt_text, max_context_tokens):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=max_context_tokens,
            return_dict=True,
            return_tensors="pt",
        )

    return tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_tokens,
    )


def generate_answer(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                    max_context_tokens=8192):
    tokenizer.truncation_side = "left"
    prompt_text = build_prompt(prompt["context"], prompt["question"])
    messages = format_messages(prompt["context"], prompt["question"])
    inputs = tokenize_messages(tokenizer, messages, prompt_text, max_context_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=model.config.pad_token_id,
            use_cache=True,
        )
    start = inputs["input_ids"].shape[1]
    new_tokens = out[0][start:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    token_ids = new_tokens.tolist()
    return extract_short_answer(raw_text), raw_text, token_ids


def load_ranked_heads_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    heads = []
    for row in data:
        head_str = row[0] if isinstance(row, list) else row.get("head")
        layer, head = map(int, head_str.split("-"))
        heads.append((layer, head))
    return heads


def resolve_topk_export_from_manifest(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    exports = manifest.get("exports", {})
    if not exports:
        return None, None

    best_k = max(int(k) for k in exports.keys())
    export_path = exports[str(best_k)]
    if not os.path.exists(export_path):
        export_path = os.path.join(os.path.dirname(manifest_path), os.path.basename(export_path))
    if not os.path.exists(export_path):
        return None, best_k
    return export_path, best_k


def ensure_full_ranking(heads, num_layers, num_heads_per_layer, seed):
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads_per_layer)]
    seen = set(heads)
    remaining = [x for x in all_heads if x not in seen]
    random.Random(seed).shuffle(remaining)
    return heads + remaining


def export_top_k_from_ranking(path, label, top_ks, export_dir):
    if not os.path.exists(path):
        return
    os.makedirs(export_dir, exist_ok=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    manifest = {
        "source_file": path,
        "label": label,
        "top_k_values": sorted(set(k for k in top_ks if k > 0)),
        "exports": {},
    }
    for k in manifest["top_k_values"]:
        out_path = os.path.join(export_dir, f"{label}_top{k}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data[: min(k, len(data))], f, indent=2)
        manifest["exports"][str(k)] = out_path

    manifest_path = os.path.join(export_dir, f"{label}_heads_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def model_specific_train_ranking_candidates(model_spec, results_dir, label, train_ranking_dir=None):
    """Return candidate train-ranking paths for the current model only.

    The old ablation workflow reused Llama-3.1-8B LME/NQ rankings for every
    target model. That is misleading for cross-model causal ablations: Qwen
    should use Qwen-detected train heads, OLMo should use OLMo-detected train
    heads, and so on. Keep the legacy method labels for result compatibility,
    but resolve their files in a model-aware way.
    """
    label = label.lower()
    if label not in {"lme", "nq"}:
        raise ValueError(f"Unsupported train ranking label: {label}")

    file_stems = {
        "lme": [
            "lme_train_heads.json",
            "LME_train_heads.json",
            "long_context_lme_train_heads.json",
            "lme_TRAIN.json",
        ],
        "nq": [
            "nq_train_heads.json",
            "NQ_train_heads.json",
            "long_context_nq_train_heads.json",
            "nq_TRAIN.json",
        ],
    }[label]

    candidates = []
    if train_ranking_dir:
        candidates.extend(os.path.join(train_ranking_dir, name) for name in file_stems)
        if model_spec.model_family == "qwen":
            candidates.extend(
                os.path.join(train_ranking_dir, name)
                for name in [
                    f"{label}_TRAIN_qwen.json",
                    f"{label.upper()}_TRAIN_qwen.json",
                    f"{label}_train_qwen.json",
                ]
            )

    if model_spec.model_family == "llama":
        candidates.extend(os.path.join(results_dir, name) for name in file_stems)
        candidates.extend(
            [
                os.path.join(PROJECT_DIR, "Llama-3.1-8B-Instruct", f"{label}_TRAIN.json"),
                os.path.join(PROJECT_DIR, "Llama-3.1-8B-Instruct", f"{label.upper()}_TRAIN.json"),
            ]
        )
    elif model_spec.model_family == "qwen":
        qwen_dirs = [
            "Qwen-2.5-7B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen__Qwen2.5-7B-Instruct",
        ]
        qwen_files = [
            f"{label}_TRAIN_qwen.json",
            f"{label.upper()}_TRAIN_qwen.json",
            f"{label}_train_qwen.json",
            f"{label}_TRAIN.json",
            f"{label.upper()}_TRAIN.json",
        ]
        for directory in qwen_dirs:
            candidates.extend(os.path.join(PROJECT_DIR, directory, filename) for filename in qwen_files)
            candidates.extend(os.path.join(os.sep, directory, filename) for filename in qwen_files)
        candidates.extend(os.path.join(results_dir, name) for name in file_stems)
    else:
        candidates.extend(os.path.join(results_dir, name) for name in file_stems)

    return candidates


def resolve_model_specific_train_ranking(model_spec, results_dir, label, train_ranking_dir=None):
    candidates = model_specific_train_ranking_candidates(
        model_spec, results_dir, label, train_ranking_dir=train_ranking_dir
    )
    return next((path for path in candidates if os.path.exists(path)), None), candidates


def sanitize_ranking(heads, num_layers, num_heads_per_layer, label):
    """Drop any head indices that are invalid for the current model shape."""
    valid = []
    dropped = 0
    for layer, head in heads:
        if 0 <= layer < num_layers and 0 <= head < num_heads_per_layer:
            valid.append((layer, head))
        else:
            dropped += 1
    if dropped:
        print(
            f"  {label}: dropped {dropped} out-of-range heads "
            f"for model shape {num_layers}x{num_heads_per_layer}"
        )
    return valid


def compute_jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def save_head_similarity(task_rankings, top_ks, output_path, model_metadata=None):
    tasks = sorted(task_rankings.keys())
    payload = {
        "tasks": tasks,
        "top_k": {},
    }
    if model_metadata is not None:
        payload.update(model_metadata)
    for k in sorted(set(top_ks)):
        matrix = []
        for src in tasks:
            row = []
            src_set = set(task_rankings[src][:k])
            for tgt in tasks:
                tgt_set = set(task_rankings[tgt][:k])
                row.append(compute_jaccard(src_set, tgt_set))
            matrix.append(row)
        payload["top_k"][str(k)] = matrix

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_random_heads(num_layers, num_heads_per_layer, total_heads, seed=42):
    rng = random.Random(seed)
    all_heads = [(l, h) for l in range(num_layers)
                 for h in range(num_heads_per_layer)]
    rng.shuffle(all_heads)
    return all_heads[:total_heads]


def load_method_rankings(args, model_spec, num_layers, num_heads_per_layer):
    results_dir = resolve_detection_dir(PROJECT_DIR, model_spec, args.ranking_dir)
    legacy_results_dir = os.path.join(PROJECT_DIR, "results", "detection")
    methods = {}

    # 1) SEC combined ranking
    sec_candidates = [os.path.join(results_dir, "long_context_combined_heads.json")]
    if model_spec.model_family == "llama":
        sec_candidates.append(os.path.join(legacy_results_dir, "long_context_combined_heads.json"))
    sec_path = next((path for path in sec_candidates if os.path.exists(path)), None)
    if sec_path is not None:
        methods["QRScore-SEC"] = load_ranked_heads_json(sec_path)
        print(f"  QRScore-SEC: loaded from {sec_path} ({len(methods['QRScore-SEC'])} heads)")
    else:
        print(f"  QRScore-SEC: not found under {results_dir} (run same-model detection first)")

    # 2) Same-model LME/NQ train rankings.
    if model_spec.allow_external_rankings:
        lme_train_path, lme_candidates = resolve_model_specific_train_ranking(
            model_spec, results_dir, "lme", train_ranking_dir=args.train_ranking_dir
        )
        nq_train_path, nq_candidates = resolve_model_specific_train_ranking(
            model_spec, results_dir, "nq", train_ranking_dir=args.train_ranking_dir
        )
        if args.lme_ranking_file is not None:
            lme_candidates = [args.lme_ranking_file]
            lme_train_path = args.lme_ranking_file if os.path.exists(args.lme_ranking_file) else None
        if args.nq_ranking_file is not None:
            nq_candidates = [args.nq_ranking_file]
            nq_train_path = args.nq_ranking_file if os.path.exists(args.nq_ranking_file) else None

        if lme_train_path is not None:
            lme_heads = sanitize_ranking(
                load_ranked_heads_json(lme_train_path),
                num_layers,
                num_heads_per_layer,
                "QRScore-8B-LME-TRAIN",
            )
            methods["QRScore-8B-LME-TRAIN"] = ensure_full_ranking(
                lme_heads, num_layers, num_heads_per_layer, seed=42
            )
            print(f"  QRScore-8B-LME-TRAIN: loaded from {lme_train_path}")
        else:
            print(
                "  QRScore-8B-LME-TRAIN: not found for this model. Checked: "
                + ", ".join(lme_candidates)
            )

        if nq_train_path is not None:
            nq_heads = sanitize_ranking(
                load_ranked_heads_json(nq_train_path),
                num_layers,
                num_heads_per_layer,
                "QRScore-8B-NQ-TRAIN",
            )
            methods["QRScore-8B-NQ-TRAIN"] = ensure_full_ranking(
                nq_heads, num_layers, num_heads_per_layer, seed=43
            )
            print(f"  QRScore-8B-NQ-TRAIN: loaded from {nq_train_path}")
        else:
            print(
                "  QRScore-8B-NQ-TRAIN: not found for this model. Checked: "
                + ", ".join(nq_candidates)
            )

        # Export deterministic top-K slices for train rankings.
        external_export_dir = os.path.join(results_dir, "topk", "external")
        if lme_train_path is not None:
            export_top_k_from_ranking(lme_train_path, "lme_train", args.export_top_k, external_export_dir)
        if nq_train_path is not None:
            export_top_k_from_ranking(nq_train_path, "nq_train", args.export_top_k, external_export_dir)

    # 3) Optional cross-task transfer methods.
    transfer_rankings = {}
    if args.enable_cross_task_transfer and not args.transfer_only_extra_sources:
        for task in args.tasks:
            candidate_paths = [
                os.path.join(results_dir, f"long_context_{task}_heads.json"),
                os.path.join(results_dir, f"{task}_heads.json"),
            ]
            if model_spec.model_family == "llama":
                candidate_paths.extend(
                    [
                        os.path.join(legacy_results_dir, f"long_context_{task}_heads.json"),
                        os.path.join(legacy_results_dir, f"{task}_heads.json"),
                    ]
                )
            task_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            ranking_source = None
            if task_path is not None:
                task_heads = load_ranked_heads_json(task_path)
                ranking_source = task_path
            else:
                manifest_candidates = [
                    os.path.join(results_dir, "topk", f"long_context_{task}_heads_manifest.json"),
                    os.path.join(results_dir, "topk", f"{task}_heads_manifest.json"),
                ]
                if model_spec.model_family == "llama":
                    manifest_candidates.extend(
                        [
                            os.path.join(legacy_results_dir, "topk", f"long_context_{task}_heads_manifest.json"),
                            os.path.join(legacy_results_dir, "topk", f"{task}_heads_manifest.json"),
                        ]
                    )
                manifest_path = next((p for p in manifest_candidates if os.path.exists(p)), None)
                if manifest_path is None:
                    raise FileNotFoundError(
                        "Transfer mode requires a per-task ranking file or exported top-K manifest. Checked: "
                        + ", ".join(candidate_paths + manifest_candidates)
                    )
                export_path, recovered_k = resolve_topk_export_from_manifest(manifest_path)
                if export_path is None:
                    raise FileNotFoundError(
                        f"Found manifest for task `{task}` but could not resolve a local exported ranking from "
                        f"{manifest_path}."
                    )
                task_heads = load_ranked_heads_json(export_path)
                ranking_source = f"{export_path} (top-{recovered_k} fallback)"
                print(
                    f"  Transfer-{task}: full ranking missing, using exported top-{recovered_k} "
                    f"fallback from {export_path}"
                )
            transfer_rankings[task] = ensure_full_ranking(
                task_heads, num_layers, num_heads_per_layer, seed=100 + args.tasks.index(task)
            )
            print(f"  Transfer-{task}: loaded from {ranking_source}")
            methods[f"Transfer-{task}"] = transfer_rankings[task]

    # 4) Optional random baselines.
    if args.include_random_baselines:
        for seed in [42, 123, 456]:
            name = f"Random-seed{seed}"
            methods[name] = generate_random_heads(
                num_layers,
                num_heads_per_layer,
                total_heads=max(args.knockout_sizes) + 50,
                seed=seed,
            )
            print(f"  {name}: {len(methods[name])} heads")

    return methods, transfer_rankings


def validate_ranking_compatibility(method_name, ranking, model_spec, num_layers, num_heads):
    invalid = [
        (layer_idx, head_idx)
        for layer_idx, head_idx in ranking
        if layer_idx >= num_layers or head_idx >= num_heads
    ]
    if invalid:
        sample = ", ".join(f"({l},{h})" for l, h in invalid[:5])
        return (
            False,
            f"Ranking `{method_name}` is incompatible with the loaded model "
            f"`{model_spec.model_name}` ({num_layers} layers x {num_heads} heads). "
            f"Example invalid entries: {sample}. You likely need same-model rankings."
        )
    return True, None


def add_extra_transfer_sources(
    transfer_rankings,
    method_rankings,
    extra_sources,
    model_spec,
    num_layers,
    num_heads,
):
    if not extra_sources:
        return

    for source in extra_sources:
        if source not in method_rankings:
            available = ", ".join(sorted(method_rankings.keys()))
            raise ValueError(
                f"Requested transfer source `{source}` is not available. "
                f"Available method rankings: {available}"
            )

        ranking = method_rankings[source]
        is_compatible, reason = validate_ranking_compatibility(
            source, ranking, model_spec, num_layers, num_heads
        )
        if not is_compatible:
            raise ValueError(reason)

        if source in transfer_rankings:
            print(f"  Extra transfer source `{source}` already present; skipping duplicate.")
            continue

        transfer_rankings[source] = ranking
        print(f"  Extra transfer source `{source}`: added to transfer matrix")


def run_single_sweep(model, tokenizer, test_instances, head_ranking,
                     knockout_sizes, max_context_tokens=8192,
                     progress_every=20, sweep_label=""):
    results = {}
    total_steps = len(knockout_sizes) * len(test_instances)
    completed_steps = 0
    sweep_start = time.time()

    if sweep_label:
        print(f"    Sweep: {sweep_label}")
    print(
        f"    Progress plan: {len(knockout_sizes)} K values x "
        f"{len(test_instances)} instances = {total_steps} steps"
    )

    for K in knockout_sizes:
        if K == 0:
            model.set_head_mask(None)
        else:
            masked = head_ranking[:min(K, len(head_ranking))]
            model.set_head_mask(masked)

        correct = 0
        total = 0
        per_task = defaultdict(lambda: {"correct": 0, "total": 0})
        details = []

        for idx, inst in enumerate(test_instances, start=1):
            prompt = {"context": inst["context"], "question": inst["question"]}
            pred, raw_text, token_ids = generate_answer(
                model, tokenizer, prompt,
                max_context_tokens=max_context_tokens)
            gold = inst["needle_value"]
            match = answers_match(pred, gold)
            correct += int(match)
            total += 1
            per_task[inst["task"]]["correct"] += int(match)
            per_task[inst["task"]]["total"] += 1
            details.append({
                "idx": inst["idx"],
                "task": inst["task"],
                "gold": gold,
                "pred": pred,
                "raw_text": raw_text,
                "token_ids": token_ids,
                "correct": int(match),
            })
            clear_device_cache(model.device.type)

            completed_steps += 1
            should_log = (
                completed_steps == 1
                or completed_steps % max(1, progress_every) == 0
                or completed_steps == total_steps
            )
            if should_log:
                elapsed = time.time() - sweep_start
                rate = completed_steps / elapsed if elapsed > 0 else 0.0
                remaining = max(total_steps - completed_steps, 0)
                eta_sec = remaining / rate if rate > 0 else 0.0
                print(
                    f"      progress {completed_steps}/{total_steps} "
                    f"({(100.0 * completed_steps / total_steps):5.1f}%) | "
                    f"K={K} inst={idx}/{len(test_instances)} | "
                    f"elapsed={elapsed/60.0:.1f}m eta={eta_sec/60.0:.1f}m"
                )

        accuracy = correct / total if total else 0
        print(f"    K={K:3d}: accuracy={accuracy:.4f} ({correct}/{total})")

        results[K] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_task": {
                t: {
                    "accuracy": d["correct"] / d["total"] if d["total"] else 0,
                    "correct": d["correct"],
                    "total": d["total"],
                }
                for t, d in per_task.items()
            },
            "details": details,
        }

    model.set_head_mask(None)
    return results


def run_cross_task_transfer(model, tokenizer, per_task_instances, transfer_rankings,
                            knockout_sizes, max_context_tokens, progress_every):
    matrix = {
        "knockout_sizes": knockout_sizes,
        "sources": sorted(transfer_rankings.keys()),
        "targets": sorted(per_task_instances.keys()),
        "results": {},
    }

    for source_task, ranking in transfer_rankings.items():
        matrix["results"][source_task] = {}
        print(f"\n[Transfer Source] {source_task}")
        for target_task, instances in per_task_instances.items():
            print(f"  -> Target {target_task} ({len(instances)} instances)")
            sweep = run_single_sweep(
                model,
                tokenizer,
                instances,
                ranking,
                knockout_sizes,
                max_context_tokens=max_context_tokens,
                progress_every=progress_every,
                sweep_label=f"source={source_task} -> target={target_task}",
            )
            baseline = sweep[0]["accuracy"] if 0 in sweep else None
            by_k = {}
            for k in knockout_sizes:
                acc = sweep[k]["accuracy"]
                drop = (baseline - acc) if baseline is not None else None
                by_k[str(k)] = {
                    "accuracy": acc,
                    "drop_from_k0": drop,
                }
            matrix["results"][source_task][target_task] = {
                "baseline": baseline,
                "by_k": by_k,
            }

    return matrix


def compute_specificity_metrics(transfer_matrix, summary_k):
    sources = transfer_matrix["sources"]
    targets = transfer_matrix["targets"]
    results = {
        "summary_k": summary_k,
        "sources": {},
    }

    for source in sources:
        source_block = transfer_matrix["results"][source]
        on_target_drop = None
        off_target_drops = []
        for target in targets:
            drop = source_block[target]["by_k"].get(str(summary_k), {}).get("drop_from_k0")
            if drop is None:
                continue
            if source == target:
                on_target_drop = drop
            else:
                off_target_drops.append(drop)

        off_target_mean = float(np.mean(off_target_drops)) if off_target_drops else 0.0
        eps = 1e-9
        specificity_index = None if on_target_drop is None else on_target_drop - off_target_mean
        surgicality_ratio = None if on_target_drop is None else on_target_drop / max(off_target_mean, eps)

        results["sources"][source] = {
            "on_target_drop": on_target_drop,
            "off_target_mean_drop": off_target_mean,
            "specificity_index": specificity_index,
            "surgicality_ratio": surgicality_ratio,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Comparison ablation across detection methods")
    parser.add_argument("--niah_dir", default=os.path.join(PROJECT_DIR, "data", "niah_input"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--ranking_dir", default=None, help="Optional override for model-specific detection rankings.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_slug", default=None, help="Optional slug override used for result directory resolution.")
    parser.add_argument("--tokenizer_name", default=None, help="Optional tokenizer override.")
    parser.add_argument(
        "--lme_ranking_file",
        default=None,
        help=(
            "Optional override for QRScore-8B-LME-TRAIN ranking JSON "
            "(list of [\"layer-head\", score] rows)."
        ),
    )
    parser.add_argument(
        "--nq_ranking_file",
        default=None,
        help=(
            "Optional override for QRScore-8B-NQ-TRAIN ranking JSON "
            "(list of [\"layer-head\", score] rows)."
        ),
    )
    parser.add_argument(
        "--train_ranking_dir",
        default=None,
        help=(
            "Optional directory containing same-model LME/NQ ranking JSON files, "
            "such as lme_TRAIN_qwen.json and nq_TRAIN_qwen.json."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device. `auto` prefers CUDA, otherwise falls back to CPU.",
    )
    parser.add_argument("--knockout_sizes", nargs="+", type=int, default=DEFAULT_KNOCKOUT_SIZES)
    parser.add_argument("--max_instances_per_task", type=int, default=None)
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=8192,
        help="Max prompt tokens; full NIAH ~6500. Lower to 4096 if OOM.",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Specific methods to run (default: all available)")
    parser.add_argument("--enable_cross_task_transfer", action="store_true")
    parser.add_argument(
        "--transfer_extra_sources",
        nargs="+",
        default=[],
        help=(
            "Additional loaded method rankings to include as source rows in "
            "cross-task transfer, e.g. QRScore-8B-LME-TRAIN."
        ),
    )
    parser.add_argument(
        "--transfer_only_extra_sources",
        action="store_true",
        help="Run cross-transfer only for --transfer_extra_sources, skipping SEC per-task source rows.",
    )
    parser.add_argument("--include_random_baselines", action="store_true")
    parser.add_argument("--transfer_summary_k", type=int, default=16)
    parser.add_argument("--export_top_k", nargs="+", type=int, default=DEFAULT_EXPORT_TOP_K)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=20,
        help="Print progress every N generated answers (default: 20)",
    )
    parser.add_argument(
        "--log_tokens",
        action="store_true",
        help="Write per-method JSONL token logs (idx, K, token_ids, raw_text) for analysis.",
    )
    args = parser.parse_args()
    if args.transfer_extra_sources and not args.enable_cross_task_transfer:
        parser.error("--transfer_extra_sources requires --enable_cross_task_transfer")
    if args.transfer_only_extra_sources and not args.transfer_extra_sources:
        parser.error("--transfer_only_extra_sources requires --transfer_extra_sources")

    model_spec = resolve_model_spec(
        model_name=args.model_name,
        model_slug=args.model_slug,
        tokenizer_name=args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    args.output_dir = resolve_ablation_dir(PROJECT_DIR, model_spec, args.output_dir)
    args.ranking_dir = resolve_detection_dir(PROJECT_DIR, model_spec, args.ranking_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resolved project_dir={PROJECT_DIR}")
    print(f"Resolved niah_dir={args.niah_dir}")
    print(f"Resolved output_dir={args.output_dir}")
    print(f"Resolved ranking_dir={args.ranking_dir}")
    print(f"Resolved model={model_spec.model_name} ({model_spec.model_family})")

    # Load test instances by task and pooled.
    per_task_instances = {}
    test_instances = []
    for task in args.tasks:
        test_path = os.path.join(args.niah_dir, f"{task}_test.json")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing required test file: {test_path}")
        with open(test_path, encoding="utf-8") as f:
            data = json.load(f)
        if args.max_instances_per_task and len(data) > args.max_instances_per_task:
            data = data[:args.max_instances_per_task]
        for inst in data:
            inst["task"] = task
        per_task_instances[task] = data
        test_instances.extend(data)

    print(f"Loaded {len(test_instances)} test instances across {len(args.tasks)} tasks")

    # Load model.
    print(f"\nLoading model: {args.model_name}")
    resolved_device = resolve_device(args.device)
    print(f"Resolved device: {resolved_device}")
    tokenizer = load_tokenizer(model_spec)
    model = load_stock_causal_lm(model_spec, resolved_device, for_detection=False)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id or model.config.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, attr):
                setattr(model.generation_config, attr, None)
    model.eval()
    install_head_masking(model)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Model: {num_layers} layers x {num_heads} heads")

    # Load method rankings.
    print("\nLoading method rankings...")
    all_methods, transfer_rankings = load_method_rankings(args, model_spec, num_layers, num_heads)
    available_methods = dict(all_methods)

    if args.methods:
        missing_requested = [method for method in args.methods if method not in all_methods]
        if missing_requested:
            print(
                "ERROR: Requested method ranking(s) are unavailable for this model: "
                + ", ".join(missing_requested)
            )
            print("Generate the same-model ranking files first, or remove the missing methods from --methods.")
            sys.exit(1)
        all_methods = {k: v for k, v in all_methods.items() if k in args.methods}

    if args.enable_cross_task_transfer:
        add_extra_transfer_sources(
            transfer_rankings,
            available_methods,
            args.transfer_extra_sources,
            model_spec,
            num_layers,
            num_heads,
        )

    if not all_methods:
        print("ERROR: No methods available. Run detection first.")
        sys.exit(1)

    compatible_methods = {}
    skipped_methods = {}
    for method_name, ranking in all_methods.items():
        is_compatible, reason = validate_ranking_compatibility(
            method_name, ranking, model_spec, num_layers, num_heads
        )
        if is_compatible:
            compatible_methods[method_name] = ranking
        else:
            skipped_methods[method_name] = reason

    if skipped_methods:
        print("\nSkipping incompatible ranking methods for this model:")
        for method_name, reason in skipped_methods.items():
            print(f"  - {method_name}: {reason}")

    all_methods = compatible_methods

    if not all_methods:
        print("ERROR: No compatible ranking methods are available for this model.")
        print(
            f"For `{model_spec.model_name}`, generate same-model SEC detection rankings first, "
            "or rerun with --include_random_baselines for a smoke test."
        )
        sys.exit(1)

    print(f"\nMethods to evaluate: {list(all_methods.keys())}")
    total_method_steps = len(all_methods) * len(args.knockout_sizes) * len(test_instances)
    print(
        f"Planned pooled workload: {len(all_methods)} methods x "
        f"{len(args.knockout_sizes)} K values x {len(test_instances)} instances = "
        f"{total_method_steps} steps"
    )

    # Run pooled sweeps.
    all_results = {}
    for method_idx, (method_name, head_ranking) in enumerate(all_methods.items(), start=1):
        print(f"\n{'=' * 60}")
        print(f"Method {method_idx}/{len(all_methods)}: {method_name}")
        print(f"{'=' * 60}")

        method_results = run_single_sweep(
            model,
            tokenizer,
            test_instances,
            head_ranking,
            args.knockout_sizes,
            max_context_tokens=args.max_context_tokens,
            progress_every=args.progress_every,
            sweep_label=f"method={method_name}",
        )
        all_results[method_name] = method_results

        # Write per-method token log (JSONL) if requested.
        if args.log_tokens:
            token_log_path = os.path.join(
                args.output_dir, f"{method_name.replace(' ', '_')}_token_log.jsonl")
            with open(token_log_path, "w", encoding="utf-8") as tl:
                for k in args.knockout_sizes:
                    for d in method_results[k]["details"]:
                        tl.write(json.dumps({
                            "method": method_name,
                            "K": k,
                            "idx": d["idx"],
                            "task": d["task"],
                            "gold": d["gold"],
                            "pred": d["pred"],
                            "raw_text": d["raw_text"],
                            "token_ids": d["token_ids"],
                            "correct": d["correct"],
                        }) + "\n")
            print(f"  Token log: {token_log_path}")

        # Strip raw_text / token_ids from the summary JSON to keep it compact.
        summary_details = {}
        for k in args.knockout_sizes:
            summary_details[str(k)] = [
                {key: val for key, val in d.items() if key not in ("raw_text", "token_ids")}
                for d in method_results[k]["details"]
            ]

        method_path = os.path.join(args.output_dir, f"{method_name.replace(' ', '_')}_results.json")
        with open(method_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_spec.model_name,
                    "model_slug": model_spec.model_slug,
                    "model_family": model_spec.model_family,
                    "method": method_name,
                    "knockout_sizes": args.knockout_sizes,
                    "accuracy_curve": {
                        str(k): method_results[k]["accuracy"]
                        for k in args.knockout_sizes
                    },
                    "per_task_curves": {
                        task: {
                            str(k): method_results[k]["per_task"].get(task, {}).get("accuracy", 0)
                            for k in args.knockout_sizes
                        }
                        for task in args.tasks
                    },
                    "details": summary_details,
                },
                f,
                indent=2,
            )

    # Build main summary.
    summary = {
        "model": args.model_name,
        "model_name": model_spec.model_name,
        "model_slug": model_spec.model_slug,
        "model_family": model_spec.model_family,
        "ranking_dir": args.ranking_dir,
        "num_instances": len(test_instances),
        "knockout_sizes": args.knockout_sizes,
        "methods": {},
    }

    for method_name, method_results in all_results.items():
        curve = {str(k): method_results[k]["accuracy"] for k in args.knockout_sizes}
        baseline_acc = method_results[0]["accuracy"] if 0 in method_results else None
        k16_acc = method_results[16]["accuracy"] if 16 in method_results else None
        drop_at_16 = (baseline_acc - k16_acc) if baseline_acc is not None and k16_acc is not None else None

        summary["methods"][method_name] = {
            "accuracy_curve": curve,
            "baseline_accuracy": baseline_acc,
            "accuracy_at_k16": k16_acc,
            "drop_at_k16": drop_at_16,
        }

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    # Merge with existing summary to avoid overwriting results from prior runs.
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("model_slug") == summary["model_slug"]:
            existing.setdefault("methods", {}).update(summary["methods"])
            summary["methods"] = existing["methods"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Optional cross-task transfer analysis.
    if args.enable_cross_task_transfer:
        if not transfer_rankings:
            raise RuntimeError("Cross-task transfer enabled, but no per-task rankings were loaded.")

        transfer_matrix = run_cross_task_transfer(
            model,
            tokenizer,
            per_task_instances,
            transfer_rankings,
            args.knockout_sizes,
            args.max_context_tokens,
            args.progress_every,
        )
        transfer_matrix.update(
            {
                "model_name": model_spec.model_name,
                "model_slug": model_spec.model_slug,
                "model_family": model_spec.model_family,
            }
        )

        transfer_path = os.path.join(args.output_dir, "cross_task_transfer_matrix.json")
        with open(transfer_path, "w", encoding="utf-8") as f:
            json.dump(transfer_matrix, f, indent=2)
        print(f"Saved transfer matrix: {transfer_path}")

        specificity = compute_specificity_metrics(transfer_matrix, args.transfer_summary_k)
        specificity.update(
            {
                "model_name": model_spec.model_name,
                "model_slug": model_spec.model_slug,
                "model_family": model_spec.model_family,
            }
        )
        specificity_path = os.path.join(args.output_dir, "cross_task_specificity_metrics.json")
        with open(specificity_path, "w", encoding="utf-8") as f:
            json.dump(specificity, f, indent=2)
        print(f"Saved specificity metrics: {specificity_path}")

        similarity_path = os.path.join(args.output_dir, "cross_task_head_similarity_topk.json")
        save_head_similarity(transfer_rankings, args.export_top_k, similarity_path, model_spec.as_metadata())
        print(f"Saved head similarity matrices: {similarity_path}")


if __name__ == "__main__":
    main()
