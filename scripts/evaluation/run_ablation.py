"""
Comparison ablation for SEC NIAH tasks using precomputed head rankings.

Methods supported:
- QRScore-SEC (combined SEC detection)
- QRScore-8B-LME-TRAIN (external ranking from <external_rankings_dir>/lme_TRAIN.json)
- QRScore-8B-NQ-TRAIN (external ranking from <external_rankings_dir>/nq_TRAIN.json)
- Transfer-<task> (optional cross-task transfer from per-task SEC rankings)
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

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_EXTERNAL_RANKINGS_DIR = "Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 30
DEFAULT_KNOCKOUT_SIZES = [0, 8, 16, 32, 48, 64, 96, 128]
DEFAULT_EXPORT_TOP_K = [8, 16, 32, 48, 64, 96, 128]


# ---------------------------------------------------------------------------
# Lightweight head-masking on top of a stock HF CausalLM model.
# We register a pre-forward hook on each attention layer's o_proj that zeros
# out the head slices listed in _masked_head_indices before the projection.
# Works for decoder architectures exposing model.layers[*].self_attn.o_proj.
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


def install_head_masking(model):
    """Patch a stock CausalLM with set_head_mask / per-layer hooks."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError(
            "Unsupported model architecture for head masking: expected model.layers"
        )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer in model.model.layers:
        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "o_proj"):
            raise ValueError(
                "Unsupported attention module for head masking: expected self_attn.o_proj"
            )
        attn = layer.self_attn
        attn._masked_head_indices = None
        attn.o_proj.register_forward_pre_hook(_make_o_proj_hook(attn, head_dim))

    def set_head_mask(masked_heads):
        for layer in model.model.layers:
            layer.self_attn._masked_head_indices = None
        if not masked_heads:
            return
        per_layer = defaultdict(list)
        for layer_idx, head_idx in masked_heads:
            per_layer[layer_idx].append(head_idx)
        for layer_idx, head_indices in per_layer.items():
            model.model.layers[layer_idx].self_attn._masked_head_indices = head_indices

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


def build_prompt(tokenizer, context, question):
    messages = [
        {
            "role": "user",
            "content": (
                "Read the following document and answer the question.\n\n"
                f"Document:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely with just the answer value."
            ),
        }
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return messages[0]["content"]


def generate_answer(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                    max_context_tokens=8192):
    tokenizer.truncation_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_context_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
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


def compute_jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def save_head_similarity(task_rankings, top_ks, output_path):
    tasks = sorted(task_rankings.keys())
    payload = {
        "tasks": tasks,
        "top_k": {},
    }
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


def load_method_rankings(args, num_layers, num_heads_per_layer):
    results_dir = args.detection_results_dir
    methods = {}

    # 1) SEC combined ranking
    sec_path = os.path.join(results_dir, "long_context_combined_heads.json")
    if os.path.exists(sec_path):
        sec_heads = load_ranked_heads_json(sec_path)
        sec_heads = sanitize_ranking(sec_heads, num_layers, num_heads_per_layer, "QRScore-SEC")
        methods["QRScore-SEC"] = sec_heads
        print(f"  QRScore-SEC: loaded from {sec_path} ({len(methods['QRScore-SEC'])} heads)")
    else:
        print("  QRScore-SEC: not found (run detection first)")

    # 2) External LME/NQ train rankings
    lme_train_path = os.path.join(args.external_rankings_dir, "lme_TRAIN.json")
    nq_train_path = os.path.join(args.external_rankings_dir, "nq_TRAIN.json")

    if os.path.exists(lme_train_path):
        lme_heads = load_ranked_heads_json(lme_train_path)
        lme_heads = sanitize_ranking(lme_heads, num_layers, num_heads_per_layer, "QRScore-8B-LME-TRAIN")
        methods["QRScore-8B-LME-TRAIN"] = ensure_full_ranking(
            lme_heads, num_layers, num_heads_per_layer, seed=42
        )
        print(f"  QRScore-8B-LME-TRAIN: loaded from {lme_train_path}")

    if os.path.exists(nq_train_path):
        nq_heads = load_ranked_heads_json(nq_train_path)
        nq_heads = sanitize_ranking(nq_heads, num_layers, num_heads_per_layer, "QRScore-8B-NQ-TRAIN")
        methods["QRScore-8B-NQ-TRAIN"] = ensure_full_ranking(
            nq_heads, num_layers, num_heads_per_layer, seed=43
        )
        print(f"  QRScore-8B-NQ-TRAIN: loaded from {nq_train_path}")

    # Export deterministic top-K slices for external rankings.
    external_export_dir = os.path.join(results_dir, "topk", "external")
    export_top_k_from_ranking(lme_train_path, "lme_train", args.export_top_k, external_export_dir)
    export_top_k_from_ranking(nq_train_path, "nq_train", args.export_top_k, external_export_dir)

    # 3) Optional cross-task transfer methods.
    transfer_rankings = {}
    if args.enable_cross_task_transfer:
        for task in args.tasks:
            candidate_paths = [
                os.path.join(results_dir, f"long_context_{task}_heads.json"),
                os.path.join(results_dir, f"{task}_heads.json"),
            ]
            task_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if task_path is None:
                raise FileNotFoundError(
                    "Transfer mode requires per-task ranking file. Checked: "
                    + ", ".join(candidate_paths)
                )
            task_heads = load_ranked_heads_json(task_path)
            task_heads = sanitize_ranking(
                task_heads,
                num_layers,
                num_heads_per_layer,
                f"Transfer-{task}",
            )
            transfer_rankings[task] = ensure_full_ranking(
                task_heads, num_layers, num_heads_per_layer, seed=100 + args.tasks.index(task)
            )
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
            prompt = build_prompt(tokenizer, inst["context"], inst["question"])
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
    parser = argparse.ArgumentParser(description="Comparison ablation across head-ranking methods")
    parser.add_argument("--niah_dir", default=os.path.join(PROJECT_DIR, "data", "niah_input"))
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_DIR, "results", "comparison_ablation"))
    parser.add_argument(
        "--detection_results_dir",
        default=os.path.join(PROJECT_DIR, "results", "detection"),
        help="Directory containing long_context_*_heads.json rankings to ablate",
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument(
        "--external_rankings_dir",
        default=os.path.join(PROJECT_DIR, DEFAULT_EXTERNAL_RANKINGS_DIR),
        help="Directory containing external ranking files: lme_TRAIN.json and nq_TRAIN.json",
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend passed to AutoModelForCausalLM.from_pretrained",
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

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resolved project_dir={PROJECT_DIR}")
    print(f"Resolved niah_dir={args.niah_dir}")
    print(f"Resolved output_dir={args.output_dir}")
    print(f"Resolved detection_results_dir={args.detection_results_dir}")
    print(f"Resolved external_rankings_dir={args.external_rankings_dir}")

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
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            attn_implementation=args.attn_implementation,
            device_map="auto",
        )
    except (ImportError, RuntimeError) as e:
        if args.attn_implementation == "flash_attention_2":
            print(
                "flash_attention_2 unavailable or incompatible in this environment; "
                "retrying with sdpa."
            )
            print(f"Original error: {e}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                device_map="auto",
            )
        else:
            raise
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    install_head_masking(model)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Model: {num_layers} layers x {num_heads} heads")

    # Load method rankings.
    print("\nLoading method rankings...")
    all_methods, transfer_rankings = load_method_rankings(args, num_layers, num_heads)

    if args.methods:
        all_methods = {k: v for k, v in all_methods.items() if k in args.methods}

    if not all_methods:
        print("ERROR: No methods available. Run detection first.")
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

        transfer_path = os.path.join(args.output_dir, "cross_task_transfer_matrix.json")
        with open(transfer_path, "w", encoding="utf-8") as f:
            json.dump(transfer_matrix, f, indent=2)
        print(f"Saved transfer matrix: {transfer_path}")

        specificity = compute_specificity_metrics(transfer_matrix, args.transfer_summary_k)
        specificity_path = os.path.join(args.output_dir, "cross_task_specificity_metrics.json")
        with open(specificity_path, "w", encoding="utf-8") as f:
            json.dump(specificity, f, indent=2)
        print(f"Saved specificity metrics: {specificity_path}")

        similarity_path = os.path.join(args.output_dir, "cross_task_head_similarity_topk.json")
        save_head_similarity(transfer_rankings, args.export_top_k, similarity_path)
        print(f"Saved head similarity matrices: {similarity_path}")


if __name__ == "__main__":
    main()
