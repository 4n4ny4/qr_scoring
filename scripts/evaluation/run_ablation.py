"""
Comparison ablation for SEC NIAH tasks using only Llama-3.1-8B-derived head rankings.

Methods supported:
- QRScore-SEC (combined SEC detection)
- QRScore-8B-LME-TRAIN (external 8B ranking from Llama-3.1-8B-Instruct/lme_TRAIN.json)
- QRScore-8B-NQ-TRAIN (external 8B ranking from Llama-3.1-8B-Instruct/nq_TRAIN.json)
- Transfer-<task> (optional cross-task transfer from per-task SEC rankings)
- Random-seed{42,123,456} (optional)
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import transformers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qrretriever.custom_modeling_llama import LlamaForCausalLM

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 30
DEFAULT_KNOCKOUT_SIZES = [0, 8, 16, 32, 48, 64, 96, 128]
DEFAULT_EXPORT_TOP_K = [8, 16, 32, 48, 64, 96, 128]


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
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Read the following document and answer the question.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely with just the answer value."
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_answer(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                    max_context_tokens=8192):
    tokenizer.truncation_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_context_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    start = inputs["input_ids"].shape[1]
    new_tokens = out[0][start:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_short_answer(text)


def load_ranked_heads_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    heads = []
    for row in data:
        head_str = row[0] if isinstance(row, list) else row.get("head")
        layer, head = map(int, head_str.split("-"))
        heads.append((layer, head))
    return heads


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
    results_dir = os.path.join(PROJECT_DIR, "results", "detection")
    methods = {}

    # 1) SEC combined ranking
    sec_path = os.path.join(results_dir, "long_context_combined_heads.json")
    if os.path.exists(sec_path):
        methods["QRScore-SEC"] = load_ranked_heads_json(sec_path)
        print(f"  QRScore-SEC: loaded from {sec_path} ({len(methods['QRScore-SEC'])} heads)")
    else:
        print("  QRScore-SEC: not found (run detection first)")

    # 2) External 8B LME/NQ train rankings
    lme_train_path = os.path.join(PROJECT_DIR, "Llama-3.1-8B-Instruct", "lme_TRAIN.json")
    nq_train_path = os.path.join(PROJECT_DIR, "Llama-3.1-8B-Instruct", "nq_TRAIN.json")

    if os.path.exists(lme_train_path):
        lme_heads = load_ranked_heads_json(lme_train_path)
        methods["QRScore-8B-LME-TRAIN"] = ensure_full_ranking(
            lme_heads, num_layers, num_heads_per_layer, seed=42
        )
        print(f"  QRScore-8B-LME-TRAIN: loaded from {lme_train_path}")

    if os.path.exists(nq_train_path):
        nq_heads = load_ranked_heads_json(nq_train_path)
        methods["QRScore-8B-NQ-TRAIN"] = ensure_full_ranking(
            nq_heads, num_layers, num_heads_per_layer, seed=43
        )
        print(f"  QRScore-8B-NQ-TRAIN: loaded from {nq_train_path}")

    # Export deterministic top-K slices for external 8B rankings.
    external_export_dir = os.path.join(results_dir, "topk", "8b_external")
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
                     knockout_sizes, max_context_tokens=8192):
    results = {}

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

        for inst in test_instances:
            prompt = build_prompt(inst["context"], inst["question"])
            pred = generate_answer(model, tokenizer, prompt,
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
                "correct": int(match),
            })

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
                            knockout_sizes, max_context_tokens):
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
    parser = argparse.ArgumentParser(description="8B-only comparison ablation across detection methods")
    parser.add_argument("--niah_dir", default=os.path.join(PROJECT_DIR, "data", "niah_input"))
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_DIR, "results", "comparison_ablation"))
    parser.add_argument("--model_name", default=MODEL_NAME)
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resolved project_dir={PROJECT_DIR}")
    print(f"Resolved niah_dir={args.niah_dir}")
    print(f"Resolved output_dir={args.output_dir}")

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
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.config.pad_token_id = model.config.eos_token_id
    model._num_logits_to_keep = 1
    model.eval()

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

    # Run pooled sweeps.
    all_results = {}
    for method_name, head_ranking in all_methods.items():
        print(f"\n{'=' * 60}")
        print(f"Method: {method_name}")
        print(f"{'=' * 60}")

        method_results = run_single_sweep(
            model,
            tokenizer,
            test_instances,
            head_ranking,
            args.knockout_sizes,
            max_context_tokens=args.max_context_tokens,
        )
        all_results[method_name] = method_results

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
                    "details": {
                        str(k): method_results[k]["details"]
                        for k in args.knockout_sizes
                    },
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
