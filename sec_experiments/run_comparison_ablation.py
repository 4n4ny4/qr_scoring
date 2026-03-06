"""
Comparison ablation: run head-knockout sweeps across multiple detection methods.

For each method (QRScore-SEC, QRScore-Paper-LME, QRScore-Paper-NQ, RETHEAD, Random):
  For each knockout level K:
    - Mask the top-K heads during generation
    - Feed NIAH context + question to LM
    - Generate answer, compare to needle_value
    - Record accuracy

The method that identifies heads whose removal causes the steepest accuracy drop
is the most effective at finding true retrieval heads.

Usage:
  python sec_experiments/run_comparison_ablation.py \
    --niah_dir data/niah_input \
    --output_dir results/comparison_ablation \
    --max_instances_per_task 20
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
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qrretriever.custom_modeling_llama import LlamaForCausalLM

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 30
DEFAULT_KNOCKOUT_SIZES = [0, 8, 16, 32, 48, 64, 96, 128]


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
    # Keep question (at end of prompt); truncate from start if over limit
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


# --- Head ranking loaders ---

def load_heads_from_detection_json(path):
    """Load ranked heads from detection output JSON. Returns list of (layer, head)."""
    with open(path) as f:
        data = json.load(f)
    heads = []
    for head_str, score in data:
        layer, head = map(int, head_str.split("-"))
        heads.append((layer, head))
    return heads


def load_heads_from_yaml(path):
    """Load heads from paper's YAML config. Returns list of (layer, head)."""
    with open(path) as f:
        config = yaml.safe_load(f)
    heads = []
    for h in config["attn_head_set"].split(","):
        layer, head = map(int, h.strip().split("-"))
        heads.append((layer, head))
    return heads


def generate_random_heads(num_layers, num_heads_per_layer, total_heads, seed=42):
    """Generate a random head ranking."""
    rng = random.Random(seed)
    all_heads = [(l, h) for l in range(num_layers)
                 for h in range(num_heads_per_layer)]
    rng.shuffle(all_heads)
    return all_heads[:total_heads]


def load_method_rankings(args, num_layers, num_heads_per_layer):
    """Load all method rankings. Returns dict: method_name -> list of (layer, head)."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs_dir = os.path.join(project_dir, "src", "qrretriever", "configs")
    results_dir = os.path.join(project_dir, "results", "detection")

    methods = {}

    # 1. QRScore-SEC (our long-context detection)
    qrscore_sec_paths = [
        os.path.join(results_dir, "long_context_combined_heads.json"),
        os.path.join(results_dir, "niah_combined_heads.json"),
    ]
    for p in qrscore_sec_paths:
        if os.path.exists(p):
            methods["QRScore-SEC"] = load_heads_from_detection_json(p)
            print(f"  QRScore-SEC: loaded from {p} ({len(methods['QRScore-SEC'])} heads)")
            break
    if "QRScore-SEC" not in methods:
        print("  QRScore-SEC: not found (run detection first)")

    # 2. QRScore-Paper-LME
    lme_path = os.path.join(configs_dir, "Llama-3.1-8B-Instruct_qr_head_LME.yaml")
    if os.path.exists(lme_path):
        paper_lme = load_heads_from_yaml(lme_path)
        all_heads = [(l, h) for l in range(num_layers)
                     for h in range(num_heads_per_layer)]
        remaining = [x for x in all_heads if x not in set(paper_lme)]
        random.Random(42).shuffle(remaining)
        methods["QRScore-Paper-LME"] = paper_lme + remaining
        print(f"  QRScore-Paper-LME: {len(paper_lme)} paper heads + {len(remaining)} remaining")

    # 3. QRScore-Paper-NQ
    nq_path = os.path.join(configs_dir, "Llama-3.1-8B-Instruct_qr_head_NQ.yaml")
    if os.path.exists(nq_path):
        paper_nq = load_heads_from_yaml(nq_path)
        all_heads = [(l, h) for l in range(num_layers)
                     for h in range(num_heads_per_layer)]
        remaining = [x for x in all_heads if x not in set(paper_nq)]
        random.Random(43).shuffle(remaining)
        methods["QRScore-Paper-NQ"] = paper_nq + remaining
        print(f"  QRScore-Paper-NQ: {len(paper_nq)} paper heads + {len(remaining)} remaining")

    # 4. RETHEAD
    rethead_path = os.path.join(results_dir, "rethead_combined_heads.json")
    if os.path.exists(rethead_path):
        methods["RETHEAD"] = load_heads_from_detection_json(rethead_path)
        print(f"  RETHEAD: loaded from {rethead_path} ({len(methods['RETHEAD'])} heads)")
    else:
        print("  RETHEAD: not found (run detect_retrieval_heads.py first)")

    # 5. Random heads (average over 3 seeds)
    for seed in [42, 123, 456]:
        name = f"Random-seed{seed}"
        methods[name] = generate_random_heads(
            num_layers, num_heads_per_layer,
            total_heads=max(DEFAULT_KNOCKOUT_SIZES) + 50,
            seed=seed,
        )
        print(f"  {name}: {len(methods[name])} heads")

    return methods


def run_single_sweep(model, tokenizer, test_instances, head_ranking,
                     knockout_sizes, method_name, max_context_tokens=8192):
    """Run ablation sweep for one method. Returns dict: K -> accuracy info."""
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
                t: {"accuracy": d["correct"]/d["total"] if d["total"] else 0,
                    "correct": d["correct"], "total": d["total"]}
                for t, d in per_task.items()
            },
            "details": details,
        }

    model.set_head_mask(None)
    return results


def main():
    parser = argparse.ArgumentParser(description="Comparison ablation across detection methods")
    parser.add_argument("--niah_dir", default="data/niah_input")
    parser.add_argument("--output_dir", default="results/comparison_ablation")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--knockout_sizes", nargs="+", type=int,
                        default=DEFAULT_KNOCKOUT_SIZES)
    parser.add_argument("--max_instances_per_task", type=int, default=None)
    parser.add_argument("--max_context_tokens", type=int, default=8192,
                        help="Max prompt tokens; full NIAH ~6500. Lower to 4096 if OOM.")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Specific methods to run (default: all available)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test instances
    test_instances = []
    for task in args.tasks:
        test_path = os.path.join(args.niah_dir, f"{task}_test.json")
        if not os.path.exists(test_path):
            print(f"WARNING: {test_path} not found, skipping", file=sys.stderr)
            continue
        with open(test_path) as f:
            data = json.load(f)
        if args.max_instances_per_task and len(data) > args.max_instances_per_task:
            data = data[:args.max_instances_per_task]
        for inst in data:
            inst["task"] = task
        test_instances.extend(data)

    print(f"Loaded {len(test_instances)} test instances across {len(args.tasks)} tasks")

    # Load model
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

    # Load all method rankings
    print("\nLoading method rankings...")
    all_methods = load_method_rankings(args, num_layers, num_heads)

    if args.methods:
        all_methods = {k: v for k, v in all_methods.items() if k in args.methods}

    if not all_methods:
        print("ERROR: No methods available. Run detection first.")
        sys.exit(1)

    print(f"\nMethods to evaluate: {list(all_methods.keys())}")

    # Run sweeps
    all_results = {}
    for method_name, head_ranking in all_methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        method_results = run_single_sweep(
            model, tokenizer, test_instances, head_ranking,
            args.knockout_sizes, method_name,
            max_context_tokens=args.max_context_tokens,
        )
        all_results[method_name] = method_results

        method_path = os.path.join(args.output_dir, f"{method_name.replace(' ', '_')}_results.json")
        with open(method_path, "w") as f:
            json.dump({
                "method": method_name,
                "knockout_sizes": args.knockout_sizes,
                "accuracy_curve": {str(K): method_results[K]["accuracy"]
                                   for K in args.knockout_sizes},
                "per_task_curves": {},
                "details": {str(K): method_results[K]["details"]
                            for K in args.knockout_sizes},
            }, f, indent=2)

    # Build combined summary
    summary = {
        "model": args.model_name,
        "num_instances": len(test_instances),
        "knockout_sizes": args.knockout_sizes,
        "methods": {},
    }

    for method_name, method_results in all_results.items():
        curve = {str(K): method_results[K]["accuracy"] for K in args.knockout_sizes}
        per_task_curves = {}
        for task in args.tasks:
            task_curve = {}
            for K in args.knockout_sizes:
                if task in method_results[K]["per_task"]:
                    task_curve[str(K)] = method_results[K]["per_task"][task]["accuracy"]
            if task_curve:
                per_task_curves[task] = task_curve

        baseline_acc = method_results[0]["accuracy"] if 0 in method_results else None
        k16_acc = method_results[16]["accuracy"] if 16 in method_results else None
        drop_at_16 = (baseline_acc - k16_acc) if baseline_acc is not None and k16_acc is not None else None

        summary["methods"][method_name] = {
            "accuracy_curve": curve,
            "per_task_curves": per_task_curves,
            "baseline_accuracy": baseline_acc,
            "accuracy_at_k16": k16_acc,
            "drop_at_k16": drop_at_16,
        }

    # For random methods, compute average
    random_methods = [m for m in summary["methods"] if m.startswith("Random-")]
    if random_methods:
        avg_curve = {}
        for k_str in [str(K) for K in args.knockout_sizes]:
            vals = [summary["methods"][m]["accuracy_curve"].get(k_str, 0) for m in random_methods]
            avg_curve[k_str] = float(np.mean(vals))
        summary["methods"]["Random-avg"] = {
            "accuracy_curve": avg_curve,
            "baseline_accuracy": avg_curve.get("0"),
            "accuracy_at_k16": avg_curve.get("16"),
            "drop_at_k16": (avg_curve.get("0", 0) - avg_curve.get("16", 0))
                           if "0" in avg_curve and "16" in avg_curve else None,
            "note": f"Average over {len(random_methods)} random seeds",
        }

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25s} {'Baseline':>10s} {'K=16':>10s} {'Drop':>10s}")
    print(f"{'='*80}")
    for method_name, info in summary["methods"].items():
        baseline = f"{info['baseline_accuracy']:.4f}" if info.get('baseline_accuracy') is not None else "N/A"
        k16 = f"{info['accuracy_at_k16']:.4f}" if info.get('accuracy_at_k16') is not None else "N/A"
        drop = f"{info['drop_at_k16']:.4f}" if info.get('drop_at_k16') is not None else "N/A"
        print(f"{method_name:<25s} {baseline:>10s} {k16:>10s} {drop:>10s}")

    print(f"\nFull accuracy curves:")
    for method_name, info in summary["methods"].items():
        curve_str = " | ".join(
            f"K={K}:{info['accuracy_curve'].get(str(K), 0):.3f}"
            for K in args.knockout_sizes
        )
        print(f"  {method_name}: {curve_str}")


if __name__ == "__main__":
    main()
