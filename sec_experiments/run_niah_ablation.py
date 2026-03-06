"""
NIAH-style ablation: mask QRHeads during generation, sweep knockout sizes.

For each knockout level K (0, 10, 20, ... 100):
  - Mask the top-K heads (by QRScore rank) during generation
  - Feed the full NIAH context + question to the LM
  - Generate an answer and compare to needle_value
  - Record accuracy

Uses the custom LlamaForCausalLM with head masking support.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qrretriever.custom_modeling_llama import LlamaForCausalLM


TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 30
TEMPERATURE = 0.0


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


def load_head_ranking(head_file):
    """Load head ranking from detection output.

    Returns list of (layer, head) tuples sorted from most important to least.
    """
    with open(head_file) as f:
        data = json.load(f)
    heads = []
    for head_str, score in data:
        layer, head = map(int, head_str.split("-"))
        heads.append((layer, head))
    return heads


def load_paper_heads(config_path):
    """Load paper's pre-computed head set from YAML config."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    head_str = config["attn_head_set"]
    heads = []
    for h in head_str.split(","):
        layer, head = map(int, h.strip().split("-"))
        heads.append((layer, head))
    return heads


def build_prompt(context, question):
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Read the following document and answer the question.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely with just the answer value."
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_answer(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
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


def run_sweep(
    niah_dir,
    head_ranking,
    knockout_sizes,
    model_name=MODEL_NAME,
    output_dir="results/niah_ablation",
    max_instances_per_task=None,
):
    os.makedirs(output_dir, exist_ok=True)

    test_instances = []
    for task in TASKS:
        test_path = os.path.join(niah_dir, f"{task}_test.json")
        if not os.path.exists(test_path):
            print(f"WARNING: {test_path} not found, skipping", file=sys.stderr)
            continue
        with open(test_path) as f:
            data = json.load(f)
        if max_instances_per_task and len(data) > max_instances_per_task:
            data = data[:max_instances_per_task]
        for inst in data:
            inst["task"] = task
        test_instances.extend(data)

    print(f"Loaded {len(test_instances)} test instances across {len(TASKS)} tasks")

    print(f"Loading model: {model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.config.pad_token_id = model.config.eos_token_id
    model._num_logits_to_keep = 1
    model.eval()

    all_results = {}

    for K in knockout_sizes:
        print(f"\n{'='*60}")
        print(f"Knockout K={K} heads")
        print(f"{'='*60}")

        if K == 0:
            model.set_head_mask(None)
        else:
            masked = head_ranking[:K]
            model.set_head_mask(masked)
            layers_used = sorted(set(l for l, h in masked))
            print(f"  Masking heads in layers: {layers_used}")

        results = []
        correct = 0
        total = 0

        for i, inst in enumerate(test_instances):
            prompt = build_prompt(inst["context"], inst["question"])
            pred = generate_answer(model, tokenizer, prompt)
            gold = inst["needle_value"]
            match = answers_match(pred, gold)
            correct += int(match)
            total += 1

            results.append({
                "idx": inst["idx"],
                "task": inst["task"],
                "gold": gold,
                "pred": pred,
                "correct": int(match),
            })

            if (i + 1) % 10 == 0 or i == 0:
                acc_so_far = correct / total
                print(f"  [{i+1}/{len(test_instances)}] acc={acc_so_far:.3f} "
                      f"(last: gold='{gold[:30]}' pred='{pred[:30]}' {'OK' if match else 'MISS'})",
                      flush=True)

        accuracy = correct / total if total else 0
        print(f"  K={K}: accuracy = {correct}/{total} = {accuracy:.4f}")

        per_task = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            per_task[r["task"]]["correct"] += r["correct"]
            per_task[r["task"]]["total"] += 1

        all_results[K] = {
            "knockout_size": K,
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
            "details": results,
        }

    summary = {
        "model": model_name,
        "num_instances": len(test_instances),
        "knockout_sizes": knockout_sizes,
        "accuracy_curve": {
            str(K): all_results[K]["accuracy"] for K in knockout_sizes
        },
        "per_task_curves": {},
        "masked_heads_per_level": {},
    }

    for task in TASKS:
        curve = {}
        for K in knockout_sizes:
            if task in all_results[K]["per_task"]:
                curve[str(K)] = all_results[K]["per_task"][task]["accuracy"]
        if curve:
            summary["per_task_curves"][task] = curve

    for K in knockout_sizes:
        if K > 0:
            summary["masked_heads_per_level"][str(K)] = [
                f"{l}-{h}" for l, h in head_ranking[:K]
            ]

    summary_path = os.path.join(output_dir, "niah_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    details_path = os.path.join(output_dir, "niah_ablation_details.json")
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Details saved to {details_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="NIAH-style head ablation sweep")
    parser.add_argument("--niah_dir", default="data/niah_input")
    parser.add_argument("--head_ranking_file", default=None,
                        help="Path to detection output JSON with ranked heads")
    parser.add_argument("--paper_heads_config", default=None,
                        help="Path to paper's QRHead YAML config (used as top-priority heads)")
    parser.add_argument("--output_dir", default="results/niah_ablation")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--knockout_sizes", nargs="+", type=int,
                        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    parser.add_argument("--max_instances_per_task", type=int, default=None)
    args = parser.parse_args()

    head_ranking = []

    if args.paper_heads_config:
        paper_heads = load_paper_heads(args.paper_heads_config)
        head_ranking.extend(paper_heads)
        print(f"Loaded {len(paper_heads)} paper QRHeads from {args.paper_heads_config}")

    if args.head_ranking_file:
        detected = load_head_ranking(args.head_ranking_file)
        paper_set = set(head_ranking)
        for h in detected:
            if h not in paper_set:
                head_ranking.append(h)
        print(f"Extended ranking to {len(head_ranking)} heads using {args.head_ranking_file}")
    elif not head_ranking:
        parser.error("Must provide --head_ranking_file and/or --paper_heads_config")

    if len(head_ranking) < max(args.knockout_sizes):
        print(f"WARNING: Only {len(head_ranking)} ranked heads available, "
              f"but max knockout is {max(args.knockout_sizes)}. "
              f"Will use all available heads for large K values.")

    summary = run_sweep(
        niah_dir=args.niah_dir,
        head_ranking=head_ranking,
        knockout_sizes=args.knockout_sizes,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_instances_per_task=args.max_instances_per_task,
    )

    print("\n=== Accuracy Curve ===")
    for K in args.knockout_sizes:
        acc = summary["accuracy_curve"][str(K)]
        bar = "#" * int(acc * 50)
        print(f"  K={K:3d}: {acc:.4f} {bar}")


if __name__ == "__main__":
    main()
