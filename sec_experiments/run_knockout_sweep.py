"""
Knockout sweep: progressively remove top-K QR heads (K=100,200,...,900)
and measure pooled one-word answer accuracy vs the full-head baseline.

Loads the retrieval model once, sweeps knockout sizes by changing the
attn_head_set attribute, runs retrieval + LM generation, computes paired
bootstrap significance at each step, and stops when p < 0.05.

Output: results/ablation/knockout_sweep_summary.json
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]
NUM_LAYERS = 32
NUM_HEADS = 32
ALL_HEADS = [f"{l}-{h}" for l in range(NUM_LAYERS) for h in range(NUM_HEADS)]
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

def normalize_answer(s):
    """Lowercase, strip, collapse whitespace, remove commas from numbers."""
    if s is None or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"[,.;:!?\"']+$", "", s)
    return " ".join(s.split())

def extract_short_answer(text):
    """Extract a short answer from model output (up to first newline/sentence break)."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.split("\n")[0].strip()
    text = re.split(r"[.!?]\s", text)[0].strip()
    text = re.sub(r"[.!?]+$", "", text)
    return text

def answers_match(pred, gold):
    """Flexible matching: exact, containment, or normalized containment."""
    if not pred or not gold:
        return False
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    if g in p or p in g:
        return True
    return False

def get_top1_doc(scores_dict):
    if not scores_dict:
        return None
    return max(scores_dict.items(), key=lambda x: x[1])[0]

def get_paragraph_text(paragraphs, doc_id):
    for p in paragraphs:
        if str(p.get("idx")) == str(doc_id):
            return p.get("paragraph_text", "")
    return ""

def build_prompt(doc_text, question):
    return f"Context:\n{doc_text}\n\nQuestion: {question}\n\nAnswer concisely:"

def paired_bootstrap(correct_a, correct_b, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Return (diff, ci_lo, ci_hi, p_value). diff = mean(a) - mean(b)."""
    a = np.array(correct_a, dtype=float)
    b = np.array(correct_b, dtype=float)
    diff_per = a - b
    observed = diff_per.mean()
    rng = np.random.RandomState(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(diff_per), size=len(diff_per))
        boot_diffs.append(diff_per[idx].mean())
    boot_diffs = np.array(boot_diffs)
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    p_val = float((boot_diffs <= 0).mean())
    return float(observed), ci_lo, ci_hi, p_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/detection_input")
    parser.add_argument("--detection_dir", default="results/detection")
    parser.add_argument("--ablation_dir", default="results/ablation")
    parser.add_argument("--knockout_sizes", type=int, nargs="+",
                        default=list(range(0, 101, 10)))  # 0, 10, 20, ..., 100
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--stop_at_significance", type=float, default=0.05,
                        help="Stop when p < this threshold")
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from qrretriever.attn_retriever import QRRetriever

    # ---- Load test data and head rankings per task ----
    task_data = {}
    task_head_rankings = {}
    for task in TASKS:
        test_path = os.path.join(args.data_dir, f"{task}_test.json")
        heads_path = os.path.join(args.detection_dir, f"{task}_heads.json")
        if not os.path.exists(test_path) or not os.path.exists(heads_path):
            print(f"Skipping {task}: missing files")
            continue
        with open(test_path) as f:
            task_data[task] = json.load(f)
        with open(heads_path) as f:
            task_head_rankings[task] = [h[0] for h in json.load(f)]

    active_tasks = sorted(task_data.keys())
    print(f"Active tasks: {active_tasks}")

    # ---- Load existing full-head retrieval results ----
    full_scores_per_task = {}
    for task in active_tasks:
        full_path = os.path.join(args.ablation_dir, f"{task}_full_head.json")
        with open(full_path) as f:
            full_scores_per_task[task] = json.load(f)

    # ---- Load retrieval model once ----
    print("Loading retrieval model...")
    base_config_path = os.path.join(args.ablation_dir, "configs",
                                     f"{active_tasks[0]}_knockout_top16.yaml")
    retriever = QRRetriever(config_or_config_path=base_config_path)
    print("Retrieval model loaded.")

    # ---- Run retrieval for each knockout size ----
    knockout_retrieval = {}  # {K: {task: {qid: {doc_id: score}}}}

    for K in sorted(args.knockout_sizes):
        print(f"\n{'='*60}")
        print(f"KNOCKOUT SIZE: {K}  (removing top-{K} QR heads, using {1024-K})")
        print(f"{'='*60}")
        knockout_retrieval[K] = {}

        for task in active_tasks:
            if K == 0:
                # K=0 means no knockout: use full-head results
                knockout_retrieval[K][task] = dict(full_scores_per_task[task])
                print(f"  {task}: using full-head results (K=0)")
                continue

            out_path = os.path.join(args.ablation_dir, f"{task}_knockout_top{K}.json")
            if os.path.exists(out_path):
                print(f"  {task}: loading cached results")
                with open(out_path) as f:
                    knockout_retrieval[K][task] = json.load(f)
                continue

            top_k_heads = set(task_head_rankings[task][:K])
            remaining = [h for h in ALL_HEADS if h not in top_k_heads]
            retriever.attn_head_set = ",".join(remaining)

            results = {}
            instances = task_data[task]
            t0 = time.time()
            for i, inst in enumerate(instances):
                qid = str(inst["idx"])
                docs = inst["paragraphs"]
                scores = retriever.score_docs(inst["question"], docs)
                results[qid] = {str(k): v for k, v in scores.items()}
            elapsed = time.time() - t0
            print(f"  {task}: {len(instances)} instances in {elapsed:.1f}s")

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            knockout_retrieval[K][task] = results

    # ---- Build instance rows for generation ----
    # For full-head, we use existing retrieval. For each K, use knockout retrieval.
    all_rows = []
    for task in active_tasks:
        for inst in task_data[task]:
            qid = str(inst["idx"])
            gold = normalize_answer(inst.get("answer", ""))
            if not gold or qid not in full_scores_per_task[task]:
                continue
            top1_full_doc = get_top1_doc(full_scores_per_task[task].get(qid, {}))
            if top1_full_doc is None:
                continue
            text_full = get_paragraph_text(inst["paragraphs"], top1_full_doc)

            ko_docs = {}
            for K in sorted(args.knockout_sizes):
                ko_scores = knockout_retrieval[K].get(task, {}).get(qid, {})
                top1_ko = get_top1_doc(ko_scores)
                if top1_ko is not None:
                    ko_docs[K] = get_paragraph_text(inst["paragraphs"], top1_ko)
                else:
                    ko_docs[K] = ""

            all_rows.append({
                "task": task, "idx": qid, "question": inst["question"],
                "gold": gold, "text_full": text_full, "ko_docs": ko_docs,
            })

    n_total = len(all_rows)
    print(f"\nTotal instances for generation: {n_total}")

    # ---- Generate answers ----
    if args.skip_generation:
        print("--skip_generation: using retrieval-match heuristic (gold doc in top-1)")
        for row in all_rows:
            row["correct_full"] = 0
            for K in args.knockout_sizes:
                row[f"correct_ko{K}"] = 0
    else:
        print("Loading generation model...")
        gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        gen_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("Generation model loaded.")

        # Free retrieval model to save VRAM
        del retriever
        torch.cuda.empty_cache()
        import gc; gc.collect()

        def generate_answer(prompt):
            inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = gen_model.generate(
                    **inputs, max_new_tokens=20, do_sample=False,
                    pad_token_id=gen_tokenizer.eos_token_id,
                )
            start = inputs["input_ids"].shape[1]
            text = gen_tokenizer.decode(out[0][start:], skip_special_tokens=True)
            return extract_short_answer(text)

        # Generate for full-head condition
        print("Generating full-head answers...")
        for i, row in enumerate(all_rows):
            pred = generate_answer(build_prompt(row["text_full"], row["question"]))
            row["correct_full"] = 1 if answers_match(pred, row["gold"]) else 0
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_total}")

        # Generate for each knockout size
        for K in sorted(args.knockout_sizes):
            print(f"Generating knockout-{K} answers...")
            for i, row in enumerate(all_rows):
                ko_text = row["ko_docs"].get(K, "")
                if ko_text == row["text_full"]:
                    row[f"correct_ko{K}"] = row["correct_full"]
                else:
                    pred = generate_answer(build_prompt(ko_text, row["question"]))
                    row[f"correct_ko{K}"] = 1 if answers_match(pred, row["gold"]) else 0
            acc = sum(r[f"correct_ko{K}"] for r in all_rows) / n_total
            print(f"  Knockout-{K} accuracy: {acc:.4f}")

    # ---- Compute results ----
    acc_full = sum(r["correct_full"] for r in all_rows) / n_total
    print(f"\nFull-head accuracy: {acc_full:.4f}")

    sweep_results = []
    found_significant = False

    for K in sorted(args.knockout_sizes):
        correct_full = [r["correct_full"] for r in all_rows]
        correct_ko = [r[f"correct_ko{K}"] for r in all_rows]
        acc_ko = sum(correct_ko) / n_total

        diff, ci_lo, ci_hi, p_val = paired_bootstrap(correct_full, correct_ko)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        entry = {
            "knockout_size": K,
            "heads_remaining": 1024 - K,
            "accuracy_full": round(acc_full, 5),
            "accuracy_knockout": round(acc_ko, 5),
            "diff": round(diff, 5),
            "ci_lo": round(ci_lo, 5),
            "ci_hi": round(ci_hi, 5),
            "p_value": round(p_val, 5),
            "significant": sig,
        }
        sweep_results.append(entry)
        print(f"  K={K:4d}  remaining={1024-K:4d}  "
              f"acc_full={acc_full:.4f}  acc_ko={acc_ko:.4f}  "
              f"diff={diff:+.4f}  p={p_val:.4f} {sig}")

        if p_val < args.stop_at_significance and not found_significant:
            found_significant = True
            print(f"\n  *** SIGNIFICANT at K={K} (p={p_val:.4f}) ***")

    summary = {
        "accuracy_full_head": round(acc_full, 5),
        "n_instances": n_total,
        "sweep": sweep_results,
        "first_significant_K": next(
            (r["knockout_size"] for r in sweep_results if r["p_value"] < args.stop_at_significance),
            None
        ),
    }

    out_path = os.path.join(args.ablation_dir, "knockout_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
