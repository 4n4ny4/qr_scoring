"""
Answer-accuracy knockout ablation.

Load test JSONs (with answer), full-head and knockout retrieval results per task.
For each instance: take top-1 doc per condition, prompt LM for one-word answer,
compare to gold. Compute answer accuracy per condition, paired bootstrap (full vs knockout),
save summary JSON and optional bar chart.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# Optional: transformers for generation
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CONDITIONS = ["full_head", "knockout_top16"]
MODEL_NAME_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.0
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


def normalize_answer(s):
    """Lowercase, strip, collapse whitespace, remove commas from numbers."""
    if s is None or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)  # "2,356" -> "2356"
    s = re.sub(r"[,.;:!?\"']+$", "", s)   # trailing punctuation
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
    """Return doc_id with highest score. scores_dict: doc_id -> score."""
    if not scores_dict:
        return None
    return max(scores_dict.items(), key=lambda x: x[1])[0]


def get_paragraph_text(paragraphs, doc_id):
    """Return paragraph_text for paragraph with idx == doc_id."""
    for p in paragraphs:
        if str(p.get("idx")) == str(doc_id):
            return p.get("paragraph_text", "")
    return ""


def build_prompt(doc_text, question):
    return f"Context:\n{doc_text}\n\nQuestion: {question}\n\nAnswer concisely:"


def load_model_and_tokenizer(model_name_or_path, device=None):
    if not HAS_TORCH:
        raise RuntimeError("transformers and torch required for generation. pip install torch transformers")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model.device.type != device and device == "cpu":
        model = model.to(device)
    return model, tokenizer


def generate_answer(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Generate and return short answer from model output."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    start = inputs["input_ids"].shape[1]
    new_tokens = out[0][start:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_short_answer(text)


def run_answer_ablation(
    data_dir,
    ablation_dir,
    tasks,
    model_name_or_path=MODEL_NAME_DEFAULT,
    skip_generation=False,
    output_dir=None,
):
    if output_dir is None:
        output_dir = ablation_dir

    # Load test data and retrieval results per task
    task_data = {}
    for task in tasks:
        test_path = os.path.join(data_dir, f"{task}_test.json")
        full_path = os.path.join(ablation_dir, f"{task}_full_head.json")
        knockout_path = os.path.join(ablation_dir, f"{task}_knockout_top16.json")

        if not os.path.exists(test_path):
            print(f"WARNING: {test_path} not found, skipping {task}", file=sys.stderr)
            continue
        if not os.path.exists(full_path):
            print(f"WARNING: {full_path} not found, skipping {task}", file=sys.stderr)
            continue
        if not os.path.exists(knockout_path):
            print(f"WARNING: {knockout_path} not found, skipping {task}", file=sys.stderr)
            continue

        with open(test_path) as f:
            test_instances = json.load(f)
        with open(full_path) as f:
            full_scores = json.load(f)
        with open(knockout_path) as f:
            knockout_scores = json.load(f)

        task_data[task] = {
            "instances": test_instances,
            "full_scores": full_scores,
            "knockout_scores": knockout_scores,
        }

    if not task_data:
        raise SystemExit("No task data loaded. Ensure test JSONs and full_head + knockout_top16 results exist.")

    # Collect (instance, doc_full, doc_knockout, gold_answer) for shared instance set
    rows = []
    for task, data in task_data.items():
        instances = data["instances"]
        full_scores = data["full_scores"]
        knockout_scores = data["knockout_scores"]
        for inst in instances:
            idx = str(inst["idx"])
            if idx not in full_scores or idx not in knockout_scores:
                continue
            gold = normalize_answer(inst.get("answer", ""))
            if not gold:
                continue
            top1_full = get_top1_doc(full_scores[idx])
            top1_kill = get_top1_doc(knockout_scores[idx])
            if top1_full is None or top1_kill is None:
                continue
            text_full = get_paragraph_text(inst["paragraphs"], top1_full)
            text_kill = get_paragraph_text(inst["paragraphs"], top1_kill)
            rows.append({
                "task": task,
                "idx": idx,
                "question": inst["question"],
                "gold": gold,
                "doc_text_full": text_full,
                "doc_text_knockout": text_kill,
            })
            assert "correct_full" not in rows[-1]
            assert "correct_knockout" not in rows[-1]

    n_total = len(rows)
    print(f"Total instances with both conditions: {n_total}")

    if skip_generation:
        # Dummy correctness for eval-only run (e.g. no GPU)
        for r in rows:
            r["correct_full"] = 0
            r["correct_knockout"] = 0
        print("Skipping generation (--skip_generation); writing placeholder results.")
    else:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path)
        for i, r in enumerate(rows):
            prompt_full = build_prompt(r["doc_text_full"], r["question"])
            prompt_kill = build_prompt(r["doc_text_knockout"], r["question"])
            pred_full = generate_answer(model, tokenizer, prompt_full)
            pred_kill = generate_answer(model, tokenizer, prompt_kill)
            r["pred_full"] = pred_full
            r["pred_knockout"] = pred_kill
            r["correct_full"] = 1 if answers_match(pred_full, r["gold"]) else 0
            r["correct_knockout"] = 1 if answers_match(pred_kill, r["gold"]) else 0
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{n_total}")

    # Per-task and pooled accuracy
    acc_full_per_task = {}
    acc_kill_per_task = {}
    for task in task_data:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        n = len(task_rows)
        acc_full_per_task[task] = sum(r["correct_full"] for r in task_rows) / n
        acc_kill_per_task[task] = sum(r["correct_knockout"] for r in task_rows) / n

    acc_full_pooled = sum(r["correct_full"] for r in rows) / n_total if n_total else 0.0
    acc_kill_pooled = sum(r["correct_knockout"] for r in rows) / n_total if n_total else 0.0

    # Paired bootstrap: difference = full - knockout (positive means full better)
    diff_per_instance = np.array([r["correct_full"] - r["correct_knockout"] for r in rows])
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    boot_diffs = []
    for _ in range(BOOTSTRAP_N):
        inds = rng.randint(0, len(diff_per_instance), size=len(diff_per_instance))
        boot_diffs.append(diff_per_instance[inds].mean())
    boot_diffs = np.array(boot_diffs)
    ci_lo = np.percentile(boot_diffs, 100 * (1 - CI_LEVEL) / 2)
    ci_hi = np.percentile(boot_diffs, 100 * (1 + CI_LEVEL) / 2)
    # p-value: P(diff <= 0) under bootstrap (one-sided: knockout worse)
    p_value = (boot_diffs <= 0).mean()

    summary = {
        "accuracy_full_head": acc_full_pooled,
        "accuracy_knockout_top16": acc_kill_pooled,
        "accuracy_diff_full_minus_knockout": acc_full_pooled - acc_kill_pooled,
        "paired_bootstrap_ci_lower": float(ci_lo),
        "paired_bootstrap_ci_upper": float(ci_hi),
        "p_value_full_better_than_knockout": float(p_value),
        "n_instances": n_total,
        "per_task": {
            task: {
                "accuracy_full_head": acc_full_per_task[task],
                "accuracy_knockout_top16": acc_kill_per_task[task],
                "n": len([r for r in rows if r["task"] == task]),
            }
            for task in sorted(acc_full_per_task.keys())
        },
    }

    out_path = os.path.join(output_dir, "answer_accuracy_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_path}")
    print(f"  Pooled: full={acc_full_pooled:.4f}, knockout={acc_kill_pooled:.4f}, diff={acc_full_pooled - acc_kill_pooled:.4f}, p={p_value:.4f}")

    # Optional bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None:
        fig, ax = plt.subplots()
        tasks_sorted = sorted(acc_full_per_task.keys())
        x = np.arange(len(tasks_sorted))
        w = 0.35
        ax.bar(x - w / 2, [acc_full_per_task[t] for t in tasks_sorted], width=w, label="Full heads")
        ax.bar(x + w / 2, [acc_kill_per_task[t] for t in tasks_sorted], width=w, label="Knockout top-16")
        ax.axhline(acc_full_pooled, color="C0", linestyle="--", alpha=0.7)
        ax.axhline(acc_kill_pooled, color="C1", linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_sorted, rotation=45, ha="right")
        ax.set_ylabel("Answer accuracy")
        ax.legend()
        ax.set_title("Answer accuracy: full heads vs knockout top-16")
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "answer_accuracy_knockout.png")
        fig.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Answer-accuracy knockout ablation")
    parser.add_argument("--data_dir", default="data/detection_input", help="Directory with {task}_test.json")
    parser.add_argument("--ablation_dir", default="results/ablation", help="Directory with full_head and knockout_top16 JSONs")
    parser.add_argument("--tasks", nargs="+", default=[
        "registrant_name", "headquarters_city", "headquarters_state", "incorporation_state",
        "incorporation_year", "employees_count_total", "ceo_lastname", "holder_record_amount",
    ])
    parser.add_argument("--model_name_or_path", default=MODEL_NAME_DEFAULT, help="LM for one-word generation")
    parser.add_argument("--skip_generation", action="store_true", help="Skip LM calls; write placeholder (for eval-only)")
    parser.add_argument("--output_dir", default=None, help="Write summary/plot here (default: ablation_dir)")
    args = parser.parse_args()

    run_answer_ablation(
        data_dir=args.data_dir,
        ablation_dir=args.ablation_dir,
        tasks=args.tasks,
        model_name_or_path=args.model_name_or_path,
        skip_generation=args.skip_generation,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
