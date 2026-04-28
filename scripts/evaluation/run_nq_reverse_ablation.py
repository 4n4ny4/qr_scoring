"""
Reverse-transfer ablation on Natural Questions.

This script mirrors `run_ablation.py`, but the evaluation target is a single
NQ QA-style JSON file while the ablated head rankings come from:

- QRScore-8B-NQ-TRAIN (external NQ control)
- QRScore-SEC (combined SEC heads)
- Transfer-<SEC task> (category-specific SEC heads)
- Random-seed* controls

Example:

python scripts/evaluation/run_nq_reverse_ablation.py \\
  --model_name Qwen/Qwen2.5-7B-Instruct \\
  --nq_file data/nq_input/nq_test.json \\
  --max_instances 512 \\
  --knockout_sizes 0 8 16 32 48 64 96 128
"""

import argparse
import csv
import json
import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from qrretriever.model_runtime import (
    load_stock_causal_lm,
    load_tokenizer,
    resolve_detection_dir,
    resolve_model_spec,
)

from run_ablation import (
    DEFAULT_KNOCKOUT_SIZES,
    TASKS as SEC_TASKS,
    answers_match,
    clear_device_cache,
    ensure_full_ranking,
    generate_answer,
    generate_random_heads,
    install_head_masking,
    load_ranked_heads_json,
    resolve_device,
    sanitize_ranking,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_DIR, "results", "nq_reverse_ablation")
DEFAULT_NQ_FILE = os.path.join(PROJECT_DIR, "data", "nq_input", "nq_test.json")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RANDOM_SEEDS = [42, 123, 456]


def load_nq_instances(path, max_instances=None, shuffle=False, seed=42):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"NQ file must contain a list or JSONL records: {path}")

    instances = []
    for row in data:
        missing = [key for key in ("idx", "question", "context", "needle_value") if not row.get(key)]
        if missing:
            raise ValueError(f"NQ instance is missing required fields {missing}: {row}")
        row = dict(row)
        row["task"] = "nq"
        aliases = row.get("answer_aliases") or [row["needle_value"]]
        if isinstance(aliases, str):
            aliases = [aliases]
        row["answer_aliases"] = aliases
        instances.append(row)

    if shuffle:
        random.Random(seed).shuffle(instances)
    if max_instances and len(instances) > max_instances:
        instances = instances[:max_instances]
    return instances


def answer_matches_any(pred, gold, aliases):
    candidates = [gold]
    candidates.extend(aliases or [])
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        if answers_match(pred, str(candidate)):
            return True
    return False


def method_filter(method_name, requested):
    if not requested:
        return True
    if method_name in requested:
        return True
    if method_name.startswith("Transfer-") and "Transfer-all" in requested:
        return True
    if method_name.startswith("Random-seed") and "Random" in requested:
        return True
    return False


def load_reverse_rankings(args, model_spec, num_layers, num_heads):
    ranking_dir = resolve_detection_dir(PROJECT_DIR, model_spec, args.ranking_dir)
    methods = {}
    sources = {}

    nq_path = os.path.join(PROJECT_DIR, "Llama-3.1-8B-Instruct", "nq_TRAIN.json")
    if os.path.exists(nq_path):
        nq_heads = sanitize_ranking(
            load_ranked_heads_json(nq_path),
            num_layers,
            num_heads,
            "QRScore-8B-NQ-TRAIN",
        )
        methods["QRScore-8B-NQ-TRAIN"] = ensure_full_ranking(
            nq_heads, num_layers, num_heads, seed=43
        )
        sources["QRScore-8B-NQ-TRAIN"] = nq_path
    else:
        print(f"  QRScore-8B-NQ-TRAIN: missing {nq_path}")

    sec_path = os.path.join(ranking_dir, "long_context_combined_heads.json")
    if os.path.exists(sec_path):
        sec_heads = sanitize_ranking(
            load_ranked_heads_json(sec_path),
            num_layers,
            num_heads,
            "QRScore-SEC",
        )
        methods["QRScore-SEC"] = ensure_full_ranking(
            sec_heads, num_layers, num_heads, seed=99
        )
        sources["QRScore-SEC"] = sec_path
    else:
        print(f"  QRScore-SEC: missing {sec_path}")

    for idx, task in enumerate(args.transfer_source_tasks):
        method_name = f"Transfer-{task}"
        task_path = os.path.join(ranking_dir, f"long_context_{task}_heads.json")
        if not os.path.exists(task_path):
            print(f"  {method_name}: missing {task_path}")
            continue
        heads = sanitize_ranking(
            load_ranked_heads_json(task_path),
            num_layers,
            num_heads,
            method_name,
        )
        methods[method_name] = ensure_full_ranking(
            heads, num_layers, num_heads, seed=100 + idx
        )
        sources[method_name] = task_path

    if args.include_random_baselines:
        for seed in RANDOM_SEEDS:
            method_name = f"Random-seed{seed}"
            methods[method_name] = generate_random_heads(
                num_layers,
                num_heads,
                total_heads=max(args.knockout_sizes) + 50,
                seed=seed,
            )
            sources[method_name] = f"random_seed={seed}"

    if args.methods:
        methods = {
            name: ranking
            for name, ranking in methods.items()
            if method_filter(name, set(args.methods))
        }
        sources = {name: sources[name] for name in methods}

    return methods, sources, ranking_dir


def run_nq_sweep(
    model,
    tokenizer,
    instances,
    head_ranking,
    knockout_sizes,
    max_context_tokens,
    progress_every,
    sweep_label,
):
    results = {}
    total_steps = len(knockout_sizes) * len(instances)
    completed_steps = 0
    start_time = time.time()

    print(f"    Sweep: {sweep_label}")
    print(
        f"    Progress plan: {len(knockout_sizes)} K values x "
        f"{len(instances)} NQ instances = {total_steps} steps"
    )

    for k in knockout_sizes:
        if k == 0:
            model.set_head_mask(None)
        else:
            model.set_head_mask(head_ranking[: min(k, len(head_ranking))])

        correct = 0
        details = []
        for idx, inst in enumerate(instances, start=1):
            pred, raw_text, token_ids = generate_answer(
                model,
                tokenizer,
                {"context": inst["context"], "question": inst["question"]},
                max_context_tokens=max_context_tokens,
            )
            match = answer_matches_any(pred, inst["needle_value"], inst.get("answer_aliases"))
            correct += int(match)
            details.append(
                {
                    "idx": inst["idx"],
                    "task": "nq",
                    "gold": inst["needle_value"],
                    "answer_aliases": inst.get("answer_aliases", []),
                    "pred": pred,
                    "raw_text": raw_text,
                    "token_ids": token_ids,
                    "correct": int(match),
                }
            )
            clear_device_cache(model.device.type)

            completed_steps += 1
            should_log = (
                completed_steps == 1
                or completed_steps % max(1, progress_every) == 0
                or completed_steps == total_steps
            )
            if should_log:
                elapsed = time.time() - start_time
                rate = completed_steps / elapsed if elapsed > 0 else 0.0
                eta = (total_steps - completed_steps) / rate if rate > 0 else 0.0
                print(
                    f"      progress {completed_steps}/{total_steps} "
                    f"({100.0 * completed_steps / total_steps:5.1f}%) | "
                    f"K={k} inst={idx}/{len(instances)} | "
                    f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m"
                )

        total = len(instances)
        accuracy = correct / total if total else 0.0
        print(f"    K={k:3d}: accuracy={accuracy:.4f} ({correct}/{total})")
        results[k] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_task": {"nq": {"accuracy": accuracy, "correct": correct, "total": total}},
            "details": details,
        }

    model.set_head_mask(None)
    return results


def write_method_result(output_dir, model_spec, method_name, knockout_sizes, method_results, log_tokens):
    if log_tokens:
        token_log_path = os.path.join(output_dir, f"{method_name.replace(' ', '_')}_token_log.jsonl")
        with open(token_log_path, "w", encoding="utf-8") as tl:
            for k in knockout_sizes:
                for d in method_results[k]["details"]:
                    tl.write(
                        json.dumps(
                            {
                                "method": method_name,
                                "K": k,
                                "idx": d["idx"],
                                "task": d["task"],
                                "gold": d["gold"],
                                "answer_aliases": d["answer_aliases"],
                                "pred": d["pred"],
                                "raw_text": d["raw_text"],
                                "token_ids": d["token_ids"],
                                "correct": d["correct"],
                            }
                        )
                        + "\n"
                    )
        print(f"  Token log: {token_log_path}")

    summary_details = {}
    for k in knockout_sizes:
        summary_details[str(k)] = [
            {key: val for key, val in d.items() if key not in ("raw_text", "token_ids")}
            for d in method_results[k]["details"]
        ]

    payload = {
        "model_name": model_spec.model_name,
        "model_slug": model_spec.model_slug,
        "model_family": model_spec.model_family,
        "method": method_name,
        "task": "nq",
        "knockout_sizes": knockout_sizes,
        "accuracy_curve": {str(k): method_results[k]["accuracy"] for k in knockout_sizes},
        "per_task_curves": {
            "nq": {str(k): method_results[k]["accuracy"] for k in knockout_sizes}
        },
        "details": summary_details,
    }
    out_path = os.path.join(output_dir, f"{method_name.replace(' ', '_')}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_summary(output_dir, model_spec, nq_file, ranking_dir, sources, knockout_sizes, all_results):
    summary = {
        "model": model_spec.model_name,
        "model_name": model_spec.model_name,
        "model_slug": model_spec.model_slug,
        "model_family": model_spec.model_family,
        "task": "nq",
        "nq_file": nq_file,
        "ranking_dir": ranking_dir,
        "knockout_sizes": knockout_sizes,
        "ranking_sources": sources,
        "num_instances": next(iter(all_results.values()))[knockout_sizes[0]]["total"] if all_results else 0,
        "methods": {},
    }
    for method_name, method_results in all_results.items():
        curve = {str(k): method_results[k]["accuracy"] for k in knockout_sizes}
        baseline = method_results[0]["accuracy"] if 0 in method_results else None
        k16_acc = method_results[16]["accuracy"] if 16 in method_results else None
        summary["methods"][method_name] = {
            "accuracy_curve": curve,
            "baseline_accuracy": baseline,
            "accuracy_at_k16": k16_acc,
            "drop_at_k16": (baseline - k16_acc) if baseline is not None and k16_acc is not None else None,
        }

    path = os.path.join(output_dir, "comparison_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def write_tables(output_dir, knockout_sizes, all_results):
    accuracy_path = os.path.join(output_dir, "accuracy_table.csv")
    drop_path = os.path.join(output_dir, "drop_from_baseline_table.csv")

    with open(accuracy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "K", "accuracy", "correct", "total"])
        writer.writeheader()
        for method_name, method_results in all_results.items():
            for k in knockout_sizes:
                row = method_results[k]
                writer.writerow(
                    {
                        "method": method_name,
                        "K": k,
                        "accuracy": row["accuracy"],
                        "correct": row["correct"],
                        "total": row["total"],
                    }
                )

    with open(drop_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "K", "drop_from_baseline"])
        writer.writeheader()
        for method_name, method_results in all_results.items():
            baseline = method_results[0]["accuracy"] if 0 in method_results else None
            for k in knockout_sizes:
                acc = method_results[k]["accuracy"]
                writer.writerow(
                    {
                        "method": method_name,
                        "K": k,
                        "drop_from_baseline": (baseline - acc) if baseline is not None else None,
                    }
                )

    return accuracy_path, drop_path


def plot_accuracy(output_dir, knockout_sizes, all_results):
    if not HAS_MPL:
        print("matplotlib not available; skipping accuracy_vs_knockout.png")
        return None
    if not all_results:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    transfer_names = [name for name in all_results if name.startswith("Transfer-")]
    transfer_colors = plt.cm.tab20.colors

    for idx, (method_name, method_results) in enumerate(all_results.items()):
        accs = [method_results[k]["accuracy"] for k in knockout_sizes]
        if method_name == "QRScore-8B-NQ-TRAIN":
            style = {"color": "#2E7D32", "linewidth": 2.8, "marker": "D", "linestyle": "--"}
        elif method_name == "QRScore-SEC":
            style = {"color": "#1565C0", "linewidth": 2.8, "marker": "o", "linestyle": "-"}
        elif method_name.startswith("Random"):
            style = {"color": "#9E9E9E", "linewidth": 1.4, "marker": "x", "linestyle": ":"}
        else:
            transfer_idx = transfer_names.index(method_name) if method_name in transfer_names else idx
            style = {
                "color": transfer_colors[transfer_idx % len(transfer_colors)],
                "linewidth": 1.6,
                "marker": ".",
                "linestyle": "-",
            }
        ax.plot(knockout_sizes, accs, label=method_name, markersize=6, **style)

    ax.set_xlabel("Number of Knocked-Out Heads (K)", fontsize=12)
    ax.set_ylabel("NQ Answer Accuracy", fontsize=12)
    ax.set_title("NQ Reverse-Transfer Head Ablation", fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(knockout_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()

    path = os.path.join(output_dir, "accuracy_vs_knockout.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run NQ reverse-transfer head ablation.")
    parser.add_argument("--nq_file", default=DEFAULT_NQ_FILE)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--ranking_dir", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--model_slug", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--knockout_sizes", nargs="+", type=int, default=DEFAULT_KNOCKOUT_SIZES)
    parser.add_argument("--max_instances", type=int, default=512)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--shuffle_instances", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Methods to run. Use Transfer-all for all SEC category heads and Random "
            "for all random seeds. Default: all available methods."
        ),
    )
    parser.add_argument("--transfer_source_tasks", nargs="+", default=SEC_TASKS)
    parser.add_argument(
        "--include_random_baselines",
        dest="include_random_baselines",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_random_baselines",
        dest="include_random_baselines",
        action="store_false",
    )
    parser.add_argument("--log_tokens", action="store_true")
    args = parser.parse_args()

    model_spec = resolve_model_spec(
        model_name=args.model_name,
        model_slug=args.model_slug,
        tokenizer_name=args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, model_spec.model_slug)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Resolved project_dir={PROJECT_DIR}")
    print(f"Resolved nq_file={args.nq_file}")
    print(f"Resolved output_dir={output_dir}")
    print(f"Resolved model={model_spec.model_name} ({model_spec.model_family})")

    instances = load_nq_instances(
        args.nq_file,
        max_instances=args.max_instances,
        shuffle=args.shuffle_instances,
        seed=args.seed,
    )
    print(f"Loaded {len(instances)} NQ instances")

    resolved_device = resolve_device(args.device)
    print(f"\nLoading model: {args.model_name}")
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

    print("\nLoading reverse-transfer rankings...")
    rankings, sources, ranking_dir = load_reverse_rankings(args, model_spec, num_layers, num_heads)
    if not rankings:
        raise RuntimeError("No compatible ranking methods found.")
    print(f"Methods to evaluate: {list(rankings.keys())}")

    all_results = {}
    for method_idx, (method_name, ranking) in enumerate(rankings.items(), start=1):
        print(f"\n{'=' * 60}")
        print(f"Method {method_idx}/{len(rankings)}: {method_name}")
        print(f"{'=' * 60}")
        method_results = run_nq_sweep(
            model=model,
            tokenizer=tokenizer,
            instances=instances,
            head_ranking=ranking,
            knockout_sizes=args.knockout_sizes,
            max_context_tokens=args.max_context_tokens,
            progress_every=args.progress_every,
            sweep_label=f"method={method_name}",
        )
        all_results[method_name] = method_results
        path = write_method_result(
            output_dir=output_dir,
            model_spec=model_spec,
            method_name=method_name,
            knockout_sizes=args.knockout_sizes,
            method_results=method_results,
            log_tokens=args.log_tokens,
        )
        print(f"  Results: {path}")

    summary_path = write_summary(
        output_dir=output_dir,
        model_spec=model_spec,
        nq_file=args.nq_file,
        ranking_dir=ranking_dir,
        sources=sources,
        knockout_sizes=args.knockout_sizes,
        all_results=all_results,
    )
    accuracy_path, drop_path = write_tables(output_dir, args.knockout_sizes, all_results)
    plot_path = plot_accuracy(output_dir, args.knockout_sizes, all_results)

    print(f"\nSummary saved to {summary_path}")
    print(f"Accuracy table saved to {accuracy_path}")
    print(f"Drop table saved to {drop_path}")
    if plot_path:
        print(f"Accuracy plot saved to {plot_path}")


if __name__ == "__main__":
    main()
