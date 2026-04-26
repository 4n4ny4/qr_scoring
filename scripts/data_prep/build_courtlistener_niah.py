"""
Convert CourtListener CSV into NIAH JSON files for ablation evaluation.

Reads niah_formatted_dataset.csv, filters to verified rows, does an 80/20
split by unique case ID, and outputs the 20% test split as per-task JSON
files matching the schema expected by run_ablation.py.

Output:
    data/niah_input_courtlistener/{Task}_test.json

Usage:
    python scripts/data_prep/build_courtlistener_niah.py
    python scripts/data_prep/build_courtlistener_niah.py --csv path/to/file.csv
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

RANDOM_SEED = 42
TRAIN_FRACTION = 0.8


def main():
    parser = argparse.ArgumentParser(
        description="Build CourtListener NIAH test data for ablation evaluation"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))

    parser.add_argument(
        "--csv",
        default=os.path.join(project_dir, "niah_formatted_dataset.csv"),
        help="Path to the CourtListener CSV file",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_dir, "data", "niah_input_courtlistener"),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train_fraction", type=float, default=TRAIN_FRACTION)
    args = parser.parse_args()

    random.seed(args.seed)
    csv.field_size_limit(sys.maxsize)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading: {args.csv}")
    print(f"Output:  {args.output_dir}")

    # -- Read and filter CSV --
    rows_by_case = defaultdict(list)
    total_rows = 0
    skipped_unverified = 0

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if row.get("evidence_verified", "").strip().lower() != "true":
                skipped_unverified += 1
                continue

            idx = row["idx"]
            case_id = idx.split("_")[0]
            rows_by_case[case_id].append(row)

    verified_count = total_rows - skipped_unverified
    unique_cases = sorted(rows_by_case.keys(), key=int)
    print(f"\nCSV stats:")
    print(f"  Total rows:      {total_rows}")
    print(f"  Verified rows:   {verified_count}")
    print(f"  Skipped:         {skipped_unverified}")
    print(f"  Unique cases:    {len(unique_cases)}")

    # -- 80/20 split by case ID --
    random.shuffle(unique_cases)
    split_idx = int(len(unique_cases) * args.train_fraction)
    train_cases = set(unique_cases[:split_idx])
    test_cases = set(unique_cases[split_idx:])

    print(f"\nSplit (seed={args.seed}, train_fraction={args.train_fraction}):")
    print(f"  Train cases: {len(train_cases)}")
    print(f"  Test cases:  {len(test_cases)}")

    # -- Build test instances per task --
    task_instances = defaultdict(list)
    needle_missing = 0

    for case_id in sorted(test_cases, key=int):
        for row in rows_by_case[case_id]:
            task = row["task"]
            context = row["context"].strip()
            needle_sentence = row.get("needle_sentence", "").strip()
            needle_value = row["needle_value"].strip()
            question = row["question"].strip()
            idx = row["idx"]

            if not context or not needle_value or not question:
                continue

            if needle_sentence and needle_sentence not in context:
                needle_missing += 1
                continue

            task_instances[task].append({
                "idx": idx,
                "question": question,
                "needle_value": needle_value,
                "needle_sentence": needle_sentence,
                "context": context,
            })

    # -- Write per-task JSON files --
    print(f"\nTest instances per task:")
    total_test = 0
    for task in sorted(task_instances.keys()):
        instances = task_instances[task]
        total_test += len(instances)
        out_path = os.path.join(args.output_dir, f"{task}_test.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)
        print(f"  {task:15s}: {len(instances):4d} instances -> {out_path}")

    print(f"\n  Total test:      {total_test}")
    if needle_missing:
        print(f"  Skipped (needle not in context): {needle_missing}")

    print(f"\nDone. Files written to {args.output_dir}/")
    print(f"Run ablation with:")
    print(f"  python scripts/evaluation/run_ablation.py \\")
    print(f"    --niah_dir {args.output_dir} \\")
    print(f"    --tasks {' '.join(sorted(task_instances.keys()))} \\")
    print(f"    --output_dir results/courtlistener_nq_ablation \\")
    print(f"    --model_name meta-llama/Llama-3.1-8B-Instruct \\")
    print(f"    --knockout_sizes 0 8 16 32 64 \\")
    print(f"    --methods QRScore-8B-NQ-TRAIN \\")
    print(f"    --include_random_baselines \\")
    print(f"    --log_tokens \\")
    print(f"    --max_context_tokens 8192")


if __name__ == "__main__":
    main()
