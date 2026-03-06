"""
Build long-context detection data for QRHead detection.

Unlike the short-context data in data/detection_input/ (~3K tokens per instance),
this script creates instances with ~25K-32K tokens by concatenating multiple
haystack_text entries from haystack_plan.csv. This matches the paper's detection
scale (32K-128K tokens) and should produce meaningful QRScore rankings with
top heads in middle layers (8-17) rather than noisy early layers (0-2).

For each detection instance:
  - 1 "gold" row provides the needle_sentence to insert
  - 4-5 additional rows provide extra distractor haystack_text
  - The combined context is split into ~400-word paragraph chunks
  - The chunk containing the needle is marked as gt_doc

Outputs:
  data/long_context_detection/{task}_detection.json  (for QRHead detection)
  data/long_context_detection/combined_detection.json (pooled across tasks)
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict

TASK_QUESTIONS = {
    "registrant_name": "What is the registrant name?",
    "headquarters_city": "What is the headquarters city?",
    "headquarters_state": "What is the headquarters state?",
    "incorporation_state": "What is the incorporation state?",
    "incorporation_year": "What is the incorporation year?",
    "employees_count_total": "What is the total employee count?",
    "ceo_lastname": "What is the CEO's last name?",
    "holder_record_amount": "What is the holder record amount?",
}

RANDOM_SEED = 42
MIN_HAYSTACK_WORDS = 500
TARGET_TOTAL_WORDS = 20000  # ~25K-32K tokens (1 token ~ 0.75 words)
MAX_INSTANCES_PER_TASK = 200
CHUNK_WORDS = 400

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def insert_needle(haystack: str, needle: str, position: str = "middle") -> str:
    """Insert needle sentence into haystack at a given relative position."""
    sentences = _SENT_BOUNDARY.split(haystack)
    if not sentences:
        return needle + " " + haystack

    if position == "start":
        idx = max(1, len(sentences) // 10)
    elif position == "end":
        idx = len(sentences) - max(1, len(sentences) // 10)
    else:
        idx = len(sentences) // 2

    idx = max(0, min(idx, len(sentences)))
    before = ". ".join(sentences[:idx])
    after = ". ".join(sentences[idx:])

    parts = []
    if before:
        if not before.rstrip().endswith("."):
            before = before.rstrip() + "."
        parts.append(before)
    parts.append(needle)
    if after:
        parts.append(after)

    return " ".join(parts)


def chunk_text_with_needle(full_context: str, needle_sentence: str,
                           chunk_words: int = CHUNK_WORDS):
    """Split context into paragraph chunks, keeping needle in a single chunk.

    Returns list of {"idx": ..., "paragraph_text": ...} dicts and gt_docs list.
    """
    needle_char_start = full_context.index(needle_sentence)
    needle_char_end = needle_char_start + len(needle_sentence)

    words = full_context.split()
    char_pos = 0
    word_char_starts = []
    for w in words:
        idx = full_context.index(w, char_pos)
        word_char_starts.append(idx)
        char_pos = idx + len(w)

    needle_start_word = 0
    needle_end_word = len(words)
    for i, cs in enumerate(word_char_starts):
        if cs <= needle_char_start:
            needle_start_word = i
        if cs + len(words[i]) >= needle_char_end:
            needle_end_word = i + 1
            break

    paragraphs = []
    chunk_idx = 0
    i = 0
    while i < len(words):
        if i <= needle_start_word < i + chunk_words:
            end = max(i + chunk_words, needle_end_word)
            chunk_text = " ".join(words[i:end])
            paragraphs.append({"idx": "needle_chunk", "paragraph_text": chunk_text})
            i = end
        else:
            chunk_text = " ".join(words[i:i + chunk_words])
            if chunk_text.strip():
                paragraphs.append({
                    "idx": f"chunk_{chunk_idx}",
                    "paragraph_text": chunk_text,
                })
            chunk_idx += 1
            i += chunk_words

    gt_docs = [p["idx"] for p in paragraphs
               if needle_sentence in p["paragraph_text"]]
    if not gt_docs:
        gt_docs = ["needle_chunk"]

    return paragraphs, gt_docs


def build_long_context_instance(gold_row, distractor_rows, task, question,
                                row_idx, target_words=TARGET_TOTAL_WORDS):
    """Build one detection instance by combining gold + distractor haystacks."""
    haystack_text = gold_row["haystack_text"].strip()
    needle_sentence = gold_row["needle_sentence"].strip()
    needle_value = gold_row["needle_value"].strip()

    gold_with_needle = insert_needle(haystack_text, needle_sentence, "middle")
    if needle_sentence not in gold_with_needle:
        return None

    context_parts = []
    gold_words = len(gold_with_needle.split())

    words_so_far = gold_words
    distractor_texts = []
    for dr in distractor_rows:
        dt = dr["haystack_text"].strip()
        dw = len(dt.split())
        if dw < MIN_HAYSTACK_WORDS:
            continue
        distractor_texts.append(dt)
        words_so_far += dw
        if words_so_far >= target_words:
            break

    if words_so_far < target_words * 0.5:
        return None

    random.shuffle(distractor_texts)
    insert_pos = random.randint(0, len(distractor_texts))
    for i, dt in enumerate(distractor_texts):
        if i == insert_pos:
            context_parts.append(gold_with_needle)
        context_parts.append(dt)
    if insert_pos >= len(distractor_texts):
        context_parts.append(gold_with_needle)

    full_context = "\n\n".join(context_parts)

    if needle_sentence not in full_context:
        return None

    paragraphs, gt_docs = chunk_text_with_needle(
        full_context, needle_sentence, CHUNK_WORDS
    )

    instance_id = f"{gold_row['filename']}_{task}_{row_idx}"
    return {
        "idx": instance_id,
        "question": question,
        "needle_value": needle_value,
        "needle_sentence": needle_sentence,
        "paragraphs": paragraphs,
        "gt_docs": gt_docs,
        "context_words": len(full_context.split()),
        "num_paragraphs": len(paragraphs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build long-context detection data for QRHead detection"
    )
    parser.add_argument("--target_words", type=int, default=TARGET_TOTAL_WORDS,
                        help="Target total words per instance (default: 20000)")
    parser.add_argument("--max_instances", type=int, default=MAX_INSTANCES_PER_TASK,
                        help="Max instances per task (default: 200)")
    parser.add_argument("--chunk_words", type=int, default=CHUNK_WORDS,
                        help="Words per paragraph chunk (default: 400)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(data_dir, "long_context_detection")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading haystack_plan.csv ...")
    task_rows = defaultdict(list)
    with open(os.path.join(data_dir, "haystack_plan.csv"),
              newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            task_rows[row["task"]].append((row_idx, row))
    total = sum(len(v) for v in task_rows.values())
    print(f"  Loaded {total} rows across {len(task_rows)} tasks")

    all_instances = []
    stats = {}

    for task, question in TASK_QUESTIONS.items():
        rows = task_rows.get(task, [])
        print(f"\nProcessing task: {task} ({len(rows)} raw rows)")

        valid_gold = []
        all_distractors = []

        for row_idx, row in rows:
            ht = row["haystack_text"].strip()
            ns = row["needle_sentence"].strip()
            nv = row["needle_value"].strip()

            if not nv or not ns:
                continue
            if len(ht.split()) < MIN_HAYSTACK_WORDS:
                continue

            if row["needle_unique_in_section"] == "True":
                valid_gold.append((row_idx, row))

            all_distractors.append(row)

        random.shuffle(valid_gold)
        valid_gold = valid_gold[:args.max_instances]

        instances = []
        skip_count = 0

        for i, (row_idx, gold_row) in enumerate(valid_gold):
            candidates = [r for r in all_distractors
                          if r["filename"] != gold_row["filename"]]
            random.shuffle(candidates)
            num_distractors = max(3, args.target_words // 5000)
            distractor_sample = candidates[:num_distractors + 2]

            inst = build_long_context_instance(
                gold_row, distractor_sample, task, question, row_idx,
                target_words=args.target_words,
            )
            if inst is None:
                skip_count += 1
                continue
            instances.append(inst)

        out_path = os.path.join(output_dir, f"{task}_detection.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)

        all_instances.extend(instances)

        avg_words = (sum(inst["context_words"] for inst in instances) /
                     len(instances)) if instances else 0
        avg_paras = (sum(inst["num_paragraphs"] for inst in instances) /
                     len(instances)) if instances else 0

        stats[task] = {
            "valid_gold": len(valid_gold),
            "built": len(instances),
            "skipped": skip_count,
            "avg_context_words": int(avg_words),
            "avg_paragraphs": int(avg_paras),
        }
        print(f"  Built: {len(instances)} instances "
              f"(avg {int(avg_words)} words, {int(avg_paras)} paragraphs)")
        if skip_count:
            print(f"  Skipped: {skip_count}")

    random.shuffle(all_instances)
    combined_path = os.path.join(output_dir, "combined_detection.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"Total instances: {len(all_instances)}")
    for task, s in stats.items():
        print(f"  {task:30s}: {s['built']:4d} instances "
              f"(avg {s['avg_context_words']} words, "
              f"{s['avg_paragraphs']} paras)")

    stats_path = os.path.join(output_dir, "build_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print(f"Combined detection data: {combined_path}")


if __name__ == "__main__":
    main()
