"""
Build NIAH-style evaluation data from haystack_plan.csv.

For each row:
  - haystack_text provides ~5000 words of distractor SEC filing sections
  - needle_sentence is inserted into the haystack at a controlled position
  - needle_value is the ground-truth answer
  - The task question is asked over the combined context

Outputs:
  data/niah_input/{task}_test.json   (100% of test_plan.csv for ablation evaluation)

Instance format:
{
    "idx": "<filename>_<task>_<row_index>",
    "question": "<TASK_QUESTIONS[task]>",
    "needle_value": "...",
    "needle_sentence": "...",
    "context": "<haystack with needle inserted>",
    "needle_position": "middle",
    "paragraphs": [{"idx": "<section_id>", "paragraph_text": "..."}],
    "gt_docs": ["<needle_section_id>"]
}
"""

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


def build_paragraphs_from_context(haystack_text: str, needle_sentence: str,
                                  needle_section_id: str,
                                  haystack_sections_used: str,
                                  chunk_words: int = 400) -> list:
    """Split the combined context into paragraph-style chunks for QRHead detection.

    Ensures the needle_sentence is never split across chunks.
    Returns a list of {"idx": ..., "paragraph_text": ...} dicts plus a gt_docs list.
    """
    full_context = insert_needle(haystack_text, needle_sentence, position="middle")

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
                paragraphs.append({"idx": f"chunk_{chunk_idx}", "paragraph_text": chunk_text})
            chunk_idx += 1
            i += chunk_words

    gt_docs = [p["idx"] for p in paragraphs if needle_sentence in p["paragraph_text"]]
    if not gt_docs:
        gt_docs = ["needle_chunk"]

    return paragraphs, gt_docs


def main():
    random.seed(RANDOM_SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(data_dir, "niah_input")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resolved project_dir={project_dir}")
    print(f"Resolved data_dir={data_dir}")
    print(f"Resolved output_dir={output_dir}")

    print("Loading test_plan.csv ...")
    task_rows = defaultdict(list)
    with open(os.path.join(data_dir, "test_plan.csv"), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            task_rows[row["task"]].append((row_idx, row))
    total = sum(len(v) for v in task_rows.values())
    print(f"  Loaded {total} rows across {len(task_rows)} tasks")

    stats = {}
    for task, question in TASK_QUESTIONS.items():
        rows = task_rows.get(task, [])
        print(f"\nProcessing task: {task} ({len(rows)} raw rows)")

        valid = []
        skip_reasons = defaultdict(int)

        for row_idx, row in rows:
            haystack_text = row["haystack_text"].strip()
            needle_sentence = row["needle_sentence"].strip()
            needle_value = row["needle_value"].strip()

            if not needle_value:
                skip_reasons["no_needle_value"] += 1
                continue
            if not needle_sentence:
                skip_reasons["no_needle_sentence"] += 1
                continue
            if len(haystack_text.split()) < MIN_HAYSTACK_WORDS:
                skip_reasons["haystack_too_short"] += 1
                continue
            if row["needle_unique_in_section"] != "True":
                skip_reasons["not_unique"] += 1
                continue

            full_context = insert_needle(haystack_text, needle_sentence, position="middle")
            if needle_sentence not in full_context:
                skip_reasons["needle_lost_in_insert"] += 1
                continue

            paragraphs, gt_docs = build_paragraphs_from_context(
                haystack_text, needle_sentence,
                row["needle_section_id"],
                row.get("haystack_sections_used", ""),
            )

            instance_id = f"{row['filename']}_{task}_{row_idx}"
            valid.append({
                "idx": instance_id,
                "question": question,
                "needle_value": needle_value,
                "needle_sentence": needle_sentence,
                "context": full_context,
                "needle_position": "middle",
                "context_words": len(full_context.split()),
                "paragraphs": paragraphs,
                "gt_docs": gt_docs,
            })

        random.shuffle(valid)
        test = valid

        test_path = os.path.join(output_dir, f"{task}_test.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2, ensure_ascii=False)

        stats[task] = {
            "raw": len(rows),
            "valid": len(valid),
            "test": len(test),
            "skipped": dict(skip_reasons),
        }
        print(f"  Valid: {len(valid)} (test={len(test)})")
        if skip_reasons:
            print(f"  Skipped: {dict(skip_reasons)}")

    print("\n=== Summary ===")
    for task, s in stats.items():
        print(f"  {task:30s}: {s['valid']:4d} total (test={s['test']:3d})")


if __name__ == "__main__":
    main()
