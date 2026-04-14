"""
Option A: Build long-context detection data for QRHead detection using
single-filing distractors only.

Each detection instance uses:
  - One filing (filename) from haystack_plan.csv
  - Gold section: needle_section_id from sections.csv (contains needle_sentence)
  - Distractor sections: haystack_sections_used from the SAME filename

We concatenate these sections into a long context, then split into fixed
~400-word chunks. gt_docs is defined as all chunks whose text contains the
needle_sentence (primary) or, if that fails, a normalized form of needle_value
for numeric robustness.

Outputs:
    data/long_context_detection_optionA/{task}_detection.json
    data/long_context_detection_optionA/combined_detection.json
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
CHUNK_WORDS = 400
MAX_INSTANCES_PER_TASK = None

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def order_sections_by_filing(section_ids, section_order):
    """Return unique section ids in the filing order defined by sections.csv."""
    seen = set()
    ordered = []
    for sec_id in section_ids:
        sec_id = (sec_id or "").strip()
        if not sec_id or sec_id in seen:
            continue
        seen.add(sec_id)
        ordered.append(sec_id)

    return sorted(
        ordered,
        key=lambda sec_id: (section_order.get(sec_id, float("inf")), sec_id),
    )


def chunk_text_with_needle(full_context: str, needle_sentence: str, chunk_words: int = CHUNK_WORDS):
    """Split context into chunks, keeping needle_sentence within a single chunk."""
    needle_char_start = full_context.find(needle_sentence)
    if needle_char_start == -1:
        return [], []

    needle_char_end = needle_char_start + len(needle_sentence)

    words = full_context.split()
    char_pos = 0
    word_char_starts = []
    for w in words:
        idx = full_context.find(w, char_pos)
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
            chunk_text = " ".join(words[i : i + chunk_words])
            if chunk_text.strip():
                paragraphs.append(
                    {
                        "idx": f"chunk_{chunk_idx}",
                        "paragraph_text": chunk_text,
                    }
                )
            chunk_idx += 1
            i += chunk_words

    return paragraphs


def normalize_numeric(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas in numbers
    s = re.sub(r"[,.;:!?\"']+$", "", s)
    return " ".join(s.split())


def compute_gt_docs(paragraphs, needle_sentence: str, needle_value: str):
    """Select gold doc ids: prefer sentence match, then numeric value match."""
    gt = []
    for p in paragraphs:
        if needle_sentence and needle_sentence in p["paragraph_text"]:
            gt.append(p["idx"])

    if gt:
        return gt

    # Fallback: numeric / value-based match
    nv_norm = normalize_numeric(needle_value)
    if not nv_norm:
        return gt

    for p in paragraphs:
        text_norm = normalize_numeric(p["paragraph_text"])
        if nv_norm and nv_norm in text_norm:
            gt.append(p["idx"])

    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Build Option A long-context detection data (single filing, sections-based)"
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=MAX_INSTANCES_PER_TASK,
        help="Optional max instances per task. If omitted, use all valid train samples.",
    )
    parser.add_argument(
        "--chunk_words",
        type=int,
        default=CHUNK_WORDS,
        help="Words per paragraph chunk (default: 400)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(data_dir, "long_context_detection_optionA")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resolved project_dir={project_dir}")
    print(f"Resolved data_dir={data_dir}")
    print(f"Resolved output_dir={output_dir}")

    # Load sections.csv into a dict: filename -> row dict
    sections_path = os.path.join(data_dir, "sections.csv")
    sections_by_filename = {}
    with open(sections_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        section_field_order = {
            field: idx
            for idx, field in enumerate(reader.fieldnames or [])
            if field.startswith("section_")
        }
        for row in reader:
            sections_by_filename[row["filename"]] = row

    print("Loading train_plan.csv ...")
    task_rows = defaultdict(list)
    with open(os.path.join(data_dir, "train_plan.csv"), newline="", encoding="utf-8") as f:
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

        valid_rows = []
        for row_idx, row in rows:
            if not row["needle_value"].strip() or not row["needle_sentence"].strip():
                continue
            if row["needle_unique_in_section"] != "True":
                continue
            if row["filename"] not in sections_by_filename:
                continue
            valid_rows.append((row_idx, row))

        random.shuffle(valid_rows)
        if args.max_instances is not None:
            valid_rows = valid_rows[: args.max_instances]

        instances = []
        skip_count = 0

        for row_idx, row in valid_rows:
            filename = row["filename"]
            sec_row = sections_by_filename.get(filename)
            if not sec_row:
                skip_count += 1
                continue

            needle_section_id = row["needle_section_id"]
            needle_text = sec_row.get(needle_section_id, "") or ""
            needle_text = needle_text.strip()
            if not needle_text:
                skip_count += 1
                continue

            # Build context in filing order so the gold section stays in a natural position.
            used_sections = (row.get("haystack_sections_used") or "").split("|")
            ordered_section_ids = order_sections_by_filing(
                [needle_section_id, *used_sections],
                section_field_order,
            )

            parts = []
            for sec_id in ordered_section_ids:
                txt = (sec_row.get(sec_id, "") or "").strip()
                if txt:
                    parts.append(txt)

            full_context = "\n\n".join(parts).strip()
            if not full_context:
                skip_count += 1
                continue

            needle_sentence = row["needle_sentence"].strip()
            needle_value = row["needle_value"].strip()

            if needle_sentence not in full_context:
                # Needle sentence should be in the needle section; if not, skip for safety.
                skip_count += 1
                continue

            paragraphs = chunk_text_with_needle(full_context, needle_sentence, args.chunk_words)
            if not paragraphs:
                skip_count += 1
                continue

            gt_docs = compute_gt_docs(paragraphs, needle_sentence, needle_value)
            if not gt_docs:
                skip_count += 1
                continue

            instance_id = f"{filename}_{task}_{row_idx}"
            instances.append(
                {
                    "idx": instance_id,
                    "question": question,
                    "needle_value": needle_value,
                    "needle_sentence": needle_sentence,
                    "paragraphs": paragraphs,
                    "gt_docs": gt_docs,
                    "context_words": len(full_context.split()),
                    "num_paragraphs": len(paragraphs),
                }
            )

        task_out = os.path.join(output_dir, f"{task}_detection.json")
        with open(task_out, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)

        all_instances.extend(instances)

        avg_words = int(sum(inst["context_words"] for inst in instances) / len(instances)) if instances else 0
        avg_paras = int(sum(inst["num_paragraphs"] for inst in instances) / len(instances)) if instances else 0

        stats[task] = {
            "raw": len(rows),
            "valid": len(valid_rows),
            "built": len(instances),
            "skipped": skip_count,
            "avg_context_words": avg_words,
            "avg_paragraphs": avg_paras,
        }
        print(
            f"  Built: {len(instances)} instances "
            f"(avg {avg_words} words, {avg_paras} paragraphs, skipped={skip_count})"
        )

    combined_path = os.path.join(output_dir, "combined_detection.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Total instances: {len(all_instances)}")
    for task, s in stats.items():
        print(
            f"  {task:30s}: raw={s['raw']:4d}, valid={s['valid']:4d}, "
            f"built={s['built']:4d}, skipped={s['skipped']:4d}, "
            f"avg_words={s['avg_context_words']}, avg_paras={s['avg_paragraphs']}"
        )

    stats_path = os.path.join(output_dir, "build_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print(f"Combined detection data: {combined_path}")


if __name__ == "__main__":
    main()
