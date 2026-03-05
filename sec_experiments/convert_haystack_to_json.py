"""
Convert haystack_plan.csv + sections.csv into per-task JSON files
for QRHead detection and ablation evaluation.

Outputs per task:
  data/detection_input/{task}_train.json  (80% for QR head detection)
  data/detection_input/{task}_test.json   (20% for ablation evaluation)

Instance format:
{
    "idx": "<filename>_<task>_<row_index>",
    "question": "<TASK_QUESTIONS[task]>",
    "paragraphs": [
        {"idx": "<section_id>", "paragraph_text": "..."},
        ...
    ],
    "gt_docs": ["<needle_section_id>"]
}
"""

import csv
import json
import random
import os
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

ALL_SECTION_IDS = [
    "section_1", "section_1A", "section_1B",
    "section_2", "section_3", "section_4",
    "section_5", "section_6", "section_7", "section_7A",
    "section_8", "section_9", "section_9A", "section_9B",
    "section_10", "section_11", "section_12",
    "section_13", "section_14", "section_15",
]

MIN_DISTRACTOR_WORDS = 50
MIN_DISTRACTORS = 3
TRUNCATE_WORDS = 400
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42


def truncate_first_n_words(text, n):
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n])


_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _trim_to_sentence_start(text):
    """Remove a partial sentence from the beginning of text."""
    m = _SENTENCE_BOUNDARY_RE.search(text)
    if m:
        return text[m.end():]
    return text


def window_around_needle(section_text, needle_sentence, window_words=400):
    """
    Extract a ~window_words window from section_text that contains needle_sentence.
    Prefer starting from the beginning of the section; only shift forward
    when the needle would be truncated away. When the window starts mid-section,
    trim the leading partial sentence so the result begins at a sentence boundary.
    """
    if needle_sentence not in section_text:
        return None

    needle_start_char = section_text.index(needle_sentence)
    needle_end_char = needle_start_char + len(needle_sentence)

    words = section_text.split()

    if len(words) <= window_words:
        return section_text

    first_n = " ".join(words[:window_words])
    if needle_sentence in first_n:
        return first_n

    char_count = 0
    needle_start_word = 0
    for i, w in enumerate(words):
        if char_count >= needle_start_char:
            needle_start_word = i
            break
        char_count += len(w) + 1

    char_count = 0
    needle_end_word = len(words)
    for i, w in enumerate(words):
        char_count += len(w) + 1
        if char_count >= needle_end_char:
            needle_end_word = i + 1
            break

    needle_word_len = needle_end_word - needle_start_word
    remaining = window_words - needle_word_len

    before = remaining // 2
    after = remaining - before

    start = max(0, needle_start_word - before)
    end = start + window_words

    if end > len(words):
        end = len(words)
        start = max(0, end - window_words)

    result = " ".join(words[start:end])

    if start > 0:
        result = _trim_to_sentence_start(result)

    if needle_sentence not in result:
        return None

    return result


def main():
    random.seed(RANDOM_SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(data_dir, "detection_input")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading sections.csv...")
    sections_by_file = {}
    with open(os.path.join(data_dir, "sections.csv"), newline="", encoding="utf-8") as sf:
        reader = csv.DictReader(sf)
        for row in reader:
            sections_by_file[row["filename"]] = row
    print(f"  Loaded {len(sections_by_file)} filings")

    print("Loading haystack_plan.csv...")
    task_instances = defaultdict(list)
    with open(os.path.join(data_dir, "haystack_plan.csv"), newline="", encoding="utf-8") as hf:
        reader = csv.DictReader(hf)
        for row_idx, row in enumerate(reader):
            task_instances[row["task"]].append((row_idx, row))
    print(f"  Loaded {sum(len(v) for v in task_instances.values())} instances across {len(task_instances)} tasks")

    conversion_log = []
    stats = {}

    for task, question in TASK_QUESTIONS.items():
        instances = task_instances.get(task, [])
        print(f"\nProcessing task: {task} ({len(instances)} raw instances)")

        valid_instances = []
        skipped_reasons = defaultdict(int)

        for row_idx, row in instances:
            filename = row["filename"]
            needle_section_id = row["needle_section_id"]
            needle_sentence = row["needle_sentence"]

            file_sections = sections_by_file.get(filename)
            if file_sections is None:
                skipped_reasons["filing_not_found"] += 1
                continue

            gold_section_text = file_sections.get(needle_section_id, "")
            if not gold_section_text:
                skipped_reasons["gold_section_empty"] += 1
                continue

            gold_doc_text = window_around_needle(gold_section_text, needle_sentence, TRUNCATE_WORDS)
            if gold_doc_text is None:
                skipped_reasons["needle_not_in_window"] += 1
                continue

            distractors = []
            distractor_ids = []
            for sec_id in ALL_SECTION_IDS:
                if sec_id == needle_section_id:
                    continue
                sec_text = file_sections.get(sec_id, "")
                if not sec_text or len(sec_text.split()) < MIN_DISTRACTOR_WORDS:
                    continue
                truncated = truncate_first_n_words(sec_text, TRUNCATE_WORDS)
                distractors.append({"idx": sec_id, "paragraph_text": truncated})
                distractor_ids.append(sec_id)

            if len(distractors) < MIN_DISTRACTORS:
                skipped_reasons["too_few_distractors"] += 1
                continue

            paragraphs = [{"idx": needle_section_id, "paragraph_text": gold_doc_text}] + distractors

            instance_id = f"{filename}_{task}_{row_idx}"
            valid_instances.append({
                "idx": instance_id,
                "question": question,
                "paragraphs": paragraphs,
                "gt_docs": [needle_section_id],
            })

            conversion_log.append({
                "idx": instance_id,
                "task": task,
                "filename": filename,
                "needle_section_id": needle_section_id,
                "needle_sentence": needle_sentence,
                "gold_doc_word_count": len(gold_doc_text.split()),
                "num_distractors": len(distractors),
                "distractor_section_ids": distractor_ids,
            })

        random.shuffle(valid_instances)

        for inst in valid_instances:
            random.shuffle(inst["paragraphs"])

        split_idx = int(len(valid_instances) * TRAIN_FRACTION)
        train_instances = valid_instances[:split_idx]
        test_instances = valid_instances[split_idx:]

        train_path = os.path.join(output_dir, f"{task}_train.json")
        test_path = os.path.join(output_dir, f"{task}_test.json")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_instances, f, indent=2, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_instances, f, indent=2, ensure_ascii=False)

        train_ids = {inst["idx"] for inst in train_instances}
        test_ids = {inst["idx"] for inst in test_instances}
        for entry in conversion_log:
            if entry["task"] == task:
                if entry["idx"] in train_ids:
                    entry["split"] = "train"
                elif entry["idx"] in test_ids:
                    entry["split"] = "test"

        stats[task] = {
            "raw_instances": len(instances),
            "valid_instances": len(valid_instances),
            "train_instances": len(train_instances),
            "test_instances": len(test_instances),
            "skipped": dict(skipped_reasons),
        }
        print(f"  Valid: {len(valid_instances)} (train={len(train_instances)}, test={len(test_instances)})")
        print(f"  Skipped: {dict(skipped_reasons)}")
        print(f"  Saved to {train_path} and {test_path}")

    log_path = os.path.join(output_dir, "conversion_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(conversion_log, f, indent=2, ensure_ascii=False)
    print(f"\nConversion log saved to {log_path}")

    print("\n=== Summary ===")
    for task, s in stats.items():
        print(f"  {task:30s}: {s['valid_instances']:4d} total  (train={s['train_instances']:4d}, test={s['test_instances']:3d})  from {s['raw_instances']} raw")


if __name__ == "__main__":
    main()
