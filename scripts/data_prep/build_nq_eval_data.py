"""
Build an NQ-style answer-generation evaluation file for reverse-transfer
head ablation.

Output format matches the fields consumed by the ablation evaluators:

[
  {
    "idx": "nq_<id>",
    "question": "...",
    "context": "...",
    "needle_value": "primary gold answer",
    "answer_aliases": ["all acceptable short answers"],
    "task": "nq"
  }
]

Preferred input is the original/simplified Natural Questions JSONL with
short-answer annotations. The script also supports Hugging Face datasets
when `datasets` is installed, including passage-pair variants such as
`sentence-transformers/natural-questions`; those are useful for smoke tests
but weaker as answer-generation gold because they do not expose short answers.
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

HTML_TAG_RE = re.compile(r"^<[^>]+>$")
WHITESPACE_RE = re.compile(r"\s+")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_json_records(path: Path) -> Iterator[Dict]:
    """Yield records from JSON, JSONL, JSONL.GZ, or a dict/list JSON wrapper."""
    with open_text(path) as f:
        if path.suffix in {".jsonl", ".gz"}:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return

        payload = json.load(f)

    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("data", "examples", "records", "train", "validation", "test"):
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        yield payload
    else:
        raise ValueError(f"Unsupported JSON payload in {path}: {type(payload)}")


def iter_hf_records(dataset_name: str, dataset_config: Optional[str], split: str) -> Iterator[Dict]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hugging Face loading requires `datasets`. Install it with "
            "`python3 -m pip install datasets`, or pass --input_file for a local NQ JSONL."
        ) from exc

    args = [dataset_name]
    if dataset_config:
        args.append(dataset_config)
    dataset = load_dataset(*args, split=split)
    for row in dataset:
        yield dict(row)


def clean_text(text: str) -> str:
    text = text.replace("``", '"').replace("''", '"')
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def first_present(row: Dict, keys: Sequence[str]):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def extract_question(row: Dict) -> Optional[str]:
    value = first_present(row, ("question_text", "question", "query", "questionText"))
    if isinstance(value, dict):
        value = first_present(value, ("text", "question", "query"))
    return clean_text(str(value)) if value else None


def extract_example_id(row: Dict, fallback_idx: int) -> str:
    value = first_present(row, ("example_id", "id", "query_id", "question_id"))
    if value is None:
        value = fallback_idx
    return f"nq_{value}"


def token_text(token) -> Tuple[str, bool]:
    """Return (token, is_html) for NQ token variants."""
    if isinstance(token, dict):
        tok = token.get("token") or token.get("text") or ""
        return str(tok), bool(token.get("html_token", False))
    return str(token), bool(HTML_TAG_RE.match(str(token)))


def document_tokens(row: Dict) -> List[str]:
    tokens = first_present(row, ("document_tokens", "tokens"))
    if tokens is not None:
        cleaned = []
        for token in tokens:
            tok, is_html = token_text(token)
            if tok and not is_html and not HTML_TAG_RE.match(tok):
                cleaned.append(tok)
        return cleaned

    text = first_present(row, ("document_text", "document", "context", "passage", "answer", "positive"))
    if not text:
        return []

    tokens = []
    for tok in str(text).split():
        if tok and not HTML_TAG_RE.match(tok):
            tokens.append(tok)
    return tokens


def normalize_annotation_list(annotations) -> List[Dict]:
    """Handle list-of-dicts and HF dict-of-lists annotation encodings."""
    if isinstance(annotations, list):
        return [a for a in annotations if isinstance(a, dict)]
    if not isinstance(annotations, dict):
        return []

    length = 0
    for value in annotations.values():
        if isinstance(value, list):
            length = max(length, len(value))
    normalized = []
    for i in range(length):
        item = {}
        for key, value in annotations.items():
            item[key] = value[i] if isinstance(value, list) and i < len(value) else value
        normalized.append(item)
    return normalized


def span_text(tokens: Sequence[str], start: int, end: int) -> Optional[str]:
    if start is None or end is None:
        return None
    if start < 0 or end <= start or start >= len(tokens):
        return None
    end = min(end, len(tokens))
    text = clean_text(" ".join(tokens[start:end]))
    return text or None


def span_bounds(span) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(span, dict):
        return None, None
    return span.get("start_token"), span.get("end_token")


def extract_short_answers(row: Dict, tokens: Sequence[str], include_yes_no: bool) -> Tuple[List[str], Optional[Tuple[int, int]]]:
    """Return answer aliases and a preferred token span for context selection."""
    direct_answer = first_present(
        row,
        ("short_answer", "short_answers_text", "answers", "answer_text", "needle_value"),
    )
    if direct_answer is not None:
        answers = []
        for value in as_list(direct_answer):
            if isinstance(value, dict):
                value = first_present(value, ("text", "answer"))
            if value:
                answers.append(clean_text(str(value)))
        return dedupe_keep_order(answers), None

    annotations = normalize_annotation_list(row.get("annotations"))
    answers = []
    preferred_span = None

    for ann in annotations:
        if include_yes_no:
            yes_no = str(ann.get("yes_no_answer", "NONE")).strip().lower()
            if yes_no in {"yes", "no"}:
                answers.append(yes_no)

        short_answers = ann.get("short_answers") or []
        if isinstance(short_answers, dict):
            short_answers = normalize_annotation_list(short_answers)

        for short in as_list(short_answers):
            if isinstance(short, dict):
                text = first_present(short, ("text", "answer"))
                if text:
                    answers.append(clean_text(str(text)))
                    continue
                start, end = span_bounds(short)
                text = span_text(tokens, start, end)
                if text:
                    answers.append(text)
                    if preferred_span is None:
                        preferred_span = (start, end)

    return dedupe_keep_order(answers), preferred_span


def extract_long_answer_span(row: Dict) -> Optional[Tuple[int, int]]:
    annotations = normalize_annotation_list(row.get("annotations"))
    for ann in annotations:
        long_answer = ann.get("long_answer")
        if isinstance(long_answer, dict):
            start, end = span_bounds(long_answer)
            if start is not None and end is not None and start >= 0 and end > start:
                return start, end
    return None


def build_context(
    row: Dict,
    tokens: Sequence[str],
    answer_span: Optional[Tuple[int, int]],
    max_context_words: int,
    context_padding_words: int,
) -> Optional[str]:
    direct_context = first_present(row, ("context", "passage", "positive"))
    if direct_context:
        return truncate_words(clean_text(str(direct_context)), max_context_words)

    if not tokens:
        return None

    span = extract_long_answer_span(row) or answer_span
    if span:
        start, end = span
        start = max(0, start - context_padding_words)
        end = min(len(tokens), end + context_padding_words)
        if end - start > max_context_words:
            center = (span[0] + span[1]) // 2
            half = max_context_words // 2
            start = max(0, center - half)
            end = min(len(tokens), start + max_context_words)
        return clean_text(" ".join(tokens[start:end]))

    return clean_text(" ".join(tokens[:max_context_words]))


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        cleaned = clean_text(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def build_instances(
    records: Iterable[Dict],
    max_instances: int,
    max_context_words: int,
    context_padding_words: int,
    include_yes_no: bool,
    allow_passage_answer: bool,
    shuffle: bool,
    seed: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    instances = []
    rng = random.Random(seed)
    stats = {
        "seen": 0,
        "built": 0,
        "eligible": 0,
        "missing_question": 0,
        "missing_context": 0,
        "missing_short_answer": 0,
        "passage_answer_fallback": 0,
    }

    for idx, row in enumerate(records):
        stats["seen"] += 1
        question = extract_question(row)
        if not question:
            stats["missing_question"] += 1
            continue

        tokens = document_tokens(row)
        answers, answer_span = extract_short_answers(row, tokens, include_yes_no)

        if not answers and allow_passage_answer:
            passage_answer = first_present(row, ("answer", "positive", "passage", "context"))
            if passage_answer:
                answers = [truncate_words(clean_text(str(passage_answer)), max_context_words)]
                stats["passage_answer_fallback"] += 1

        if not answers:
            stats["missing_short_answer"] += 1
            continue

        context = build_context(row, tokens, answer_span, max_context_words, context_padding_words)
        if not context:
            stats["missing_context"] += 1
            continue

        example_id = extract_example_id(row, idx)
        instance = {
            "idx": example_id,
            "task": "nq",
            "question": question,
            "context": context,
            "needle_value": answers[0],
            "answer_aliases": answers,
            "context_words": len(context.split()),
        }
        stats["eligible"] += 1

        if not max_instances:
            instances.append(instance)
            continue

        if shuffle:
            # Reservoir sample so a capped shuffled run does not keep every
            # long-context example in memory.
            if len(instances) < max_instances:
                instances.append(instance)
            else:
                replace_idx = rng.randrange(stats["eligible"])
                if replace_idx < max_instances:
                    instances[replace_idx] = instance
        else:
            instances.append(instance)
            if len(instances) >= max_instances:
                break

    if shuffle:
        rng.shuffle(instances)
    stats["built"] = len(instances)
    return instances, stats


def main():
    parser = argparse.ArgumentParser(description="Build NQ QA-style evaluation JSON for reverse ablation.")
    parser.add_argument("--input_file", default=None, help="Local NQ JSON/JSONL/JSONL.GZ file.")
    parser.add_argument(
        "--hf_dataset",
        default=None,
        help="Optional Hugging Face dataset name, e.g. google-research-datasets/natural_questions.",
    )
    parser.add_argument("--hf_config", default=None, help="Optional Hugging Face dataset config/subset.")
    parser.add_argument("--split", default="validation", help="HF split name (default: validation).")
    parser.add_argument(
        "--output_file",
        default=os.path.join(PROJECT_DIR, "data", "nq_input", "nq_test.json"),
    )
    parser.add_argument("--max_instances", type=int, default=512)
    parser.add_argument("--max_context_words", type=int, default=3500)
    parser.add_argument("--context_padding_words", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Reservoir-sample and shuffle capped instances.")
    parser.add_argument("--include_yes_no", action="store_true", help="Keep yes/no NQ annotations.")
    parser.add_argument(
        "--allow_passage_answer",
        action="store_true",
        help=(
            "Fallback for passage-pair datasets without short answers. The whole answer passage "
            "is used as broad gold, which is suitable only for smoke tests."
        ),
    )
    args = parser.parse_args()

    if not args.input_file and not args.hf_dataset:
        raise ValueError("Provide either --input_file or --hf_dataset.")

    if args.input_file:
        records = iter_json_records(Path(args.input_file))
        source = args.input_file
    else:
        records = iter_hf_records(args.hf_dataset, args.hf_config, args.split)
        source = f"{args.hf_dataset}/{args.hf_config or 'default'}:{args.split}"

    instances, stats = build_instances(
        records=records,
        max_instances=args.max_instances,
        max_context_words=args.max_context_words,
        context_padding_words=args.context_padding_words,
        include_yes_no=args.include_yes_no,
        allow_passage_answer=args.allow_passage_answer,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)

    stats_path = output_path.with_name(output_path.stem + "_build_stats.json")
    stats_payload = {
        "source": source,
        "output_file": str(output_path),
        "max_instances": args.max_instances,
        "max_context_words": args.max_context_words,
        "context_padding_words": args.context_padding_words,
        **stats,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    print(f"Wrote {len(instances)} NQ instances to {output_path}")
    print(f"Stats: {stats_path}")
    if stats["passage_answer_fallback"]:
        print(
            "WARNING: Some examples used passage-answer fallback. Prefer original NQ short-answer "
            "data for the main scientific run."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
