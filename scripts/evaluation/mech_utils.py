"""Utilities for mechanistic QRHead intervention experiments.

The scripts in this directory use teacher-forced answer scoring rather than
free generation. This keeps interventions comparable after the first answer
token, where generation-based rollouts can diverge for uninteresting reasons.
"""

import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qrretriever.model_runtime import (  # noqa: E402
    load_stock_causal_lm,
    load_tokenizer,
    resolve_ablation_dir,
    resolve_detection_dir,
    resolve_model_spec,
)


DEFAULT_TASKS = [
    "employees_count_total",
    "ceo_lastname",
    "registrant_name",
    "headquarters_state",
]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def build_prompt(context: str, question: str) -> str:
    return (
        "Read the following document and answer the question.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely with just the answer value."
    )


def render_prompt(tokenizer, context: str, question: str) -> str:
    prompt_text = build_prompt(context, question)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt_text


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    answer_ids: List[int]
    answer_positions: List[int]
    prediction_positions: List[int]
    rendered_prompt: str
    prompt_offsets: List[Tuple[int, int]]


def encode_prompt_and_answer(
    tokenizer,
    inst: dict,
    *,
    max_context_tokens: int,
    answer_prefix: str = "",
) -> EncodedExample:
    rendered_prompt = render_prompt(tokenizer, inst["context"], inst["question"])
    tokenizer.truncation_side = "left"
    prompt_encoding = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_context_tokens,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    answer_text = f"{answer_prefix}{inst['needle_value']}"
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    if not answer_ids:
        raise ValueError(f"Gold answer tokenized to an empty sequence: {answer_text!r}")

    prompt_ids = prompt_encoding["input_ids"]
    answer_tensor = torch.tensor([answer_ids], dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, answer_tensor], dim=1)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = int(prompt_ids.shape[1])
    answer_positions = list(range(prompt_len, prompt_len + len(answer_ids)))
    prediction_positions = [pos - 1 for pos in answer_positions]
    offsets = [(int(a), int(b)) for a, b in prompt_encoding["offset_mapping"][0].tolist()]
    return EncodedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_len=prompt_len,
        answer_ids=answer_ids,
        answer_positions=answer_positions,
        prediction_positions=prediction_positions,
        rendered_prompt=rendered_prompt,
        prompt_offsets=offsets,
    )


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model structure: expected `model.layers` or `transformer.h`.")


def get_num_heads(model) -> int:
    return int(getattr(model.config, "num_attention_heads"))


def get_head_dim(model) -> int:
    return int(model.config.hidden_size // model.config.num_attention_heads)


def load_ranked_heads_json(path: Path) -> List[Tuple[int, int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    heads = []
    for row in data:
        head_str = row[0] if isinstance(row, list) else row.get("head")
        layer, head = map(int, head_str.split("-"))
        heads.append((layer, head))
    return heads


def default_ranking_path(project_dir: Path, model_spec, ranking_name: str) -> Path:
    detection_dir = Path(resolve_detection_dir(str(project_dir), model_spec))
    full_path = detection_dir / f"{ranking_name}_heads.json"
    if full_path.exists():
        return full_path
    # Some collected artifacts only keep top-k exports. This fallback is enough
    # to run the QR patch, but bottom-ranked controls require a full ranking.
    return detection_dir / "topk" / f"{ranking_name}_top128.json"


def all_heads(num_layers: int, num_heads: int) -> List[Tuple[int, int]]:
    return [(layer, head) for layer in range(num_layers) for head in range(num_heads)]


def random_control_heads(
    *,
    qr_heads: Sequence[Tuple[int, int]],
    num_layers: int,
    num_heads: int,
    seed: int,
) -> List[Tuple[int, int]]:
    qr_set = set(qr_heads)
    candidates = [head for head in all_heads(num_layers, num_heads) if head not in qr_set]
    rng = random.Random(seed)
    return rng.sample(candidates, k=len(qr_heads))


def bottom_control_heads(
    *,
    ranking: Sequence[Tuple[int, int]],
    qr_heads: Sequence[Tuple[int, int]],
    k: int,
) -> List[Tuple[int, int]]:
    qr_set = set(qr_heads)
    out = []
    for head in reversed(ranking):
        if head not in qr_set:
            out.append(head)
        if len(out) == k:
            break
    if len(out) < k:
        raise ValueError("Could not build bottom-ranked control head set.")
    return out


def same_layer_control_heads(
    *,
    qr_heads: Sequence[Tuple[int, int]],
    num_heads: int,
) -> List[Tuple[int, int]]:
    qr_set = set(qr_heads)
    used = set(qr_set)
    controls = []
    for layer, head in qr_heads:
        chosen = None
        for offset in range(1, num_heads + 1):
            candidate = (layer, (head + offset) % num_heads)
            if candidate not in used:
                chosen = candidate
                break
        if chosen is None:
            raise ValueError(f"Could not find same-layer control for head {(layer, head)}")
        controls.append(chosen)
        used.add(chosen)
    return controls


def group_heads_by_layer(heads: Optional[Sequence[Tuple[int, int]]]) -> Dict[int, List[int]]:
    grouped = defaultdict(list)
    if not heads:
        return grouped
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    return grouped


class HeadPatchController:
    """Cache, ablate, and patch per-head activations immediately before o_proj."""

    def __init__(self, model):
        self.model = model
        self.layers = get_decoder_layers(model)
        self.num_heads = get_num_heads(model)
        self.head_dim = get_head_dim(model)
        self.handles = []
        self.mode = "none"
        self.prediction_positions: List[int] = []
        self.cache: Dict[int, torch.Tensor] = {}
        self.masked_by_layer: Dict[int, List[int]] = defaultdict(list)
        self.patch_by_layer: Dict[int, List[int]] = defaultdict(list)

    def install(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            handle = attn.o_proj.register_forward_pre_hook(
                self._make_hook(layer_idx),
                with_kwargs=False,
            )
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset(self) -> None:
        self.mode = "none"
        self.prediction_positions = []
        self.masked_by_layer = defaultdict(list)
        self.patch_by_layer = defaultdict(list)

    def set_cache_mode(self, prediction_positions: Sequence[int]) -> None:
        self.cache = {}
        self.mode = "cache"
        self.prediction_positions = list(prediction_positions)
        self.masked_by_layer = defaultdict(list)
        self.patch_by_layer = defaultdict(list)

    def set_ablate_mode(
        self,
        masked_heads: Sequence[Tuple[int, int]],
        *,
        prediction_positions: Optional[Sequence[int]] = None,
    ) -> None:
        self.mode = "ablate"
        self.prediction_positions = list(prediction_positions or [])
        self.masked_by_layer = group_heads_by_layer(masked_heads)
        self.patch_by_layer = defaultdict(list)

    def set_patch_mode(
        self,
        masked_heads: Sequence[Tuple[int, int]],
        patch_heads: Sequence[Tuple[int, int]],
        prediction_positions: Sequence[int],
    ) -> None:
        if not self.cache:
            raise ValueError("Patch mode requires a clean cache. Run cache mode first.")
        self.mode = "patch"
        self.prediction_positions = list(prediction_positions)
        self.masked_by_layer = group_heads_by_layer(masked_heads)
        self.patch_by_layer = group_heads_by_layer(patch_heads)

    def _make_hook(self, layer_idx: int):
        def hook(_module, args):
            if self.mode == "none":
                return args
            x = args[0]
            bsz, seq_len, hidden = x.shape
            if hidden != self.num_heads * self.head_dim:
                raise ValueError(
                    f"Unexpected o_proj input width {hidden}; expected "
                    f"{self.num_heads * self.head_dim}."
                )
            x_heads = x.clone().view(bsz, seq_len, self.num_heads, self.head_dim)

            positions = [p for p in self.prediction_positions if 0 <= p < seq_len]
            if self.mode == "cache":
                if positions:
                    self.cache[layer_idx] = x_heads[:, positions, :, :].detach().cpu()
                return (x_heads.view(bsz, seq_len, hidden),) + args[1:]

            masked = self.masked_by_layer.get(layer_idx, [])
            if masked:
                x_heads[:, :, masked, :] = 0

            patch_heads = self.patch_by_layer.get(layer_idx, [])
            if self.mode == "patch" and patch_heads and positions:
                if layer_idx not in self.cache:
                    raise ValueError(f"Missing clean cache for layer {layer_idx}.")
                cached = self.cache[layer_idx].to(device=x_heads.device, dtype=x_heads.dtype)
                pos_index = torch.tensor(positions, device=x_heads.device)
                head_index = torch.tensor(patch_heads, device=x_heads.device)
                # cached shape: (1, n_positions, n_heads, head_dim)
                x_heads[:, pos_index[:, None], head_index[None, :], :] = cached[
                    :, :, head_index, :
                ]

            return (x_heads.view(bsz, seq_len, hidden),) + args[1:]

        return hook


def answer_logprobs(logits: torch.Tensor, encoded: EncodedExample) -> Tuple[float, float, List[float]]:
    pred_pos = torch.tensor(encoded.prediction_positions, device=logits.device)
    ans_ids = torch.tensor(encoded.answer_ids, device=logits.device)
    selected_logits = logits[0, pred_pos, :]
    log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(1, ans_ids[:, None]).squeeze(1)
    values = [float(x) for x in token_logprobs.detach().cpu()]
    return float(token_logprobs.sum().detach().cpu()), float(token_logprobs.mean().detach().cpu()), values


def score_gold_answer(model, encoded: EncodedExample) -> Tuple[float, float, List[float]]:
    device = next(model.parameters()).device
    inputs = {
        "input_ids": encoded.input_ids.to(device),
        "attention_mask": encoded.attention_mask.to(device),
        "use_cache": False,
    }
    with torch.inference_mode():
        outputs = model(**inputs)
    return answer_logprobs(outputs.logits, encoded)


def load_task_instances(project_dir: Path, tasks: Sequence[str]) -> List[dict]:
    instances = []
    for task in tasks:
        path = project_dir / "data" / "niah_input" / f"{task}_test.json"
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            row = dict(row)
            row.setdefault("task", task)
            instances.append(row)
    return instances


def load_clean_to_ablated_failure_ids(results_path: Path, k: int) -> Optional[set]:
    if not results_path.exists():
        return None
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", {})
    clean_rows = details.get("0") or details.get(0)
    ablated_rows = details.get(str(k)) or details.get(k)
    if not clean_rows or not ablated_rows:
        return None
    clean_correct = {row["idx"] for row in clean_rows if int(row.get("correct", 0)) == 1}
    ablated_wrong = {row["idx"] for row in ablated_rows if int(row.get("correct", 0)) == 0}
    return clean_correct & ablated_wrong


def select_instances(
    instances: Sequence[dict],
    *,
    failure_ids: Optional[set],
    max_examples_per_task: int,
    seed: int,
) -> List[dict]:
    by_task = defaultdict(list)
    for inst in instances:
        if failure_ids is not None and inst["idx"] not in failure_ids:
            continue
        by_task[inst["task"]].append(inst)
    rng = random.Random(seed)
    selected = []
    for task in sorted(by_task):
        rows = list(by_task[task])
        rng.shuffle(rows)
        if max_examples_per_task > 0:
            rows = rows[:max_examples_per_task]
        selected.extend(rows)
    return selected


def safe_div(num: float, den: float) -> Optional[float]:
    if den is None or abs(den) < 1e-8 or math.isnan(den):
        return None
    return num / den


def summarize_by_task_and_condition(rows: Sequence[dict], value_key: str) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is None or math.isnan(float(value)):
            continue
        grouped[(row["task"], row["condition"])].append(float(value))
    out = []
    for (task, condition), values in sorted(grouped.items()):
        out.append({
            "task": task,
            "condition": condition,
            "n": len(values),
            f"mean_{value_key}": sum(values) / len(values),
        })
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_and_tokenizer(args, *, for_detection: bool = False):
    model_spec = resolve_model_spec(
        args.model_name,
        model_slug=getattr(args, "model_slug", None),
        tokenizer_name=getattr(args, "tokenizer_name", None),
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(model_spec)
    model = load_stock_causal_lm(
        model_spec,
        device,
        for_detection=for_detection,
        load_in_8bit=getattr(args, "load_in_8bit", False),
    )
    model.eval()
    return model_spec, tokenizer, model


def default_ablation_results_path(project_dir: Path, model_spec) -> Path:
    return Path(resolve_ablation_dir(str(project_dir), model_spec)) / "QRScore-SEC_results.json"


def normalize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def char_span_to_token_span(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Optional[Tuple[int, int]]:
    hits = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start < char_end and end > char_start:
            hits.append(idx)
    if not hits:
        return None
    return min(hits), max(hits) + 1


def find_gold_span(encoded: EncodedExample, inst: dict, *, mode: str) -> Optional[Tuple[int, int]]:
    sentence = inst.get("needle_sentence") or ""
    value = str(inst.get("needle_value") or "")
    rendered = encoded.rendered_prompt
    if mode == "value" and value:
        value_start = rendered.find(value)
        if value_start >= 0:
            return char_span_to_token_span(encoded.prompt_offsets, value_start, value_start + len(value))
    if sentence:
        sent_start = rendered.find(sentence)
        if sent_start >= 0:
            if mode == "value_in_sentence" and value and value in sentence:
                rel = sentence.find(value)
                start = sent_start + rel
                return char_span_to_token_span(encoded.prompt_offsets, start, start + len(value))
            return char_span_to_token_span(encoded.prompt_offsets, sent_start, sent_start + len(sentence))
    if value:
        value_start = rendered.find(value)
        if value_start >= 0:
            return char_span_to_token_span(encoded.prompt_offsets, value_start, value_start + len(value))
    return None


def find_distractor_span(encoded: EncodedExample, inst: dict, length: int) -> Optional[Tuple[int, int]]:
    rendered = encoded.rendered_prompt
    value = str(inst.get("needle_value") or "")
    for para in inst.get("paragraphs", []):
        if para.get("idx") == "needle_chunk":
            continue
        text = para.get("paragraph_text") or ""
        if not text or (value and value in text):
            continue
        char_start = rendered.find(text[: min(len(text), 200)])
        if char_start < 0:
            continue
        span = char_span_to_token_span(encoded.prompt_offsets, char_start, char_start + len(text))
        if span is None:
            continue
        start, end = span
        if end - start >= length:
            mid = start + max(0, (end - start - length) // 2)
            return mid, mid + length
    return None


def find_random_span(
    encoded: EncodedExample,
    *,
    length: int,
    exclude: Optional[Tuple[int, int]],
    seed: int,
) -> Optional[Tuple[int, int]]:
    max_start = max(0, encoded.prompt_len - length)
    candidates = []
    for start in range(max_start):
        end = start + length
        if exclude is not None and start < exclude[1] and end > exclude[0]:
            continue
        # Skip special-token-ish offsets.
        if any(b <= a for a, b in encoded.prompt_offsets[start:end]):
            continue
        candidates.append(start)
    if not candidates:
        return None
    rng = random.Random(seed)
    start = rng.choice(candidates)
    return start, start + length
