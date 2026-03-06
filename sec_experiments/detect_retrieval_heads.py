"""
RETHEAD detection: Wu et al. (2025b) copy-paste retrieval head scoring.

For each NIAH-style instance:
  - Run autoregressive generation with output_attentions=True
  - For each generated token t, check each head h:
    - Find the input position with max attention weight from h
    - If that position is within the needle span AND the input token there
      matches t, count it as a "copy" by head h
  - Retrieval_Score(h) = |copied tokens| / |generated answer tokens|
  - Average across all instances

This requires eager attention (not flash) to access attention weights during
generation, so memory usage is higher. Use --max_instances to limit.

Uses use_cache=False to avoid transformers cache dimension mismatch (4 vs 5)
with output_attentions=True during manual generation loops. Slower but robust.

Outputs: same format as QRScore detection: list of ["layer-head", score] sorted
descending.

Usage:
  python sec_experiments/detect_retrieval_heads.py \
    --niah_dir data/niah_input \
    --output_file results/detection/rethead_combined_heads.json \
    --max_instances 50 \
    --model_name meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import transformers

# Use qrretriever's custom Llama (better cache compatibility with output_attentions)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qrretriever.custom_modeling_llama import LlamaForCausalLM

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 30
MAX_INPUT_TOKENS = 4096


def build_prompt(context, question):
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Read the following document and answer the question.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely with just the answer value."
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def find_needle_token_positions(input_ids, tokenizer, needle_sentence):
    """Find the token positions in input_ids that correspond to needle_sentence."""
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    needle_start_char = full_text.find(needle_sentence)
    if needle_start_char == -1:
        needle_tokens = tokenizer.encode(needle_sentence, add_special_tokens=False)
        needle_len = len(needle_tokens)
        ids_list = input_ids[0].tolist()
        for start in range(len(ids_list) - needle_len + 1):
            window = ids_list[start:start + needle_len]
            match_count = sum(1 for a, b in zip(window, needle_tokens) if a == b)
            if match_count >= needle_len * 0.8:
                return list(range(start, start + needle_len))
        return []

    offsets = tokenizer(full_text, return_offsets_mapping=True)["offset_mapping"]
    needle_end_char = needle_start_char + len(needle_sentence)
    positions = []
    for i, (s, e) in enumerate(offsets):
        if e > needle_start_char and s < needle_end_char:
            positions.append(i)
    return positions


def detect_rethead_single_instance(model, tokenizer, input_ids, needle_positions,
                                   num_layers, num_heads, max_new_tokens=MAX_NEW_TOKENS,
                                   use_cache=True):
    """Run generation and track copy-paste behavior per head.

    Returns: dict mapping (layer, head) -> number of tokens copied from needle.
    Also returns total generated tokens.

    When use_cache=False: avoids transformers cache dimension bug (4 vs 5) with
    output_attentions=True, but does full forward pass each step (slower).
    """
    device = model.device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    needle_set = set(needle_positions)
    if not needle_set:
        return {}, 0

    needle_token_ids = {}
    for pos in needle_positions:
        if pos < input_ids.shape[1]:
            needle_token_ids[pos] = input_ids[0, pos].item()

    copy_counts = defaultdict(int)
    total_generated = 0

    past_key_values = None
    current_ids = input_ids

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=use_cache,
            )

        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1)
        next_token_id = next_token.item()

        if next_token_id == tokenizer.eos_token_id:
            break

        total_generated += 1

        attentions = outputs.attentions
        for layer_idx, attn in enumerate(attentions):
            # attn shape: (batch, num_heads, q_len, kv_len)
            last_token_attn = attn[0, :, -1, :]  # (num_heads, kv_len)

            for head_idx in range(last_token_attn.shape[0]):
                max_pos = last_token_attn[head_idx].argmax().item()

                if max_pos in needle_set and needle_token_ids.get(max_pos) == next_token_id:
                    copy_counts[(layer_idx, head_idx)] += 1

        if use_cache:
            past_key_values = outputs.past_key_values
            current_ids = next_token.unsqueeze(0).unsqueeze(0)
        else:
            # No cache: append new token and run full forward on extended sequence
            current_ids = torch.cat(
                [current_ids, next_token.unsqueeze(0).unsqueeze(0)],
                dim=1
            ).to(device)
            # Update needle_token_ids for positions in the extended sequence
            # (needle positions stay the same; we only care about last-token attention)
            past_key_values = None

    return copy_counts, total_generated


def main():
    parser = argparse.ArgumentParser(description="RETHEAD (copy-paste) retrieval head detection")
    parser.add_argument("--niah_dir", default="data/niah_input")
    parser.add_argument("--output_file", default="results/detection/rethead_combined_heads.json")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--max_instances", type=int, default=50,
                        help="Max instances total (sampled across tasks)")
    parser.add_argument("--max_input_tokens", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--use_cache", action="store_true",
                        help="Use KV cache (faster). Default: no cache to avoid transformers "
                             "cache dimension bug with output_attentions.")
    args = parser.parse_args()

    test_instances = []
    for task in args.tasks:
        test_path = os.path.join(args.niah_dir, f"{task}_test.json")
        if not os.path.exists(test_path):
            print(f"WARNING: {test_path} not found, skipping", file=sys.stderr)
            continue
        with open(test_path) as f:
            data = json.load(f)
        for inst in data:
            inst["task"] = task
        test_instances.extend(data)

    import random
    random.seed(42)
    random.shuffle(test_instances)
    if args.max_instances and len(test_instances) > args.max_instances:
        test_instances = test_instances[:args.max_instances]

    print(f"Using {len(test_instances)} instances for RETHEAD detection")
    if not args.use_cache:
        print("Using use_cache=False (avoids transformers cache bug; slower but robust)")

    print(f"Loading model: {args.model_name} (eager attention for output_attentions)")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Model: {num_layers} layers x {num_heads} heads = {num_layers * num_heads} total")

    all_scores = defaultdict(list)

    for i, inst in enumerate(test_instances):
        prompt = build_prompt(inst["context"], inst["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_input_tokens)
        input_ids = inputs["input_ids"]

        needle_positions = find_needle_token_positions(
            input_ids, tokenizer, inst["needle_sentence"]
        )

        if not needle_positions:
            print(f"  [{i+1}] WARNING: needle not found in tokenized input, skipping")
            continue

        copy_counts, total_gen = detect_rethead_single_instance(
            model, tokenizer, input_ids, needle_positions,
            num_layers, num_heads, max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )

        if total_gen == 0:
            continue

        for layer in range(num_layers):
            for head in range(num_heads):
                score = copy_counts.get((layer, head), 0) / total_gen
                all_scores[(layer, head)].append(score)

        if (i + 1) % 5 == 0 or i == 0:
            top_this = sorted(copy_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"L{l}H{h}={c}" for (l, h), c in top_this)
            print(f"  [{i+1}/{len(test_instances)}] "
                  f"needle_tokens={len(needle_positions)} "
                  f"gen_tokens={total_gen} "
                  f"top_copiers: {top_str}")

    head_scores = {}
    for (layer, head), scores in all_scores.items():
        head_scores[(layer, head)] = np.mean(scores)

    head_scores_list = [
        (f"{layer}-{head}", float(score))
        for (layer, head), score in head_scores.items()
    ]
    head_scores_list.sort(key=lambda x: x[1], reverse=True)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(head_scores_list, f, indent=4)

    print(f"\nResults saved to {args.output_file}")
    print(f"\nTop-16 RETHEAD heads:")
    for rank, (head_str, score) in enumerate(head_scores_list[:16]):
        layer, head = head_str.split("-")
        print(f"  {rank+1:2d}. layer {layer:>2s} head {head:>2s}  score={score:.6f}")

    top_layers = [int(h.split("-")[0]) for h, s in head_scores_list[:16]]
    print(f"\nLayer distribution (top-16): "
          f"early(0-3)={sum(1 for l in top_layers if l<=3)}, "
          f"middle(8-17)={sum(1 for l in top_layers if 8<=l<=17)}, "
          f"late(25+)={sum(1 for l in top_layers if l>=25)}")


if __name__ == "__main__":
    main()
