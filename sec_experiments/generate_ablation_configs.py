"""
Generate YAML config files for ablation experiments.

For each task, reads the detection results and creates:
  - {task}_qr_head_top16.yaml: config using top-16 detected QR heads
  - {task}_random_head.yaml: config using 16 randomly selected heads
"""

import argparse
import json
import os
import random

MODEL_BASE_CLASS = "Llama-3.1-8B-Instruct"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
NUM_LAYERS = 32
NUM_HEADS = 32
RANDOM_SEED = 123


def write_yaml(path, attn_head_set):
    with open(path, "w") as f:
        f.write(f'attn_head_set: "{attn_head_set}"\n')
        f.write(f'model_base_class: "{MODEL_BASE_CLASS}"\n')
        f.write(f'model_name_or_path: "{MODEL_NAME_OR_PATH}"\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_dir", required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    os.makedirs(args.config_dir, exist_ok=True)
    random.seed(RANDOM_SEED)

    all_heads = [f"{l}-{h}" for l in range(NUM_LAYERS) for h in range(NUM_HEADS)]

    for task in args.tasks:
        heads_file = os.path.join(args.detection_dir, f"{task}_heads.json")
        if not os.path.exists(heads_file):
            print(f"WARNING: {heads_file} not found, skipping {task}")
            continue

        with open(heads_file) as f:
            head_scores = json.load(f)

        top_k_heads = [head_id for head_id, _ in head_scores[:args.top_k]]
        qr_head_set = ",".join(top_k_heads)

        qr_path = os.path.join(args.config_dir, f"{task}_qr_head_top16.yaml")
        write_yaml(qr_path, qr_head_set)
        print(f"  {task} QR top-{args.top_k}: {qr_head_set}")

        random_heads = random.sample(all_heads, args.top_k)
        random_head_set = ",".join(random_heads)

        rand_path = os.path.join(args.config_dir, f"{task}_random_head.yaml")
        write_yaml(rand_path, random_head_set)
        print(f"  {task} random-{args.top_k}: {random_head_set}")


if __name__ == "__main__":
    main()
