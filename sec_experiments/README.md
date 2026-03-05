# SEC 10-K QR Head Detection and Ablation

This directory contains our experiment scripts for detecting query-focused retrieval heads (QRHeads) on SEC 10-K filings and evaluating them through ablation experiments.

These scripts build on the QRHead library (`src/qrretriever/`) and detection framework (`exp_scripts/`) from [Zhang et al. (EMNLP 2025)](https://arxiv.org/pdf/2506.09944). See the [repo separation](#repo-structure) section below for the boundary between their code and ours.

## Tasks (8 total)

| Task | Question |
|------|----------|
| `registrant_name` | What is the registrant name? |
| `headquarters_city` | What is the headquarters city? |
| `headquarters_state` | What is the headquarters state? |
| `incorporation_state` | What is the incorporation state? |
| `incorporation_year` | What is the incorporation year? |
| `employees_count_total` | What is the total employee count? |
| `ceo_lastname` | What is the CEO's last name? |
| `holder_record_amount` | What is the holder record amount? |

## Pipeline

```
haystack_plan.csv + sections.csv
        |
        v
[1] convert_haystack_to_json.py        (local)
        |
        +--> {task}_train.json (80%)
        |         |
        |         v
        |    [2] run_all_detection.sh    (GPU)
        |         |
        |         v
        |    {task}_heads.json
        |         |
        +--> {task}_test.json  (20%)    |
                  |                     |
                  v                     v
             [3] run_ablation.sh        (GPU)
                  |
                  v
             ablation_summary.json
```

### Stage 1: Data Conversion (local, no GPU)

```bash
python sec_experiments/convert_haystack_to_json.py
```

Reads `data/haystack_plan.csv` and `data/sections.csv` (from [c3po-ai/edgar-corpus](https://huggingface.co/datasets/c3po-ai/edgar-corpus)). For each instance:
- Extracts a 400-word window from the gold section centered around the needle sentence, trimmed to start at a sentence boundary
- Collects all other sections from the same filing as distractors (>= 50 words, truncated to 400 words)
- Shuffles document order randomly
- Splits 80/20 into train and test sets

**Output:** `data/detection_input/{task}_train.json` and `{task}_test.json`

### Stage 2: QR Head Detection (GPU required)

```bash
bash sec_experiments/run_all_detection.sh
```

Runs `exp_scripts/detection/detect_qrhead_lme.py` on each task's training split. Computes QRScore for all 1024 attention heads (32 layers x 32 heads) in Llama-3.1-8B-Instruct.

**Output:** `results/detection/{task}_heads.json`

### Stage 3: Ablation Evaluation (GPU required)

```bash
bash sec_experiments/run_ablation.sh
```

Validates detected heads by comparing three retrieval conditions on the held-out test split:

| Condition | Head set | Expected result |
|-----------|----------|-----------------|
| `full_head` | All 1024 heads | Baseline |
| `qr_head_top16` | Top-16 detected QR heads | Best (signal without noise) |
| `random_head` | 16 randomly selected heads | Worst (no retrieval signal) |

Computes Recall@1 and Recall@3 with 95% bootstrap confidence intervals and paired significance tests.

**Output:** `results/ablation/ablation_summary.json`

### Cross-Task Analysis (CPU, after detection)

```bash
python sec_experiments/compare_heads_across_tasks.py
```

Compares detected heads across tasks: Jaccard overlap, Spearman correlation, universal head identification, heatmap visualization.

**Output:** `results/detection/cross_task_summary.json` and `head_comparison_heatmap.png`

## Script Reference

| Script | Purpose |
|--------|---------|
| `convert_haystack_to_json.py` | Convert raw CSV data to JSON with 80/20 split |
| `run_all_detection.sh` | Run QR head detection on all 8 tasks |
| `run_ablation.sh` | Run full ablation pipeline |
| `generate_ablation_configs.py` | Create YAML configs for ablation conditions |
| `eval_ablation.py` | Compute Recall metrics, bootstrap CIs, significance tests |
| `compare_heads_across_tasks.py` | Cross-task head overlap and correlation analysis |

## Repo Structure

```
QRHead/
│
├── # ──── Original Zhang et al. (EMNLP 2025) ────
│   src/qrretriever/             # Core library (attn_retriever, configs, custom modeling)
│   exp_scripts/                 # Their experiment scripts (detection, retrieval, generation)
│   setup.py                     # Package installer
│   README.md                    # Original repo README
│
├── # ──── Our SEC 10-K Experiments ────
│   sec_experiments/             # Our scripts (this directory)
│   data/
│     haystack_plan.csv          # Task definitions
│     sections.csv               # Full section text per filing (from edgar-corpus)
│     needles.csv                # Needle metadata
│     detection_input/           # Generated train/test JSON files
│   results/
│     detection/                 # Head rankings per task
│     ablation/                  # Retrieval scores and evaluation summary
```

## Requirements

- Python >= 3.9, PyTorch with CUDA, `transformers >= 4.44.0`, `flash_attn`
- `numpy`, `matplotlib`
- HuggingFace access to `meta-llama/Llama-3.1-8B-Instruct`

```bash
pip install -e .
pip install numpy matplotlib
huggingface-cli login
```
