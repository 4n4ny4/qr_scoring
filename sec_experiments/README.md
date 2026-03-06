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

#### Interpreting detection results

- **QRScore direction:** **Higher is better.** QRScore is the mean (over instances) of the sum of retrieval scores over gold documents for that head. Heads are ranked by this score in descending order. Values are often small and negative (e.g. around -1e-4); the *least negative* (closest to zero or positive) heads are the best for retrieval on that task.
- **Top-20 heads:** Each task’s `*_heads.json` lists all 1024 heads sorted best-first. The top 20 are the strongest QR heads for that task. The plots in `top20_heads_plot.png` show these per task.
- **Overlap:** The task×task heatmap and “heads in multiple tasks” chart in `head_overlap_plot.png` show which tasks share retrieval heads. High overlap suggests similar retrieval mechanisms; little overlap suggests task-specific heads.
- **Layer patterns:** Early layers (e.g. 0–2) and late layers (e.g. 28–31) often appear in top-20 lists; the exact mix is task-dependent and can suggest where the model does query–document matching.

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

**Output:** `results/ablation/ablation_summary.json` plus recall figures:
- `recall_per_task_recall1.png`, `recall_per_task_recall3.png` — bar charts by task (3 conditions per task)
- `recall_pooled_recall1.png`, `recall_pooled_recall3.png` — pooled bar chart per metric
- `recall_pooled_both.png` — Recall@1 and Recall@3 pooled side by side

### Cross-Task Analysis (CPU, after detection)

```bash
python sec_experiments/compare_heads_across_tasks.py
```

Compares detected heads across tasks: Jaccard overlap, Spearman correlation, universal head identification, heatmap visualization.

**Output:** `results/detection/cross_task_summary.json` and `head_comparison_heatmap.png`

### Answer-Accuracy Knockout Ablation (GPU for retrieval + generation)

This experiment measures whether **removing** the top-16 QR heads from retrieval (retrieving with the remaining 1008 heads only) **reduces** the model’s ability to output the correct one-word answer. Metric: answer accuracy (generated answer vs `needle_value`) on the test set, with paired bootstrap (full vs knockout).

**Prerequisites:** Train/test JSONs must include an `answer` field (re-run `convert_haystack_to_json.py`; it now adds `answer` from `needle_value`). Detection and full-head ablation should be done (`run_ablation.sh`).

1. **Generate knockout configs** (1008 heads = all minus top-16 per task):

   ```bash
   python sec_experiments/generate_ablation_configs.py \
     --detection_dir results/detection --config_dir results/ablation/configs \
     --tasks registrant_name headquarters_city headquarters_state incorporation_state \
            incorporation_year employees_count_total ceo_lastname holder_record_amount
   ```

   This creates `results/ablation/configs/{task}_knockout_top16.yaml` in addition to the existing QR and random configs.

2. **Run retrieval with knockout config** on the test set (GPU):

   ```bash
   bash sec_experiments/run_knockout_retrieval.sh
   ```

   Requires `PYTHONPATH=src` or `pip install -e .`. Writes `results/ablation/{task}_knockout_top16.json` per task.

3. **Run answer ablation** (loads test JSONs + full-head and knockout retrieval, takes top-1 doc per condition, prompts LM for one-word answer, compares to gold, computes accuracy and paired bootstrap):

   ```bash
   python sec_experiments/run_answer_ablation.py \
     --data_dir data/detection_input --ablation_dir results/ablation
   ```

   Uses `meta-llama/Llama-3.1-8B-Instruct` by default. With `--skip_generation`, skips LM calls and writes placeholder accuracies (useful if you only have retrieval results and want to test the eval pipeline).

**Output:**
- `results/ablation/answer_accuracy_summary.json` — per-task and pooled accuracy (full vs knockout), difference, 95% CI, and p-value (full better than knockout).
- `results/ablation/answer_accuracy_knockout.png` — bar chart of answer accuracy (full vs knockout) per task and pooled.

## Script Reference

| Script | Purpose |
|--------|---------|
| `convert_haystack_to_json.py` | Convert raw CSV data to JSON with 80/20 split; includes `answer` (needle_value) |
| `run_all_detection.sh` | Run QR head detection on all 8 tasks |
| `run_ablation.sh` | Run full ablation pipeline (full_head, qr_head_top16, random_head) |
| `generate_ablation_configs.py` | Create YAML configs (QR top-16, random 16, knockout 1008 heads) |
| `run_knockout_retrieval.sh` | Run retrieval on test set with knockout config per task |
| `run_answer_ablation.py` | Top-1 doc per condition → LM one-word answer → accuracy + paired bootstrap |
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
