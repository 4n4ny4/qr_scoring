# QRScore: ICML 2026 Workshop Submission Repo

## What This Project Does

This branch is the cleaned submission repository for our ICML 2026 workshop experiments on **query-relevant attention heads** in long-context language models. It identifies which attention heads are responsible for retrieval, then tests that claim by knocking those heads out and measuring the accuracy drop.

The core idea: if a set of heads truly drives retrieval, zeroing them out at inference time should destroy the model's ability to answer questions about the document. By comparing head rankings from different data sources, we measure whether head importance is domain-specific or universal.

**Submission artifacts included:** `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`

**Code support retained:** `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-7b`, `allenai/OLMo-7B`

OLMo support remains in the codebase, but OLMo result artifacts are intentionally not bundled in this submission-cleanup branch because the completed outputs were not available in the locally discoverable committed refs.

### The Three Experiments

1. **Pooled ablation comparison** — Knock out top-K heads (from 3 different ranking sources) and measure accuracy on SEC extraction tasks. Shows domain-matched detection outperforms out-of-domain rankings.
2. **Cross-task transfer** — Knock out heads detected for task A and evaluate on tasks B–H. Reveals whether heads are task-specific or broadly shared.
3. **Head similarity** — Jaccard overlap between per-task head rankings. Identifies functional head clusters for semantically related tasks.

### The Eight SEC Tasks

Each test instance presents a long SEC 10-K filing and asks the model to extract one fact:

| Task | Example Answer | Train (Detection) | Test (NIAH) |
|------|---------------|-------------------|-------------|
| `registrant_name` | "Vishay Intertechnology, Inc." | 149 | 41 |
| `headquarters_city` | "Malvern" | 110 | 28 |
| `headquarters_state` | "Pennsylvania" | 106 | 27 |
| `incorporation_state` | "Delaware" | 123 | 32 |
| `incorporation_year` | "1962" | 124 | 30 |
| `employees_count_total` | "25,600" | 118 | 24 |
| `ceo_lastname` | "Zandman" | 123 | 35 |
| `holder_record_amount` | "7,543" | 121 | 24 |
| **Total** | | **974** | **241** |

- **Train (Detection)** — instances in `data/long_context_detection_optionA/`, built by `build_detection_data.py` from `train_plan.csv`. Each instance concatenates sections from a single SEC filing into a long context with chunked paragraphs, used in Step 5 to score all 1024 heads. Some raw instances are dropped during build (e.g. needle not found in chunked context).
- **Test (NIAH)** — instances in `data/niah_input/*_test.json`, built by `build_niah_data.py` from `test_plan.csv`. Each instance inserts a needle sentence into a haystack of distractor filing sections. Used for all ablation experiments. The evaluation caps at 24 per task (`--max_instances_per_task 24`) for balanced comparison, giving 192 evaluated instances.

---

## The Three Head Ranking Sources

The experiments compare head rankings produced from three different data domains. All three use the same QRScore detection algorithm (attention-based scoring of gold vs. distractor passages) but on different training data:

### 1. QRScore-SEC (in-domain)

**Source data:** SEC 10-K filings from the training split of this project's dataset.

**How it's built:** `scripts/detection/detect_qrhead.py` processes the training instances in `data/long_context_detection_optionA/`. For each instance, the model reads a long document (~5K–30K tokens) containing a gold passage (with the answer) and distractor passages (from the same filing). The script calls `score_docs_per_head_for_detection()` which computes a per-head retrieval score: how much each individual head's attention to the query tokens helps rank the gold passage above distractors. Heads are ranked by their aggregated QRScore across all training instances.

**Files:**
- Per-task: `results/detection/<model_slug>/long_context_{task}_heads.json`
- Combined (pooled across all tasks): `results/detection/<model_slug>/long_context_combined_heads.json`

**Why it matters:** This is the in-domain ranking. If QRScore works, these heads should be the most damaging to knock out on SEC test data.

### 2. QRScore-8B-LME-TRAIN (cross-domain, legal/manual)

**Source data:** LM-Eval (LME) benchmark passages — a mix of general-purpose reading comprehension and language modeling evaluation tasks. These are **not** SEC filings; they include diverse text genres.

**How it's built:** The same QRScore detection algorithm was run on LME benchmark data using Llama-3.1-8B-Instruct. The resulting head ranking was saved externally and is provided as a pre-computed file.

**File:** `Llama-3.1-8B-Instruct/lme_TRAIN.json`

**Why it matters:** Tests whether heads important for general-purpose retrieval also matter for SEC document extraction. If they do, it suggests a universal retrieval mechanism; if not, it suggests domain-specific head circuits.

### 3. QRScore-8B-NQ-TRAIN (cross-domain, open QA)

**Source data:** Natural Questions (NQ) — Google's open-domain question-answering dataset. Passages are Wikipedia articles; questions are real user queries from Google Search. Very different from SEC filings in both style and content.

**How it's built:** Same QRScore detection algorithm, run on NQ training data using Llama-3.1-8B-Instruct. Pre-computed externally.

**File:** `Llama-3.1-8B-Instruct/nq_TRAIN.json`

**Why it matters:** NQ is the most distant domain from SEC filings. If NQ-detected heads barely affect SEC performance, it's strong evidence that QRScore detects task-relevant heads rather than generic attention patterns.

---

## Repository Structure

```
qr_scoring/
├── data/
│   ├── haystack_plan.csv                  # All SEC instances (pre-split)
│   ├── train_plan.csv                     # Training split (80% by filing)
│   ├── test_plan.csv                      # Test split (20% by filing)
│   ├── needles.csv                        # Extracted facts per SEC filing
│   ├── sections.csv                       # Section-level text from filings
│   ├── long_context_detection_optionA/    # Training detection instances (generated)
│   └── niah_input/                        # Test NIAH instances (*_test.json, from test_plan.csv)
├── Llama-3.1-8B-Instruct/
│   ├── lme_TRAIN.json                     # External LME head ranking
│   └── nq_TRAIN.json                      # External NQ head ranking
├── results/
│   ├── detection/
│   │   ├── <model_slug>/                  # Canonical per-model rankings + top-K exports
│   ├── comparison_ablation/
│   │   └── <model_slug>/                  # Canonical per-model ablation outputs
│   │       ├── *_results.json             # Per-method accuracy curves
│   │       ├── cross_task_*.json          # Transfer matrix + specificity
│   │       ├── *.png                      # Plots
│   │       ├── *.csv                      # Summary tables
│   │       └── experiment_manifest.json   # Provenance + labeling metadata
│   └── cross_ablation_index.json          # Repo-level index of collected workshop artifacts
├── scripts/
│   ├── data_prep/
│   │   ├── split_dataset.py               # 80/20 train/test split
│   │   ├── build_detection_data.py        # Build training detection instances
│   │   └── build_niah_data.py             # Build test NIAH instances
│   ├── detection/
│   │   ├── detect_qrhead.py               # Score all 1024 heads
│   │   └── run_detection.sh               # Wrapper for per-task + combined
│   └── evaluation/
│       ├── run_ablation.py                # Main ablation + transfer + specificity
│       ├── plot_ablation.py               # Accuracy curves/heatmaps/tables
│       ├── plot_transfer.py               # Transfer heatmaps/similarity/specificity
│       └── verify_leakage.py              # Confirm no train/test leakage
├── src/qrretriever/                       # Core package
│   ├── attn_retriever.py                  # AttnBasedRetriever, QRRetriever
│   ├── custom_modeling_llama.py           # Custom LlamaForCausalLM (detection only)
│   ├── custom_cache.py                    # DynamicCacheWithQuery
│   ├── config.py                          # YAML config loader
│   ├── predefined_heads.py               # Hardcoded head sets
│   └── configs/                           # YAML configs per dataset
├── examples/
│   └── qrretriever_example.py             # Basic retriever usage
└── setup.py                               # Package metadata (Apache 2.0)
```

---

## Setup

**Requirements:** Python ≥ 3.9, CUDA GPU, HuggingFace access to the supported checkpoints you want to run.

```bash
pip install -e .
```

Dependencies: `torch`, `transformers>=4.44.0,<5.0.0`, `flash_attn`, `jinja2>=3.1.0`, `pyyaml>=5.1`, `tqdm`, `Pillow>=9.1.0`

Gemma and OLMo use stock Hugging Face model loading in this repo. If model import fails before download, check your local `torch` / `torchvision` install first; the scripts now preflight that runtime and surface a targeted error.

For plotting: `pip install matplotlib pandas`

---

## Workshop Artifact Collection

If you have completed cross-ablation experiments spread across branches, collect
them into the canonical per-model layout with:

```bash
python scripts/evaluation/collect_cross_ablation_experiments.py
```

This imports the locally available historical results into:

- `results/detection/meta-llama__Llama-3.1-8B-Instruct/`
- `results/detection/Qwen__Qwen2.5-7B-Instruct/`
- `results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct/`
- `results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct/`

and writes `results/cross_ablation_index.json`, which labels each experiment by
model family, source branch, status, and available methods. OLMo is listed in
the index as excluded from the bundled submission artifacts, rather than being
presented as a completed included result set.

## Running the Full Pipeline

### Step 1: Split Data

Splits `data/haystack_plan.csv` into 80% train / 20% test by unique SEC filing filename (document-level split prevents leakage).

```bash
python scripts/data_prep/split_dataset.py
```

**Output:** `data/train_plan.csv`, `data/test_plan.csv`

### Step 2: Build Detection (Training) Data

Constructs long-context detection instances from the training split. For each instance, combines the gold section (containing the answer) with distractor sections from the same filing, chunked into ~400-word paragraphs.

```bash
python scripts/data_prep/build_detection_data.py --chunk_words 400
```

**Output:**
- `data/long_context_detection_optionA/{task}_detection.json` (one per task)
- `data/long_context_detection_optionA/combined_detection.json` (all tasks pooled)

### Step 3: Build NIAH Evaluation (Test) Data

Constructs needle-in-a-haystack test instances from the test split. Each instance is a full SEC filing with the answer sentence embedded in context.

```bash
python scripts/data_prep/build_niah_data.py --chunk_words 400
```

**Output:** `data/niah_input/{task}_test.json`

### Step 4: Verify No Leakage

Confirms zero overlap between SEC filings used in train detection and test evaluation.

```bash
python scripts/evaluation/verify_leakage.py
```

### Step 5: Run Head Detection (Training)

Scores all 1024 heads on the training data. For each head, computes how well it ranks gold passages above distractors. Produces a ranked list of heads by QRScore.

```bash
bash scripts/detection/run_detection.sh
```

Gemma example:

```bash
MODEL_NAME=google/gemma-7b \
bash scripts/detection/run_detection.sh --combined-only
```

OLMo example:

```bash
MODEL_NAME=allenai/OLMo-7B \
TRUST_REMOTE_CODE=1 \
bash scripts/detection/run_detection.sh --combined-only
```

This runs detection on all 8 per-task files and the combined file. Use `--combined-only` to skip per-task detection.

**Output:**
- `results/detection/<model_slug>/long_context_combined_heads.json`
- `results/detection/<model_slug>/long_context_{task}_heads.json` (8 files)

Qwen example:

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
TRUNCATE_BY_SPACE=20 \
bash scripts/detection/run_detection.sh --combined-only
```

### Step 6: Pooled Ablation Comparison

Compares all three ranking sources by knocking out their top-K heads and measuring accuracy on the test set:

```bash
python scripts/evaluation/run_ablation.py \
  --niah_dir data/niah_input \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --max_instances_per_task 24 \
  --max_context_tokens 8192 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --progress_every 20 \
  --log_tokens \
  --methods QRScore-SEC QRScore-8B-LME-TRAIN QRScore-8B-NQ-TRAIN
```

**Output:** `{method}_results.json`, `comparison_summary.json`

Gemma example:

```bash
python scripts/evaluation/run_ablation.py \
  --model_name google/gemma-7b \
  --ranking_dir results/detection/google__gemma-7b \
  --max_instances_per_task 24 \
  --methods QRScore-SEC
```

OLMo example:

```bash
python scripts/evaluation/run_ablation.py \
  --model_name allenai/OLMo-7B \
  --trust_remote_code \
  --ranking_dir results/detection/allenai__OLMo-7B \
  --max_instances_per_task 24 \
  --methods QRScore-SEC
```

Qwen example:

```bash
python scripts/evaluation/run_ablation.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --ranking_dir results/detection/Qwen__Qwen2.5-7B-Instruct \
  --max_instances_per_task 24 \
  --methods QRScore-SEC QRScore-8B-LME-TRAIN QRScore-8B-NQ-TRAIN
```

For non-Llama models, `LME` / `NQ` comparisons should use same-model rankings.
For Qwen, place native train rankings under the model-specific detection layout
or a Qwen model folder, for example:

- `results/detection/Qwen__Qwen2.5-7B-Instruct/lme_train_heads.json`
- `results/detection/Qwen__Qwen2.5-7B-Instruct/nq_train_heads.json`
- `Qwen-2.5-7B-Instruct/lme_TRAIN_qwen.json`
- `Qwen-2.5-7B-Instruct/nq_TRAIN_qwen.json`

The ablation script intentionally does not fall back to Llama train rankings
for Qwen, because raw head indices are not comparable across model families.

### Step 7: Cross-Task Transfer Ablation

Runs an 8×8 source-task × target-task ablation matrix using per-task SEC head rankings:

```bash
python scripts/evaluation/run_ablation.py \
  --niah_dir data/niah_input \
  --output_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct \
  --max_instances_per_task 24 \
  --max_context_tokens 8192 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --progress_every 20 \
  --log_tokens \
  --enable_cross_task_transfer \
  --transfer_summary_k 16
```

To add a same-model LME source row to the transfer matrix, include:

```bash
  --transfer_extra_sources QRScore-8B-LME-TRAIN
```

For Qwen, this resolves to `Qwen-2.5-7B-Instruct/lme_TRAIN_qwen.json` by
default when that file is present. The resulting matrix has the usual SEC
target-task columns plus an additional LME source row.

To run only a same-model Qwen NQ source row against all SEC target tasks, use
`--transfer_extra_sources QRScore-8B-NQ-TRAIN --transfer_only_extra_sources`.
For Qwen, this resolves to `Qwen-2.5-7B-Instruct/nq_TRAIN_qwen.json` by
default when that file is present.

**Output:**
- `cross_task_transfer_matrix.json` — accuracy drop for each (source, target, K) triple
- `cross_task_specificity_metrics.json` — on-target drop, off-target mean drop, specificity index
- `cross_task_head_similarity_topk.json` — Jaccard overlap between per-task head sets at each K

### Step 8: Reverse-Transfer NQ Ablation

Tests whether SEC-derived heads causally matter on Natural Questions by
evaluating NQ answer generation while ablating Qwen SEC head rankings:

**Dataset source:** Prefer the original/simplified Natural Questions files from
Google Research (`https://ai.google.com/research/NaturalQuestions/dataset`) so
the evaluation has real short-answer gold. The builder accepts local
JSON/JSONL/JSONL.GZ records with NQ-style `annotations.short_answers`.

```bash
python scripts/data_prep/build_nq_eval_data.py \
  --input_file /path/to/simplified-nq-dev.jsonl.gz \
  --output_file data/nq_input/nq_test.json \
  --max_instances 512 \
  --shuffle
```

For a Hugging Face smoke test, install `datasets` and use a short-answer NQ
dataset when available. Passage-pair datasets such as
`sentence-transformers/natural-questions`
(`https://huggingface.co/datasets/sentence-transformers/natural-questions`) can
be used with `--allow_passage_answer`, but they are weaker scientific controls
because they do not expose true short-answer gold.

```bash
python scripts/evaluation/run_nq_reverse_ablation.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --nq_file data/nq_input/nq_test.json \
  --max_instances 512 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --progress_every 20
```

**Output:** `results/nq_reverse_ablation/<model_slug>/` containing
`*_results.json`, `comparison_summary.json`, `accuracy_table.csv`,
`drop_from_baseline_table.csv`, and `accuracy_vs_knockout.png`.

The key metric is `Drop@K = accuracy(K=0) - accuracy(K)`. A large NQ drop from
SEC category heads supports reverse transfer; a small drop suggests the
category heads are not general NQ recall heads.

### Step 9: Generate Plots and Tables

```bash
python scripts/evaluation/plot_ablation.py \
  --results_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct \
  --output_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct

python scripts/evaluation/plot_transfer.py \
  --results_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct \
  --output_dir results/comparison_ablation/meta-llama__Llama-3.1-8B-Instruct
```

**Output:** The plots and tables are written into the selected per-model ablation directory.

To compute cross-dataset head overlap, including the SEC--NQ Jaccard analysis
used to explain cross-dataset transfer, run:

```bash
python scripts/evaluation/compute_cross_method_overlap.py --allow-missing
```

By default this discovers available per-model detection directories, uses only
same-model ranking files such as `nq_train_heads.json`, and writes
`results/comparison_ablation/cross_model_head_overlap.{json,csv}` plus
per-model `cross_method_head_overlap.json` files. Models without same-model NQ
rankings are reported as missing rather than silently using Llama-derived heads.
To check a specific set of models as their SEC or NQ rankings become available:

```bash
python scripts/evaluation/compute_cross_method_overlap.py \
  --model-slugs Qwen__Qwen2.5-7B-Instruct allenai__OLMo-7B mistralai__Mistral-7B-Instruct-v0.3 \
  --pairs SEC:NQ \
  --allow-missing
```

For same-model SEC--NQ Jaccard, each model must have both files below:

- `results/detection/<model_slug>/long_context_combined_heads.json` — SEC heads
- `results/detection/<model_slug>/nq_train_heads.json` — same-model NQ heads

Qwen is ready to evaluate because its NQ heads are available from the
PrincetonPLI QRHead release:

```bash
python scripts/evaluation/compute_cross_method_overlap.py \
  --model-slugs Qwen__Qwen2.5-7B-Instruct \
  --pairs SEC:NQ \
  --allow-missing
```

OLMo and Mistral are reported as missing until same-model SEC and NQ rankings
exist. For OLMo SEC detection, run:

```bash
MODEL_NAME=allenai/OLMo-7B \
TRUST_REMOTE_CODE=1 \
bash scripts/detection/run_detection.sh --combined-only
```

Then run NQ detection against an NQ detection file in the same JSON format used
by `detect_qrhead.py` (`idx`, `question`, `paragraphs`, `gt_docs`) and write the
ranking to `results/detection/allenai__OLMo-7B/nq_train_heads.json`. Mistral
requires adding model-runtime support before running the same SEC/NQ detection
pattern.

If you intentionally want the legacy transferred/external rankings, for example
to label a Llama-NQ transferred-head baseline, opt in explicitly:

```bash
python scripts/evaluation/compute_cross_method_overlap.py \
  --pairs SEC:NQ \
  --include-transferred-external \
  --allow-missing
```

### Smoke Test (Quick Validation)

Run a minimal version to verify the pipeline works before committing to a full run:

```bash
python scripts/evaluation/run_ablation.py \
  --max_instances_per_task 4 \
  --knockout_sizes 0 8 16 \
  --max_context_tokens 4096 \
  --log_tokens
```

---

## How the Ablation Works

The ablation script loads the **stock** `transformers.LlamaForCausalLM` (no custom model needed) and installs lightweight `forward_pre_hook` functions on each layer's `o_proj` projection. When a head mask is active, the hook zeroes out the output dimensions corresponding to the masked heads before the projection is applied. This is equivalent to removing those heads' contribution to the residual stream.

For each K in `--knockout_sizes`:
1. Mask the top-K heads from the ranking
2. Run generation on all test instances
3. Extract a short answer from the model's output (first sentence)
4. Compare against the gold answer (normalised substring match)
5. Compute accuracy overall and per-task

The custom model in `src/qrretriever/custom_modeling_llama.py` is only used by the **detection** pipeline (Step 5), which needs per-head attention score access.

---

## Key CLI Arguments for `run_ablation.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--niah_dir` | `data/niah_input` | Directory with test JSON files |
| `--output_dir` | `results/comparison_ablation/<model_slug>` | Where to write results |
| `--model_name` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model |
| `--knockout_sizes` | `0 8 16 32 48 64 96 128` | Number of heads to knock out |
| `--max_instances_per_task` | all | Cap instances per task (use 24 for balanced comparison) |
| `--max_context_tokens` | `8192` | Max prompt tokens (left-truncation if exceeded) |
| `--methods` | all detected | Which ranking methods to evaluate |
| `--enable_cross_task_transfer` | off | Run the 8×8 transfer matrix |
| `--transfer_extra_sources` | none | Add method rankings, e.g. `QRScore-8B-LME-TRAIN`, as extra transfer source rows |
| `--transfer_only_extra_sources` | off | Run transfer only for extra source rows, skipping SEC per-task sources |
| `--transfer_summary_k` | `16` | K value for specificity metric computation |
| `--include_random_baselines` | off | Include random head rankings as control |
| `--log_tokens` | off | Write per-instance JSONL token logs |
| `--progress_every` | `20` | Print progress every N instances |

---

## Interpreting the Results

### Output Files

| File | Contents |
|------|----------|
| `{method}_results.json` | Full accuracy curves, per-task breakdowns, per-instance details |
| `comparison_summary.json` | Baseline accuracy, K=16 accuracy, and drop for each method |
| `cross_task_transfer_matrix.json` | Source × target × K matrix: (source ranking, target task, K) → accuracy + drop |
| `cross_task_specificity_metrics.json` | Per-source: on-target drop, off-target mean, specificity index, surgicality ratio |
| `cross_task_head_similarity_topk.json` | Jaccard similarity matrices at each top-K |
| `accuracy_vs_knockout.png` | Overall accuracy curves (3 methods compared) |
| `per_task_accuracy_curves.png` | 8 subplots showing per-task degradation |
| `per_task_heatmaps.png` | Tasks × K heatmaps annotated with accuracy % |
| `transfer_drop_heatmap_K{k}.png` | Source × Target drop heatmaps (one per K) |
| `head_similarity_heatmaps.png` | Jaccard similarity panels at each top-K |
| `specificity_bars.png` | On-target vs off-target drop bar chart |
| `accuracy_table.csv` | Method × Task × K accuracy matrix |
| `drop_from_baseline_table.csv` | Drop from K=0 baseline for each cell |
| `specificity_table.csv` | Per-task specificity index and surgicality ratio |
| `experiment_manifest.json` | Provenance, inclusion status, and labeling metadata for the bundled run |

### Key Metrics

- **Drop@K** = accuracy(K=0) − accuracy(K). How much accuracy falls when K heads are removed.
- **Specificity Index** = on-target drop − off-target mean drop. Positive means the ablation is task-specific; negative means it causes more collateral damage to other tasks.
- **Surgicality Ratio** = on-target drop / off-target mean drop. >1 means surgical; <1 means broad.
- **Jaccard Similarity** = |intersection| / |union| of two tasks' top-K head sets. 1 = identical heads, 0 = no overlap.

Use the per-model `comparison_summary.json`, transfer metrics, and figures inside `results/comparison_ablation/<model_slug>/` as the canonical submission artifacts.

---

## Practical Notes

- Full runs are GPU-heavy. Start with `--max_instances_per_task 4` and `--knockout_sizes 0 8 16` for smoke tests.
- Always include `K=0` in `--knockout_sizes` so all drop metrics have a baseline anchor.
- Use `--max_context_tokens 8192` for final metrics (NIAH contexts are typically >4K tokens).
- `--log_tokens` writes per-method JSONL token logs with `raw_text` and `token_ids` for post-hoc analysis (e.g. `QRScore-SEC_token_log.jsonl`).

## Troubleshooting

If you see:
```
FileNotFoundError: ... data/long_context_detection_optionA/combined_detection.json
```
Run Steps 1–3 first:
```bash
python scripts/data_prep/split_dataset.py
python scripts/data_prep/build_detection_data.py --chunk_words 400
python scripts/data_prep/build_niah_data.py --chunk_words 400
```

If detection files are missing (`long_context_*_heads.json`), run Step 5:
```bash
bash scripts/detection/run_detection.sh
```

## Quick Smoke Test

```bash
python scripts/evaluation/run_ablation.py \
  --niah_dir data/niah_input \
  --output_dir results/comparison_ablation \
  --max_instances_per_task 5 \
  --max_context_tokens 8192 \
  --knockout_sizes 0 16 \
  --progress_every 5 \
  --log_tokens \
  --methods QRScore-SEC QRScore-8B-LME-TRAIN
```
