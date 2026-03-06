## SEC QRScore vs QRHead Experiments (Minimal Repo)

This repository is a **trimmed-down copy** of the original QRRetriever project, focused on a single set of experiments:

- **Long-context QRScore detection on SEC 10-K filings (Option A)** to obtain SEC-specific retrieval heads (QRScore-SEC).
- **Generation-side head ablation** (Part B in the QRHead paper): compare how much answer accuracy drops when knocking out heads ranked by:
  - **QRScore-SEC** (SEC-specific detection),
  - **QRScore-Paper-LME** (heads from LongMemEval),
  - **QRScore-Paper-NQ** (heads from NQ),
  - **Random** (random head sets).

---

### 1. Environment

- Python 3.10+
- GPU with enough memory for `meta-llama/Llama-3.1-8B-Instruct`
- Key packages:
  - `torch`
  - `transformers` (4.44–4.48 recommended)
  - `flash_attn`
  - `matplotlib` (for plotting)

Install the project in editable mode:

```bash
pip install -e .
```

Make sure your Hugging Face credentials (if needed) are set so `meta-llama/Llama-3.1-8B-Instruct` can be loaded.

---

### 2. Data layout (kept for these experiments)

- `data/sections.csv`, `data/needles.csv`, `data/haystack_plan.csv`  
Source SEC metadata used to build:
  - NIAH-style evaluation data, and
  - Option A long-context detection inputs.
- `data/long_context_detection_optionA/`  
**Option A long-context detection data** (per-task and combined JSON) built by:
  - `sec_experiments/build_long_context_detection_optionA.py`
- `data/niah_input/`  
**NIAH-style SEC tasks** built by:
  - `sec_experiments/build_niah_data.py`  
  Contains `{task}_train.json` and `{task}_test.json`.

---

### 3. Scripts and configs that matter

- **Detection (QRScore-SEC on Option A)**
  - `sec_experiments/build_long_context_detection_optionA.py`  
  Builds `data/long_context_detection_optionA/{task}_detection.json` and `combined_detection.json` from `sections.csv` and `haystack_plan.csv`.
  - `sec_experiments/run_long_context_detection.sh`  
  Wraps `exp_scripts/detection/detect_qrhead_lme.py` to run QRScore detection on the long-context SEC data and produce:
    - `results/detection/long_context_combined_heads.json` (your **QRScore-SEC** ranking).
- **NIAH generation ablation (Part B)**
  - `sec_experiments/build_niah_data.py`  
  Builds `data/niah_input/{task}_{train,test}.json` from `haystack_plan.csv`.
  - `sec_experiments/run_comparison_ablation.py`  
  Main script that:
    - Loads head rankings for:
      - `QRScore-SEC` from `results/detection/long_context_combined_heads.json`,
      - `QRScore-Paper-LME` from `src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml`,
      - `QRScore-Paper-NQ` from `src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_NQ.yaml`,
      - `Random-seed{42,123,456}` from internal sampling.
    - For each method and each knockout size `K`, masks the top-`K` heads during generation on NIAH test instances.
    - Records answer accuracy.
  - `sec_experiments/plot_comparison_ablation.py`  
  Reads `results/comparison_ablation/*_results.json` and produces:
    - `results/comparison_ablation/accuracy_vs_knockout.png`
    - (and potentially other summary plots).
- **Model + QRHead configs**
  - `src/qrretriever/custom_modeling_llama.py`  
  LLaMA model wrapper with support for head masking.
  - `src/qrretriever/attn_retriever.py`, `src/qrretriever/custom_cache.py`, `src/qrretriever/config.py`  
  Core QRRetriever internals used by detection scripts.
  - `src/qrretriever/configs/`  
    - `Llama-3.1-8B-Instruct_full_head.yaml` (full-head detection config),
    - `Llama-3.1-8B-Instruct_qr_head_LME.yaml` (paper LME heads),
    - `Llama-3.1-8B-Instruct_qr_head_NQ.yaml` (paper NQ heads),
    - `Llama-3.1-8B-Instruct_qr_head_SEC.yaml` (optional: SEC heads, if you saved them).

---

### 4. Reproducing experiments

#### 4.1 Build NIAH data (optional if already present)

```bash
python sec_experiments/build_niah_data.py \
  --max_instances_per_task 200 \
  --chunk_words 400
```

This reads `data/haystack_plan.csv` and writes `data/niah_input/{task}_{train,test}.json`.

#### 4.2 Build Option A long-context detection data (optional if already present)

```bash
python sec_experiments/build_long_context_detection_optionA.py \
  --max_instances 200 \
  --chunk_words 400
```

This reads `data/sections.csv` and `data/haystack_plan.csv` and writes JSONs under `data/long_context_detection_optionA/`.

#### 4.3 Run QRScore detection on Option A (QRScore-SEC)

```bash
bash sec_experiments/run_long_context_detection.sh
```

This uses `data/long_context_detection_optionA/combined_detection.json` and writes:

- `results/detection/long_context_combined_heads.json`

which is your **QRScore-SEC** head ranking (layer–head + score).

#### 4.4 Run generation ablation (Part B)

Use NIAH test data and multiple head rankings:

```bash
python sec_experiments/run_comparison_ablation.py \
  --niah_dir data/niah_input \
  --output_dir results/comparison_ablation \
  --max_instances_per_task 20 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --max_context_tokens 8192 \
  --methods QRScore-SEC QRScore-Paper-LME QRScore-Paper-NQ \
            Random-seed42 Random-seed123 Random-seed456
```

This will produce one `*_results.json` per method under `results/comparison_ablation/`, each containing:

- The accuracy curve vs `K`,
- Per-instance details, and
- Per-task breakdown (if enabled).

#### 4.5 Plot accuracy vs knockout size

```bash
python sec_experiments/plot_comparison_ablation.py \
  --results_dir results/comparison_ablation \
  --output_dir results/comparison_ablation
```

This creates:

- `results/comparison_ablation/accuracy_vs_knockout.png`

showing answer accuracy vs `K` for all methods on the same plot. A **steeper drop** indicates that the corresponding head ranking finds more critical retrieval heads.

---

### 5. Interpreting results

- At `K = 0` (no heads knocked out), all methods share the same accuracy (full model, no masking).
- As `K` increases (especially around `K ≈ 16`), compare:
  - **QRScore-SEC**,
  - **QRScore-Paper-LME**,
  - **QRScore-Paper-NQ**,
  - **Random-avg`or individual`Random-seed*`.
- If QRScore-SEC (or the paper QRHeads) shows a **larger accuracy drop** than Random, it indicates that those heads are more important for retrieval in SEC NIAH tasks.

