# Mechanistic QRHead Experiments

These scripts implement the minimum mechanistic experiment package for the
ICML 2026 Mechanistic Interpretability workshop submission.

## 1. Activation-patching rescue

```bash
python scripts/evaluation/run_activation_patching.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --k 16 \
  --tasks employees_count_total ceo_lastname registrant_name headquarters_state \
  --max_examples_per_task 16
```

Outputs are written to:

```text
results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct/activation_patching/
```

Key files:

- `activation_patching_rescue.csv`: per-example rescue rates.
- `activation_patching_summary.json`: task/condition summaries.
- `activation_patching_rescue_bar.png`: paper-ready control comparison.

## 2. Gold-evidence edge masking

```bash
python scripts/evaluation/run_edge_masking.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --k 16 \
  --tasks employees_count_total ceo_lastname registrant_name headquarters_state \
  --max_examples_per_task 12
```

Outputs are written to:

```text
results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct/edge_masking/
```

Key files:

- `edge_masking_damage.csv`: per-example fraction of full-ablation damage.
- `edge_masking_summary.json`: task/condition summaries.
- `edge_masking_damage_bar.png`: paper-ready edge-localization comparison.

## 3. Qwen patching replication

```bash
python scripts/evaluation/run_activation_patching.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --k 16 \
  --tasks employees_count_total ceo_lastname registrant_name headquarters_state \
  --max_examples_per_task 16
```

The patching script uses existing `QRScore-SEC_results.json` files to select
examples where clean generation was correct and K=16 QRHead ablation was wrong.
Use `--use_all_examples` to disable that filter.

