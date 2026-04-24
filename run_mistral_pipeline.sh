#!/bin/bash
set -e

cd /root/qr_scoring
export HF_TOKEN=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRUNCATE_BY_SPACE=50

echo "============================================"
echo " Mistral QRScore Pipeline"
echo " Started: $(date)"
echo "============================================"
echo ""

# Step 1: Detection (~3-6 hrs on A100)
echo ">>> STEP 1/5: Detection"
echo "    Started: $(date)"
bash scripts/detection/run_detection_mistral.sh
echo "    Finished: $(date)"
echo ""

# Step 2: Pooled ablation (~2-4 hrs)
echo ">>> STEP 2/5: Pooled ablation"
echo "    Started: $(date)"
python scripts/evaluation/run_ablation.py \
  --detection_results_dir results/detection_mistral \
  --output_dir results/comparison_ablation_mistral \
  --max_instances_per_task 24 \
  --max_context_tokens 8192 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --log_tokens \
  --methods QRScore-SEC \
  --include_random_baselines
echo "    Finished: $(date)"
echo ""

# Step 3: Cross-task transfer (~8-12 hrs)
echo ">>> STEP 3/5: Cross-task transfer"
echo "    Started: $(date)"
python scripts/evaluation/run_ablation.py \
  --detection_results_dir results/detection_mistral \
  --output_dir results/comparison_ablation_mistral \
  --max_instances_per_task 24 \
  --max_context_tokens 8192 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --log_tokens \
  --enable_cross_task_transfer \
  --transfer_summary_k 16
echo "    Finished: $(date)"
echo ""

# Step 4: Plot ablation
echo ">>> STEP 4/5: Plot ablation"
echo "    Started: $(date)"
python scripts/evaluation/plot_ablation.py \
  --results_dir results/comparison_ablation_mistral \
  --output_dir results/comparison_ablation_mistral
echo "    Finished: $(date)"
echo ""

# Step 5: Plot transfer
echo ">>> STEP 5/5: Plot transfer"
echo "    Started: $(date)"
python scripts/evaluation/plot_transfer.py \
  --results_dir results/comparison_ablation_mistral \
  --output_dir results/comparison_ablation_mistral
echo "    Finished: $(date)"
echo ""

echo "============================================"
echo " Pipeline complete: $(date)"
echo "============================================"
