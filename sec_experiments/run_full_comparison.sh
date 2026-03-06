#!/bin/bash
# Full comparison pipeline: retrieval + generation ablation.
#
# This script runs the complete evaluation to compare with the paper:
#   Part A: Retrieval recall (paper Tables 2-4)
#   Part B: Generation head-knockout ablation (paper Section 5)
#
# Prerequisites:
#   - pip install -e . (qrretriever installed)
#   - HuggingFace access to meta-llama/Llama-3.1-8B-Instruct
#   - QRScore detection done: results/detection/long_context_combined_heads.json
#   - NIAH test data: data/niah_input/{task}_test.json
#
# Usage:
#   bash sec_experiments/run_full_comparison.sh                # run everything
#   bash sec_experiments/run_full_comparison.sh --retrieval     # Part A only
#   bash sec_experiments/run_full_comparison.sh --generation    # Part B only
#   bash sec_experiments/run_full_comparison.sh --rethead-only  # just RETHEAD detection
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

RUN_RETRIEVAL=true
RUN_GENERATION=true
RETHEAD_ONLY=false
MAX_INSTANCES=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --retrieval) RUN_RETRIEVAL=true; RUN_GENERATION=false; shift ;;
        --generation) RUN_RETRIEVAL=false; RUN_GENERATION=true; shift ;;
        --rethead-only) RETHEAD_ONLY=true; RUN_RETRIEVAL=false; RUN_GENERATION=false; shift ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " SEC QRHead Full Comparison Pipeline"
echo "============================================================"
echo "  Retrieval evaluation:  $RUN_RETRIEVAL"
echo "  Generation ablation:   $RUN_GENERATION"
echo "  Max instances/task:    $MAX_INSTANCES"
echo "============================================================"
echo ""

# ---- RETHEAD Detection (needed for Part B) ----
RETHEAD_FILE="$PROJECT_DIR/results/detection/rethead_combined_heads.json"
if [ "$RUN_GENERATION" = true ] || [ "$RETHEAD_ONLY" = true ]; then
    if [ -f "$RETHEAD_FILE" ]; then
        echo "[RETHEAD] Already exists: $RETHEAD_FILE (skipping)"
    else
        echo "[RETHEAD] Running RETHEAD detection (eager attention, ~50 instances)..."
        python "$PROJECT_DIR/sec_experiments/detect_retrieval_heads.py" \
            --niah_dir "$PROJECT_DIR/data/niah_input" \
            --output_file "$RETHEAD_FILE" \
            --max_instances 50
        echo "[RETHEAD] Done."
    fi
fi

if [ "$RETHEAD_ONLY" = true ]; then
    echo "RETHEAD detection only. Done."
    exit 0
fi

# ---- Part A: Retrieval Comparison ----
if [ "$RUN_RETRIEVAL" = true ]; then
    echo ""
    echo "============================================================"
    echo " Part A: Retrieval Comparison"
    echo "============================================================"
    bash "$PROJECT_DIR/sec_experiments/run_retrieval_comparison.sh"
fi

# ---- Part B: Generation Head-Knockout Ablation ----
if [ "$RUN_GENERATION" = true ]; then
    echo ""
    echo "============================================================"
    echo " Part B: Generation Head-Knockout Ablation"
    echo "============================================================"

    python "$PROJECT_DIR/sec_experiments/run_comparison_ablation.py" \
        --niah_dir "$PROJECT_DIR/data/niah_input" \
        --output_dir "$PROJECT_DIR/results/comparison_ablation" \
        --max_instances_per_task "$MAX_INSTANCES" \
        --knockout_sizes 0 8 16 32 48 64 96 128

    echo ""
    echo "[PLOTS] Generating comparison plots and significance tests..."
    python "$PROJECT_DIR/sec_experiments/plot_comparison.py" \
        --summary "$PROJECT_DIR/results/comparison_ablation/comparison_summary.json" \
        --output_dir "$PROJECT_DIR/results/comparison_ablation"
fi

echo ""
echo "============================================================"
echo " All done!"
echo "============================================================"
if [ "$RUN_RETRIEVAL" = true ]; then
    echo "  Retrieval results: results/retrieval_comparison/"
    echo "    Summary:  results/retrieval_comparison/retrieval_comparison_summary.json"
    echo "    Plot:     results/retrieval_comparison/retrieval_comparison.png"
fi
if [ "$RUN_GENERATION" = true ]; then
    echo "  Generation results: results/comparison_ablation/"
    echo "    Summary:  results/comparison_ablation/comparison_summary.json"
    echo "    Curves:   results/comparison_ablation/accuracy_curves.png"
    echo "    Heatmap:  results/comparison_ablation/per_task_heatmap_k16.png"
    echo "    Overlap:  results/comparison_ablation/head_overlap_jaccard_top16.png"
fi
