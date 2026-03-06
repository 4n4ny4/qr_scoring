#!/bin/bash
# Run retrieval with multiple head configs on NIAH test data,
# then evaluate and produce a summary table comparing Recall@k.
#
# This mirrors the paper's retrieval evaluation (Tables 2-4):
#   - Full head (all 1024 heads)
#   - QR-head SEC (our detected heads from Option A)
#   - QR-head Paper-LME (paper's LME-detected heads)
#   - QR-head Paper-NQ  (paper's NQ-detected heads)
#
# Usage:
#   bash sec_experiments/run_retrieval_comparison.sh
#   bash sec_experiments/run_retrieval_comparison.sh --tasks registrant_name ceo_lastname
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

NIAH_DIR="$PROJECT_DIR/data/niah_input"
OUTPUT_DIR="$PROJECT_DIR/results/retrieval_comparison"
CONFIGS_DIR="$PROJECT_DIR/src/qrretriever/configs"

mkdir -p "$OUTPUT_DIR"

TASKS=(
    "registrant_name"
    "headquarters_city"
    "headquarters_state"
    "incorporation_state"
    "incorporation_year"
    "employees_count_total"
    "ceo_lastname"
    "holder_record_amount"
)

# Allow overriding tasks from command line
if [ "$1" = "--tasks" ]; then
    shift
    TASKS=("$@")
fi

declare -A METHODS
METHODS=(
    ["full_head"]="full_head:$CONFIGS_DIR/Llama-3.1-8B-Instruct_full_head.yaml"
    ["qr_sec"]="qr_head:$CONFIGS_DIR/Llama-3.1-8B-Instruct_qr_head_SEC.yaml"
    ["qr_lme"]="qr_head:$CONFIGS_DIR/Llama-3.1-8B-Instruct_qr_head_LME.yaml"
    ["qr_nq"]="qr_head:$CONFIGS_DIR/Llama-3.1-8B-Instruct_qr_head_NQ.yaml"
)

echo "========================================"
echo "Retrieval Comparison"
echo "  Tasks: ${TASKS[*]}"
echo "  Methods: ${!METHODS[*]}"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

for TASK in "${TASKS[@]}"; do
    INPUT_FILE="$NIAH_DIR/${TASK}_test.json"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "WARNING: $INPUT_FILE not found, skipping $TASK"
        continue
    fi

    for METHOD_NAME in "${!METHODS[@]}"; do
        IFS=':' read -r RETRIEVER_TYPE CONFIG_PATH <<< "${METHODS[$METHOD_NAME]}"
        OUTPUT_FILE="$OUTPUT_DIR/${TASK}_${METHOD_NAME}.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping $TASK / $METHOD_NAME (output exists)"
            continue
        fi

        echo ""
        echo "--- $TASK / $METHOD_NAME ---"
        echo "  retriever_type=$RETRIEVER_TYPE"
        echo "  config=$CONFIG_PATH"

        python "$PROJECT_DIR/exp_scripts/retrieval/run_retrieval.py" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --data_type lme \
            --retriever_type "$RETRIEVER_TYPE" \
            --config_or_config_path "$CONFIG_PATH"

        echo "  Done: $OUTPUT_FILE"
    done
done

echo ""
echo "========================================"
echo "Retrieval runs complete."
echo "Now evaluating..."
echo "========================================"

# Run evaluation
python "$PROJECT_DIR/sec_experiments/eval_retrieval_comparison.py" \
    --retrieval_dir "$OUTPUT_DIR" \
    --niah_dir "$NIAH_DIR" \
    --output_file "$OUTPUT_DIR/retrieval_comparison_summary.json"

echo ""
echo "All done. Summary: $OUTPUT_DIR/retrieval_comparison_summary.json"
