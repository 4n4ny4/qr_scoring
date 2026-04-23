#!/bin/bash
# Run QRHead detection on long-context data with Mistral-7B-Instruct-v0.3.
#
# Usage:
#   bash scripts/detection/run_detection_mistral.sh
#   bash scripts/detection/run_detection_mistral.sh --combined-only

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
CONFIG="$PROJECT_DIR/src/qrretriever/configs/Mistral-7B-Instruct-v0.3_full_head.yaml"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/long_context_detection_optionA}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/detection_mistral}"
TOPK_EXPORT_DIR="$OUTPUT_DIR/topk"
EXPORT_TOP_K=(8 16 32 48 64 96 128)
# No truncation by default (matches Llama detection). Set >0 if OOM.
TRUNCATE_BY_SPACE="${TRUNCATE_BY_SPACE:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TOPK_EXPORT_DIR"

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

COMBINED_ONLY=false
if [ "${1}" = "--combined-only" ]; then
    COMBINED_ONLY=true
fi

if [ "$COMBINED_ONLY" = false ]; then
    for TASK in "${TASKS[@]}"; do
        INPUT_FILE="$INPUT_DIR/${TASK}_detection.json"
        OUTPUT_FILE="$OUTPUT_DIR/long_context_${TASK}_heads.json"

        if [ ! -f "$INPUT_FILE" ]; then
            echo "WARNING: $INPUT_FILE not found, skipping $TASK"
            continue
        fi

        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping $TASK (output already exists: $OUTPUT_FILE)"
            continue
        fi

        echo "========================================"
        echo "Running Mistral long-context detection for: $TASK"
        echo "  Input:  $INPUT_FILE"
        echo "  Output: $OUTPUT_FILE"
        echo "========================================"

        python "$PROJECT_DIR/scripts/detection/detect_qrhead.py" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --config_or_config_path "$CONFIG" \
            --task_name "long_context_${TASK}" \
            --export_dir "$TOPK_EXPORT_DIR" \
            --export_top_k "${EXPORT_TOP_K[@]}" \
            --truncate_by_space "$TRUNCATE_BY_SPACE"

        echo "Done: $TASK"
        echo ""
    done
fi

COMBINED_INPUT="$INPUT_DIR/combined_detection.json"
COMBINED_OUTPUT="$OUTPUT_DIR/long_context_combined_heads.json"

if [ -f "$COMBINED_INPUT" ]; then
    if [ -f "$COMBINED_OUTPUT" ]; then
        echo "Skipping combined (output already exists: $COMBINED_OUTPUT)"
    else
        echo "========================================"
        echo "Running Mistral long-context COMBINED detection"
        echo "  Input:  $COMBINED_INPUT"
        echo "  Output: $COMBINED_OUTPUT"
        echo "========================================"

        python "$PROJECT_DIR/scripts/detection/detect_qrhead.py" \
            --input_file "$COMBINED_INPUT" \
            --output_file "$COMBINED_OUTPUT" \
            --config_or_config_path "$CONFIG" \
            --task_name "long_context_combined" \
            --export_dir "$TOPK_EXPORT_DIR" \
            --export_top_k "${EXPORT_TOP_K[@]}" \
            --truncate_by_space "$TRUNCATE_BY_SPACE"

        echo "Done: combined"
    fi
else
    echo "WARNING: $COMBINED_INPUT not found. Run scripts/data_prep/build_detection_data.py first."
fi

echo ""
echo "All Mistral detection complete. Results in $OUTPUT_DIR"
echo "Used TRUNCATE_BY_SPACE=$TRUNCATE_BY_SPACE"
