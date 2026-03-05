#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_DIR/src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml"
INPUT_DIR="$PROJECT_DIR/data/detection_input"
OUTPUT_DIR="$PROJECT_DIR/results/detection"

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

for TASK in "${TASKS[@]}"; do
    INPUT_FILE="$INPUT_DIR/${TASK}_train.json"
    OUTPUT_FILE="$OUTPUT_DIR/${TASK}_heads.json"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $TASK (output already exists: $OUTPUT_FILE)"
        continue
    fi

    echo "========================================"
    echo "Running detection for: $TASK"
    echo "  Input:  $INPUT_FILE"
    echo "  Output: $OUTPUT_FILE"
    echo "========================================"

    python "$PROJECT_DIR/exp_scripts/detection/detect_qrhead_lme.py" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --config_or_config_path "$CONFIG"

    echo "Done: $TASK"
    echo ""
done

echo "All tasks complete. Results in $OUTPUT_DIR"
