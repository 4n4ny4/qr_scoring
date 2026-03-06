#!/bin/bash
# Run QRHead detection on long-context data (~30K tokens per instance).
# This should produce proper QRScore rankings with top heads in middle layers.
#
# Usage:
#   bash sec_experiments/run_long_context_detection.sh
#   bash sec_experiments/run_long_context_detection.sh --combined-only  # only run combined
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
CONFIG="$PROJECT_DIR/src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml"
# Allow overriding INPUT_DIR from environment; default to original long_context_detection
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/long_context_detection}"
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
        echo "Running long-context detection for: $TASK"
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
fi

# Run combined detection (pooled across all tasks -- recommended)
COMBINED_INPUT="$INPUT_DIR/combined_detection.json"
COMBINED_OUTPUT="$OUTPUT_DIR/long_context_combined_heads.json"

if [ -f "$COMBINED_INPUT" ]; then
    if [ -f "$COMBINED_OUTPUT" ]; then
        echo "Skipping combined (output already exists: $COMBINED_OUTPUT)"
    else
        echo "========================================"
        echo "Running long-context COMBINED detection"
        echo "  Input:  $COMBINED_INPUT"
        echo "  Output: $COMBINED_OUTPUT"
        echo "========================================"

        python "$PROJECT_DIR/exp_scripts/detection/detect_qrhead_lme.py" \
            --input_file "$COMBINED_INPUT" \
            --output_file "$COMBINED_OUTPUT" \
            --config_or_config_path "$CONFIG"

        echo "Done: combined"
    fi
else
    echo "WARNING: $COMBINED_INPUT not found. Run build_long_context_detection_data.py first."
fi

echo ""
echo "All detection complete. Results in $OUTPUT_DIR"
echo ""
echo "Validate with: python sec_experiments/validate_detection.py"
