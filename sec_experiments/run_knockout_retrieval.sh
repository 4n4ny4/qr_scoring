#!/bin/bash
# Run retrieval on test set with knockout config (1008 heads) per task.
# Requires: PYTHONPATH=src or pip install -e . from repo root.
# Output: results/ablation/{task}_knockout_top16.json

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="$PROJECT_DIR/data/detection_input"
ABLATION_DIR="$PROJECT_DIR/results/ablation"
CONFIG_DIR="$ABLATION_DIR/configs"

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
    OUT="$ABLATION_DIR/${TASK}_knockout_top16.json"
    if [ -f "$OUT" ]; then
        echo "Skipping $TASK (already exists: $OUT)"
        continue
    fi
    echo "========================================"
    echo "Knockout retrieval: $TASK"
    echo "========================================"
    python "$PROJECT_DIR/exp_scripts/retrieval/run_retrieval.py" \
        --input_file "$INPUT_DIR/${TASK}_test.json" \
        --output_file "$OUT" \
        --data_type lme \
        --retriever_type qr_head \
        --config_or_config_path "$CONFIG_DIR/${TASK}_knockout_top16.yaml"
    echo "Done: $TASK"
done
echo "Knockout retrieval complete."
