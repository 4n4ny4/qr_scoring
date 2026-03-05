#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="$PROJECT_DIR/data/detection_input"
DETECTION_DIR="$PROJECT_DIR/results/detection"
ABLATION_DIR="$PROJECT_DIR/results/ablation"
CONFIG_DIR="$ABLATION_DIR/configs"

mkdir -p "$ABLATION_DIR" "$CONFIG_DIR"

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

FULL_HEAD_CONFIG="$PROJECT_DIR/src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml"

# Step 1: Generate per-task QR head configs and random head configs from detection results
echo "Generating ablation configs..."
python "$PROJECT_DIR/sec_experiments/generate_ablation_configs.py" \
    --detection_dir "$DETECTION_DIR" \
    --config_dir "$CONFIG_DIR" \
    --top_k 16 \
    --tasks "${TASKS[@]}"

# Step 2: Run retrieval for each task x condition
for TASK in "${TASKS[@]}"; do
    TEST_FILE="$INPUT_DIR/${TASK}_test.json"

    for CONDITION in "full_head" "qr_head_top16" "random_head"; do
        OUTPUT_FILE="$ABLATION_DIR/${TASK}_${CONDITION}.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping $TASK / $CONDITION (already exists)"
            continue
        fi

        if [ "$CONDITION" = "full_head" ]; then
            RETRIEVER_TYPE="full_head"
            CONFIG_FILE="$FULL_HEAD_CONFIG"
        elif [ "$CONDITION" = "qr_head_top16" ]; then
            RETRIEVER_TYPE="qr_head"
            CONFIG_FILE="$CONFIG_DIR/${TASK}_qr_head_top16.yaml"
        elif [ "$CONDITION" = "random_head" ]; then
            RETRIEVER_TYPE="qr_head"
            CONFIG_FILE="$CONFIG_DIR/${TASK}_random_head.yaml"
        fi

        echo "========================================"
        echo "Ablation: $TASK / $CONDITION"
        echo "  Test file: $TEST_FILE"
        echo "  Config:    $CONFIG_FILE"
        echo "  Output:    $OUTPUT_FILE"
        echo "========================================"

        python "$PROJECT_DIR/exp_scripts/retrieval/run_retrieval.py" \
            --input_file "$TEST_FILE" \
            --output_file "$OUTPUT_FILE" \
            --data_type lme \
            --retriever_type "$RETRIEVER_TYPE" \
            --config_or_config_path "$CONFIG_FILE"

        echo "Done: $TASK / $CONDITION"
        echo ""
    done
done

# Step 3: Evaluate all results
echo "========================================"
echo "Running evaluation..."
echo "========================================"
python "$PROJECT_DIR/sec_experiments/eval_ablation.py" \
    --ablation_dir "$ABLATION_DIR" \
    --data_dir "$INPUT_DIR" \
    --tasks "${TASKS[@]}"

echo "All ablation experiments complete."
