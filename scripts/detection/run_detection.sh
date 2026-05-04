PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Default model if not set from environment
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
CONFIG=""
if [ "$MODEL_NAME" = "meta-llama/Llama-3.1-8B-Instruct" ]; then
    CONFIG="$PROJECT_DIR/src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml"
elif [ "$MODEL_NAME" = "Qwen/Qwen2.5-7B-Instruct" ]; then
    CONFIG="$PROJECT_DIR/src/qrretriever/configs/Qwen2.5-7B-Instruct_full_head.yaml"
elif [ -f "$PROJECT_DIR/src/qrretriever/configs/${MODEL_NAME}_full_head.yaml" ]; then
    CONFIG="$PROJECT_DIR/src/qrretriever/configs/${MODEL_NAME}_full_head.yaml"
fi
# Allow overriding INPUT_DIR from environment; default to Option A dataset
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/long_context_detection_optionA}"
OUTPUT_DIR="${DETECTION_DIR:-}"
TOPK_EXPORT_DIR="${TOPK_EXPORT_DIR:-}"
EXPORT_TOP_K=(8 16 32 48 64 96 128)
TRUNCATE_BY_SPACE="${TRUNCATE_BY_SPACE:-512}"
MODEL_LOAD_IN_8BIT="${MODEL_LOAD_IN_8BIT:-1}"

check_python_dependencies() {
    # Use the same interpreter as the run command so checks match runtime behavior.
    python - <<'PY'
import importlib
import sys

required = ["tqdm", "numpy", "torch", "transformers", "yaml", "bitsandbytes"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

if missing:
    print("ERROR: Missing Python dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install project dependencies with:", file=sys.stderr)
    print("  python -m pip install -e .", file=sys.stderr)
    print("Or install only missing packages with:", file=sys.stderr)
    print("  python -m pip install " + " ".join(missing), file=sys.stderr)
    sys.exit(1)

try:
    import PIL.Image
except ModuleNotFoundError:
    print("ERROR: Missing Python dependency: Pillow", file=sys.stderr)
    print("Install project dependencies with:", file=sys.stderr)
    print("  python -m pip install -e .", file=sys.stderr)
    print("Or install only missing package with:", file=sys.stderr)
    print("  python -m pip install Pillow>=9.1.0", file=sys.stderr)
    sys.exit(1)

if not hasattr(PIL.Image, "Resampling"):
    print("ERROR: Pillow is too old. `PIL.Image.Resampling` is required.", file=sys.stderr)
    print("Detected Pillow without Resampling support.", file=sys.stderr)
    print("Upgrade with:", file=sys.stderr)
    print("  python -m pip install --upgrade 'Pillow>=9.1.0'", file=sys.stderr)
    sys.exit(1)
PY
}

check_python_dependencies

if [ -z "$MODEL_SLUG" ]; then
    MODEL_SLUG="$(MODEL_NAME="$MODEL_NAME" python - <<'PY'
import os
from qrretriever.model_runtime import resolve_model_slug
print(resolve_model_slug(os.environ["MODEL_NAME"]))
PY
)"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_DIR/results/detection/$MODEL_SLUG"
fi
if [ -z "$TOPK_EXPORT_DIR" ]; then
    TOPK_EXPORT_DIR="$OUTPUT_DIR/topk"
fi

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
        echo "Running long-context detection for: $TASK"
        echo "  Model:  $MODEL_NAME"
        echo "  Input:  $INPUT_FILE"
        echo "  Output: $OUTPUT_FILE"
        echo "========================================"

        DETECTION_ARGS=(
            --input_file "$INPUT_FILE"
            --output_file "$OUTPUT_FILE"
            --detection_dir "$OUTPUT_DIR"
            --model_name_or_path "$MODEL_NAME"
            --model_slug "$MODEL_SLUG"
            --task_name "long_context_${TASK}"
            --export_dir "$TOPK_EXPORT_DIR"
            --export_top_k "${EXPORT_TOP_K[@]}"
            --truncate_by_space "$TRUNCATE_BY_SPACE"
        )
        if [ -n "$TOKENIZER_NAME" ]; then
            DETECTION_ARGS+=(--tokenizer_name_or_path "$TOKENIZER_NAME")
        fi
        if [ -n "$CONFIG" ]; then
            DETECTION_ARGS+=(--config_or_config_path "$CONFIG")
        fi
        if [ "$TRUST_REMOTE_CODE" = "1" ] || [ "$TRUST_REMOTE_CODE" = "true" ]; then
            DETECTION_ARGS+=(--trust_remote_code)
        fi
        if [ "$MODEL_LOAD_IN_8BIT" = "1" ] || [ "$MODEL_LOAD_IN_8BIT" = "true" ]; then
            DETECTION_ARGS+=(--model_load_in_8bit)
        fi

        python "$PROJECT_DIR/scripts/detection/detect_qrhead.py" \
            "${DETECTION_ARGS[@]}"

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
        echo "  Model:  $MODEL_NAME"
        echo "  Input:  $COMBINED_INPUT"
        echo "  Output: $COMBINED_OUTPUT"
        echo "========================================"

        DETECTION_ARGS=(
            --input_file "$COMBINED_INPUT"
            --output_file "$COMBINED_OUTPUT"
            --detection_dir "$OUTPUT_DIR"
            --model_name_or_path "$MODEL_NAME"
            --model_slug "$MODEL_SLUG"
            --task_name "long_context_combined"
            --export_dir "$TOPK_EXPORT_DIR"
            --export_top_k "${EXPORT_TOP_K[@]}"
            --truncate_by_space "$TRUNCATE_BY_SPACE"
        )
        if [ -n "$TOKENIZER_NAME" ]; then
            DETECTION_ARGS+=(--tokenizer_name_or_path "$TOKENIZER_NAME")
        fi
        if [ -n "$CONFIG" ]; then
            DETECTION_ARGS+=(--config_or_config_path "$CONFIG")
        fi
        if [ "$TRUST_REMOTE_CODE" = "1" ] || [ "$TRUST_REMOTE_CODE" = "true" ]; then
            DETECTION_ARGS+=(--trust_remote_code)
        fi
        if [ "$MODEL_LOAD_IN_8BIT" = "1" ] || [ "$MODEL_LOAD_IN_8BIT" = "true" ]; then
            DETECTION_ARGS+=(--model_load_in_8bit)
        fi

        python "$PROJECT_DIR/scripts/detection/detect_qrhead.py" \
            "${DETECTION_ARGS[@]}"

        echo "Done: combined"
    fi
else
    echo "WARNING: $COMBINED_INPUT not found. Run scripts/data_prep/build_detection_data.py first."
fi

echo ""
echo "All detection complete. Results in $OUTPUT_DIR"
echo "Used TRUNCATE_BY_SPACE=$TRUNCATE_BY_SPACE"
echo ""
