#!/usr/bin/env bash
# Run OLMo per-task SEC retrieval-head detection, task-head Jaccard, and
# optional cross-task ablations.
#
# Full run:
#   bash scripts/evaluation/run_olmo_cross_task.sh
#
# Dry run:
#   DRY_RUN=1 bash scripts/evaluation/run_olmo_cross_task.sh
#
# Smoke run:
#   MAX_INSTANCES_PER_TASK=1 K_VALUES="8" KNOCKOUT_SIZES="0 8" \
#     bash scripts/evaluation/run_olmo_cross_task.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-allenai/OLMo-7B}"
MODEL_SLUG="${MODEL_SLUG:-allenai__OLMo-7B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/long_context_detection_optionA}"
NIAH_DIR="${NIAH_DIR:-$PROJECT_DIR/data/niah_input}"
DETECTION_DIR="${DETECTION_DIR:-$PROJECT_DIR/results/detection/$MODEL_SLUG}"
ABLATION_DIR="${ABLATION_DIR:-$PROJECT_DIR/results/comparison_ablation/$MODEL_SLUG}"
TOPK_EXPORT_DIR="${TOPK_EXPORT_DIR:-$DETECTION_DIR/topk}"
GENERATED_INPUT_DIR="${GENERATED_INPUT_DIR:-$DETECTION_DIR/_inputs}"

RUN_DETECTION="${RUN_DETECTION:-1}"
RUN_JACCARD="${RUN_JACCARD:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

MAX_INSTANCES_PER_TASK="${MAX_INSTANCES_PER_TASK:-24}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-8192}"
K_VALUES="${K_VALUES:-8 16 32 48 64 96 128}"
KNOCKOUT_SIZES="${KNOCKOUT_SIZES:-0 8 16 32 48 64 96 128}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
DEVICE="${DEVICE:-auto}"
TRUNCATE_BY_SPACE="${TRUNCATE_BY_SPACE:-0}"
TRANSFER_SUMMARY_K="${TRANSFER_SUMMARY_K:-16}"
LOG_TOKENS="${LOG_TOKENS:-0}"

CACHE_ROOT="${CACHE_ROOT:-${TMPDIR:-/tmp}/qr_scoring_olmo_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"

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

read -r -a K_VALUES_ARRAY <<< "$K_VALUES"
read -r -a KNOCKOUT_SIZES_ARRAY <<< "$KNOCKOUT_SIZES"

is_truthy() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

print_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if is_truthy "$DRY_RUN"; then
        return 0
    fi
    "$@"
}

ensure_dirs() {
    local dirs=("$DETECTION_DIR" "$TOPK_EXPORT_DIR" "$ABLATION_DIR" "$GENERATED_INPUT_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME")
    print_cmd mkdir -p "${dirs[@]}"
    if ! is_truthy "$DRY_RUN"; then
        mkdir -p "${dirs[@]}"
    fi
}

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "ERROR: required file not found: $path" >&2
        exit 1
    fi
}

topk_exports_complete() {
    local prefix="$1"
    local k
    for k in "${K_VALUES_ARRAY[@]}"; do
        [ -f "$TOPK_EXPORT_DIR/${prefix}_top${k}.json" ] || return 1
    done
    return 0
}

detection_args_common() {
    local input_file="$1"
    local output_file="$2"
    local task_name="$3"
    local args=(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/detection/detect_qrhead.py"
        --input_file "$input_file"
        --output_file "$output_file"
        --detection_dir "$DETECTION_DIR"
        --model_name_or_path "$MODEL_NAME"
        --model_slug "$MODEL_SLUG"
        --task_name "$task_name"
        --export_dir "$TOPK_EXPORT_DIR"
        --export_top_k "${K_VALUES_ARRAY[@]}"
        --truncate_by_space "$TRUNCATE_BY_SPACE"
    )
    if [ -n "$TOKENIZER_NAME" ]; then
        args+=(--tokenizer_name_or_path "$TOKENIZER_NAME")
    fi
    if is_truthy "$TRUST_REMOTE_CODE"; then
        args+=(--trust_remote_code)
    fi
    run_cmd "${args[@]}"
}

run_task_detection() {
    local task="$1"
    local prefix="long_context_${task}"
    local input_file="$INPUT_DIR/${task}_detection.json"
    local output_file="$DETECTION_DIR/${prefix}_heads.json"

    require_file "$input_file"

    if [ -f "$output_file" ] && topk_exports_complete "$prefix" && ! is_truthy "$FORCE"; then
        echo "Skipping OLMo detection for $task; outputs already exist."
        return 0
    fi

    echo
    echo "Running OLMo retrieval-head detection for task: $task"
    detection_args_common "$input_file" "$output_file" "$prefix"
}

generate_combined_detection_input() {
    local generated_path="$1"
    echo
    echo "Generating combined SEC detection input: $generated_path"

    print_cmd "$PYTHON_BIN" - "$INPUT_DIR" "$generated_path" "${TASKS[@]}"
    if is_truthy "$DRY_RUN"; then
        return 0
    fi

    "$PYTHON_BIN" - "$INPUT_DIR" "$generated_path" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
tasks = sys.argv[3:]

combined = []
for task in tasks:
    path = input_dir / f"{task}_detection.json"
    with path.open(encoding="utf-8") as f:
        combined.extend(json.load(f))

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)
    f.write("\n")

print(f"Wrote {len(combined)} combined detection instances to {output_path}")
PY
}

resolve_combined_input() {
    local canonical="$INPUT_DIR/combined_detection.json"
    local generated="$GENERATED_INPUT_DIR/combined_detection.generated.json"

    if [ -n "${COMBINED_INPUT:-}" ]; then
        require_file "$COMBINED_INPUT"
        printf '%s\n' "$COMBINED_INPUT"
        return 0
    fi

    if [ -f "$canonical" ]; then
        printf '%s\n' "$canonical"
        return 0
    fi

    if [ ! -f "$generated" ] || is_truthy "$FORCE"; then
        generate_combined_detection_input "$generated" >&2
    fi
    printf '%s\n' "$generated"
}

run_combined_detection() {
    local input_file="$1"
    local prefix="long_context_combined"
    local output_file="$DETECTION_DIR/${prefix}_heads.json"

    if [ -f "$output_file" ] && topk_exports_complete "$prefix" && ! is_truthy "$FORCE"; then
        echo "Skipping OLMo combined detection; outputs already exist."
        return 0
    fi

    echo
    echo "Running OLMo combined SEC retrieval-head detection"
    detection_args_common "$input_file" "$output_file" "$prefix"
}

ensure_task_rankings_exist() {
    local task
    if is_truthy "$DRY_RUN"; then
        for task in "${TASKS[@]}"; do
            print_cmd test -f "$DETECTION_DIR/long_context_${task}_heads.json"
        done
        return 0
    fi

    for task in "${TASKS[@]}"; do
        require_file "$DETECTION_DIR/long_context_${task}_heads.json"
    done
}

compute_task_jaccard() {
    local output_json="$ABLATION_DIR/cross_task_head_similarity_topk.json"

    if [ -f "$output_json" ] && ! is_truthy "$FORCE"; then
        echo "Skipping OLMo task-head Jaccard; output already exists: $output_json"
        return 0
    fi

    echo
    echo "Computing OLMo per-task retrieval-head Jaccard similarity"
    print_cmd "$PYTHON_BIN" - "$DETECTION_DIR" "$output_json" "$MODEL_NAME" "$MODEL_SLUG" "${K_VALUES_ARRAY[@]}" -- "${TASKS[@]}"
    if is_truthy "$DRY_RUN"; then
        return 0
    fi

    "$PYTHON_BIN" - "$DETECTION_DIR" "$output_json" "$MODEL_NAME" "$MODEL_SLUG" "${K_VALUES_ARRAY[@]}" -- "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

sep = sys.argv.index("--")
detection_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
model_name = sys.argv[3]
model_slug = sys.argv[4]
k_values = [int(x) for x in sys.argv[5:sep]]
tasks = sys.argv[sep + 1:]


def parse_head(row):
    head_str = row[0] if isinstance(row, list) else row.get("head")
    layer, head = str(head_str).split("-", 1)
    return int(layer), int(head)


rankings = {}
ranking_paths = {}
for task in tasks:
    path = detection_dir / f"long_context_{task}_heads.json"
    with path.open(encoding="utf-8") as f:
        rankings[task] = [parse_head(row) for row in json.load(f)]
    ranking_paths[task] = str(path)

payload = {
    "model_name": model_name,
    "model_slug": model_slug,
    "model_family": "olmo",
    "tasks": tasks,
    "ranking_paths": ranking_paths,
    "top_k": {},
}

for k in sorted(set(k_values)):
    matrix = []
    for src in tasks:
        src_set = set(rankings[src][: min(k, len(rankings[src]))])
        row = []
        for tgt in tasks:
            tgt_set = set(rankings[tgt][: min(k, len(rankings[tgt]))])
            union = src_set | tgt_set
            row.append((len(src_set & tgt_set) / len(union)) if union else 0.0)
        matrix.append(row)
    payload["top_k"][str(k)] = matrix

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")

print(f"Wrote OLMo task-head Jaccard matrices to {output_path}")
PY
}

plot_task_jaccard() {
    local input_json="$ABLATION_DIR/cross_task_head_similarity_topk.json"
    local output_png="$ABLATION_DIR/head_similarity_jaccard.png"

    if [ ! -f "$input_json" ] && ! is_truthy "$DRY_RUN"; then
        echo "WARNING: skipping Jaccard plot; missing $input_json" >&2
        return 0
    fi

    if [ -f "$output_png" ] && ! is_truthy "$FORCE"; then
        echo "Skipping OLMo Jaccard plot; output already exists: $output_png"
        return 0
    fi

    run_cmd \
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluation/plot_head_similarity.py" \
        --input_json "$input_json" \
        --output_png "$output_png" \
        --title "OLMo Retrieval-Head Jaccard Across SEC Tasks" \
        --model_name "$MODEL_NAME"
}

run_cross_task_ablation() {
    local combined_ranking="$DETECTION_DIR/long_context_combined_heads.json"
    if is_truthy "$DRY_RUN"; then
        print_cmd test -f "$combined_ranking"
    else
        require_file "$combined_ranking"
    fi
    ensure_task_rankings_exist

    echo
    echo "Running OLMo QRScore-SEC pooled and cross-task ablations"

    local args=(
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluation/run_ablation.py"
        --niah_dir "$NIAH_DIR"
        --output_dir "$ABLATION_DIR"
        --ranking_dir "$DETECTION_DIR"
        --model_name "$MODEL_NAME"
        --model_slug "$MODEL_SLUG"
        --device "$DEVICE"
        --max_instances_per_task "$MAX_INSTANCES_PER_TASK"
        --max_context_tokens "$MAX_CONTEXT_TOKENS"
        --knockout_sizes "${KNOCKOUT_SIZES_ARRAY[@]}"
        --tasks "${TASKS[@]}"
        --methods QRScore-SEC
        --enable_cross_task_transfer
        --transfer_summary_k "$TRANSFER_SUMMARY_K"
        --export_top_k "${K_VALUES_ARRAY[@]}"
        --progress_every "$PROGRESS_EVERY"
    )
    if [ -n "$TOKENIZER_NAME" ]; then
        args+=(--tokenizer_name "$TOKENIZER_NAME")
    fi
    if is_truthy "$TRUST_REMOTE_CODE"; then
        args+=(--trust_remote_code)
    fi
    if is_truthy "$LOG_TOKENS"; then
        args+=(--log_tokens)
    fi

    run_cmd "${args[@]}"
}

plot_ablation_outputs() {
    run_cmd \
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluation/plot_ablation.py" \
        --results_dir "$ABLATION_DIR" \
        --output_dir "$ABLATION_DIR" \
        --method_filter QRScore-SEC

    run_cmd \
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluation/plot_transfer.py" \
        --results_dir "$ABLATION_DIR" \
        --output_dir "$ABLATION_DIR"
}

echo "OLMo SEC head workflow"
echo "  Project:       $PROJECT_DIR"
echo "  Model:         $MODEL_NAME"
echo "  Detection dir: $DETECTION_DIR"
echo "  Ablation dir:  $ABLATION_DIR"
echo "  K values:      ${K_VALUES_ARRAY[*]}"
echo "  Knockouts:     ${KNOCKOUT_SIZES_ARRAY[*]}"
echo "  Dry run:       $DRY_RUN"

ensure_dirs

if is_truthy "$RUN_DETECTION"; then
    for task in "${TASKS[@]}"; do
        run_task_detection "$task"
    done
    combined_input="$(resolve_combined_input)"
    run_combined_detection "$combined_input"
else
    echo "Skipping detection because RUN_DETECTION=$RUN_DETECTION"
fi

if is_truthy "$RUN_JACCARD"; then
    ensure_task_rankings_exist
    compute_task_jaccard
    if is_truthy "$RUN_PLOTS"; then
        plot_task_jaccard
    fi
else
    echo "Skipping Jaccard because RUN_JACCARD=$RUN_JACCARD"
fi

if is_truthy "$RUN_ABLATION"; then
    run_cross_task_ablation
else
    echo "Skipping ablation because RUN_ABLATION=$RUN_ABLATION"
fi

if is_truthy "$RUN_PLOTS"; then
    plot_ablation_outputs
else
    echo "Skipping plots because RUN_PLOTS=$RUN_PLOTS"
fi

echo
echo "Done."
echo "Detection outputs: $DETECTION_DIR"
echo "Ablation/Jaccard outputs: $ABLATION_DIR"
