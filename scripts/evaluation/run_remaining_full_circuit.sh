#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-meta-llama__Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda}"
K="${K:-16}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-8192}"
TASKS="${TASKS:-employees_count_total ceo_lastname}"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-13-18 14-31}"

ROOT_DIR="results/mech_experiments/${MODEL_SLUG}"
PAIRS_PATH="${ROOT_DIR}/circuit_pairs/circuit_pairs.jsonl"
RESIDUAL_SUMMARY="${ROOT_DIR}/residual_circuit_patching/residual_patch_summary.csv"

if [[ ! -f "${PAIRS_PATH}" ]]; then
  echo "Missing circuit pairs: ${PAIRS_PATH}" >&2
  echo "Run build_circuit_pairs.py first." >&2
  exit 1
fi

if [[ ! -f "${RESIDUAL_SUMMARY}" ]]; then
  echo "Missing residual patching summary: ${RESIDUAL_SUMMARY}" >&2
  echo "Run run_residual_stream_patching.py first." >&2
  exit 1
fi

echo "=== Component node search ==="
python scripts/evaluation/run_component_circuit_patching.py \
  --model_name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --pairs_path "${PAIRS_PATH}" \
  --tasks ${TASKS} \
  --candidate_heads ${CANDIDATE_HEADS} \
  --include_top_qr 4 \
  --position_group query \
  --k "${K}" \
  --max_context_tokens "${MAX_CONTEXT_TOKENS}" \
  --output_dir "${ROOT_DIR}/component_circuit_patching"

echo "=== Narrow path/edge tests ==="
python scripts/evaluation/run_path_patching.py \
  --model_name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --pairs_path "${PAIRS_PATH}" \
  --tasks ${TASKS} \
  --candidate_heads ${CANDIDATE_HEADS} \
  --source_position_groups gold_value gold_sentence \
  --target_position_groups query answer \
  --max_context_tokens "${MAX_CONTEXT_TOKENS}" \
  --output_dir "${ROOT_DIR}/path_circuit_patching"

echo "=== Final necessity/sufficiency validation ==="
python scripts/evaluation/run_component_circuit_patching.py \
  --model_name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --pairs_path "${PAIRS_PATH}" \
  --tasks ${TASKS} \
  --candidate_heads ${CANDIDATE_HEADS} \
  --include_discovered_downstream_nodes \
  --run_final_validation \
  --k "${K}" \
  --max_context_tokens "${MAX_CONTEXT_TOKENS}" \
  --output_dir "${ROOT_DIR}/final_circuit_validation"

echo "=== Analysis summary ==="
python scripts/evaluation/analyze_full_circuit.py \
  --model_slug "${MODEL_SLUG}" \
  --output_dir "${ROOT_DIR}/full_circuit_summary"

echo "=== Top residual sites ==="
python - <<'PY'
import pandas as pd

p = "results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct/residual_circuit_patching/residual_patch_summary.csv"
df = pd.read_csv(p)
cols = ["task", "position_group", "layer", "n", "median_margin_recovery", "mean_margin_recovery"]
print(df.sort_values("median_margin_recovery", ascending=False)[cols].head(20).to_string(index=False))
PY

echo "=== Top component nodes ==="
python - <<'PY'
import pandas as pd

p = "results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct/component_circuit_patching/component_patch_summary.csv"
df = pd.read_csv(p)
cols = ["task", "component_type", "condition", "n", "median_margin_recovery", "mean_margin_recovery"]
print(df.sort_values("median_margin_recovery", ascending=False)[cols].head(30).to_string(index=False))
PY

echo "=== Top path edges ==="
python - <<'PY'
import pandas as pd

p = "results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct/path_circuit_patching/path_patch_edge_summary.csv"
df = pd.read_csv(p)
cols = [
    "task",
    "head",
    "source_position_group",
    "target_position_group",
    "condition",
    "n",
    "median_margin_damage_fraction",
]
print(df.sort_values("median_margin_damage_fraction", ascending=False)[cols].head(30).to_string(index=False))
PY

echo "Done. To push results, run:"
echo "git add ${ROOT_DIR}/component_circuit_patching ${ROOT_DIR}/path_circuit_patching ${ROOT_DIR}/final_circuit_validation ${ROOT_DIR}/full_circuit_summary"
echo "git commit -m 'Add remaining full QRHead circuit results'"
echo "git push"
