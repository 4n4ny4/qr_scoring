#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SLUG="${MODEL_SLUG:-meta-llama__Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-cuda}"
K="${K:-16}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-8192}"
TASKS="${TASKS:-employees_count_total ceo_lastname}"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-13-18 14-31}"
ANALYSIS_ONLY="${ANALYSIS_ONLY:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qr_scoring_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

ROOT_DIR="results/mech_experiments/${MODEL_SLUG}"
export ROOT_DIR
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

if [[ "${ANALYSIS_ONLY}" != "1" ]]; then
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
else
  echo "ANALYSIS_ONLY=1: skipping component/path/final validation reruns."
fi

echo "=== Analysis summary ==="
python scripts/evaluation/analyze_full_circuit.py \
  --model_slug "${MODEL_SLUG}" \
  --output_dir "${ROOT_DIR}/full_circuit_summary"

echo "=== Top residual sites ==="
python - <<'PY'
import csv
import os

root = os.environ.get("ROOT_DIR", "results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct")
p = f"{root}/residual_circuit_patching/residual_patch_summary.csv"
cols = ["task", "position_group", "layer", "n", "median_margin_recovery", "mean_margin_recovery"]
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r.get("median_margin_recovery") or "-inf"), reverse=True)
rows = rows[:20]
widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in cols}
print(" ".join(col.ljust(widths[col]) for col in cols))
for row in rows:
    print(" ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))
PY

echo "=== Top component nodes ==="
python - <<'PY'
import csv
import os

root = os.environ.get("ROOT_DIR", "results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct")
p = f"{root}/component_circuit_patching/component_patch_summary.csv"
cols = ["task", "component_type", "condition", "n", "median_margin_recovery", "mean_margin_recovery"]
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r.get("median_margin_recovery") or "-inf"), reverse=True)
rows = rows[:30]
widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in cols}
print(" ".join(col.ljust(widths[col]) for col in cols))
for row in rows:
    print(" ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))
PY

echo "=== Top path edges ==="
python - <<'PY'
import csv
import os

root = os.environ.get("ROOT_DIR", "results/mech_experiments/meta-llama__Llama-3.1-8B-Instruct")
p = f"{root}/path_circuit_patching/path_patch_edge_summary.csv"
cols = [
    "task",
    "head",
    "source_position_group",
    "target_position_group",
    "condition",
    "n",
    "median_margin_damage_fraction",
]
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r.get("median_margin_damage_fraction") or "-inf"), reverse=True)
rows = rows[:30]
widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in cols}
print(" ".join(col.ljust(widths[col]) for col in cols))
for row in rows:
    print(" ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))
PY

echo "Done. To push results, run:"
echo "git add ${ROOT_DIR}/component_circuit_patching ${ROOT_DIR}/path_circuit_patching ${ROOT_DIR}/final_circuit_validation ${ROOT_DIR}/full_circuit_summary"
echo "git commit -m 'Add remaining full QRHead circuit results'"
echo "git push"
