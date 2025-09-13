#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for MSE comparison across all targets.
# Usage: ./scripts/run_compare_mse_all.sh [data_csv|--dataset base|cot] [results_csv]

# Dataset selection: allow path or shortcut via --dataset base|cot
if [[ "${1:-}" == "--dataset" ]]; then
  CHOICE=${2:-base}
  shift 2
  if [[ "$CHOICE" == "cot" ]]; then
    DATA_CSV="archive/predictions/NPORS_2024_cot_optimized_lr06_step560_20250911_232934.csv"
  else
    DATA_CSV="archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv"
  fi
else
  DATA_CSV=${1:-archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv}
  shift 1 || true
fi

RESULTS_CSV=${1:-compare_runs_log_mse.csv}

# Default params (can be overridden by env vars)
NEXP=${NEXP:-100}
LABELS=${LABELS:-500}
GAMMA=${GAMMA:-0.5}
ALPHA=${ALPHA:-0.95}

TARGETS=(
  ECON1MOD
  UNITY
  GPT1
  MOREGUNIMPACT
  GAMBLERESTR
)

echo "Dataset: ${DATA_CSV}"
echo "Results CSV: ${RESULTS_CSV}"
echo "Experiments=${NEXP} Labels=${LABELS} Gamma=${GAMMA} Alpha=${ALPHA}"

for TGT in "${TARGETS[@]}"; do
  echo "\n=== Running target: ${TGT} ==="
  python estimators/compare_estimators.py "${DATA_CSV}" \
    --target "${TGT}" \
    --experiments "${NEXP}" \
    --labels "${LABELS}" \
    --gamma "${GAMMA}" \
    --results-csv "${RESULTS_CSV}" \
    --alpha "${ALPHA}"
done

echo "\nAll targets completed. Results appended to ${RESULTS_CSV}."
