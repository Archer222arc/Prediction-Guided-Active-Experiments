#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for CI-width cost comparisons across all targets.
# Usage:
#   ./scripts/run_compare_ci_all.sh [data_csv|--dataset base|cot] [results_csv]
#
# Defaults can be overridden by environment variables:
#   ALPHA=0.95 NEXP=50 MINLABELS=100 MAXLABELS=10000 TOL=0.00 \
#   METHODS="Active_Inference Naive PGAE Adaptive_PGAE" CIWIDTH=0.1 CIWIDTH_GAMBLERESTR=0.05 \
#   ./scripts/run_compare_ci_all.sh --dataset base compare_runs_log_ci.csv

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

RESULTS_CSV=${1:-compare_runs_log_ci.csv}

# Defaults
ALPHA=${ALPHA:-0.95}
NEXP=${NEXP:-50}
MINLABELS=${MINLABELS:-100}
MAXLABELS=${MAXLABELS:-10000}
TOL=${TOL:-0.00}
METHODS=${METHODS:-"Active_Inference Naive PGAE Adaptive_PGAE"}
CIWIDTH=${CIWIDTH:-0.1}
CIWIDTH_GAMBLERESTR=${CIWIDTH_GAMBLERESTR:-0.05}

TARGETS=(ECON1MOD UNITY GPT1 MOREGUNIMPACT GAMBLERESTR)

echo "Dataset: ${DATA_CSV}"
echo "Results CSV: ${RESULTS_CSV}"
echo "Alpha=${ALPHA} Experiments=${NEXP} MinLabels=${MINLABELS} MaxLabels=${MAXLABELS} Tol=${TOL}"
echo "Methods: ${METHODS}"

for TGT in "${TARGETS[@]}"; do
  if [[ "$TGT" == "GAMBLERESTR" ]]; then
    CW="$CIWIDTH_GAMBLERESTR"
  else
    CW="$CIWIDTH"
  fi
  echo "\n=== Running CI-cost target: ${TGT} (ci-width=${CW}) ==="
  python ./estimators/compare_estimators.py "${DATA_CSV}" \
    --target "${TGT}" \
    --alpha "${ALPHA}" \
    --ci-width "${CW}" \
    --experiments "${NEXP}" \
    --methods ${METHODS} \
    --min-labels "${MINLABELS}" \
    --max-labels "${MAXLABELS}" \
    --ci-tolerance "${TOL}" \
    --results-csv "${RESULTS_CSV}"
done

echo "\nAll CI-cost targets completed. Results appended to ${RESULTS_CSV}."
