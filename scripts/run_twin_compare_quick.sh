#!/usr/bin/env bash
set -euo pipefail

# Quick runner for testing Digital Twin comparison on subset of questions
# Usage:
#   ./scripts/run_twin_compare_quick.sh [mse|ci] [cot|base]

MODE=${1:-mse}
DATASET=${2:-cot}

if [[ "$DATASET" == "base" ]]; then
  DATA_CSV="data/NPORS_2024_base_gpt5mini_20250913_133642.csv"
else
  DATA_CSV="data/NPORS_2024_cot_gpt5mini_20250913_133629.csv"
fi

# Quick test parameters
NEXP=5
LABELS=100

echo "======================================================================"
echo "Digital Twin Dataset Quick Test ($MODE mode, $DATASET predictions)"
echo "======================================================================"
echo "Dataset: ${DATA_CSV}"
echo "Mode: ${MODE}"
echo "Quick params: ${NEXP} experiments, ${LABELS} labels"
echo "Questions: CARBONTAX, CLEANENERGY (subset for speed)"
echo "======================================================================"

if [[ ! -f "${DATA_CSV}" ]]; then
  echo "❌ Error: Dataset file not found: ${DATA_CSV}"
  exit 1
fi

if [[ "$MODE" == "ci" ]]; then
  echo "🔍 Running CI Width Cost Analysis (quick test)..."
  python estimators/twin_compare.py "${DATA_CSV}" \
    --questions CARBONTAX CLEANENERGY \
    --ci-width 0.2 \
    --experiments "${NEXP}" \
    --max-labels 500 \
    --label-step 50
else
  echo "📊 Running MSE Performance Analysis (quick test)..."
  python estimators/twin_compare.py "${DATA_CSV}" \
    --questions CARBONTAX CLEANENERGY \
    --experiments "${NEXP}" \
    --labels "${LABELS}"
fi

echo ""
echo "✅ Quick test completed!"
echo "For full analysis, use:"
echo "  ./scripts/run_twin_compare_mse_all.sh --dataset ${DATASET}"
echo "  ./scripts/run_twin_compare_ci_all.sh --dataset ${DATASET}"
echo "======================================================================"