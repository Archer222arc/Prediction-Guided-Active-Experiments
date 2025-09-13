#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for CI-width cost comparisons on Digital Twin dataset.
# Usage:
#   ./scripts/run_twin_compare_ci_all.sh [OPTIONS]

show_help() {
    cat << EOF
Digital Twin Dataset CI Width Cost Comparison Script

USAGE:
    ./scripts/run_twin_compare_ci_all.sh [OPTIONS]

OPTIONS:
    --dataset cot|base       Use preset dataset (default: cot)
    --data-csv FILE          Direct path to CSV file
    --ci-width N             Target CI width (default: 0.1)
    --ci-tolerance N         CI width tolerance (default: 0.000)
    --experiments N          Number of experiments (default: 20)
    --min-labels N           Minimum labels to search (default: 200)
    --max-labels N           Maximum labels to search (default: 2000)
    --label-step N           Labels search step (default: 50)
    --alpha N                Confidence level (default: 0.95)
    --max-workers N          Max concurrent workers (default: 10)
    --help                   Show this help

EXAMPLES:
    # Basic usage with CoT dataset
    ./scripts/run_twin_compare_ci_all.sh --dataset cot

    # Custom CI width target
    ./scripts/run_twin_compare_ci_all.sh --dataset base --ci-width 0.05

    # High precision search
    ./scripts/run_twin_compare_ci_all.sh --dataset cot --ci-width 0.08 --max-labels 5000

ENVIRONMENT VARIABLES:
    You can also set parameters via environment variables:
    CIWIDTH=0.15 NEXP=30 ./scripts/run_twin_compare_ci_all.sh --dataset cot
EOF
}

# Defaults (can be overridden by env vars or command line)
DATA_CSV=""
ALPHA=${ALPHA:-0.95}
NEXP=${NEXP:-20}
MINLABELS=${MINLABELS:-200}
MAXLABELS=${MAXLABELS:-10000}
TOL=${TOL:-0.000}
CIWIDTH=${CIWIDTH:-0.15}
LABELSTEP=${LABELSTEP:-50}
MAXWORKERS=${MAXWORKERS:-10}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            CHOICE=${2:-cot}
            if [[ "$CHOICE" == "base" ]]; then
                DATA_CSV="data/NPORS_2024_base_gpt5mini_20250913_133642.csv"
            elif [[ "$CHOICE" == "cot" ]]; then
                DATA_CSV="data/NPORS_2024_cot_gpt5mini_20250913_133629.csv"
            else
                echo "❌ Error: --dataset must be 'cot' or 'base', got: $CHOICE"
                exit 1
            fi
            shift 2
            ;;
        --data-csv)
            DATA_CSV="$2"
            shift 2
            ;;
        --ci-width)
            CIWIDTH="$2"
            shift 2
            ;;
        --ci-tolerance)
            TOL="$2"
            shift 2
            ;;
        --experiments)
            NEXP="$2"
            shift 2
            ;;
        --min-labels)
            MINLABELS="$2"
            shift 2
            ;;
        --max-labels)
            MAXLABELS="$2"
            shift 2
            ;;
        --label-step)
            LABELSTEP="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --max-workers)
            MAXWORKERS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            echo "❌ Unexpected argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default dataset if none specified
if [[ -z "$DATA_CSV" ]]; then
    DATA_CSV="data/NPORS_2024_cot_gpt5mini_20250913_133629.csv"
fi

# Digital Twin questions - start with subset for CI analysis
# Use fewer questions for CI mode as it's computationally intensive
QUESTIONS=(
    CARBONTAX
    CLEANENERGY
    MEDICAREALL
    IMMIGRATION
    WEALTHTAX
)

echo "======================================================================"
echo "Digital Twin Dataset CI Width Cost Comparison"
echo "======================================================================"
echo "Dataset: ${DATA_CSV}"
echo "Alpha=${ALPHA} Experiments=${NEXP} MinLabels=${MINLABELS} MaxLabels=${MAXLABELS}"
echo "CI Width=${CIWIDTH} Tolerance=${TOL} Label Step=${LABELSTEP}"
echo "Max Workers=${MAXWORKERS}"
echo "Questions: ${QUESTIONS[*]}"
echo "======================================================================"

# Check if dataset exists
if [[ ! -f "${DATA_CSV}" ]]; then
    echo "❌ Error: Dataset file not found: ${DATA_CSV}"
    echo "Please ensure the dataset exists or use --dataset cot|base"
    exit 1
fi

echo ""
echo "🔍 Running CI Width Cost Analysis..."
echo "This will find the minimum labels needed to achieve CI width ≤ ${CIWIDTH}"
echo ""

# Run twin CI width comparison
python estimators/twin_compare.py "${DATA_CSV}" \
    --questions "${QUESTIONS[@]}" \
    --ci-width "${CIWIDTH}" \
    --ci-tolerance "${TOL}" \
    --experiments "${NEXP}" \
    --min-labels "${MINLABELS}" \
    --max-labels "${MAXLABELS}" \
    --label-step "${LABELSTEP}" \
    --alpha "${ALPHA}" \
    --max-workers "${MAXWORKERS}"

echo ""
echo "✅ Digital Twin CI width cost analysis completed!"
echo "Results saved with timestamp prefix."
echo ""
echo "💡 To run full 10 questions, modify QUESTIONS array in script"
echo "   But expect longer runtime (~30-60 minutes)"
echo "======================================================================"