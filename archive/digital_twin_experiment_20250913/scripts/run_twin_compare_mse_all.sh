#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for MSE comparison on Digital Twin dataset.
# Usage:
#   ./scripts/run_twin_compare_mse_all.sh [OPTIONS]
#
# Options:
#   --dataset cot|base             Use preset dataset (default: cot)
#   --data-csv FILE                Direct path to CSV file
#   --results-csv FILE             Summary CSV output (default: twin_compare_mse.csv)
#   --runs-csv FILE                Per-repeat CSV output (default: twin_compare_runs.csv)
#   --experiments N                Number of experiments (default: 50, env: NEXP)
#   --labels N                     Labels per experiment (default: 600, env: LABELS)
#   --gamma N                      Gamma parameter (default: 0.5, env: GAMMA)
#   --alpha N                      Confidence level (default: 0.95, env: ALPHA)
#   --max-workers N                Max concurrent workers (default: 10, env: MAXWORKERS)
#   --help                         Show this help

show_help() {
    cat << EOF
Digital Twin Dataset MSE Comparison Script

USAGE:
    ./scripts/run_twin_compare_mse_all.sh [OPTIONS]

OPTIONS:
    --dataset cot|base       Use preset dataset (default: cot)
    --data-csv FILE          Direct path to CSV file
    --results-csv FILE       Summary CSV output (default: twin_compare_mse.csv)
    --runs-csv FILE          Per-repeat CSV output (default: twin_compare_runs.csv)
    --experiments N          Number of experiments (default: 50)
    --labels N               Labels per experiment (default: 600)
    --gamma N                Gamma parameter (default: 0.5)
    --alpha N                Confidence level (default: 0.95)
    --max-workers N          Max concurrent workers (default: 10)
    --help                   Show this help

EXAMPLES:
    # Use CoT dataset with defaults
    ./scripts/run_twin_compare_mse_all.sh --dataset cot

    # Use base dataset with custom labels
    ./scripts/run_twin_compare_mse_all.sh --dataset base --labels 400

    # Custom file with specific output names
    ./scripts/run_twin_compare_mse_all.sh --data-csv my_data.csv --results-csv my_results.csv

    # High precision run
    ./scripts/run_twin_compare_mse_all.sh --dataset cot --experiments 100 --labels 800

ENVIRONMENT VARIABLES:
    You can also set parameters via environment variables:
    NEXP=100 LABELS=500 ./scripts/run_twin_compare_mse_all.sh --dataset cot
EOF
}

# Defaults (can be overridden by env vars or command line)
DATA_CSV=""
RESULTS_CSV="twin_compare_mse.csv"
RUNS_CSV="twin_compare_runs.csv"
NEXP=${NEXP:-50}
LABELS=${LABELS:-600}
GAMMA=${GAMMA:-0.5}
ALPHA=${ALPHA:-0.95}
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
        --results-csv)
            RESULTS_CSV="$2"
            shift 2
            ;;
        --runs-csv)
            RUNS_CSV="$2"
            shift 2
            ;;
        --experiments)
            NEXP="$2"
            shift 2
            ;;
        --labels)
            LABELS="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
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

# Digital Twin questions (all 10 policy questions)
QUESTIONS=(
    CARBONTAX
    CLEANENERGY
    CLEANELEC
    MEDICAREALL
    PUBLICOPTION
    IMMIGRATION
    FAMILYLEAVE
    WEALTHTAX
    DEPORTATIONS
    MEDICVOUCHER
)

echo "======================================================================"
echo "Digital Twin Dataset MSE Comparison"
echo "======================================================================"
echo "Dataset: ${DATA_CSV}"
echo "Results CSV: ${RESULTS_CSV}"
echo "Runs CSV: ${RUNS_CSV}"
echo "Experiments=${NEXP} Labels=${LABELS} Gamma=${GAMMA} Alpha=${ALPHA}"
echo "Max Workers=${MAXWORKERS}"
echo "Questions: ${QUESTIONS[*]}"
echo "======================================================================"

# Check if dataset exists
if [[ ! -f "${DATA_CSV}" ]]; then
    echo "❌ Error: Dataset file not found: ${DATA_CSV}"
    echo "Please ensure the dataset exists or use --dataset cot|base"
    exit 1
fi

# Run twin comparison for all questions
python estimators/twin_compare.py "${DATA_CSV}" \
    --questions "${QUESTIONS[@]}" \
    --experiments "${NEXP}" \
    --labels "${LABELS}" \
    --alpha "${ALPHA}" \
    --gamma "${GAMMA}" \
    --max-workers "${MAXWORKERS}" \
    --summary-csv "${RESULTS_CSV}" \
    --runs-csv "${RUNS_CSV}"

echo ""
echo "✅ Digital Twin MSE comparison completed!"
echo "📊 Summary results: ${RESULTS_CSV}"
echo "📈 Detailed runs: ${RUNS_CSV}"
echo "======================================================================"