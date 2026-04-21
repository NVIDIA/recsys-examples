#!/bin/bash
# ============================================================================
# Batch Run All Experiments (Single Node)
# 
# Usage: ./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=<file> [options]
# 
# Environment Variables:
#   HSTU_ROOT            Path to examples/hstu directory (optional, defaults to pwd)
# 
# Options:
#   --exp-file=FILE      Experiment list file (required, format: exp_name,gin_options)
#   --hstu-root=PATH     Specify examples/hstu directory path (overrides env var and pwd)
#   --results-dir=PATH   Output directory (default: training/benchmark/results)
#   --nproc=N            Number of processes/GPUs (default: 8)
#   --nsys               Enable nsys profile sampling (traces all processes)
#   --dry-run            Print commands only, do not execute
#   --help               Show help information
# 
# Experiment List File Format:
#   # Comment lines start with #
#   exp_name,generate_gin_config_options
#   exp0_baseline,
#   exp1_cutlass,--kernel_backend cutlass
#   exp4_caching,--kernel_backend cutlass --recompute_layernorm --balanced_shuffler --caching
# 
# Notes:
#   - Executes all experiments in the list sequentially
#   - 10 second interval between experiments
#   - All outputs organized by timestamp: results/{batch_timestamp}/{exp_name}/
# 
# Output Directory Structure:
#   {results_dir}/
#   └── {batch_timestamp}/           # Timestamp of this batch run
#       ├── exp0_baseline/           # First experiment
#       │   ├── exp0_baseline_*.log
#       │   ├── exp0_baseline_*.gin
#       │   └── exp0_baseline_*.nsys-rep
#       ├── exp1_cutlass/            # Second experiment
#       │   ├── ...
#       └── summary.txt              # Batch experiment summary
# 
# Examples:
#   ./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=training/benchmark/experiments.txt
#   ./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=training/benchmark/experiments.txt --nproc=4
#   ./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=training/benchmark/experiments.txt --nsys
#   ./training/benchmark/scripts/run_all_experiments_local.sh --hstu-root=/path/to/examples/hstu --exp-file=training/benchmark/experiments.txt
#   ./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=training/benchmark/experiments.txt --results-dir=/data/results
# ============================================================================

set -e

# Default parameters
NPROC=8
EXP_FILE=""
ENABLE_NSYS=0
DRY_RUN=0
CUSTOM_HSTU_ROOT=""
CUSTOM_RESULTS_DIR=""

# Parse command line arguments (support both --arg value and --arg=value)
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-file=*)
            EXP_FILE="${1#*=}"
            shift
            ;;
        --exp-file)
            EXP_FILE="$2"
            shift 2
            ;;
        --hstu-root=*)
            CUSTOM_HSTU_ROOT="${1#*=}"
            shift
            ;;
        --hstu-root)
            CUSTOM_HSTU_ROOT="$2"
            shift 2
            ;;
        --results-dir=*)
            CUSTOM_RESULTS_DIR="${1#*=}"
            shift
            ;;
        --results-dir)
            CUSTOM_RESULTS_DIR="$2"
            shift 2
            ;;
        --nproc=*)
            NPROC="${1#*=}"
            shift
            ;;
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --nsys)
            ENABLE_NSYS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            head -50 "$0" | tail -47
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Set HSTU_ROOT (Priority: command line arg > env var > pwd)
# ============================================================================
if [ -n "$CUSTOM_HSTU_ROOT" ]; then
    HSTU_ROOT="$CUSTOM_HSTU_ROOT"
elif [ -z "$HSTU_ROOT" ]; then
    HSTU_ROOT=$(pwd)
fi

# Verify HSTU_ROOT directory exists (skip in dry-run mode)
if [ ${DRY_RUN} -eq 0 ]; then
    if [ ! -d "$HSTU_ROOT" ]; then
        echo "❌ Error: HSTU_ROOT directory does not exist: $HSTU_ROOT"
        exit 1
    fi

    # Verify directory structure
    if [ ! -d "$HSTU_ROOT/training" ]; then
        echo "❌ Error: Invalid HSTU_ROOT - missing 'training' subdirectory"
        echo "  HSTU_ROOT: $HSTU_ROOT"
        echo ""
        echo "Please ensure HSTU_ROOT points to 'recsys-examples/examples/hstu'"
        exit 1
    fi
fi

# Path configuration
SCRIPT_DIR="${HSTU_ROOT}/training/benchmark/scripts"
BENCHMARK_DIR="${HSTU_ROOT}/training/benchmark"

# Set output directory
if [ -n "$CUSTOM_RESULTS_DIR" ]; then
    if [[ ! "$CUSTOM_RESULTS_DIR" = /* ]]; then
        RESULTS_BASE="${HSTU_ROOT}/${CUSTOM_RESULTS_DIR}"
    else
        RESULTS_BASE="${CUSTOM_RESULTS_DIR}"
    fi
else
    RESULTS_BASE="${BENCHMARK_DIR}/results"
fi

# Create timestamped batch experiment directory
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUTPUT_DIR="${RESULTS_BASE}/${BATCH_TIMESTAMP}"

# Check experiment list file
if [ -z "$EXP_FILE" ]; then
    echo "⚠️  Missing experiment list file (--exp-file=<file>)"
    echo ""
    head -48 "$0" | tail -46
    exit 0
fi

# If relative path, make it relative to examples/hstu
if [[ ! "$EXP_FILE" = /* ]]; then
    EXP_FILE="${HSTU_ROOT}/${EXP_FILE}"
fi

# Read experiment list
declare -a EXP_NAMES
declare -a GIN_OPTIONS

if [ ! -f "$EXP_FILE" ]; then
    if [ ${DRY_RUN} -eq 1 ]; then
        echo "⚠️  Experiment list file not found: $EXP_FILE"
        echo "   No experiments to run."
        exit 0
    else
        echo "❌ Error: Experiment list file not found: $EXP_FILE"
        exit 1
    fi
fi

while IFS=',' read -r exp_name gin_opts || [ -n "$exp_name" ]; do
    # Skip empty lines and comments
    [[ -z "$exp_name" || "$exp_name" =~ ^[[:space:]]*# ]] && continue
    # Trim leading/trailing whitespace
    exp_name=$(echo "$exp_name" | xargs)
    gin_opts=$(echo "$gin_opts" | xargs)
    EXP_NAMES+=("$exp_name")
    GIN_OPTIONS+=("$gin_opts")
done < "$EXP_FILE"

if [ ${#EXP_NAMES[@]} -eq 0 ]; then
    if [ ${DRY_RUN} -eq 1 ]; then
        echo "⚠️  No experiments found in $EXP_FILE"
        exit 0
    else
        echo "❌ Error: No experiments found in $EXP_FILE"
        exit 1
    fi
fi

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 HSTU Benchmark Suite (Single Node)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - GPUs/Processes:   ${NPROC}"
echo "  - Experiment file:  ${EXP_FILE}"
echo "  - Batch timestamp:  ${BATCH_TIMESTAMP}"
echo ""
echo "NSYS Profiling: $([ ${ENABLE_NSYS} -eq 1 ] && echo 'ENABLED (all processes)' || echo 'DISABLED')"
echo ""
if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - Commands will be printed but not executed${NC}"
    echo ""
fi
echo "Experiments to run (${#EXP_NAMES[@]} total):"
for i in "${!EXP_NAMES[@]}"; do
    echo "  - ${EXP_NAMES[$i]}: ${GIN_OPTIONS[$i]:-'(defaults)'}"
done
echo ""
echo "Output directory:  ${BATCH_OUTPUT_DIR}"
echo ""

# ============================================================================
# DRY RUN Mode Processing
# ============================================================================
if [ ${DRY_RUN} -eq 1 ]; then
    echo "=========================================="
    echo -e "${YELLOW}DRY RUN: Commands that would be executed:${NC}"
    echo "=========================================="
    echo ""
    
    for i in "${!EXP_NAMES[@]}"; do
        exp="${EXP_NAMES[$i]}"
        gin_opts="${GIN_OPTIONS[$i]}"
        exp_num=$((i + 1))
        EXP_OUTPUT_DIR="${BATCH_OUTPUT_DIR}/${exp}"
        
        echo -e "[${exp_num}/${#EXP_NAMES[@]}] ${YELLOW}${exp}${NC}"
        echo "  Options:    ${gin_opts:-'(defaults)'}"
        echo "  Output dir: ${EXP_OUTPUT_DIR}"
        echo "  Command:"
        if [ ${ENABLE_NSYS} -eq 1 ]; then
            echo "    ${SCRIPT_DIR}/run_single_experiment_local.sh ${exp} \\"
            echo "        ${gin_opts} \\"
            echo "        --nproc=${NPROC} \\"
            echo "        --output-dir=${EXP_OUTPUT_DIR} \\"
            echo "        --hstu-root=${HSTU_ROOT} \\"
            echo "        --nsys"
        else
            echo "    ${SCRIPT_DIR}/run_single_experiment_local.sh ${exp} \\"
            echo "        ${gin_opts} \\"
            echo "        --nproc=${NPROC} \\"
            echo "        --output-dir=${EXP_OUTPUT_DIR} \\"
            echo "        --hstu-root=${HSTU_ROOT}"
        fi
        echo ""
    done
    
    echo "=========================================="
    echo -e "${YELLOW}DRY RUN completed. No experiments were executed.${NC}"
    echo "=========================================="
    exit 0
fi

# ============================================================================
# Actual Execution Mode
# ============================================================================

# Create batch output directory
mkdir -p ${BATCH_OUTPUT_DIR}

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date)

# Create summary file
SUMMARY_FILE="${BATCH_OUTPUT_DIR}/summary.txt"
cat > ${SUMMARY_FILE} << EOF
================================================================================
HSTU Benchmark Suite Summary
================================================================================

Batch Timestamp: ${BATCH_TIMESTAMP}
Started:         ${START_DATE}
GPUs/Processes:  ${NPROC}
NSYS Profiling:  $([ ${ENABLE_NSYS} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')
Experiment File: ${EXP_FILE}

Experiments (${#EXP_NAMES[@]} total):
$(for i in "${!EXP_NAMES[@]}"; do echo "  $((i+1)). ${EXP_NAMES[$i]} -> ${GIN_OPTIONS[$i]:-'(defaults)'}"; done)

--------------------------------------------------------------------------------
Results:
--------------------------------------------------------------------------------
EOF

# Run each experiment
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_EXPS=()

for i in "${!EXP_NAMES[@]}"; do
    exp="${EXP_NAMES[$i]}"
    gin_opts="${GIN_OPTIONS[$i]}"
    exp_num=$((i + 1))
    
    # Output directory for each experiment
    EXP_OUTPUT_DIR="${BATCH_OUTPUT_DIR}/${exp}"
    mkdir -p ${EXP_OUTPUT_DIR}
    
    echo ""
    echo "=========================================="
    echo -e "${YELLOW}[${exp_num}/${#EXP_NAMES[@]}] Running ${exp}...${NC}"
    echo "  Options:    ${gin_opts:-'(defaults)'}"
    echo "  Output dir: ${EXP_OUTPUT_DIR}"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo "  NSYS:       ENABLED"
    fi
    echo "=========================================="
    
    EXP_START=$(date +%s)
    EXP_START_DATE=$(date)
    
    # Build run command, pass output directory and HSTU_ROOT
    RUN_CMD="${SCRIPT_DIR}/run_single_experiment_local.sh ${exp} ${gin_opts} --nproc=${NPROC} --output-dir=${EXP_OUTPUT_DIR} --hstu-root=${HSTU_ROOT}"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        RUN_CMD="${RUN_CMD} --nsys"
    fi
    
    # Run experiment
    if ${RUN_CMD}; then
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        echo ""
        echo -e "${GREEN}✅ ${exp} completed successfully in ${EXP_DURATION}s${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Record to summary file
        echo "[${exp_num}/${#EXP_NAMES[@]}] ${exp}: ✅ SUCCESS (${EXP_DURATION}s)" >> ${SUMMARY_FILE}
        echo "    Started:  ${EXP_START_DATE}" >> ${SUMMARY_FILE}
        echo "    Duration: ${EXP_DURATION}s" >> ${SUMMARY_FILE}
        echo "    Output:   ${EXP_OUTPUT_DIR}" >> ${SUMMARY_FILE}
    else
        EXP_END=$(date +%s)
        EXP_DURATION=$((EXP_END - EXP_START))
        echo ""
        echo -e "${RED}❌ ${exp} failed after ${EXP_DURATION}s${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_EXPS+=("${exp}")
        
        # Record to summary file
        echo "[${exp_num}/${#EXP_NAMES[@]}] ${exp}: ❌ FAILED (${EXP_DURATION}s)" >> ${SUMMARY_FILE}
        echo "    Started:  ${EXP_START_DATE}" >> ${SUMMARY_FILE}
        echo "    Duration: ${EXP_DURATION}s" >> ${SUMMARY_FILE}
        echo "    Output:   ${EXP_OUTPUT_DIR}" >> ${SUMMARY_FILE}
        
        # Ask whether to continue
        echo ""
        echo "Do you want to continue with the next experiment? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Benchmark suite interrupted by user." >> ${SUMMARY_FILE}
            echo "Benchmark suite interrupted by user."
            break
        fi
    fi
    
    # Wait interval (except for the last experiment)
    if [ $i -lt $((${#EXP_NAMES[@]} - 1)) ]; then
        echo ""
        echo "⏱️  Waiting 10 seconds before next experiment..."
        sleep 10
    fi
done

# Calculate total time
END_TIME=$(date +%s)
END_DATE=$(date)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Complete summary file
cat >> ${SUMMARY_FILE} << EOF

--------------------------------------------------------------------------------
Summary:
--------------------------------------------------------------------------------
Finished:    ${END_DATE}
Total Time:  ${HOURS}h ${MINUTES}m ${SECONDS}s
Successful:  ${SUCCESS_COUNT}
Failed:      ${FAILED_COUNT}
EOF

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo "" >> ${SUMMARY_FILE}
    echo "Failed experiments:" >> ${SUMMARY_FILE}
    for exp in "${FAILED_EXPS[@]}"; do
        echo "  - ${exp}" >> ${SUMMARY_FILE}
    done
fi

echo "================================================================================" >> ${SUMMARY_FILE}

# Print summary
echo ""
echo "=========================================="
echo "📊 Benchmark Suite Summary"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${GREEN}✅ Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}❌ Failed:     ${FAILED_COUNT}${NC}"

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for exp in "${FAILED_EXPS[@]}"; do
        echo "  - ${exp}"
    done
fi

echo ""
echo "Output directory: ${BATCH_OUTPUT_DIR}"
echo ""
echo "Directory structure:"
echo "  ${BATCH_OUTPUT_DIR}/"
for exp in "${EXP_NAMES[@]}"; do
    echo "  ├── ${exp}/"
    echo "  │   ├── ${exp}_*.log"
    echo "  │   ├── ${exp}_*.gin"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo "  │   └── ${exp}_*.nsys-rep"
    fi
done
echo "  └── summary.txt"
echo ""
echo "Next steps:"
echo "  1. Check summary: cat ${SUMMARY_FILE}"
echo "  2. Check individual logs in ${BATCH_OUTPUT_DIR}/{exp_name}/"
if [ ${ENABLE_NSYS} -eq 1 ]; then
    echo "  3. Analyze nsys profiles: nsys-ui ${BATCH_OUTPUT_DIR}/{exp_name}/*.nsys-rep"
fi
echo ""

if [ ${FAILED_COUNT} -eq 0 ]; then
    echo -e "${GREEN}🎉 All experiments completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some experiments failed. Please check the logs.${NC}"
    exit 1
fi
