#!/bin/bash
# ============================================================================
# 4-way comparison: pipeline_type × enable_balanced_shuffler
#
# Runs pretrain_gr_ranking.py under 4 configurations and extracts achieved
# TFLOPS from logs for easy comparison.
#
# Usage:
#   cd examples/hstu
#   bash training/benchmark/run_balanced_pipeline_comparison.sh [GIN_CONFIG] [NPROC] [EXTRA_ENVS] [--nsys]
#
# Examples:
#   bash training/benchmark/run_balanced_pipeline_comparison.sh                       # defaults
#   bash training/benchmark/run_balanced_pipeline_comparison.sh my_config.gin 4       # custom config
#   bash training/benchmark/run_balanced_pipeline_comparison.sh my_config.gin 4 "" --nsys  # with nsys profiling
#
# Note: Ensure TrainerArgs.seed is set in the gin config file to ensure reproducible
# datasets across all 4 configurations. 
#
# IMPORTANT: If datasets differ between runs or configurations, check:
# 1. TrainerArgs.seed is set and consistent in gin config (e.g., 1234)
# 2. No other random operations occur before dataset initialization
# 3. CUDA RNG tracker state is properly initialized
# 4. ZIPF distribution uses np.random.zipf - ensure numpy random state is
#    properly reset before data generation (set_random_seed should handle this)
# 5. Verify that dataset initialization happens at the same point in the
#    execution flow for all configurations
# ============================================================================

set -euo pipefail

# ── arguments ──────────────────────────────────────────────────────────────
ENABLE_NSYS=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --nsys)
            ENABLE_NSYS=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

GIN_CONFIG="${1:-./training/configs/balanced_ranking_benchmark.gin}"
NPROC="${2:-8}"
EXTRA_ENVS="${3:-}"    # e.g. "LOG_LEVEL=DEBUG PRINT_LOAD_BALANCE=1"
MASTER_PORT="${MASTER_PORT:-6000}"

# ── derived paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HSTU_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$HSTU_ROOT"
PYTHONPATH="${PYTHONPATH:-}:$(realpath ../)"
export PYTHONPATH


TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="./training/benchmark/results/balanced_pipeline_4way_comparison/${TIMESTAMP}"
mkdir -p "$RESULT_DIR"
SUMMARY_FILE="${RESULT_DIR}/summary_${TIMESTAMP}.txt"

# ── 4 configurations ──────────────────────────────────────────────────────
# Note: Ensure TrainerArgs.seed is set in the gin config file to ensure
# reproducible datasets across all 4 configurations.
declare -a PIPELINE_TYPES=("prefetch" "none"     "none"     "prefetch" )
declare -a SHUFFLER_FLAGS=("True"   "True"     "False"    "False"  )
declare -a CONFIG_NAMES=(
    "prefetch_with_shuffler"
    "no_pipeline_with_shuffler"
    "prefetch_no_shuffler"
    "no_pipeline_no_shuffler"
)

echo "============================================================"
echo " Balanced Pipeline 4-Way Comparison Benchmark"
echo " gin-config : ${GIN_CONFIG}"
echo " nproc      : ${NPROC}"
echo " result dir : ${RESULT_DIR}"
echo " timestamp  : ${TIMESTAMP}"
if [ "$ENABLE_NSYS" = true ]; then
    echo " nsys profiler: ENABLED"
fi
echo "============================================================"
echo ""

printf "%-40s %12s %12s\n" "Configuration" "TFLOPS" "Time(ms)" | tee "$SUMMARY_FILE"
printf "%-40s %12s %12s\n" "----------------------------------------" "------------" "------------" | tee -a "$SUMMARY_FILE"

for i in "${!CONFIG_NAMES[@]}"; do
    NAME="${CONFIG_NAMES[$i]}"
    PIPELINE="${PIPELINE_TYPES[$i]}"
    SHUFFLER="${SHUFFLER_FLAGS[$i]}"

    LOG_FILE="${RESULT_DIR}/${NAME}_${TIMESTAMP}.log"
    NSYS_FILE="${RESULT_DIR}/${NAME}_${TIMESTAMP}.nsys-rep"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo " [$((i+1))/4] ${NAME}"
    echo "   pipeline_type=${PIPELINE}  enable_balanced_shuffler=${SHUFFLER}"
    if [ "$ENABLE_NSYS" = true ]; then
        echo "   nsys output: ${NSYS_FILE}"
    fi
    echo "────────────────────────────────────────────────────────────"

    # Build a temp gin config: base config + overrides appended at the end
    # (gin uses last-write-wins for duplicate keys)
    # 
    # IMPORTANT: For reproducible datasets across all 4 configurations:
    # 1. TrainerArgs.seed must be set in the base gin config (e.g., 1234)
    # 2. Ensure no random operations occur before dataset initialization
    # 3. Note that ZIPF distribution uses np.random.zipf, which depends on
    #    numpy's random state. Make sure numpy random state is properly reset
    #    before data generation in the training script.
    TMP_GIN=$(mktemp /tmp/gin_balanced_pipeline_4way_XXXXXX.gin)
    cat "${GIN_CONFIG}" > "$TMP_GIN"
    cat >> "$TMP_GIN" << GIN_EOF

# ── balanced pipeline 4-way override ──
TrainerArgs.pipeline_type = '${PIPELINE}'
TrainerArgs.enable_balanced_shuffler = ${SHUFFLER}
GIN_EOF

    # Run training (with optional nsys profiling)
    set +e
    if [ "$ENABLE_NSYS" = true ]; then
        eval ${EXTRA_ENVS} \
        nsys profile \
            -f true \
            -s none \
            -t cuda,nvtx \
            -c cudaProfilerApi \
            --cpuctxsw none \
            --cuda-flush-interval 100 \
            --capture-range-end=stop \
            --cuda-graph-trace=node \
            -o "${NSYS_FILE}" \
            torchrun \
                --nproc_per_node="${NPROC}" \
                --master_addr localhost \
                --master_port "${MASTER_PORT}" \
                training/pretrain_gr_ranking.py \
                --gin-config-file "${TMP_GIN}" \
                2>&1 | tee "$LOG_FILE"
    else
        eval ${EXTRA_ENVS} \
        torchrun \
            --nproc_per_node="${NPROC}" \
            --master_addr localhost \
            --master_port "${MASTER_PORT}" \
            training/pretrain_gr_ranking.py \
            --gin-config-file "${TMP_GIN}" \
            2>&1 | tee "$LOG_FILE"
    fi
    EXIT_CODE=$?
    set -e

    rm -f "$TMP_GIN"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ⚠  Run failed (exit code ${EXIT_CODE}), skipping."
        printf "%-40s %12s %12s\n" "${NAME}" "FAILED" "-" | tee -a "$SUMMARY_FILE"
        continue
    fi

    # Extract the max TFLOPS and corresponding elapsed_time from log (rank 0 prints these)
    # Format: "[train] [iter X, tokens Y, elapsed_time Z ms, achieved FLOPS W TFLOPS]: loss ..."
    # We need to find the line with max TFLOPS and extract elapsed_time from the same line.
    # Use awk to extract both values from each line, then find the line with max TFLOPS
    # NOTE: The Python logger may wrap long lines, so "achieved FLOPS" and the
    # actual TFLOPS number can end up on separate lines, e.g.:
    #   ... elapsed_time 18895.34 ms, achieved FLOPS
    #                              900.67 TFLOPS]: loss ...
    # We use sed to join the continuation line back, then parse with awk.
    # The regex uses " +" to tolerate the extra whitespace after joining.
    RESULT=$(sed -n '/achieved FLOPS/{N;s/\n/ /;p}' "$LOG_FILE" | \
        awk '{
            # Extract elapsed_time: find pattern "elapsed_time X.XXX ms"
            time_val = ""
            if (match($0, /elapsed_time [0-9.]+ ms/)) {
                time_str = substr($0, RSTART, RLENGTH)
                gsub(/elapsed_time | ms/, "", time_str)
                time_val = time_str
            }
            # Extract TFLOPS: " +" handles extra spaces from line-join
            tflops_val = ""
            if (match($0, /achieved FLOPS +[0-9.]+ TFLOPS/)) {
                tflops_str = substr($0, RSTART, RLENGTH)
                gsub(/achieved FLOPS +| TFLOPS/, "", tflops_str)
                tflops_val = tflops_str
            }
            if (time_val != "" && tflops_val != "") {
                print tflops_val, time_val
            }
        }' | sort -g -k1 | tail -1)
    
    if [ -n "$RESULT" ]; then
        LAST_TFLOPS_LINE=$(echo "$RESULT" | awk '{print $1}')
        LAST_TIME_LINE=$(echo "$RESULT" | awk '{print $2}')
    else
        LAST_TFLOPS_LINE="N/A"
        LAST_TIME_LINE="N/A"
    fi

    printf "%-40s %12s %12s\n" "${NAME}" "${LAST_TFLOPS_LINE}" "${LAST_TIME_LINE}" | tee -a "$SUMMARY_FILE"
done

echo ""
echo "============================================================"
echo " Summary saved to: ${SUMMARY_FILE}"
echo " Full logs in:     ${RESULT_DIR}/"
if [ "$ENABLE_NSYS" = true ]; then
    echo " NSys profiles in: ${RESULT_DIR}/"
fi
echo "============================================================"
echo ""
cat "$SUMMARY_FILE"
