#!/bin/bash
# ============================================================================
# Single Experiment Runner (Single Node, unified dispatcher)
#
# Runs ONE experiment locally for the chosen benchmark type.
#
# Usage:
#   ./training/benchmark/scripts/run_single_experiment_local.sh <exp_name> [options]
#
# Common Options:
#   --benchmark-type=TYPE  e2e | hstu-layer | hstu-attn-kernel (default: e2e)
#   --exp-args=ARGS        Raw CLI args passed to the benchmark Python entry.
#                          For e2e, these are gin options for
#                          generate_gin_config.py (same as before). For
#                          hstu-layer / hstu-attn-kernel, these are passed
#                          verbatim to the Python benchmark.
#   --hstu-root=PATH       Specify examples/hstu directory path
#   --output-dir=PATH      Output directory (default: results/{timestamp}/{exp_name}/)
#   --nsys                 Run under nsys profile. For e2e: wraps torchrun.
#                          For hstu-layer: wraps the benchmark with
#                          `nsys profile -c cudaProfilerApi` and auto-injects
#                          `--profile True` into --exp-args.
#   --dry-run              Print commands only, do not execute
#   --help,-h              Show this help
#
# Examples:
#   # E2E baseline (gin args inside --exp-args)
#   ./run_single_experiment_local.sh exp0_baseline \
#       --exp-args="--kernel_backend cutlass --caching --ratio 0.1"
#
#   # HSTU layer benchmark
#   ./run_single_experiment_local.sh baseline \
#       --benchmark-type=hstu-layer \
#       --exp-args="--layer-type fused --kernel-backend cutlass --dim-per-head 256 --num-heads 4 --max-seqlen 4096 --batchsize 32 --full-sequence True --dtype bfloat16"
#
#   # HSTU attention kernel heatmap
#   ./run_single_experiment_local.sh default \
#       --benchmark-type=hstu-attn-kernel \
#       --exp-args="--gin-config-file training/configs/benchmark_ranking.gin --batch-sizes 1,2,4,8,16,32,64,128 --seqlens 64,128,256,512,1024,4096"
# ============================================================================

set -e
set -o pipefail
export PYTHONWARNINGS="ignore"
# Default values
BENCHMARK_TYPE="e2e"
EXP_ARGS=""
NPROC=${NPROC:-8}
ENABLE_NSYS=0
CUSTOM_OUTPUT_DIR=""
DRY_RUN=0
CUSTOM_HSTU_ROOT=""

# Parse arguments
EXP_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-type=*)
            BENCHMARK_TYPE="${1#*=}"
            shift
            ;;
        --benchmark-type)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        --exp-args=*)
            EXP_ARGS="${1#*=}"
            shift
            ;;
        --exp-args)
            EXP_ARGS="$2"
            shift 2
            ;;
        # Other options (support both --arg value and --arg=value)
        --hstu-root=*)
            CUSTOM_HSTU_ROOT="${1#*=}"
            shift
            ;;
        --hstu-root)
            CUSTOM_HSTU_ROOT="$2"
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
        --output-dir=*)
            CUSTOM_OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --output-dir)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            sed -n '2,43p' "$0"
            exit 0
            ;;
        -*)
            echo "❌ Error: Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$EXP_NAME" ]; then
                EXP_NAME="$1"
            fi
            shift
            ;;
    esac
done

# Argument validation
if [ -z "$EXP_NAME" ]; then
    echo "❌ Error: Missing experiment name"
    echo "Usage: $0 <exp_name> --benchmark-type=TYPE --exp-args=\"...\" [options]"
    echo "Run with --help for full usage."
    exit 1
fi

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
RESULTS_BASE="${BENCHMARK_DIR}/results"
GIN_GENERATOR="${SCRIPT_DIR}/generate_gin_config.py"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Determine output directory. Prefix the timestamped batch dir with the
# benchmark type (e2e / hstu_layer / hstu_kernel) for at-a-glance
# classification under results/.
case "${BENCHMARK_TYPE}" in
    e2e)              BATCH_PREFIX="e2e" ;;
    hstu-layer)       BATCH_PREFIX="hstu_layer" ;;
    hstu-attn-kernel) BATCH_PREFIX="hstu_kernel" ;;
    *)                BATCH_PREFIX="${BENCHMARK_TYPE//-/_}" ;;
esac

if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    # Use custom output directory (caller-provided, typically from run_all /
    # submit_all which have already prefixed the batch dir themselves).
    if [[ ! "$CUSTOM_OUTPUT_DIR" = /* ]]; then
        OUTPUT_DIR="${HSTU_ROOT}/${CUSTOM_OUTPUT_DIR}"
    else
        OUTPUT_DIR="${CUSTOM_OUTPUT_DIR}"
    fi
else
    # Default: results/<type>_{timestamp}/{exp_name}/
    OUTPUT_DIR="${RESULTS_BASE}/${BATCH_PREFIX}_${TIMESTAMP}/${EXP_NAME}"
fi

# Only create directory in non-dry-run mode
if [ ${DRY_RUN} -eq 0 ]; then
    mkdir -p ${OUTPUT_DIR}
fi

# ============================================================================
# Common env + log (shared by all benchmark types)
# ============================================================================
export PYTHONPATH="${HSTU_ROOT}/..:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export FILL_DYNAMICEMB_TABLES=1
# NOTE: Do NOT set CUDA_MODULE_LOADING=EAGER here. It causes NCCL
# "invalid resource handle" errors because eager loading pre-initializes
# all CUDA modules before fork, and those handles are not fork-safe.

LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}.log"
CONFIG_FILE=""
NSYS_OUTPUT=""

# Color output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 Running Experiment: ${EXP_NAME}"
echo "=========================================="
echo ""
echo "Benchmark type:  ${BENCHMARK_TYPE}"
echo "Output dir:      ${OUTPUT_DIR}"

case "$BENCHMARK_TYPE" in
    e2e|hstu-layer|hstu-attn-kernel) ;;
    *)
        echo "❌ Error: Unknown --benchmark-type: ${BENCHMARK_TYPE} (supported: e2e, hstu-layer, hstu-attn-kernel)"
        exit 1
        ;;
esac

case "$BENCHMARK_TYPE" in

e2e)
    GIN_GEN_ARGS="${EXP_ARGS}"
    CONFIG_FILE="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}.gin"

    echo "GIN options:     ${GIN_GEN_ARGS}"
    echo "GPUs:            ${NPROC}"
    echo "NSYS Profiling:  $([ ${ENABLE_NSYS} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
    [ ${DRY_RUN} -eq 1 ] && echo -e "${YELLOW}⚠️  DRY RUN MODE${NC}"
    echo "=========================================="
    echo ""

    echo "📝 Generating gin config file..."
    echo "   Command: python ${GIN_GENERATOR} ${GIN_GEN_ARGS} -o ${CONFIG_FILE}"
    echo ""

    if [ ${DRY_RUN} -eq 0 ]; then
        python ${GIN_GENERATOR} ${GIN_GEN_ARGS} | tee ${CONFIG_FILE}
        echo ""
        echo -e "${GREEN}✅ Config saved to: ${CONFIG_FILE}${NC}"
    else
        echo -e "${CYAN}Generated config (not saved):${NC}"
        echo ""
        python ${GIN_GENERATOR} ${GIN_GEN_ARGS}
        echo ""
        echo -e "${CYAN}Would save to: ${CONFIG_FILE}${NC}"
    fi
    echo ""

    if [ ${DRY_RUN} -eq 1 ]; then
        HOSTNAME_SHORT=$(hostname -s)
        NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}_${HOSTNAME_SHORT}"
        echo -e "${CYAN}Would execute:${NC}"
        echo ""
        if [ ${ENABLE_NSYS} -eq 1 ]; then
            echo "nsys profile -o \"${NSYS_OUTPUT}\" ... torchrun --standalone --nproc_per_node=${NPROC} training/pretrain_gr_ranking.py --gin-config-file ${CONFIG_FILE} 2>&1 | tee -a ${LOG_FILE}"
        else
            echo "torchrun --standalone --nproc_per_node=${NPROC} training/pretrain_gr_ranking.py --gin-config-file ${CONFIG_FILE} 2>&1 | tee -a ${LOG_FILE}"
        fi
        echo ""
        echo -e "${YELLOW}DRY RUN completed. No commands were executed.${NC}"
        exit 0
    fi

    echo "📝 Logging to: ${LOG_FILE}"
    nvidia-smi > ${LOG_FILE}
    echo "⏰ Started at: $(date)"
    echo ""

    cd ${HSTU_ROOT}

    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo "🔬 Running with NVIDIA Nsight Systems profiling..."
        echo ""

        HOSTNAME_SHORT=$(hostname -s)
        NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}_${HOSTNAME_SHORT}"
        echo "📊 nsys output: ${NSYS_OUTPUT}.nsys-rep"
        echo ""

        CUBLAS_NVTX_LEVEL=2 \
        nsys profile \
            -o "${NSYS_OUTPUT}" \
            -f true \
            -s none \
            -t cuda,cublas-verbose,nvtx \
            -c cudaProfilerApi \
            --cpuctxsw none \
            --cuda-flush-interval 100 \
            --capture-range-end=stop \
            --cuda-graph-trace=node \
            torchrun \
                --standalone \
                --nproc_per_node=${NPROC} \
                training/pretrain_gr_ranking.py \
                --gin-config-file ${CONFIG_FILE} \
            2>&1 | tee -a ${LOG_FILE}

        EXIT_CODE=${PIPESTATUS[0]}
        echo ""
        echo "📊 nsys profile saved to: ${NSYS_OUTPUT}.nsys-rep"
    else
        torchrun \
            --standalone \
            --nproc_per_node=${NPROC} \
            training/pretrain_gr_ranking.py \
            --gin-config-file ${CONFIG_FILE} \
            2>&1 | tee -a ${LOG_FILE}
        EXIT_CODE=${PIPESTATUS[0]}
    fi
    ;;

hstu-layer)
    # Auto-inject --output-dir so memory snapshot lands in per-exp results dir.
    if [[ ! " ${EXP_ARGS} " =~ [[:space:]]--output-dir([[:space:]=]|$) ]]; then
        EXP_ARGS="${EXP_ARGS} --output-dir ${OUTPUT_DIR}"
    fi
    # When --nsys is set, inject --profile True so the Python script calls
    # torch.cuda.profiler.start/stop at [profiler_start, profiler_end];
    # nsys's -c cudaProfilerApi hooks those calls and captures only the
    # steady-state window.
    NSYS_OUTPUT=""
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        if [[ ! " ${EXP_ARGS} " =~ [[:space:]]--profile([[:space:]=]|$) ]]; then
            EXP_ARGS="${EXP_ARGS} --profile True"
        fi
        NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}"
    fi

    echo "Exp args:        ${EXP_ARGS}"
    echo "NSYS:            $([ ${ENABLE_NSYS} -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
    [ ${DRY_RUN} -eq 1 ] && echo -e "${YELLOW}⚠️  DRY RUN MODE${NC}"
    echo "=========================================="
    echo ""

    cd ${HSTU_ROOT}
    PY_CMD="python -u training/benchmark/scripts/hstu_layer_benchmark.py run ${EXP_ARGS}"
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        CMD="nsys profile -o ${NSYS_OUTPUT} -f true -s none -t cuda,cublas-verbose,nvtx -c cudaProfilerApi --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node ${PY_CMD}"
    else
        CMD="${PY_CMD}"
    fi
    echo "📝 Logging to: ${LOG_FILE}"
    echo "   Command: ${CMD}"
    echo ""

    if [ ${DRY_RUN} -eq 1 ]; then
        echo -e "${CYAN}Would execute:${NC}"
        echo "  ${CMD} 2>&1 | tee -a ${LOG_FILE}"
        echo -e "${YELLOW}DRY RUN completed. No commands were executed.${NC}"
        exit 0
    fi

    ${CMD} 2>&1 | tee -a ${LOG_FILE}
    EXIT_CODE=${PIPESTATUS[0]}
    ;;

hstu-attn-kernel)
    # Auto-inject --output-dir so heatmap PNGs + JSON land in per-exp results dir.
    if [[ ! " ${EXP_ARGS} " =~ [[:space:]]--output-dir([[:space:]=]|$) ]]; then
        EXP_ARGS="${EXP_ARGS} --output-dir ${OUTPUT_DIR}"
    fi
    echo "Exp args:        ${EXP_ARGS}"
    [ ${DRY_RUN} -eq 1 ] && echo -e "${YELLOW}⚠️  DRY RUN MODE${NC}"
    echo "=========================================="
    echo ""

    cd ${HSTU_ROOT}
    CMD="python -u training/benchmark/scripts/hstu_attn_kernel_benchmark.py ${EXP_ARGS}"
    echo "📝 Logging to: ${LOG_FILE}"
    echo "   Command: ${CMD}"
    echo ""

    if [ ${DRY_RUN} -eq 1 ]; then
        echo -e "${CYAN}Would execute:${NC}"
        echo "  ${CMD} 2>&1 | tee -a ${LOG_FILE}"
        echo -e "${YELLOW}DRY RUN completed. No commands were executed.${NC}"
        exit 0
    fi

    ${CMD} 2>&1 | tee -a ${LOG_FILE}
    EXIT_CODE=${PIPESTATUS[0]}
    ;;

esac

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment ${EXP_NAME} completed successfully!"
else
    echo "❌ Experiment ${EXP_NAME} failed with exit code: ${EXIT_CODE}"
fi
echo "⏰ Finished at: $(date)"
echo "📝 Log saved to: ${LOG_FILE}"
[ -n "${CONFIG_FILE}" ] && echo "📄 Config file: ${CONFIG_FILE}"
if [ ${ENABLE_NSYS} -eq 1 ] && [ -n "${NSYS_OUTPUT}" ]; then
    echo "📊 nsys profile: ${NSYS_OUTPUT}.nsys-rep"
fi
echo "=========================================="

exit $EXIT_CODE
