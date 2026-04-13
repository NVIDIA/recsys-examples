#!/bin/bash
# ============================================================================
# Single Experiment Runner (Single Node)
# 
# Usage: ./training/benchmark/scripts/run_single_experiment_local.sh <exp_name> [options]
# 
# Environment Variables:
#   HSTU_ROOT         Path to examples/hstu directory (optional, defaults to pwd)
# 
# Optimization Switches (same as generate_gin_config.py):
#   --kernel_backend [triton|cutlass]   Attention kernel backend (default: triton)
#   --recompute_layernorm               Enable LayerNorm recompute (default: False)
#   --balanced_shuffler                 Enable workload balancer (default: False)
#   --caching                           Enable DynamicEmb caching (default: False)
#   --ratio FLOAT                       GPU cache ratio (default: 0, auto 0.1 if caching)
#   --evict [lru|lfu]                   Eviction strategy (default: lru)
#   --pipeline_type [none|prefetch]     Pipeline type (default: none)
#   --tp_size INT                       Tensor Parallel size (default: 1)
# 
# Other Options:
#   --hstu-root=PATH  Specify examples/hstu directory path (overrides env var and pwd)
#   --nproc=N         Number of processes/GPUs (default: 8)
#   --nsys            Enable nsys profile sampling (traces all child processes/ranks)
#   --output-dir=PATH Output directory (default: results/{timestamp}/{exp_name}/)
#   --dry-run         Print commands only, do not execute
# 
# Output Directory Structure:
#   {output_dir}/
#   ├── {exp_name}_{timestamp}.log
#   ├── {exp_name}_{timestamp}.gin          (generated config)
#   └── {exp_name}_{timestamp}_{hostname}.nsys-rep  (if --nsys enabled)
# 
# Examples:
#   # Baseline (all defaults)
#   ./training/benchmark/scripts/run_single_experiment_local.sh exp0_baseline
#   
#   # CUTLASS attention
#   ./training/benchmark/scripts/run_single_experiment_local.sh exp1_cutlass --kernel_backend cutlass
#   
#   # Full optimization
#   ./training/benchmark/scripts/run_single_experiment_local.sh exp8_full \
#       --kernel_backend cutlass --recompute_layernorm --balanced_shuffler \
#       --caching --evict lfu --pipeline_type prefetch --tp_size 2
# ============================================================================

set -e
export PYTHONWARNINGS="ignore"
# Default values
NPROC=${NPROC:-8}
ENABLE_NSYS=0
CUSTOM_OUTPUT_DIR=""
DRY_RUN=0
CUSTOM_HSTU_ROOT=""

# Optimization switch defaults (same as generate_gin_config.py)
KERNEL_BACKEND="triton"
RECOMPUTE_LAYERNORM=0
BALANCED_SHUFFLER=0
CACHING=0
RATIO=0
EVICT="lru"
PIPELINE_TYPE="none"
TP_SIZE=1
VALUE_DIST="uniform"
VALUE_DIST_ALPHA=1.2

# Parse arguments
EXP_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        # Optimization switches (support both --arg value and --arg=value)
        --kernel_backend=*)
            KERNEL_BACKEND="${1#*=}"
            shift
            ;;
        --kernel_backend)
            KERNEL_BACKEND="$2"
            shift 2
            ;;
        --recompute_layernorm)
            RECOMPUTE_LAYERNORM=1
            shift
            ;;
        --balanced_shuffler)
            BALANCED_SHUFFLER=1
            shift
            ;;
        --caching)
            CACHING=1
            shift
            ;;
        --ratio=*)
            RATIO="${1#*=}"
            shift
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --evict=*)
            EVICT="${1#*=}"
            shift
            ;;
        --evict)
            EVICT="$2"
            shift 2
            ;;
        --pipeline_type=*)
            PIPELINE_TYPE="${1#*=}"
            shift
            ;;
        --pipeline_type)
            PIPELINE_TYPE="$2"
            shift 2
            ;;
        --tp_size=*)
            TP_SIZE="${1#*=}"
            shift
            ;;
        --tp_size)
            TP_SIZE="$2"
            shift 2
            ;;
        --value_dist=*)
            VALUE_DIST="${1#*=}"
            shift
            ;;
        --value_dist)
            VALUE_DIST="$2"
            shift 2
            ;;
        --value_dist_alpha=*)
            VALUE_DIST_ALPHA="${1#*=}"
            shift
            ;;
        --value_dist_alpha)
            VALUE_DIST_ALPHA="$2"
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
            head -47 "$0" | tail -44
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
    echo "Usage: $0 <exp_name> [optimization switches] [options]"
    echo ""
    echo "Optimization Switches (support --arg VALUE or --arg=VALUE):"
    echo "  --kernel_backend VALUE   triton|cutlass (default: triton)"
    echo "  --recompute_layernorm    Enable LayerNorm recomputation (default: off)"
    echo "  --balanced_shuffler      Enable workload balancer (default: off)"
    echo "  --caching                Enable DynamicEmb caching (default: off)"
    echo "  --ratio VALUE            GPU cache ratio 0.0-1.0 (default: 0)"
    echo "  --evict VALUE            lru|lfu (default: lru)"
    echo "  --pipeline_type VALUE    none|prefetch (default: none)"
    echo "  --tp_size VALUE          Tensor Parallel degree (default: 1)"
    echo ""
    echo "Other Options (support --arg VALUE or --arg=VALUE):"
    echo "  --hstu-root PATH    Specify examples/hstu directory path"
    echo "  --nproc N           Number of processes/GPUs (default: 8)"
    echo "  --nsys              Enable nsys profile sampling"
    echo "  --output-dir PATH   Output directory"
    echo "  --dry-run           Print commands only"
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

# Determine output directory
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    # Use custom output directory
    if [[ ! "$CUSTOM_OUTPUT_DIR" = /* ]]; then
        # Relative path, relative to examples/hstu
        OUTPUT_DIR="${HSTU_ROOT}/${CUSTOM_OUTPUT_DIR}"
    else
        OUTPUT_DIR="${CUSTOM_OUTPUT_DIR}"
    fi
else
    # Default: results/{timestamp}/{exp_name}/
    OUTPUT_DIR="${RESULTS_BASE}/${TIMESTAMP}/${EXP_NAME}"
fi

# Only create directory in non-dry-run mode
if [ ${DRY_RUN} -eq 0 ]; then
    mkdir -p ${OUTPUT_DIR}
fi

# ============================================================================
# Build generate_gin_config.py arguments
# ============================================================================
GIN_GEN_ARGS=""
GIN_GEN_ARGS="${GIN_GEN_ARGS} --kernel_backend ${KERNEL_BACKEND}"
[ ${RECOMPUTE_LAYERNORM} -eq 1 ] && GIN_GEN_ARGS="${GIN_GEN_ARGS} --recompute_layernorm"
[ ${BALANCED_SHUFFLER} -eq 1 ] && GIN_GEN_ARGS="${GIN_GEN_ARGS} --balanced_shuffler"
[ ${CACHING} -eq 1 ] && GIN_GEN_ARGS="${GIN_GEN_ARGS} --caching"
GIN_GEN_ARGS="${GIN_GEN_ARGS} --ratio ${RATIO}"
GIN_GEN_ARGS="${GIN_GEN_ARGS} --evict ${EVICT}"
GIN_GEN_ARGS="${GIN_GEN_ARGS} --pipeline_type ${PIPELINE_TYPE}"
GIN_GEN_ARGS="${GIN_GEN_ARGS} --tp_size ${TP_SIZE}"
GIN_GEN_ARGS="${GIN_GEN_ARGS} --value_dist ${VALUE_DIST}"
GIN_GEN_ARGS="${GIN_GEN_ARGS} --value_dist_alpha ${VALUE_DIST_ALPHA}"

# Generated config file path
CONFIG_FILE="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}.gin"

# Color output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 Running Experiment: ${EXP_NAME}"
echo "=========================================="
echo ""
echo "Optimization Switches:"
echo "  kernel_backend:      ${KERNEL_BACKEND}"
echo "  recompute_layernorm: $([ ${RECOMPUTE_LAYERNORM} -eq 1 ] && echo 'True' || echo 'False')"
echo "  balanced_shuffler:   $([ ${BALANCED_SHUFFLER} -eq 1 ] && echo 'True' || echo 'False')"
echo "  caching:             $([ ${CACHING} -eq 1 ] && echo 'True' || echo 'False')"
echo "  ratio:               ${RATIO}"
echo "  evict:               ${EVICT}"
echo "  pipeline_type:       ${PIPELINE_TYPE}"
echo "  tp_size:             ${TP_SIZE}"
echo ""
echo "Output dir:  ${OUTPUT_DIR}"
echo "GPUs:        ${NPROC}"
echo ""
echo "NSYS Profiling: $([ ${ENABLE_NSYS} -eq 1 ] && echo 'ENABLED (all processes)' || echo 'DISABLED')"
if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - Commands will be printed but not executed${NC}"
fi
echo "=========================================="
echo ""

# ============================================================================
# Generate gin config file
# ============================================================================
echo "📝 Generating gin config file..."
echo "   Command: python ${GIN_GENERATOR} ${GIN_GEN_ARGS} -o ${CONFIG_FILE}"
echo ""

if [ ${DRY_RUN} -eq 0 ]; then
    # Generate config and also print to terminal
    python ${GIN_GENERATOR} ${GIN_GEN_ARGS} | tee ${CONFIG_FILE}
    echo ""
    echo -e "${GREEN}✅ Config saved to: ${CONFIG_FILE}${NC}"
else
    # In dry-run mode, generate and print config to terminal (not saved to file)
    echo -e "${CYAN}Generated config (not saved):${NC}"
    echo ""
    python ${GIN_GENERATOR} ${GIN_GEN_ARGS}
    echo ""
    echo -e "${CYAN}Would save to: ${CONFIG_FILE}${NC}"
fi
echo ""

# Environment variable setup (add HSTU_ROOT's parent directory to PYTHONPATH)
export PYTHONPATH="${HSTU_ROOT}/..:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FILL_DYNAMICEMB_TABLES=1
# NOTE: Do NOT set CUDA_MODULE_LOADING=EAGER here. It causes NCCL
# "invalid resource handle" errors because eager loading pre-initializes
# all CUDA modules before fork, and those handles are not fork-safe.
# export CUDA_MODULE_LOADING=EAGER


# Log file
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

# ============================================================================
# DRY RUN Mode: Print commands only, do not execute
# ============================================================================
if [ ${DRY_RUN} -eq 1 ]; then
    HOSTNAME_SHORT=$(hostname -s)
    NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}_${HOSTNAME_SHORT}"
    
    echo -e "${CYAN}Would execute:${NC}"
    echo ""
    
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo "nsys profile \\"
        echo "    -o \"${NSYS_OUTPUT}\" \\"
        echo "    -f true \\"
        echo "    -s none \\"
        echo "    -t cuda,cublas-verbose,nvtx \\"
        echo "    -c cudaProfilerApi \\"
        echo "    --cpuctxsw none \\"
        echo "    --cuda-flush-interval 100 \\"
        echo "    --capture-range-end=stop \\"
        echo "    --cuda-graph-trace=node \\"
        echo "    torchrun \\"
        echo "        --standalone \\"
        echo "        --nproc_per_node=${NPROC} \\"
        echo "        training/pretrain_gr_ranking.py \\"
        echo "        --gin-config-file ${CONFIG_FILE} \\"
        echo "    2>&1 | tee -a ${LOG_FILE}"
    else
        echo "torchrun \\"
        echo "    --standalone \\"
        echo "    --nproc_per_node=${NPROC} \\"
        echo "    training/pretrain_gr_ranking.py \\"
        echo "    --gin-config-file ${CONFIG_FILE} \\"
        echo "    2>&1 | tee -a ${LOG_FILE}"
    fi
    
    echo ""
    echo -e "${YELLOW}DRY RUN completed. No commands were executed.${NC}"
    exit 0
fi

# ============================================================================
# Actual Execution Mode
# ============================================================================

# Start training
echo "📝 Logging to: ${LOG_FILE}"
nvidia-smi > ${LOG_FILE}

echo "⏰ Started at: $(date)"
echo ""

# Change to HSTU_ROOT directory before running training
cd ${HSTU_ROOT}

if [ ${ENABLE_NSYS} -eq 1 ]; then
    # ========================================================================
    # nsys profile mode enabled
    # Use nsys to wrap the entire torchrun command, nsys will automatically trace all child processes
    # Output file format: {exp_name}_{timestamp}_{hostname}
    # ========================================================================
    echo "🔬 Running with NVIDIA Nsight Systems profiling..."
    echo ""
    
    HOSTNAME_SHORT=$(hostname -s)
    NSYS_OUTPUT="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}_${HOSTNAME_SHORT}"
    
    echo "📊 nsys output: ${NSYS_OUTPUT}.nsys-rep"
    echo ""
    
    # Use nsys to wrap torchrun, nsys will trace all child processes
    # Parameters consistent with slurm_job.sub
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
    # ========================================================================
    # Standard training mode (no nsys profile)
    # ========================================================================
    torchrun \
        --standalone \
        --nproc_per_node=${NPROC} \
        training/pretrain_gr_ranking.py \
        --gin-config-file ${CONFIG_FILE} \
        2>&1 | tee -a ${LOG_FILE}
    
    EXIT_CODE=${PIPESTATUS[0]}
fi

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment ${EXP_NAME} completed successfully!"
else
    echo "❌ Experiment ${EXP_NAME} failed with exit code: ${EXIT_CODE}"
fi
echo "⏰ Finished at: $(date)"
echo "📝 Log saved to: ${LOG_FILE}"
echo "📄 Config file: ${CONFIG_FILE}"
if [ ${ENABLE_NSYS} -eq 1 ]; then
    echo "📊 nsys profile: ${NSYS_OUTPUT}.nsys-rep"
fi
echo "=========================================="

exit $EXIT_CODE
