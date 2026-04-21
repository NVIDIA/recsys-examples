#!/bin/bash
# ============================================================================
# Batch Submit All Experiments via SLURM sbatch
# 
# Usage: 
#   ./training/benchmark/scripts/submit_all_experiments_slurm.sh --exp-file=<file> [options]
# 
# Environment Variables:
#   HSTU_ROOT            Path to examples/hstu directory (optional, defaults to pwd)
# 
# Options:
#   --exp-file=FILE      Experiment list file (required, format: exp_name,gin_options)
#   --hstu-root=PATH     Specify examples/hstu directory path (overrides env var and pwd)
#   --results-dir=PATH   Output directory (default: training/benchmark/results)
#   --nsys               Enable nsys profile sampling
#   --mem-debug          Enable GPU memory debug logging (MEM_DEBUG=1)
#   --mem-watchdog       Enable CUDA memory fragmentation watchdog (CUDA_MEM_WATCHDOG=1)
#   --cache-debug        Enable DynamicEmb cache hit rate logging (CACHE_DEBUG=1)
#   --sequential         Sequential execution (use dependencies, start next after previous completes)
#   --partition=NAME     SLURM partition name (default: batch)
#   --account=NAME       SLURM account name (optional, passed to sbatch -A)
#   --job-name=NAME      SLURM job name prefix (optional, passed to sbatch -J)
#   --container-image=IMAGE  Container image (built from docker/Dockerfile)
#   --nodes=N            Number of nodes (default: 2)
#   --ranks-per-node=N   Number of ranks/processes per node (default: 8)
#   --time=HH:MM:SS      Job time limit (default: 00:30:00)
#   -y, --yes            Skip confirmation prompt
#   --dry-run            Print sbatch commands only, do not submit
#   --wait-and-analyze   Wait for all jobs to complete and auto-analyze results
#   --poll-interval=SEC  Polling interval for job status check (default: 60)
#   --scp-dest=USER@HOST:PATH  SCP destination for results archive (optional, skip if not set)
#   --help               Show help information
# 
# Experiment List File Format:
#   # Comment lines start with #
#   exp_name,generate_gin_config_options
#   exp0_baseline,
#   exp1_cutlass,--kernel_backend cutlass
#   exp4_caching,--kernel_backend cutlass --recompute_layernorm --balanced_shuffler --caching
# 
# Output Directory Structure:
#   {results_dir}/
#   └── {batch_timestamp}/           # Timestamp of this batch submission
#       ├── exp0_baseline/           # First experiment
#       │   ├── exp0_baseline_*.log
#       │   ├── exp0_baseline_*.gin  # Generated config
#       │   ├── {job_name}_*.out     # SLURM stdout/stderr
#       │   └── exp0_baseline_*.nsys-rep  (if nsys enabled)
#       ├── exp1_cutlass/            # Second experiment
#       │   ├── ...
#       ├── summary.txt              # Batch experiment summary
#       ├── comparison.png           # Performance comparison chart (if --wait-and-analyze)
#       ├── monitor.log              # Job monitor log (if --wait-and-analyze)
#       └── ...
#   └── {batch_timestamp}.tar.gz     # Archive of all results (if --wait-and-analyze)
# 
# Examples:
#   # Run in examples/hstu directory
#   ./training/benchmark/scripts/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt
#   
#   # Use environment variable to specify HSTU_ROOT
#   export HSTU_ROOT=/path/to/recsys-examples/examples/hstu
#   ./submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt
#   
#   # Use command line argument to specify HSTU_ROOT
#   ./submit_all_experiments_slurm.sh --hstu-root=/path/to/examples/hstu --exp-file=training/benchmark/experiments.txt
#   
#   # Other options
#   ./training/benchmark/scripts/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt --nsys
#   ./training/benchmark/scripts/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt --results-dir=/data/benchmark_results
#   
#   # Wait for all jobs and auto-analyze
#   ./training/benchmark/scripts/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt --wait-and-analyze
#   ./training/benchmark/scripts/submit_all_experiments_slurm.sh --exp-file=training/benchmark/experiments.txt --wait-and-analyze --poll-interval=120
# ============================================================================

set -e

# ============================================================================
# Default Parameters
# ============================================================================
ENABLE_NSYS=0
SEQUENTIAL=0
PARTITION="batch"
ACCOUNT=""
JOB_PREFIX=""
CONTAINER_IMAGE=""  # Set to your container image, e.g. built from docker/Dockerfile
NODES=2
RANKS_PER_NODE=8
TIME_LIMIT="00:30:00"
DRY_RUN=0
YES_FLAG=0
WAIT_AND_ANALYZE=0
POLL_INTERVAL=30
EXP_FILE=""
CUSTOM_RESULTS_DIR=""
CUSTOM_HSTU_ROOT=""
SCP_DEST=""
GIT_BRANCH=""
MEM_DEBUG=${MEM_DEBUG:-0}
CUDA_MEM_WATCHDOG=${CUDA_MEM_WATCHDOG:-0}
CACHE_DEBUG=${CACHE_DEBUG:-0}

# ============================================================================
# Help Information
# ============================================================================
show_help() {
    head -71 "$0" | tail -70
    exit 0
}

# ============================================================================
# Parse Command Line Arguments (support both --arg value and --arg=value)
# ============================================================================
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
        --nsys)
            ENABLE_NSYS=1
            shift
            ;;
        --mem-debug)
            MEM_DEBUG=1
            shift
            ;;
        --mem-watchdog)
            CUDA_MEM_WATCHDOG=1
            shift
            ;;
        --cache-debug)
            CACHE_DEBUG=1
            shift
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        --partition=*)
            PARTITION="${1#*=}"
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --account=*|-A=*)
            ACCOUNT="${1#*=}"
            shift
            ;;
        --account|-A)
            ACCOUNT="$2"
            shift 2
            ;;
        --job-name=*|-J=*)
            JOB_PREFIX="${1#*=}"
            shift
            ;;
        --job-name|-J)
            JOB_PREFIX="$2"
            shift 2
            ;;
        --container-image=*)
            CONTAINER_IMAGE="${1#*=}"
            shift
            ;;
        --container-image)
            CONTAINER_IMAGE="$2"
            shift 2
            ;;
        --nodes=*)
            NODES="${1#*=}"
            shift
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --ranks-per-node=*)
            RANKS_PER_NODE="${1#*=}"
            shift
            ;;
        --ranks-per-node)
            RANKS_PER_NODE="$2"
            shift 2
            ;;
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -y|--yes)
            YES_FLAG=1
            shift
            ;;
        --wait-and-analyze)
            WAIT_AND_ANALYZE=1
            shift
            ;;
        --poll-interval=*)
            POLL_INTERVAL="${1#*=}"
            shift
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --scp-dest=*)
            SCP_DEST="${1#*=}"
            shift
            ;;
        --scp-dest)
            SCP_DEST="$2"
            shift 2
            ;;
        --branch=*)
            GIT_BRANCH="${1#*=}"
            shift
            ;;
        --branch)
            GIT_BRANCH="$2"
            shift 2
            ;;
        --help|-h)
            show_help
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
    # Command line argument has highest priority
    HSTU_ROOT="$CUSTOM_HSTU_ROOT"
elif [ -z "$HSTU_ROOT" ]; then
    # If env var not set, use pwd
    HSTU_ROOT=$(pwd)
fi
# If env var is set, use it directly (no additional action needed)

# Verify HSTU_ROOT directory exists (skip in dry-run mode)
if [ ${DRY_RUN} -eq 0 ]; then
    if [ ! -d "$HSTU_ROOT" ]; then
        echo "❌ Error: HSTU_ROOT directory does not exist: $HSTU_ROOT"
        exit 1
    fi

    # Verify directory structure (check for training subdirectory)
    if [ ! -d "$HSTU_ROOT/training" ]; then
        echo "❌ Error: Invalid HSTU_ROOT - missing 'training' subdirectory"
        echo "  HSTU_ROOT: $HSTU_ROOT"
        echo ""
        echo "Please ensure HSTU_ROOT points to 'recsys-examples/examples/hstu'"
        exit 1
    fi
fi

# Path configuration
PROJECT_ROOT="${HSTU_ROOT}/../.."
SCRIPT_DIR="${HSTU_ROOT}/training/benchmark/scripts"
BENCHMARK_DIR="${HSTU_ROOT}/training/benchmark"

# ============================================================================
# Set Output Directory
# ============================================================================
if [ -n "$CUSTOM_RESULTS_DIR" ]; then
    # If relative path, make it relative to examples/hstu
    if [[ ! "$CUSTOM_RESULTS_DIR" = /* ]]; then
        RESULTS_BASE="${HSTU_ROOT}/${CUSTOM_RESULTS_DIR}"
    else
        RESULTS_BASE="${CUSTOM_RESULTS_DIR}"
    fi
else
    # Default directory
    RESULTS_BASE="${BENCHMARK_DIR}/results"
fi

# Create timestamped batch experiment directory
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUTPUT_DIR="${RESULTS_BASE}/${BATCH_TIMESTAMP}"

# ============================================================================
# Check Experiment List File
# ============================================================================
# If --exp-file is not provided, show help
if [ -z "$EXP_FILE" ]; then
    echo "⚠️  Missing experiment list file (--exp-file=<file>)"
    echo ""
    head -71 "$0" | tail -69
    exit 0
fi

# Read experiment list if provided
declare -a EXP_NAMES
declare -a GIN_OPTIONS

if [ -n "$EXP_FILE" ]; then
    # If relative path, make it relative to examples/hstu
    if [[ ! "$EXP_FILE" = /* ]]; then
        EXP_FILE="${HSTU_ROOT}/${EXP_FILE}"
    fi

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

    # Read experiment list (skip comments and empty lines)
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
fi

# ============================================================================
# Color Output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Capture Git Information
# ============================================================================
GIT_BRANCH_REPO=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT_HASH=$(git -C "${PROJECT_ROOT}" log -1 --format="%H" 2>/dev/null || echo "unknown")
GIT_COMMIT_SHORT=$(git -C "${PROJECT_ROOT}" log -1 --format="%h" 2>/dev/null || echo "unknown")
GIT_COMMIT_DATE=$(git -C "${PROJECT_ROOT}" log -1 --format="%ai" 2>/dev/null || echo "unknown")
GIT_COMMIT_MSG=$(git -C "${PROJECT_ROOT}" log -1 --format="%s" 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | head -1)
if [ -n "$GIT_DIRTY" ]; then
    GIT_STATUS="dirty (uncommitted changes)"
else
    GIT_STATUS="clean"
fi

# ============================================================================
# Print Configuration Information
# ============================================================================
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}🚀 HSTU Benchmark - SLURM Submission${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""
echo -e "${BLUE}SLURM Configuration:${NC}"
echo "  Partition:        ${PARTITION}"
[ -n "$ACCOUNT" ] && echo "  Account:          ${ACCOUNT}"
[ -n "$JOB_PREFIX" ] && echo "  Job prefix:       ${JOB_PREFIX}"
echo "  Container:        ${CONTAINER_IMAGE}"
echo "  Nodes:            ${NODES}"
echo "  Ranks per node:   ${RANKS_PER_NODE}"
echo "  Total ranks:      $((NODES * RANKS_PER_NODE))"
echo "  Time limit:       ${TIME_LIMIT}"
echo "  Sequential mode:  $([ ${SEQUENTIAL} -eq 1 ] && echo 'YES' || echo 'NO')"
echo ""
echo -e "${BLUE}NSYS Profiling:${NC}"
echo -e "  Enabled:          $([ ${ENABLE_NSYS} -eq 1 ] && echo "${GREEN}YES${NC}" || echo 'NO')"
echo ""
echo -e "${BLUE}Auto Analysis:${NC}"
echo -e "  Wait and analyze: $([ ${WAIT_AND_ANALYZE} -eq 1 ] && echo "${GREEN}YES${NC}" || echo 'NO')"
if [ ${WAIT_AND_ANALYZE} -eq 1 ]; then
    echo "  Poll interval:    ${POLL_INTERVAL}s"
fi
echo ""

if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - Commands will be printed but not executed${NC}"
    echo ""
fi

echo -e "${BLUE}Git Information:${NC}"
echo "  Branch (repo):    ${GIT_BRANCH_REPO}"
echo "  Commit:           ${GIT_COMMIT_SHORT} (${GIT_COMMIT_DATE})"
echo "  Message:          ${GIT_COMMIT_MSG}"
echo "  Working tree:     ${GIT_STATUS}"
if [ -n "$GIT_BRANCH" ]; then
    echo -e "  ${YELLOW}Branch override:  ${GIT_BRANCH} (running from branch-specific clone)${NC}"
fi
echo ""
echo -e "${BLUE}Batch timestamp:   ${BATCH_TIMESTAMP}${NC}"
echo -e "${BLUE}Output directory:  ${BATCH_OUTPUT_DIR}${NC}"
echo ""
echo -e "${BLUE}Experiment file: ${EXP_FILE}${NC}"
echo ""
echo -e "${BLUE}Experiments to run (${#EXP_NAMES[@]} total):${NC}"
for i in "${!EXP_NAMES[@]}"; do
    echo "  - ${EXP_NAMES[$i]}: ${GIN_OPTIONS[$i]:-'(defaults)'}"
done
echo ""

# ============================================================================
# Confirm Submission
# ============================================================================
if [ ${DRY_RUN} -eq 0 ]; then
    # Skip confirmation if -y/--yes flag is set
    if [ ${YES_FLAG} -eq 0 ]; then
        echo -e "${YELLOW}Do you want to submit these jobs? (y/n)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        echo ""
    fi
    
    # Create batch output directory
    mkdir -p ${BATCH_OUTPUT_DIR}
fi

# ============================================================================
# Submit Jobs
# ============================================================================
SUBMITTED_JOBS=()
PREV_JOB_ID=""

echo -e "${BLUE}Submitting jobs...${NC}"
echo ""

for i in "${!EXP_NAMES[@]}"; do
    exp="${EXP_NAMES[$i]}"
    gin_opts="${GIN_OPTIONS[$i]}"
    exp_num=$((i + 1))
    
    # Output directory for each experiment
    EXP_OUTPUT_DIR="${BATCH_OUTPUT_DIR}/${exp}"
    
    if [ ${DRY_RUN} -eq 0 ]; then
        mkdir -p ${EXP_OUTPUT_DIR}
    fi
    
    # Build sbatch command using array (to properly handle arguments with spaces)
    # Determine job name (with optional prefix)
    if [ -n "$JOB_PREFIX" ]; then
        FULL_JOB_NAME="${JOB_PREFIX}-hstu.${exp}"
    else
        FULL_JOB_NAME="hstu_${exp}"
    fi
    
    # Use array to build sbatch arguments (preserves spaces in values)
    SBATCH_ARGS=()
    SBATCH_ARGS+=(--job-name="${FULL_JOB_NAME}")
    SBATCH_ARGS+=(--output="${EXP_OUTPUT_DIR}/${FULL_JOB_NAME}_%j.out")
    SBATCH_ARGS+=(--partition="${PARTITION}")
    
    # Add account if specified
    if [ -n "$ACCOUNT" ]; then
        SBATCH_ARGS+=(--account="${ACCOUNT}")
    fi
    
    SBATCH_ARGS+=(--nodes="${NODES}")
    SBATCH_ARGS+=(--ntasks-per-node="${RANKS_PER_NODE}")
    SBATCH_ARGS+=(--cpus-per-task=8)
    SBATCH_ARGS+=(--mem=0)
    SBATCH_ARGS+=(--time="${TIME_LIMIT}")
    SBATCH_ARGS+=(--exclusive)
    SBATCH_ARGS+=(--network=sharp)
    
    # Sequential execution mode: add dependency
    if [ ${SEQUENTIAL} -eq 1 ] && [ -n "$PREV_JOB_ID" ]; then
        SBATCH_ARGS+=(--dependency="afterany:${PREV_JOB_ID}")
    fi
    
    # Export environment variables (using array element to preserve spaces in GIN_OPTIONS)
    EXPORT_VARS="ALL,EXP_NAME=${exp},GIN_OPTIONS=${gin_opts},EXP_OUTPUT_DIR=${EXP_OUTPUT_DIR},ENABLE_NSYS=${ENABLE_NSYS},HSTU_ROOT=${HSTU_ROOT},CONTAINER_IMAGE=${CONTAINER_IMAGE},MEM_DEBUG=${MEM_DEBUG},CUDA_MEM_WATCHDOG=${CUDA_MEM_WATCHDOG},CUDA_MEM_WATCHDOG_THRESHOLD=${CUDA_MEM_WATCHDOG_THRESHOLD:-0.5},CACHE_DEBUG=${CACHE_DEBUG}"
    SBATCH_ARGS+=(--export="${EXPORT_VARS}")
    
    # Specify SLURM job script
    SBATCH_ARGS+=("${SCRIPT_DIR}/slurm_job.sub")
    
    echo -e "[${exp_num}/${#EXP_NAMES[@]}] ${YELLOW}${exp}${NC}"
    echo "  Options:    ${gin_opts:-'(defaults)'}"
    echo "  Output dir: ${EXP_OUTPUT_DIR}"
    
    if [ ${DRY_RUN} -eq 1 ]; then
        echo "  Command: sbatch ${SBATCH_ARGS[*]}"
        echo ""
    else
        # Submit job and get job ID (using array expansion to preserve spaces)
        JOB_OUTPUT=$(sbatch "${SBATCH_ARGS[@]}")
        JOB_ID=$(echo ${JOB_OUTPUT} | grep -oP '\d+$')
        
        if [ -n "$JOB_ID" ]; then
            echo -e "  ${GREEN}✅ Submitted: Job ID ${JOB_ID}${NC}"
            SUBMITTED_JOBS+=("${exp}:${JOB_ID}")
            PREV_JOB_ID=${JOB_ID}
        else
            echo -e "  ${RED}❌ Failed to submit${NC}"
            echo "  Output: ${JOB_OUTPUT}"
        fi
        echo ""
    fi
done

# ============================================================================
# Create Summary File
# ============================================================================
if [ ${DRY_RUN} -eq 0 ]; then
    SUMMARY_FILE="${BATCH_OUTPUT_DIR}/summary.txt"
    {
        echo "================================================================================"
        echo "HSTU Benchmark Suite - SLURM Submission Summary"
        echo "================================================================================"
        echo ""
        echo "Batch Timestamp: ${BATCH_TIMESTAMP}"
        echo "Submitted at:    $(date)"
        echo ""
        echo "Git Information:"
        echo "  Branch (repo):    ${GIT_BRANCH_REPO}"
        if [ -n "$GIT_BRANCH" ]; then
            echo "  Branch override:  ${GIT_BRANCH}"
        fi
        echo "  Commit:           ${GIT_COMMIT_HASH}"
        echo "  Commit (short):   ${GIT_COMMIT_SHORT}"
        echo "  Commit date:      ${GIT_COMMIT_DATE}"
        echo "  Commit message:   ${GIT_COMMIT_MSG}"
        echo "  Working tree:     ${GIT_STATUS}"
        echo ""
        echo "SLURM Configuration:"
        echo "  Partition:        ${PARTITION}"
        [ -n "$ACCOUNT" ] && echo "  Account:          ${ACCOUNT}"
        [ -n "$JOB_PREFIX" ] && echo "  Job prefix:       ${JOB_PREFIX}"
        echo "  Container:        ${CONTAINER_IMAGE}"
        echo "  Nodes:            ${NODES}"
        echo "  Ranks per node:   ${RANKS_PER_NODE}"
        echo "  Time limit:       ${TIME_LIMIT}"
        echo "  Sequential:       $([ ${SEQUENTIAL} -eq 1 ] && echo 'YES' || echo 'NO')"
        echo "  NSYS Profiling:   $([ ${ENABLE_NSYS} -eq 1 ] && echo 'YES' || echo 'NO')"
        echo ""
        echo "Experiment File: ${EXP_FILE}"
        echo ""
        echo "Submitted Jobs (${#SUBMITTED_JOBS[@]} total):"
        echo "--------------------------------------------------------------------------------"
        for job_info in "${SUBMITTED_JOBS[@]}"; do
            exp_name=$(echo ${job_info} | cut -d: -f1)
            job_id=$(echo ${job_info} | cut -d: -f2)
            echo "  ${exp_name}: Job ID ${job_id}"
            echo "    Output: ${BATCH_OUTPUT_DIR}/${exp_name}/"
        done
        echo "================================================================================"
    } > ${SUMMARY_FILE}
fi

# ============================================================================
# Summary Information
# ============================================================================
echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}📊 Submission Summary${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "${YELLOW}DRY RUN completed. No jobs were submitted.${NC}"
else
    echo -e "${GREEN}Submitted ${#SUBMITTED_JOBS[@]} jobs:${NC}"
    echo ""
    for job_info in "${SUBMITTED_JOBS[@]}"; do
        exp_name=$(echo ${job_info} | cut -d: -f1)
        job_id=$(echo ${job_info} | cut -d: -f2)
        echo "  ${exp_name}: Job ID ${job_id}"
    done
    echo ""
    
    echo "📁 Output directory: ${BATCH_OUTPUT_DIR}"
    echo ""
    echo "Directory structure:"
    echo "  ${BATCH_OUTPUT_DIR}/"
    for exp in "${EXP_NAMES[@]}"; do
        # Determine job name pattern for display
        if [ -n "$JOB_PREFIX" ]; then
            JOB_NAME_PATTERN="${JOB_PREFIX}-${exp}"
        else
            JOB_NAME_PATTERN="hstu_${exp}"
        fi
        echo "  ├── ${exp}/"
        echo "  │   ├── ${exp}_*.log"
        echo "  │   ├── ${exp}_*.gin"
        if [ ${ENABLE_NSYS} -eq 1 ]; then
            echo "  │   ├── ${JOB_NAME_PATTERN}_*.out"
            echo "  │   └── ${exp}_*.nsys-rep"
        else
            echo "  │   └── ${JOB_NAME_PATTERN}_*.out"
        fi
    done
    if [ ${WAIT_AND_ANALYZE} -eq 1 ]; then
        echo "  ├── summary.txt"
        echo "  ├── comparison.png         (auto-generated after all jobs complete)"
        echo "  └── monitor.log            (job monitor log)"
        echo ""
        echo "  📦 Archive (in parent dir):"
        echo "  └── ${BATCH_TIMESTAMP}.tar.gz  (created after analysis)"
    else
        echo "  └── summary.txt"
    fi
    echo ""
    
    echo "📝 Summary saved to: ${SUMMARY_FILE}"
    echo ""
    
    echo -e "${BLUE}Useful commands:${NC}"
    echo "  squeue -u \$USER              # View job queue"
    echo "  scancel <job_id>              # Cancel single job"
    echo "  scancel -u \$USER             # Cancel all jobs"
    echo "  scontrol show job <job_id>    # View job details"
    echo "  cat ${SUMMARY_FILE}"
    echo ""
    
    if [ ${ENABLE_NSYS} -eq 1 ]; then
        echo -e "${BLUE}To analyze nsys profiles:${NC}"
        echo "  nsys stats ${BATCH_OUTPUT_DIR}/{exp_name}/*.nsys-rep"
        echo "  nsys-ui ${BATCH_OUTPUT_DIR}/{exp_name}/*.nsys-rep"
        echo ""
    fi
    
    # ============================================================================
    # Background Monitoring and Auto-Analysis
    # ============================================================================
    if [ ${WAIT_AND_ANALYZE} -eq 1 ] && [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
        echo -e "${BLUE}🔄 Starting background job monitor...${NC}"
        echo "   Polling interval: ${POLL_INTERVAL} seconds"
        echo ""
        
        # Extract job IDs
        JOB_IDS=""
        for job_info in "${SUBMITTED_JOBS[@]}"; do
            job_id=$(echo ${job_info} | cut -d: -f2)
            if [ -n "$JOB_IDS" ]; then
                JOB_IDS="${JOB_IDS},${job_id}"
            else
                JOB_IDS="${job_id}"
            fi
        done
        
        # Create monitor script in the batch output directory
        MONITOR_SCRIPT="${BATCH_OUTPUT_DIR}/monitor_jobs.sh"
        MONITOR_LOG="${BATCH_OUTPUT_DIR}/monitor.log"
        ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_results.py"
        
        cat > "${MONITOR_SCRIPT}" << 'MONITOR_EOF'
#!/bin/bash
# Auto-generated job monitor script

JOB_IDS="$1"
BATCH_OUTPUT_DIR="$2"
POLL_INTERVAL="$3"
ANALYZE_SCRIPT="$4"
MONITOR_LOG="$5"
SCP_DEST="$6"
SUMMARY_FILE="${BATCH_OUTPUT_DIR}/summary.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MONITOR_LOG}"
}

log "Job monitor started"
log "Monitoring jobs: ${JOB_IDS}"
log "Results directory: ${BATCH_OUTPUT_DIR}"
log "Poll interval: ${POLL_INTERVAL}s"

# Function to check if all jobs are complete
check_jobs_complete() {
    # Use sacct to check job states
    # States: COMPLETED, FAILED, CANCELLED, TIMEOUT, NODE_FAIL, PREEMPTED, OUT_OF_MEMORY
    local pending_count=0
    
    IFS=',' read -ra JOB_ARRAY <<< "$JOB_IDS"
    for job_id in "${JOB_ARRAY[@]}"; do
        # Get job state using sacct (more reliable than squeue for completed jobs)
        state=$(sacct -j "$job_id" -n -o State -X 2>/dev/null | head -1 | xargs)
        
        # If sacct doesn't return anything, try squeue
        if [ -z "$state" ]; then
            state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        fi
        
        # Check if job is still running or pending
        case "$state" in
            PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|SUSPENDED)
                ((pending_count++))
                ;;
        esac
    done
    
    echo $pending_count
}

# Function to get job status summary
get_status_summary() {
    local completed=0 failed=0 running=0 pending=0 other=0
    
    IFS=',' read -ra JOB_ARRAY <<< "$JOB_IDS"
    for job_id in "${JOB_ARRAY[@]}"; do
        state=$(sacct -j "$job_id" -n -o State -X 2>/dev/null | head -1 | xargs)
        if [ -z "$state" ]; then
            state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        fi
        
        case "$state" in
            COMPLETED) ((completed++)) ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY) ((failed++)) ;;
            RUNNING|COMPLETING) ((running++)) ;;
            PENDING|CONFIGURING) ((pending++)) ;;
            *) ((other++)) ;;
        esac
    done
    
    echo "Completed: $completed, Running: $running, Pending: $pending, Failed: $failed"
}

# Main monitoring loop
while true; do
    pending=$(check_jobs_complete)
    status=$(get_status_summary)
    
    log "Status: ${status}"
    
    if [ "$pending" -eq 0 ]; then
        log "All jobs have completed!"
        break
    fi
    
    log "Waiting ${POLL_INTERVAL}s before next check..."
    sleep "${POLL_INTERVAL}"
done

# Run analysis
log ""
log "=========================================="
log "Running performance analysis..."
log "=========================================="

if [ -f "${ANALYZE_SCRIPT}" ]; then
    PLOT_OUTPUT="${BATCH_OUTPUT_DIR}/comparison.png"
    
    log "Analyzing results in: ${BATCH_OUTPUT_DIR}"
    log "Plot will be saved to: ${PLOT_OUTPUT}"
    
    {
        echo ""
        echo "================================================================================"
        echo "Performance Analysis Results"
        echo "================================================================================"
        echo ""
    } >> "${SUMMARY_FILE}"

    python3 "${ANALYZE_SCRIPT}" "${BATCH_OUTPUT_DIR}" \
        --output "${PLOT_OUTPUT}" \
        --title "HSTU Benchmark Comparison" \
        2>&1 | tee -a "${SUMMARY_FILE}"
    
    # Use PIPESTATUS to get the exit code of python3, not tee
    ANALYZE_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ ${ANALYZE_EXIT_CODE} -eq 0 ]; then
        {
            echo ""
            echo "Plot: ${PLOT_OUTPUT}"
            echo "================================================================================"
        } >> "${SUMMARY_FILE}"

        log "✅ Analysis complete! Results saved to: ${SUMMARY_FILE}"
        log "   Plot saved to: ${PLOT_OUTPUT}"
    else
        log ""
        log "❌ Analysis failed. Check logs for details."
    fi
else
    log "❌ Analysis script not found: ${ANALYZE_SCRIPT}"
fi

# Create tar.gz archive of all results
log ""
log "=========================================="
log "Creating results archive..."
log "=========================================="

ARCHIVE_NAME="$(basename ${BATCH_OUTPUT_DIR}).tar.gz"
ARCHIVE_PATH="$(dirname ${BATCH_OUTPUT_DIR})/${ARCHIVE_NAME}"

log "Archive name: ${ARCHIVE_NAME}"
log "Archive path: ${ARCHIVE_PATH}"

# Create tar.gz archive with batch_timestamp as root directory
# Use -C option to ensure clean directory structure (batch_timestamp/ as root)
tar -czvf "${ARCHIVE_PATH}" -C "$(dirname ${BATCH_OUTPUT_DIR})" "$(basename ${BATCH_OUTPUT_DIR})" 2>&1 | tail -5 | while read line; do log "  $line"; done

if [ -f "${ARCHIVE_PATH}" ]; then
    ARCHIVE_SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)
    log ""
    log "✅ Archive created successfully!"
    log "   Archive: ${ARCHIVE_PATH}"
    log "   Size: ${ARCHIVE_SIZE}"
    
    log ""
    log "=================================================="
    log "Results Archive Created!"
    log "=================================================="
    log "Archive: ${ARCHIVE_PATH}"
    log "Size:    ${ARCHIVE_SIZE}"
    log "=================================================="

    # SCP archive to remote destination if specified
    if [ -n "${SCP_DEST}" ]; then
        log ""
        log "=========================================="
        log "Transferring archive via SCP..."
        log "  Destination: ${SCP_DEST}"
        log "=========================================="
        if scp "${ARCHIVE_PATH}" "${SCP_DEST}"; then
            log "✅ SCP transfer completed: ${SCP_DEST}"
        else
            log "❌ SCP transfer failed (exit code: $?). Archive is still available locally at: ${ARCHIVE_PATH}"
        fi
    fi
else
    log ""
    log "❌ Failed to create archive."
fi

log ""
log "Monitor script finished."
MONITOR_EOF

        chmod +x "${MONITOR_SCRIPT}" 2>/dev/null || true
        
        # Start monitor in background.
        # log() inside the monitor uses tee -a to write to MONITOR_LOG,
        # so redirect nohup stdout/stderr to /dev/null to avoid double-writing.
        nohup bash "${MONITOR_SCRIPT}" "${JOB_IDS}" "${BATCH_OUTPUT_DIR}" "${POLL_INTERVAL}" "${ANALYZE_SCRIPT}" "${MONITOR_LOG}" "${SCP_DEST}" > /dev/null 2>&1 &
        MONITOR_PID=$!
        
        echo -e "${GREEN}✅ Background monitor started (PID: ${MONITOR_PID})${NC}"
        echo ""
        echo "Monitor log: ${MONITOR_LOG}"
        echo ""
        echo -e "${BLUE}To check monitor status:${NC}"
        echo "  tail -f ${MONITOR_LOG}"
        echo "  ps -p ${MONITOR_PID}"
        echo ""
        echo -e "${BLUE}To stop monitoring:${NC}"
        echo "  kill ${MONITOR_PID}"
        echo ""
        
    fi
fi

echo -e "${GREEN}🎉 Done!${NC}"
echo ""
