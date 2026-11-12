# Common environment for all spark-fusion-plugin scripts (sourced, not executed).
# Every path can be overridden from the environment; nothing here is site specific.
#
#   FUSION_HOME      root of the plugin directory (batch/plugin, auto-detected)
#   TOOLCHAIN_DIR    optional directory holding the downloaded dependencies (default: $FUSION_HOME/.toolchain, git-ignored)
#   JAVA_HOME        JDK 11 (JDK 8 also works for running)
#   SPARK_HOME       Spark 3.5.x (Scala 2.12) distribution     (default: $TOOLCHAIN_DIR/spark-3.5.9-bin-hadoop3)
#   RAPIDS_JAR       rapids-4-spark_2.12-26.08.1-cuda12.jar     (default: $TOOLCHAIN_DIR/rapids-4-spark_2.12-26.08.1-cuda12.jar)
#   RAPIDS_SHIM_JAR  spark-rapids shim classes for compiling, made by scripts/make-shim-jar.sh
#                    (default: $TOOLCHAIN_DIR/rapids-shim359.jar)
#   CUDA_HOME        CUDA 12.x install (default: /usr/local/cuda)
#   NVRTC_DIR        directory with libnvrtc.so.12 (default: $TOOLCHAIN_DIR/nvrtc if present, else $CUDA_HOME/lib64)
#   FUSION_JAR       plugin jar to use (default: newest target/gpu-fusion-plugin-*.jar)
#   FUSION_WORK      scratch dir for generated test/bench data (default: $FUSION_HOME/work, git-ignored)
#   SLURM_PARTITION  if set (or -p PART given), GPU commands are wrapped in `srun -p $SLURM_PARTITION --gres=gpu:1`;
#                    if unset, or already inside an allocation (SLURM_JOB_ID), or NOSRUN=1, they run directly on this machine.
#   SRUN_EXTRA       extra srun arguments (e.g. "-w <node>")

FUSION_HOME=${FUSION_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TOOLCHAIN_DIR=${TOOLCHAIN_DIR:-$FUSION_HOME/.toolchain}
JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}
SPARK_HOME=${SPARK_HOME:-$TOOLCHAIN_DIR/spark-3.5.9-bin-hadoop3}
RAPIDS_JAR=${RAPIDS_JAR:-$TOOLCHAIN_DIR/rapids-4-spark_2.12-26.08.1-cuda12.jar}
RAPIDS_SHIM_JAR=${RAPIDS_SHIM_JAR:-$TOOLCHAIN_DIR/rapids-shim359.jar}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [ -z "$NVRTC_DIR" ]; then
  if [ -e "$TOOLCHAIN_DIR/nvrtc/libnvrtc.so.12" ]; then NVRTC_DIR=$TOOLCHAIN_DIR/nvrtc; else NVRTC_DIR=$CUDA_HOME/lib64; fi
fi
FUSION_WORK=${FUSION_WORK:-$FUSION_HOME/work}
if [ -z "$FUSION_JAR" ]; then
  FUSION_JAR=$(ls -t "$FUSION_HOME"/target/gpu-fusion-plugin-*.jar 2>/dev/null | grep -v -E "sources|original" | head -1)
fi
export FUSION_HOME TOOLCHAIN_DIR JAVA_HOME SPARK_HOME RAPIDS_JAR RAPIDS_SHIM_JAR CUDA_HOME NVRTC_DIR FUSION_WORK FUSION_JAR
mkdir -p "$FUSION_WORK" 2>/dev/null

# gpu_run TIME cmd...   run a command on one GPU (srun if SLURM_PARTITION is set, otherwise directly)
gpu_run() {
  local t=$1; shift
  if [ -n "$SLURM_PARTITION" ] && [ -z "$SLURM_JOB_ID" ] && [ "$NOSRUN" != 1 ]; then
    srun $SRUN_EXTRA -p "$SLURM_PARTITION" --gres=gpu:1 -t "$t" --cpus-per-task=${CPUS:-8} --mem=${MEM:-64G} --export=ALL "$@"
  else
    "$@"
  fi
}
