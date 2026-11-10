#!/bin/bash
# Run spark-shell / spark-submit / spark-sql in local mode with the RAPIDS plugin (+ the fusion jar) on 1 GPU.
# Usage: run-spark.sh [-p SLURM_PARTITION] [-t TIME] [-x EXTRA_JARS(comma)] [-n LOCAL_N] <spark-shell|spark-submit|spark-sql> [args...]
# Extra --conf can be given as normal spark args. Fusion is only active if you add
#   --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension --conf spark.rapids.sql.fusion.mode=SAFE
# Env: see env.sh; SPARK_DRIVER_MEM (24g), RAPIDS_POOL_FRAC (0.5).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TIME=01:00:00; LOCALN=4; EXTRA=""
while getopts "p:t:x:n:" o; do case $o in p) SLURM_PARTITION=$OPTARG;; t) TIME=$OPTARG;; x) EXTRA=$OPTARG;; n) LOCALN=$OPTARG;; esac; done
shift $((OPTIND-1))
JARS=$RAPIDS_JAR
[ -n "$FUSION_JAR" ] && JARS=$JARS,$FUSION_JAR
[ -n "$EXTRA" ] && JARS=$JARS,$EXTRA
CMD=$1; shift
export PATH=$JAVA_HOME/bin:$PATH LD_LIBRARY_PATH=$NVRTC_DIR:$LD_LIBRARY_PATH
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
gpu_run $TIME "$SPARK_HOME/bin/$CMD" --master "local[$LOCALN]" --driver-memory ${SPARK_DRIVER_MEM:-24g} \
 --jars "$JARS" \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin \
 --conf spark.rapids.sql.enabled=true \
 --conf spark.rapids.memory.gpu.allocFraction=${RAPIDS_POOL_FRAC:-0.5} \
 --conf spark.rapids.memory.gpu.maxAllocFraction=${RAPIDS_POOL_FRAC:-0.5} \
 --conf spark.rapids.memory.pinnedPool.size=1g \
 --conf spark.sql.shuffle.partitions=8 \
 --conf spark.ui.enabled=false \
 --conf spark.local.dir=$FUSION_WORK/spark-local \
 --conf spark.executorEnv.LD_LIBRARY_PATH=$NVRTC_DIR \
 "$@"
