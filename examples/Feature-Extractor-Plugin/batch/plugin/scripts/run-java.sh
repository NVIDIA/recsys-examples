#!/bin/bash
# Run an arbitrary main class with rapids + fusion jars on the classpath (self tests, micro-benchmarks).
# Usage: run-java.sh [-p SLURM_PARTITION] [-t TIME] <MainClass> [args...]     (env: EXTRA_CP, JAVA_EXTRA; see env.sh)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TIME=00:30:00
while getopts "p:t:" o; do case $o in p) SLURM_PARTITION=$OPTARG;; t) TIME=$OPTARG;; esac; done; shift $((OPTIND-1))
export LD_LIBRARY_PATH=$NVRTC_DIR:$LD_LIBRARY_PATH FUSION_NVRTC=${FUSION_NVRTC:-$NVRTC_DIR/libnvrtc.so.12}
gpu_run $TIME "$JAVA_HOME/bin/java" -cp "$RAPIDS_JAR:$SPARK_HOME/jars/*:$FUSION_JAR:$EXTRA_CP" $JAVA_EXTRA "$@"
