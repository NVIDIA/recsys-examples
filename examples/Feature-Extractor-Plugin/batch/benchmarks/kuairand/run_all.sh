#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the five KuaiRand cases one after another on one GPU (the protocol of the published numbers).
#   DATA=<dir with the log/user/video_basic Parquet tables>  OUT=<output dir>  benchmarks/kuairand/run_all.sh [run_case.py args]
# Environment (see benchmarks/REPRODUCE.md): TOOLCHAIN_DIR, FUSION_WORK, JAVA_HOME (see plugin/scripts/env.sh).
# Optional: CASES="a b c d e" to run a subset, SPARK_DRIVER_MEM (default 96g), LOCAL_N (default 8).
set -e
BATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$BATCH"
: "${DATA:?set DATA to the Parquet directory made by prepare_data.py}" "${OUT:?set OUT to an output directory}"
export SPARK_DRIVER_MEM=${SPARK_DRIVER_MEM:-96g}
CASES=${CASES:-"c a b e d"}
run() {   # run <sql name> <output name> [extra run_case.py args]
  local sql=$1 name=$2; shift 2
  echo "=== $name  $(date +%T)"
  FUSION_DUMP_DIR=$OUT/$name/kernels plugin/scripts/run-spark.sh -n "${LOCAL_N:-8}" spark-submit \
    --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
    --conf spark.rapids.memory.pinnedPool.size=8g \
    benchmarks/kuairand/run_case.py --case benchmarks/kuairand/cases/$sql.sql --data "$DATA" --out "$OUT" "$@" \
    2>&1 | grep -v -E ' INFO |WARN' | tail -8
}
for c in $CASES; do
  case $c in
    a) run a_sample_features a_sample_features "$@" ;;
    b) run a_sample_features a_sample_features_gpuagg --sink gpu_agg "$@" ;;
    c) run c_filter_features c_filter_features "$@" ;;
    d) run d_join_string_agg d_join_string_agg "$@" ;;
    e) run e_light_scan e_light_scan "$@" ;;
    *) echo "unknown case $c"; exit 1 ;;
  esac
done
echo ALL_DONE
