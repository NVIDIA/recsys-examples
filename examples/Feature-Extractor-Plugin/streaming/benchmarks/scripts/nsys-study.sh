#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Independent kernel / copy cross-check with Nsight Systems (GPU modes) and per-thread CPU seconds from JFR (CPU modes).
# Mega-Flink build and nsys required.   usage: nsys-study.sh ROOT      (high tier, one run per case and mode)
# Env: NSYS (path to nsys, default "nsys").
ROOT=$1
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/env.sh"
fep_mega || { echo "needs a Mega-Flink build (MEGA_FLINK_CP)" >&2; exit 2; }
export WARMUP=300000
N=3000000
NS=${NSYS:-nsys}
for spec in "c1-filter:cpu,gpu-batch" "c2-hop:cpu-1p,gpu-1p" "c3-topn:cpu,gpu-batch" "c4-dedup:cpu,gpu-adaptive" "c5-join:cpu,gpu-batch" "c5-join-filter:cpu,gpu-batch-fuse" "c5-join-agg:cpu,gpu-batch-fuse" "c6-lookup:cpu,gpu"; do
  CASE=${spec%%:*}; IFS=, read -r -a MODES <<< "${spec#*:}"
  for MODE in "${MODES[@]}"; do
    OUT=$ROOT/$CASE/$MODE/r0-n$N-ns
    mkdir -p "$OUT"
    echo "$(date +%T) nsys-study $CASE $MODE"
    JO="-XX:StartFlightRecording=settings=$D/sampling2.jfc,filename=$OUT/rec.jfr,dumponexit=true -Dcudf.flink.profile.phases=true"
    if [[ $MODE == cpu* ]]; then
      JAVA_OPTS="$JO" TAG=ns "$D/bench.sh" $CASE $MODE 0 $N "$ROOT" || echo RUN_FAILED
    else
      WRAP="$NS profile -t cuda,nvtx,osrt -s none --cuda-memory-usage=false -f true -o $OUT/nsys" JAVA_OPTS="$JO" TAG=ns "$D/bench.sh" $CASE $MODE 0 $N "$ROOT" || echo RUN_FAILED
      $NS stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum --format csv "$OUT/nsys.nsys-rep" > "$OUT/nsys-stats.csv" 2>"$OUT/nsys-stats.err" || echo STATS_FAILED
    fi
    "$(fep_java)" -cp "$FEP_CLASSES" fep.ThreadCpu "$OUT/rec.jfr" > "$OUT/thread-cpu.txt" 2>&1
    head -4 "$OUT/thread-cpu.txt"
  done
done
echo NSYS_DONE
