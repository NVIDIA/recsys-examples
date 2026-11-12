#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One profiled round per case and mode: JFR sampling (jdk.ExecutionSample, grouped by thread stack) plus, in Mega-Flink
# mode, the plugin's phase timers. JFR numbers are sampled estimates; the phase timers are measurements.
# usage: breakdown.sh ROOT TIER "case:modeA,modeB" ...      (TIER = low | mid | high, see matrix.sh)
# Writes ROOT/CASE/MODE/r<RATE>-n<N>-bd/{rec.jfr,jfr-breakdown.json,result.json,...}.
ROOT=$1; TIER=$2; shift 2
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/env.sh"
export WARMUP=${WARMUP:-300000}
case $TIER in low) RATE=20000; N=600000;; mid) RATE=100000; N=3000000;; high) RATE=0; N=3000000;; *) echo "unknown tier" >&2; exit 2;; esac
for spec in "$@"; do
  CASE=${spec%%:*}; IFS=, read -r -a MODES <<< "${spec#*:}"
  for MODE in "${MODES[@]}"; do
    OUT=$ROOT/$CASE/$MODE/r$RATE-n$N-bd
    mkdir -p "$OUT"
    echo "$(date +%T) breakdown $TIER $CASE $MODE"
    JAVA_OPTS="-XX:StartFlightRecording=settings=$D/sampling.jfc,filename=$OUT/rec.jfr,dumponexit=true -Dcudf.flink.profile.phases=true -Dcudf.flink.join.persistent.profile=true ${JAVA_OPTS:-}" \
      TAG=bd "$D/bench.sh" "$CASE" "$MODE" "$RATE" "$N" "$ROOT" || echo "RUN_FAILED $CASE $MODE"
    "$(fep_java)" -cp "$FEP_CLASSES" fep.JfrBreakdown "$OUT/rec.jfr" > "$OUT/jfr-breakdown.json" 2>&1
    head -c 1500 "$OUT/jfr-breakdown.json"; echo
  done
done
echo BREAKDOWN_DONE
