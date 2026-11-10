#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One case in one mode at one input rate.
# usage: bench.sh CASE MODE RATE RECORDS OUTROOT [extra driver args]
#   CASE    c0-pass c1-filter c2-hop c2-tumble c3-topn c4-dedup c5-join c5-join-filter c5-join-agg c6-lookup c6-lookup-left
#   RATE    input records/s; 0 = unthrottled (saturated tier)
#   RECORDS measured records (the warm-up, WARMUP default 200000, comes on top)
#   MODE    cpu | cpu-mb | cpu-1p       open-source Flink (CPU baseline); cpu-mb adds mini-batch, cpu-1p one-phase aggregation
#           gpu | gpu-1p | gpu-batch | gpu-persistent | gpu-adaptive    Mega-Flink only (MEGA_FLINK_CP must be set)
#           suffix -fuse / -nofuse   cross-operator fusion on/off (c5, Mega-Flink)
#           suffix -tuned            pipeline.object-reuse=true (plus a 100000-entry TopN cache), on either side
#           suffix -min1             c6: GPU lookup for every batch size (Mega-Flink)
#           suffix -nres             c5: native residual filter (Mega-Flink)
# Result directory: OUTROOT/CASE/MODE/r<RATE>-n<RECORDS>-<TAG> (TAG env, default t0).
# Optional env: WARMUP, DIM_MAX, LOOKUP_BATCH, LOOKUP_MIN, JAVA_OPTS, WRAP (see run-one.sh and env.sh).
CASE=$1; MODE=$2; RATE=$3; RECORDS=$4; ROOT=$5; shift 5
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/env.sh"
if [[ "$MODE" == gpu* ]] && ! fep_mega; then
  echo "mode $MODE needs a Mega-Flink build: set MEGA_FLINK_CP (see REPRODUCE.md)" >&2
  exit 2
fi
A=(--case "$CASE" --label "$MODE" --rate "$RATE" --records "$RECORDS" --warmup "${WARMUP:-200000}")
MB=(--conf table.exec.mini-batch.enabled=true --conf "table.exec.mini-batch.allow-latency=5 ms" --conf table.exec.mini-batch.size=4096)
JO=""
case "$CASE" in c5-join-agg) A+=("${MB[@]}") ;; esac   # two-phase aggregate needs mini-batch on both sides
case "$MODE" in
  cpu*) A+=(--forbid-gpu true)                       # fail if the plan contains a Gpu node
        fep_mega && A+=(--conf cudf.flink.mode=cpu-row)   # Mega-Flink build: force its plain CPU path
        [[ "$MODE" == cpu-mb* ]] && A+=("${MB[@]}") ;;
  gpu*) A+=(--conf cudf.flink.mode=gpu-native --conf cudf.flink.experimental.operators.enabled=true) ;;
esac
ENGINE=""
case "$MODE" in
  gpu-batch*) ENGINE=batch; A+=(--conf cudf.flink.batch.policy=throughput) ;;
  gpu-persistent*) ENGINE=persistent; A+=(--conf cudf.flink.batch.policy=latency-safe) ;;
  gpu-adaptive*) ENGINE=adaptive; A+=(--conf cudf.flink.batch.policy=latency-safe) ;;
esac
if [[ -n "$ENGINE" ]]; then
  case "$CASE" in
    c1*) A+=(--conf cudf.flink.calc.engine=$ENGINE --conf cudf.flink.experimental.generic-calc.enabled=true) ;;
    c3*) A+=(--conf cudf.flink.topn.engine=$ENGINE) ;;
    c4*) A+=(--conf cudf.flink.deduplicate.engine=$ENGINE) ;;
    c5*) A+=(--conf cudf.flink.join.engine=$ENGINE) ;;
  esac
fi
case "$MODE" in
  *-fuse) A+=(--conf cudf.flink.cross-operator-fusion.enabled=true); JO="-Dcudf.flink.cross-operator-fusion.enabled=true" ;;
  *-nofuse) A+=(--conf cudf.flink.cross-operator-fusion.enabled=false); JO="-Dcudf.flink.cross-operator-fusion.enabled=false" ;;
esac
case "$MODE" in
  *-nres*) JO="$JO -Dcudf.flink.join.native-residual.enabled=true" ;;
esac
case "$MODE" in
  *-tuned*) A+=(--conf pipeline.object-reuse=true --conf table.exec.rank.topn-cache-size=100000) ;;
esac
case "$MODE" in
  *-1p*) A+=(--conf table.optimizer.agg-phase-strategy=ONE_PHASE) ;;
esac
case "$MODE" in *-min1) LOOKUP_MIN=1 ;; esac
case "$CASE" in
  c6*) fep_mega && A+=(--conf cudf.flink.lookup-join.dimension.max-rows=${DIM_MAX:-4194304}
           --conf cudf.flink.lookup-join.batch.rows=${LOOKUP_BATCH:-4096}
           --conf cudf.flink.lookup-join.gpu.min-batch-rows=${LOOKUP_MIN:-1024}) ;;
esac
OUT=$ROOT/$CASE/$MODE/r$RATE-n$RECORDS-${TAG:-t0}
JAVA_OPTS="$JO ${JAVA_OPTS:-}" "$D/run-one.sh" "$OUT" "${A[@]}" --data "${FEP_DATA_DIR:?set FEP_DATA_DIR}" "$@"
