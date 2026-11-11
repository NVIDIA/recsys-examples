#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The measurement matrix: tiers x cases x rounds, mode order rotated every round.
# usage: matrix.sh RESULT_ROOT ROUNDS "case:modeA,modeB" ["case:modeA,..."] ...
#   TIERS env selects tiers (default "low mid high"):
#     low  = 20,000 rec/s, 600,000 records;  mid = 100,000 rec/s, 3,000,000;  high = unthrottled, 3,000,000.
#   Each run is preceded by WARMUP (default 300000) records at the same rate.
#   Round 1 also dumps the final table (final.tsv); every round writes result.json with its SHA-256.
#   XARGS env: extra driver arguments, for example "--parallelism 2" (warm-up + records must be divisible).
# Example (CPU baseline, open-source Flink, one case, 3 rounds):
#   matrix.sh $OUT 3 "c3-topn:cpu-tuned"
# Do not edit this or the other scripts while a matrix is running (bash reads scripts incrementally).
ROOT=$1; ROUNDS=$2; shift 2
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIERS=${TIERS:-"low mid high"}
export WARMUP=${WARMUP:-300000}
for tier in $TIERS; do
  case $tier in
    low)  RATE=20000;  N=600000 ;;
    mid)  RATE=100000; N=3000000 ;;
    high) RATE=0;      N=3000000 ;;
    *) echo "unknown tier $tier" >&2; exit 2 ;;
  esac
  for spec in "$@"; do
    CASE=${spec%%:*}; IFS=, read -r -a MODES <<< "${spec#*:}"
    M=${#MODES[@]}
    for r in $(seq 1 "$ROUNDS"); do
      for k in $(seq 0 $((M-1))); do
        MODE=${MODES[$(( (k + r - 1) % M ))]}
        EXTRA=(${XARGS:-}); [[ $r == 1 ]] && EXTRA+=(--dump-final true)
        echo "$(date +%T) tier=$tier case=$CASE mode=$MODE round=$r"
        TAG=t$r "$D/bench.sh" "$CASE" "$MODE" "$RATE" "$N" "$ROOT" "${EXTRA[@]}" || echo "RUN_FAILED $CASE $MODE $tier r$r"
      done
    done
  done
done
echo MATRIX_DONE
