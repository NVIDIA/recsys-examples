#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# c3 TopN investigation (Mega-Flink build, exclusive node).   usage: c3-study.sh ROOT
#   A: source ceiling probe (c0-pass, no real work), 3 rounds each of cpu / gpu-batch
#   B: variance, c3 high tier, 8 alternating rounds of cpu / gpu-batch
#   C: GPU batch rows x max-wait sweep, c3 gpu-batch high tier, 2 rounds
# Summarize with c3_digest.py ROOT.
ROOT=$1
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$D/env.sh"
fep_mega || { echo "needs a Mega-Flink build (MEGA_FLINK_CP)" >&2; exit 2; }
export WARMUP=300000
N=3000000
B=$D/bench.sh
echo "== A: source ceiling probe (c0-pass, high), 3 rounds cpu / gpu-batch"
for r in 1 2 3; do for m in cpu gpu-batch; do echo "$(date +%T) A $m r$r"; TAG=a$r $B c0-pass $m 0 $N "$ROOT/A" || echo RUN_FAILED; done; done
echo "== B: variance, c3 high, 8 rounds cpu / gpu-batch (alternating)"
for r in 1 2 3 4 5 6 7 8; do for m in cpu gpu-batch; do echo "$(date +%T) B $m r$r"; JAVA_OPTS="-Dcudf.flink.profile.phases=true" TAG=b$r $B c3-topn $m 0 $N "$ROOT/B" --dump-final $([ $r = 1 ] && echo true || echo false) || echo RUN_FAILED; done; done
echo "== C: batch.rows x max-wait sweep, c3 gpu-batch high, 2 rounds"
for rows in 1024 4096 16384 65536; do for wait in 1 5 20; do for r in 1 2; do
  echo "$(date +%T) C rows=$rows wait=$wait r$r"
  JAVA_OPTS="-Dcudf.flink.profile.phases=true" TAG=rows$rows-w$wait-$r $B c3-topn gpu-batch 0 $N "$ROOT/C" --conf cudf.flink.batch.rows=$rows --conf cudf.flink.batch.max-wait-ms=$wait || echo RUN_FAILED
done; done; done
echo C3_STUDY_DONE
