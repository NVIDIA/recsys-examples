#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One command: run the six streaming cases (c1 filter, c2 HOP window, c3 TopN, c4 deduplicate, c5 interval join,
# c6 dimension lookup) once on the CPU baseline and once on the Mega-Flink GPU path, compare the SHA-256 of the final
# tables and print one table. The GPU side of all six is Mega-Flink (the same source as the published c2 and c6 numbers).
#
#   source <WORK_DIR>/megaflink.env           # written by build_megaflink.sh
#   export FEP_DATA_DIR=<prepared data>       # prepare_stream_data.py output
#   run_all_six.sh --quick                    # smoke: 100,000 warm-up + 300,000 records per run, minutes in total
#   run_all_six.sh                            # one round of the saturated tier: 300,000 warm-up + 3,000,000 records
#
# options:  --quick            small smoke run (the slice whose reference digests are in REPRODUCE.md section 5)
#           --out DIR          result root (default ./six-results-<time>)
#           --cases a,b,...    subset of c1-filter,c2-hop,c3-topn,c4-dedup,c5-join,c6-lookup
#           --records N --warmup N   override the sizes
#           --env FILE         source this env file first (default: $MEGA_FLINK_ENV if set)
#           --build-driver     run driver/build.sh first (needed once per MEGA_FLINK_CP)
# Exit code: 0 if every case MATCHes (and, with --quick on the standard data, also equals the reference digest),
# 1 otherwise. Never read a throughput from a MISMATCH.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QUICK=0; OUT=""; CASES="c1-filter,c2-hop,c3-topn,c4-dedup,c5-join,c6-lookup"; REC=""; WARM=""; ENVF=${MEGA_FLINK_ENV:-}; BUILD=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick) QUICK=1 ;;
    --out) OUT=$2; shift ;;
    --cases) CASES=$2; shift ;;
    --records) REC=$2; shift ;;
    --warmup) WARM=$2; shift ;;
    --env) ENVF=$2; shift ;;
    --build-driver) BUILD=1 ;;
    -h|--help) sed -n 2,20p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$ENVF" ]] && { # shellcheck disable=SC1090
  source "$ENVF"; }
[[ -n "${MEGA_FLINK_CP:-}" ]] || { echo "MEGA_FLINK_CP is not set: run build_megaflink.sh and source its megaflink.env" >&2; exit 2; }
[[ -n "${FEP_DATA_DIR:-}" && -d "$FEP_DATA_DIR/log" ]] || { echo "set FEP_DATA_DIR to the prepare_stream_data.py output" >&2; exit 2; }
if [[ $QUICK == 1 ]]; then WARM=${WARM:-100000}; REC=${REC:-300000}; else WARM=${WARM:-300000}; REC=${REC:-3000000}; fi
OUT=${OUT:-$PWD/six-results-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export WARMUP=$WARM HEAP=${HEAP:-16g}
if [[ $BUILD == 1 || ! -d "${FEP_BUILD_DIR:-$HERE/../driver/build}/classes-megaflink" ]]; then
  "$HERE/../driver/build.sh" || exit 2
fi

# GPU mode per case: the modes of the README (gpu-batch for c1/c3/c4/c5, gpu-1p for c2, gpu for c6); CPU side is the tuned CPU mode.
gpu_mode() { case $1 in c2-hop) echo gpu-1p-tuned ;; c6-lookup) echo gpu-tuned ;; *) echo gpu-batch-tuned ;; esac; }
FAIL=0
IFS=, read -r -a CASE_LIST <<< "$CASES"
for c in "${CASE_LIST[@]}"; do
  for mode in cpu-tuned "$(gpu_mode "$c")"; do
    echo "$(date +%T) case=$c mode=$mode warmup=$WARM records=$REC"
    TAG=t1 "$HERE/bench.sh" "$c" "$mode" 0 "$REC" "$OUT" --dump-final true || { echo "RUN_FAILED $c $mode"; FAIL=1; }
  done
done

python3 - "$OUT" "$QUICK" "$WARM" "$REC" "$CASES" <<'PY'
import json, os, sys
out, quick, warm, rec, cases = sys.argv[1], sys.argv[1+1] == "1", int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(",")
# digests (first 16 hex) of the 100,000 + 300,000 row smoke slice of REPRODUCE.md, section 5
REF = {"c1-filter": "2f29e3f499b32bbf", "c2-hop": "528ec5b56b375126", "c3-topn": "21d7d0915cc5edfa",
       "c4-dedup": "8593325de68ad677", "c5-join": "d9600de61e056926", "c6-lookup": "07800d9c2c117f9f"}
use_ref = quick and warm == 100000 and rec == 300000
def load(case, prefix):
    d = os.path.join(out, case)
    if not os.path.isdir(d): return None
    for mode in sorted(os.listdir(d)):
        if mode.startswith(prefix):
            for run in sorted(os.listdir(os.path.join(d, mode))):
                f = os.path.join(d, mode, run, "result.json")
                if os.path.exists(f): return json.load(open(f))
    return None
bad = 0
print()
print("| case | CPU final rows | GPU final rows | CPU sha256 (16) | GPU sha256 (16) | CPU vs GPU | GPU operator replaced | vs REPRODUCE reference |")
print("|---|---:|---:|---|---|---|---|---|")
for c in cases:
    a, b = load(c, "cpu"), load(c, "gpu")
    if not a or not b:
        print(f"| {c} | - | - | - | - | FAILED (no result) | - | - |"); bad += 1; continue
    m = a["final_sha256"] == b["final_sha256"]
    ref = "n/a (not the smoke slice)"
    if use_ref:
        ref = "equal" if a["final_sha256"].startswith(REF[c]) else "DIFFERENT (check sha256sum -c of the data)"
        if ref != "equal": bad += 1
    if not m: bad += 1
    if not any('gpu-replaced' in str(v) for v in b.get('shim_status', {}).values()): bad += 1
    print(f"| {c} | {a['final_rows']:,} | {b['final_rows']:,} | {a['final_sha256'][:16]} | {b['final_sha256'][:16]} | "
          f"{'MATCH' if m else 'MISMATCH'} | {'yes' if any('gpu-replaced' in str(v) for v in b.get('shim_status', {}).values()) else 'NO'} | {ref} |")
print()
print("Throughput and latency are in each result.json; a smoke run is not a benchmark and the numbers are not to be quoted.")
print("`GPU operator replaced` = the planner shim reports gpu-replaced (`shim_status` in result.json); a `NO` means that run did not use the GPU operator.")
sys.exit(1 if bad else 0)
PY
RC=$?
[[ $FAIL == 1 ]] && RC=1
echo "results: $OUT   exit=$RC"
exit $RC
