#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One driver run.   usage: run-one.sh OUTDIR [driver args...]
# Environment: see env.sh (FLINK_HOME or MEGA_FLINK_CP, FEP_DATA_DIR, HEAP, RUN_TIMEOUT, JAVA_OPTS for extra -D options,
# WRAP for a profiler command placed in front of java, for example "nsys profile -o <file>").
# Writes OUTDIR/{result.json,plan.txt,plan-full.txt,jobgraph.txt,gc.log,stdout.log}; prints the RESULT line.
set -o pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
OUT=$1; shift
mkdir -p "$OUT"
CP=$(fep_flink_cp) || exit 2
[[ -d "$FEP_CLASSES" ]] || { echo "driver not built: run driver/build.sh first" >&2; exit 2; }
# the driver classes come first; in Mega-Flink mode its extra classes too
CP="$FEP_CLASSES:$CP"
LIBPATH=${LD_LIBRARY_PATH:-}
MEGA_OPT=()
if fep_mega; then
  CP="$FEP_CLASSES_MEGA:$CP"
  MEGA_OPT=(-Dfep.megaflink=true)
  LIBPATH="${MEGA_FLINK_NATIVE_DIR:+$MEGA_FLINK_NATIVE_DIR:}${CUDA_LIB_DIR:+$CUDA_LIB_DIR:}$LIBPATH"
fi
TMPD=$(mktemp -d "${FEP_TMP_DIR:-${TMPDIR:-/tmp}}/fep-bench-tmp.XXXXXX")   # per run; not removed by the script
HEAP=${HEAP:-16g}
# shellcheck disable=SC2086
LD_LIBRARY_PATH=$LIBPATH \
timeout "${RUN_TIMEOUT:-900}" ${WRAP:-} "$(fep_java)" -Xms$HEAP -Xmx$HEAP -XX:+UseG1GC \
  -Djava.io.tmpdir="$TMPD" -Dorg.slf4j.simpleLogger.defaultLogLevel=warn \
  -Xlog:gc:file="$OUT/gc.log" "${MEGA_OPT[@]}" ${FEP_JVM_ARGS:-} ${JAVA_OPTS:-} \
  -cp "$CP" fep.FepKuaiRandBench --out "$OUT" "$@" > "$OUT/stdout.log" 2>&1
RC=$?
echo "rc=$RC" >> "$OUT/stdout.log"
grep -E "^RESULT|Exception|Error" "$OUT/stdout.log" | head -5
exit $RC
