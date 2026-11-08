# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared settings for the streaming benchmark scripts. Source it; do not run it.
#
# Environment (all optional unless marked):
#   FLINK_HOME         unpacked Apache Flink 2.2.1 binary distribution (required unless MEGA_FLINK_CP is set)
#   FEP_DATA_DIR       directory written by prepare_stream_data.py (required to run the driver)
#   FEP_BUILD_DIR      where build.sh writes class files (default: <benchmarks>/driver/build)
#   FEP_TMP_DIR        parent of the per-run java.io.tmpdir (default: $TMPDIR or /tmp); put it on a fast local disk
#   HEAP               JVM heap, -Xms and -Xmx (default 16g, the setting of the published runs)
#   RUN_TIMEOUT        seconds before one driver run is killed (default 900)
#   JAVA_HOME          JDK to use (JDK 11 is what was tested); `java` on PATH otherwise
#   MEGA_FLINK_CP      full class path of a Mega-Flink build (Flink 2.2.1 jars + plugin + its benchmark harness).
#                      When set, the driver runs in Mega-Flink mode (GPU modes become available).
#   MEGA_FLINK_NATIVE_DIR  directory with the Mega-Flink native libraries (added to LD_LIBRARY_PATH)
#   CUDA_LIB_DIR       CUDA runtime library directory (added to LD_LIBRARY_PATH), for example /usr/local/cuda/lib64
#   FEP_JVM_ARGS       extra JVM arguments (for JDK 17+ add the --add-opens flags that Flink's bin/config.sh uses)

FEP_BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FEP_BUILD_DIR=${FEP_BUILD_DIR:-$FEP_BENCH_DIR/driver/build}
FEP_CLASSES=$FEP_BUILD_DIR/classes
FEP_CLASSES_MEGA=$FEP_BUILD_DIR/classes-megaflink

fep_java() { if [[ -n "${JAVA_HOME:-}" ]]; then echo "$JAVA_HOME/bin/java"; else echo java; fi; }

# Class path for compiling and running: Mega-Flink's if given, otherwise the open-source Flink distribution.
fep_flink_cp() {
  if [[ -n "${MEGA_FLINK_CP:-}" ]]; then
    echo "$MEGA_FLINK_CP"
    return
  fi
  if [[ -z "${FLINK_HOME:-}" || ! -d "$FLINK_HOME/lib" ]]; then
    echo "set FLINK_HOME to an unpacked Flink 2.2.1 distribution (or MEGA_FLINK_CP for a Mega-Flink build)" >&2
    return 1
  fi
  local cp="" j
  # the table planner must be the plain jar from opt/, not the loader jar from lib/
  for j in "$FLINK_HOME"/lib/*.jar; do
    case "$j" in *flink-table-planner-loader*) continue ;; esac
    cp="$cp:$j"
  done
  for j in "$FLINK_HOME"/opt/flink-table-planner_*.jar; do cp="$cp:$j"; done
  echo "${cp#:}"
}

fep_mega() { [[ -n "${MEGA_FLINK_CP:-}" ]]; }
