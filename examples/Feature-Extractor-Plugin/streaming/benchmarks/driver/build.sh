#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Compile the benchmark driver.
#   FLINK_HOME=<Flink 2.2.1 dir> driver/build.sh                 # open-source Flink, CPU path only
#   MEGA_FLINK_CP=<class path> driver/build.sh                    # additionally compiles src/megaflink
# Output: $FEP_BUILD_DIR/classes (and classes-megaflink in the second case).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../scripts/env.sh"
CP=$(fep_flink_cp)
if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/javac" ]]; then
  JAVAC=("$JAVA_HOME/bin/javac")
elif command -v javac >/dev/null; then
  JAVAC=(javac)
else
  # JRE-only installs still ship the compiler module
  JAVAC=("$(fep_java)" -m jdk.compiler/com.sun.tools.javac.Main)
fi
compile() {   # compile <out dir> <source dir> <resource dir> [extra class path]
  mkdir -p "$1"
  find "$2" -name '*.java' > "$1/.sources.txt"
  "${JAVAC[@]}" -nowarn -encoding UTF-8 -cp "$CP${4:+:$4}" -d "$1" @"$1/.sources.txt"
  cp -r "$3"/. "$1"/
}
compile "$FEP_CLASSES" "$HERE/src/main/java" "$HERE/src/main/resources"
echo "built $FEP_CLASSES"
if fep_mega; then
  compile "$FEP_CLASSES_MEGA" "$HERE/src/megaflink/java" "$HERE/src/megaflink/resources" "$FEP_CLASSES"
  echo "built $FEP_CLASSES_MEGA"
fi
