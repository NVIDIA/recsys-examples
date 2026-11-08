#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds and runs the JVM-free JNI smoke test against libraries produced by build.sh.
# Usage: ./selftest.sh [LIB_DIR]     (default LIB_DIR: ./build); needs a GPU and a JDK's jni.h
# (JNI_INCLUDE_DIR or JAVA_HOME). The test program is written to a temporary directory.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LIB_DIR=${1:-$SCRIPT_DIR/build}
if [[ -z "${JNI_INCLUDE_DIR:-}" && -n "${JAVA_HOME:-}" ]]; then JNI_INCLUDE_DIR=$JAVA_HOME/include; fi
[[ -f "${JNI_INCLUDE_DIR:-}/jni.h" ]] || { echo "set JNI_INCLUDE_DIR or JAVA_HOME (JDK)" >&2; exit 2; }
TMP=$(mktemp -d)
g++ -std=c++17 -O1 -I"$JNI_INCLUDE_DIR" -I"$JNI_INCLUDE_DIR/linux" \
  "$SCRIPT_DIR/selftest/native_selftest.cpp" -ldl -o "$TMP/native_selftest"
LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$TMP/native_selftest" "$LIB_DIR"
