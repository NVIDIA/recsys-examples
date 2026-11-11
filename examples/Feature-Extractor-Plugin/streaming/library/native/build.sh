#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0),
# scripts/build-native.sh. Reduced to the libraries needed by window aggregation and the
# bounded INNER lookup join.
#
# Builds the JNI shared libraries that the Java side loads with System.loadLibrary:
#
#   libcudf_flink_device.so        CudaDeviceBinding        (device select / memory info)
#   libcudf_flink_device_batch.so  device-batch runtime     (linked by the groupby library)
#   libcudf_flink_groupby.so       DenseGroupbyKernel       (window aggregation, 2-stage HOP)
#   libcudf_flink_lookup_join.so   LookupJoinKernel         (bounded INNER dimension lookup)
#
# Environment / options (all optional):
#   CUDA_TOOLKIT_ROOT   CUDA toolkit root containing bin/nvcc (default: nvcc on PATH)
#   FEP_CUDA_ARCHS      space or comma separated list, default "89 90a"
#                       (89 = L20 / Ada, 90a = H20 / Hopper)
#   JNI_INCLUDE_DIR     directory holding jni.h (default: $JAVA_HOME/include, else the
#                       JDK behind `java` on PATH). A JRE without headers is not enough.
#   FEP_NATIVE_OUTPUT_DIR  output directory (default: <this dir>/build)
#   FEP_NVCC_EXTRA_FLAGS   extra flags appended to every nvcc call
#
# Usage:  ./build.sh [--archs "89 90a"] [--cuda-root DIR] [--out DIR]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CUDA_ROOT=${CUDA_TOOLKIT_ROOT:-}
ARCHS=${FEP_CUDA_ARCHS:-"89 90a"}
OUT_DIR=${FEP_NATIVE_OUTPUT_DIR:-$SCRIPT_DIR/build}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archs) ARCHS=$2; shift 2 ;;
    --cuda-root) CUDA_ROOT=$2; shift 2 ;;
    --out) OUT_DIR=$2; shift 2 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---- toolchain ---------------------------------------------------------------------------
if [[ -n "$CUDA_ROOT" ]]; then
  NVCC=$CUDA_ROOT/bin/nvcc
elif command -v nvcc >/dev/null 2>&1; then
  NVCC=$(command -v nvcc)
else
  echo "nvcc not found: set CUDA_TOOLKIT_ROOT or put nvcc on PATH." >&2
  exit 2
fi
[[ -x "$NVCC" ]] || { echo "nvcc is not executable: $NVCC" >&2; exit 2; }

if [[ -z "${JNI_INCLUDE_DIR:-}" ]]; then
  if [[ -n "${JAVA_HOME:-}" ]]; then
    JNI_INCLUDE_DIR=$JAVA_HOME/include
  elif command -v java >/dev/null 2>&1; then
    JNI_INCLUDE_DIR=$(cd -- "$(dirname -- "$(readlink -f "$(command -v java)")")/.." && pwd)/include
  fi
fi
if [[ ! -f "${JNI_INCLUDE_DIR:-}/jni.h" ]]; then
  echo "jni.h not found (looked in '${JNI_INCLUDE_DIR:-}'): install a JDK or set JNI_INCLUDE_DIR." >&2
  exit 2
fi

# ---- architectures -----------------------------------------------------------------------
GENCODE=()
ADMIT_MASK=0
for arch in ${ARCHS//,/ }; do
  if [[ ! "$arch" =~ ^[0-9]+a?$ ]]; then
    echo "bad architecture '$arch' (expected e.g. 89, 90a)" >&2
    exit 2
  fi
  GENCODE+=(-gencode "arch=compute_${arch},code=sm_${arch}")
  number=${arch%a}
  ADMIT_MASK=$((ADMIT_MASK | (1 << (10#$number / 10))))
done
[[ ${#GENCODE[@]} -gt 0 ]] || { echo "no architectures given" >&2; exit 2; }

# The runtime device-admission check (gpu_arch_compat.h) admits exactly the compute-capability
# majors that were compiled in.
DEFINES=("-DFEP_ADMITTED_DEVICE_MAJORS_MASK=${ADMIT_MASK}u")

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd -- "$OUT_DIR" && pwd)
# Hide the checkout location in compiler-emitted paths.
FLAGS=(-std=c++17 -O3 -lineinfo "${GENCODE[@]}" "${DEFINES[@]}"
       "-Xcompiler=-fPIC" "-Xcompiler=-ffile-prefix-map=$SCRIPT_DIR=."
       "-Xcompiler=-fdebug-prefix-map=$SCRIPT_DIR=."
       -shared "-I$JNI_INCLUDE_DIR" "-I$JNI_INCLUDE_DIR/linux" "-I.")
# shellcheck disable=SC2206
EXTRA=(${FEP_NVCC_EXTRA_FLAGS:-})

echo "nvcc: $NVCC ($("$NVCC" --version | tail -1))"
echo "archs: ${ARCHS}  admit-mask: ${ADMIT_MASK}  out: $OUT_DIR"

cd "$SCRIPT_DIR"

# nvcc embeds the absolute source path in device metadata; pin it to the bare file name.
build_one() {
  local src=$1; shift
  "$NVCC" "${FLAGS[@]}" "${EXTRA[@]}" \
    "-Xcudafe=--orig_src_path_name=$src" "-Xcicc=--orig_src_path_name=$src" \
    -Xptxas=-suppress-debug-info "$src" "$@"
}

build_one cuda_device_jni.cu \
  -o "$OUT_DIR/libcudf_flink_device.so"

build_one gpu_device_batch_jni.cu \
  -Xlinker -soname -Xlinker libcudf_flink_device_batch.so \
  -o "$OUT_DIR/libcudf_flink_device_batch.so"

# The groupby library depends on the device-batch library; $ORIGIN lets the loader find it
# next to this file, so only the output directory has to be on java.library.path.
build_one dense_groupby_jni.cu \
  -Xlinker --no-as-needed -L"$OUT_DIR" -lcudf_flink_device_batch \
  -Xlinker --disable-new-dtags -Xlinker -rpath -Xlinker '$ORIGIN' \
  -o "$OUT_DIR/libcudf_flink_groupby.so"

build_one lookup_join_jni.cu \
  -o "$OUT_DIR/libcudf_flink_lookup_join.so"

echo "FEP_NATIVE_BUILD_OK $OUT_DIR"
ls -l "$OUT_DIR"/libcudf_flink_*.so
