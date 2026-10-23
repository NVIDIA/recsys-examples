#!/usr/bin/env bash
# Build and run the embedding_cache tests.
#
# These need nvcc to build and a GPU to run. They used to build with a plain C++
# compiler against a stand-in for the CUDA runtime, which let them run anywhere;
# that ended when HashMapStorage started launching kernels rather than staging
# initialisation through the host. The trade was deliberate — one implementation
# of each operation instead of two — but it does mean there is no longer a way
# to check this tree without a device.
#
# What they cover: structure and protocol, one thread per operation. What they
# still do not cover: the bucket lock under contention, which wants many threads
# on one bucket and belongs in a test of its own.
#
# Usage:  ./run_tests.sh [output-dir]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-${TMPDIR:-/tmp}/embedding_cache_tests}"
mkdir -p "$out"

NVCC="${NVCC:-nvcc}"

# HashMapStorage::grid_for reads dyn_emb::DeviceProp for the SM count, and that
# singleton is defined in utils.cpp rather than the header. This is the only
# piece of the wider tree embedding_cache links against; if the list grows, that
# is a signal worth noticing rather than a build file to keep extending.
deps=("$here/../../src/utils.cpp")

# Built for whatever device is here rather than as a fat binary: these are
# tests, and they run where they were built.
#
# Not sm_70, even though that is the design's floor — the bucket lock spins and
# needs Volta's independent thread scheduling, and the scoped atomics need it to
# lower to real acquire/release forms. CUDA 13 dropped Volta, so `nvcc fatal:
# Unsupported gpu architecture 'compute_70'` is what asking for it now gets.
# The requirement is unchanged; what can still compile for it is not.
#
# `-i 0` rather than piping through `head`: under `set -e` with `pipefail`, head
# closing the pipe kills nvidia-smi with SIGPIPE and takes the whole script with
# it, before it has printed anything.
ARCH="${EC_ARCH:-}"
if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
  ARCH=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader \
         2>/dev/null | tr -d '.' || true)
fi
ARCH="${ARCH:-90}"

flags=(-std=c++17 -O1 -g -lineinfo
       -arch="sm_${ARCH}"
       --expt-relaxed-constexpr --expt-extended-lambda
       -I"$here/../../src/embedding_cache" -I"$here")

status=0
shopt -s nullglob

run() {  # run <name> <binary>
  if "$2"; then
    printf '%s: PASS\n\n' "$1"
  else
    printf '%s: FAIL\n\n' "$1"
    status=1
  fi
}

# .cpp tests cover the components that have no kernels — NodeState is a value
# type, so a plain compiler and a stand-in for the CUDA runtime reach all of it.
# This is not a fallback path: it is the whole of what those components are.
#
# Built twice, because the assertions inside them are compiled out by -DNDEBUG
# and the shipped build defines it. Once each way means neither form rots.
CXX="${CXX:-g++}"
cxx_flags=(-std=c++17 -O1 -g -Wall -Wextra -Wno-unused-parameter
           -I"$here/../../src/embedding_cache" -I"$here"
           -I"$here/host_shim")
for src in "$here"/test_*.cpp; do
  name="$(basename "$src" .cpp)"
  for mode in -UNDEBUG -DNDEBUG; do
    printf '=== %s [%s] ===\n' "$name" "$mode"
    "$CXX" "${cxx_flags[@]}" "$mode" "$src" -o "$out/$name"
    run "$name" "$out/$name"
  done
done

# .cu tests cover the components that launch kernels. These need a GPU.
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found: skipping the device tests. Set NVCC= to point at a" >&2
  echo "CUDA toolkit. The .cpp tests above ran and are not a substitute." >&2
  exit "$status"
fi
for src in "$here"/test_*.cu; do
  name="$(basename "$src" .cu)"
  printf '=== %s ===\n' "$name"
  "$NVCC" "${flags[@]}" "$src" "${deps[@]}" -o "$out/$name"
  run "$name" "$out/$name"
done

exit "$status"
