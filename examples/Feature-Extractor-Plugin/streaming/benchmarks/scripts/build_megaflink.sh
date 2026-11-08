#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One command from nothing to a usable Mega-Flink build for the GPU side of the streaming benchmarks.
#
#   WORK_DIR=$HOME/megaflink CUDA_TOOLKIT_ROOT=/usr/local/cuda JAVA_HOME=<JDK 11> build_megaflink.sh
#
# Steps: clone Mega-Flink (Apache-2.0, https://github.com/cb521/Mega-Flink) -> check out the pinned commit -> apply the
# patches in benchmarks/patches/*.patch (none are needed at the pinned commit, see below) -> compile the Java modules
# with the repository's own Maven wrapper -> compile the native CUDA/JNI libraries for one GPU architecture -> write
# $WORK_DIR/megaflink.env, which the run scripts `source` (MEGA_FLINK_CP, MEGA_FLINK_NATIVE_DIR, CUDA_LIB_DIR).
#
# Nothing is installed anywhere: the benchmark driver runs Flink's MiniCluster in its own JVM and takes Mega-Flink
# from a class path, so your Flink installation (FLINK_HOME) is not read or modified. (Installing Mega-Flink into a
# real Flink cluster is a different, optional job: see "Installing into a cluster" in REPRODUCE.md.)
#
# Environment (all optional except where noted):
#   MEGA_FLINK_REF      commit to build; default = the commit the published numbers were measured with (see DEFAULT_REF)
#   MEGA_FLINK_REPO     git URL (default https://github.com/cb521/Mega-Flink)
#   WORK_DIR            build directory (default ./megaflink-build); an existing clone is reused after its commit is checked
#   CUDA_TOOLKIT_ROOT   CUDA toolkit with bin/nvcc (needed for the native step; default: nvcc found on PATH)
#   ARCHS               ONE CUDA architecture number, default 89 (sm_89, L20 / Ada). Mega-Flink's build takes a single
#                       architecture. Its device check accepts, besides an H20 (sm_90) by name, any device whose
#                       compute-capability major equals the major of this build (89 -> 8). 90 keeps the H20-only rule.
#   JAVA_HOME           JDK to use (JDK 11 tested). A JRE without a compiler module is not enough.
#   FLINK_HOME          accepted for symmetry with the run scripts; NOT read or modified by this script
#   MAVEN_OPTS_EXTRA    extra arguments for Maven (for example -o for an offline build with a warm cache)
#
# usage: build_megaflink.sh [--java-only | --native-only] [--force]
#   --java-only    clone, patch and compile Java, write the env file without native libraries (a machine without CUDA)
#   --native-only  compile only the native libraries (a GPU node with nvcc; run --java-only first, same WORK_DIR)
#   --force        redo the steps even if their outputs exist
#
# Patches: at the pinned commit no source patch is required. The one thing that looks like a blocker, the device check
# that admits only a device named "H20", is already switchable by a build option of the upstream script
# (CUDF_FLINK_CUDA_ARCH=89 defines CUDF_FLINK_EXPERIMENTAL_DEVICE_MAJOR=8, see native/gpu_arch_compat.h); this script
# passes that option. Anything placed in benchmarks/patches/*.patch is applied in file-name order.
set -euo pipefail

# The commit of the Mega-Flink main branch that was built for the published numbers (one worktree, detached, no
# source change that affects the results; see benchmarks/REPRODUCE.md).
DEFAULT_REF=f7f06845cb6b457c83d424d6424632fbf929e230

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_DIR=${PATCH_DIR:-$HERE/../patches}
REF=${MEGA_FLINK_REF:-$DEFAULT_REF}
REPO=${MEGA_FLINK_REPO:-https://github.com/cb521/Mega-Flink}
WORK_DIR=${WORK_DIR:-$PWD/megaflink-build}
ARCHS=${ARCHS:-89}
DO_JAVA=1; DO_NATIVE=1; FORCE=0
for a in "$@"; do
  case $a in
    --java-only) DO_NATIVE=0 ;;
    --native-only) DO_JAVA=0 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n 2,45p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done
log() { echo "[build_megaflink] $*"; }
die() { echo "[build_megaflink] ERROR: $*" >&2; exit 1; }

[[ "$ARCHS" =~ ^[0-9]+a?$ ]] || die "ARCHS must be one architecture number such as 89 or 90 (got '$ARCHS')"
mkdir -p "$WORK_DIR"
WORK_DIR=$(cd "$WORK_DIR" && pwd)
SRC=$WORK_DIR/Mega-Flink
ENV_FILE=$WORK_DIR/megaflink.env
if [[ -n "${JAVA_HOME:-}" ]]; then export PATH="$JAVA_HOME/bin:$PATH"; fi
command -v git >/dev/null || die "git not found"
command -v java >/dev/null || die "java not found (set JAVA_HOME to a JDK 11)"

# ---- 1. clone and pin -------------------------------------------------------------------------------------------
if [[ ! -d "$SRC/.git" ]]; then
  log "cloning $REPO into $SRC"
  git clone --quiet "$REPO" "$SRC"
fi
cd "$SRC"
if ! git cat-file -e "$REF^{commit}" 2>/dev/null; then git fetch --quiet origin "$REF" 2>/dev/null || git fetch --quiet origin; fi
FULL=$(git rev-parse --verify "$REF^{commit}") || die "commit $REF is not in $REPO"
HEAD_NOW=$(git rev-parse HEAD)
if [[ "$HEAD_NOW" != "$FULL" ]]; then
  # an existing clone with local changes is not touched
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] \
    || die "$SRC has uncommitted changes and is at $HEAD_NOW, not $FULL; use a clean WORK_DIR"
  log "checking out $FULL"
  git checkout --quiet --detach "$FULL"
fi
[[ "$(git rev-parse HEAD)" == "$FULL" ]] || die "checkout failed"
log "Mega-Flink at $FULL"

# ---- 2. patches (idempotent) ------------------------------------------------------------------------------------
shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
shopt -u nullglob
for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
  if git apply --reverse --check "$p" >/dev/null 2>&1; then
    log "patch already applied: $(basename "$p")"
  elif git apply --check "$p" >/dev/null 2>&1; then
    git apply "$p"; log "applied patch: $(basename "$p")"
  else
    die "patch $(basename "$p") does not apply to $FULL (wrong MEGA_FLINK_REF?)"
  fi
done
[[ ${#PATCHES[@]} -gt 0 ]] || log "no patches to apply (none are needed at $DEFAULT_REF)"

# ---- 3. Java ----------------------------------------------------------------------------------------------------
if [[ $DO_JAVA == 1 ]]; then
  if [[ $FORCE == 1 || ! -f modules/benchmarks/target/classpath.txt || ! -d modules/runtime/target/classes \
        || ! -d modules/flink-2.2.1-shim/target/classes ]]; then
    log "compiling Java modules (downloads Maven and the Flink 2.2.1 dependencies on first use)"
    # shellcheck disable=SC2086
    ./mvnw -B -q -DskipTests -pl modules/benchmarks -am ${MAVEN_OPTS_EXTRA:-} \
      compile dependency:build-classpath -Dmdep.outputFile=target/classpath.txt
  else
    log "Java already built (use --force to rebuild)"
  fi
  for m in modules/runtime modules/flink-2.2.1-shim modules/benchmarks; do
    [[ -d "$m/target/classes" ]] || die "$m/target/classes missing after the Java build"
  done
  [[ -s modules/benchmarks/target/classpath.txt ]] || die "modules/benchmarks/target/classpath.txt missing or empty"
fi

# ---- 4. native --------------------------------------------------------------------------------------------------
NATIVE_DIR=$SRC/target/native
if [[ $DO_NATIVE == 1 ]]; then
  if [[ -z "${CUDA_TOOLKIT_ROOT:-}" ]]; then
    command -v nvcc >/dev/null && CUDA_TOOLKIT_ROOT=$(cd "$(dirname "$(command -v nvcc)")/.." && pwd) || true
  fi
  [[ -n "${CUDA_TOOLKIT_ROOT:-}" && -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" ]] \
    || die "no nvcc: set CUDA_TOOLKIT_ROOT to a CUDA toolkit (12.x tested: 12.4), or use --java-only on this machine"
  export CUDA_TOOLKIT_ROOT
  if [[ $FORCE == 1 || ! -f "$NATIVE_DIR/libcudf_flink_device.so" ]]; then
    log "compiling native libraries for sm_$ARCHS with $("$CUDA_TOOLKIT_ROOT/bin/nvcc" --version | tail -1)"
    # jni.h comes from the JDK; with a JRE-only java the upstream script downloads and checks the two OpenJDK headers.
    CUDF_FLINK_CUDA_ARCH=$ARCHS CUDF_FLINK_NATIVE_OUTPUT_DIR=$NATIVE_DIR ./scripts/build-native.sh
  else
    log "native libraries already built (use --force to rebuild)"
  fi
  for l in device device_batch groupby lookup_join deduplicate topn join fused generic_calc; do
    [[ -f "$NATIVE_DIR/libcudf_flink_$l.so" ]] || die "libcudf_flink_$l.so missing in $NATIVE_DIR"
  done
fi

# ---- 5. env file ------------------------------------------------------------------------------------------------
if [[ -d modules/benchmarks/target/classes && -s modules/benchmarks/target/classpath.txt ]]; then
  CP="$SRC/modules/benchmarks/target/classes:$SRC/modules/flink-2.2.1-shim/target/classes:$SRC/modules/runtime/target/classes:$(cat modules/benchmarks/target/classpath.txt)"
  {
    echo "# written by build_megaflink.sh; source this file before the run scripts"
    echo "# Mega-Flink commit $FULL, native architecture sm_$ARCHS"
    echo "export MEGA_FLINK_COMMIT=$FULL"
    echo "export MEGA_FLINK_ARCH=$ARCHS"
    echo "export MEGA_FLINK_CP='$CP'"
    [[ -d "$NATIVE_DIR" ]] && echo "export MEGA_FLINK_NATIVE_DIR='$NATIVE_DIR'"
    [[ -n "${CUDA_TOOLKIT_ROOT:-}" ]] && echo "export CUDA_LIB_DIR='$CUDA_TOOLKIT_ROOT/lib64'"
    [[ -n "${JAVA_HOME:-}" ]] && echo "export JAVA_HOME='$JAVA_HOME'"
  } > "$ENV_FILE"
  log "wrote $ENV_FILE"
  log "next:  source $ENV_FILE && <benchmarks>/scripts/run_all_six.sh --quick"
fi
