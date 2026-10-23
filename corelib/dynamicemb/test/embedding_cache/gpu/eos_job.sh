#!/bin/bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=emb_cache
#SBATCH --output=%x-%j.log
# ---------------------------------------------------------------------------
# Compile and run the embedding_cache tests on EOS.
#
# Nothing in src/embedding_cache has ever been through nvcc, so step 2 is the
# first thing worth knowing: <cuda/atomic>, cuda::atomic_ref, __nanosleep,
# cooperative_groups and the launch syntax are all taken on trust until it runs.
#
# Batch rather than an interactive allocation: salloc needs a client to stay
# connected, and the result is a log file either way.
# ---------------------------------------------------------------------------
set -uo pipefail

IMG='gitlab-master.nvidia.com#devtech-compute/distributed-recommender:devel_latest'

echo "=== node: $(hostname) ==="
date

srun --ntasks=1 --container-image="$IMG" \
     --container-mounts="$HOME/emb_cache_test:/work" \
     bash -c '
set -uo pipefail
cd /work/dynamicemb || exit 1
FAILED=0

echo "=== toolchain ==="
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")

echo
echo "=== 1. host tests (value types only) ==="
( cd test/embedding_cache && ./run_tests.sh 2>&1 | grep -E "PASS|FAIL|error:" ) || FAILED=1

echo
echo "=== 2. nvcc compile ==="
cd test/embedding_cache/gpu
rm -f test_gpu
nvcc -O2 -std=c++17 -arch=sm_${ARCH} -lineinfo \
     --expt-relaxed-constexpr --expt-extended-lambda \
     -I../../../src/embedding_cache \
     -o test_gpu test_gpu.cu ../../../src/utils.cpp 2>&1 | head -60
if [ ! -x ./test_gpu ]; then echo "COMPILE FAILED"; exit 1; fi
echo "compile OK"

echo
echo "=== 3. run ==="
timeout 600 ./test_gpu; RC=$?
[ $RC -ne 0 ] && { echo "run rc=$RC"; FAILED=1; }

echo
echo "=== 4. racecheck ==="
# The point of being on a device. The host shim serialises every atomic, so
# nothing before step 3 could see a race.
#
# Scaled down: the detector instruments every access, and the full run takes
# longer than any sensible limit. A timeout is a failure — silence is not a pass.
EC_TEST_SCALE=10 timeout 1800 compute-sanitizer --tool racecheck \
  --racecheck-report analysis ./test_gpu > /tmp/racecheck.log 2>&1; RC=$?
grep -E "RACECHECK SUMMARY|hazard|Error|FAIL|passed" /tmp/racecheck.log | head -30
[ $RC -eq 124 ] && { echo "racecheck TIMED OUT: no verdict"; FAILED=1; }
grep -q "RACECHECK SUMMARY" /tmp/racecheck.log || { echo "racecheck printed no summary"; FAILED=1; }
grep -q "FAIL" /tmp/racecheck.log && { echo "racecheck: the test reported a failure"; FAILED=1; }

echo
echo "=== 5. memcheck ==="
timeout 1800 compute-sanitizer --tool memcheck ./test_gpu > /tmp/memcheck.log 2>&1; RC=$?
grep -E "ERROR SUMMARY|Invalid|FAIL|passed" /tmp/memcheck.log | head -30
[ $RC -eq 124 ] && { echo "memcheck TIMED OUT: no verdict"; FAILED=1; }
grep -q "ERROR SUMMARY" /tmp/memcheck.log || { echo "memcheck printed no summary"; FAILED=1; }
grep -q "FAIL" /tmp/memcheck.log && { echo "memcheck: the test reported a failure"; FAILED=1; }

echo
echo "=== 6. synccheck ==="
# The block-collective push/pop put __syncthreads() inside a loop whose bound
# every thread of a block has to agree on. This is the tool that says so.
timeout 900 compute-sanitizer --tool synccheck ./test_gpu > /tmp/synccheck.log 2>&1; RC=$?
grep -E "ERROR SUMMARY|Barrier|FAIL|passed" /tmp/synccheck.log | head -20
[ $RC -eq 124 ] && { echo "synccheck TIMED OUT: no verdict"; FAILED=1; }
grep -q "ERROR SUMMARY" /tmp/synccheck.log || { echo "synccheck printed no summary"; FAILED=1; }
grep -q "FAIL" /tmp/synccheck.log && { echo "synccheck: the test reported a failure"; FAILED=1; }

echo
echo "=== 7. does nvcc already aggregate the warp-level atomics? ==="
# reserve.h bets that warp_reserve is worth writing. If nvcc already emits a
# warp-aggregated RED/ATOM for a plain atomicAdd of a constant, it is not.
nvcc -O2 -std=c++17 -arch=sm_${ARCH} --expt-relaxed-constexpr \
     --expt-extended-lambda -I../../../src/embedding_cache \
     -ptx -o /tmp/probe.ptx test_gpu.cu 2>/dev/null \
  && grep -cE "red\.|atom\." /tmp/probe.ptx | sed "s/^/  atomic ops in PTX: /"
nvcc -O2 -std=c++17 -arch=sm_${ARCH} --expt-relaxed-constexpr \
     --expt-extended-lambda -I../../../src/embedding_cache \
     -cubin -o /tmp/probe.cubin test_gpu.cu 2>/dev/null
if [ -f /tmp/probe.cubin ]; then
  cuobjdump -sass /tmp/probe.cubin 2>/dev/null \
    | grep -oE "(RED|ATOM|ATOMG)[A-Z.]*" | sort | uniq -c | head -10
fi

echo
echo "=== overall: $FAILED (0 = all good) ==="
exit $FAILED
'
RC=$?
echo
echo "=== srun exit: $RC ==="
date
exit $RC
