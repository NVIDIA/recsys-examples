#!/bin/bash
set -e

# DMPCollection shards a table over a sharding group and replicates the group.
# Two GPUs in groups of one are two replicas of whole tables; four in groups of
# two are the shape where a table is both sharded and replicated, run with
# contiguous groups and with strided ones (use_inter_host_allreduce=False).

# Which half to run: "fwd_bwd", "load_dump", or empty for both. The central
# runner (test/unit_test.sh) sets DYNAMICEMB_TEST_GROUP.
GROUP="${1:-${DYNAMICEMB_TEST_GROUP:-}}"
want() { [ -z "$GROUP" ] || [ "$GROUP" = "$1" ]; }

run() {  # run <gpus> <sharding-group-size> <args...>
  local gpus=$1 group_size=$2; shift 2
  echo "=== ${gpus} GPU(s), sharding groups of ${group_size}: $* ==="
  torchrun --nnodes 1 --nproc_per_node "${gpus}" \
    ./test/unit_tests/test_two_d_sparse.py --sharding-group-size "${group_size}" "$@" || exit 1
}

four_gpus=false
if [ "$(nvidia-smi -L | wc -l)" -ge 4 ]; then
  four_gpus=true
fi

if want fwd_bwd; then
for dtype in fp32 fp16 fp8; do
  run 2 1 --mode sync --embedding-dtype ${dtype}
done
run 2 1 --mode sync --score-strategy lru
run 2 1 --mode sync --optimizer sgd
run 2 1 --mode sync --grow --mixed-tables
run 2 1 --mode reject --score-strategy step
if ${four_gpus}; then
  run 3 1 --mode sync --optimizer adam --embedding-dtype fp16 --random-initializer --uint64-keys --resync
  run 4 2 --mode sync --embedding-dtype fp16
  run 4 2 --mode sync --embedding-dtype fp16 --strided
  run 4 2 --mode sync --grow
fi
fi  # fwd_bwd

CKPT=/tmp/dynamicemb_two_d_ckpt
if want load_dump; then
rm -rf ${CKPT}
run 2 1 --mode checkpoint --save-path ${CKPT}
if ${four_gpus}; then
  rm -rf ${CKPT}
  run 4 2 --mode checkpoint --save-path ${CKPT}
fi
rm -rf ${CKPT}
fi  # load_dump

echo "2D sparse${GROUP:+ ($GROUP)}: all modes passed"
