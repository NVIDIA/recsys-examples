#!/bin/bash
set -e

# A "node" here is LOCAL_WORLD_SIZE ranks, which TorchRec's
# intra_and_cross_node_pg and the planner's Topology both read. Two nodes of
# one rank each is a faithful table-row-wise topology on a two-GPU box -- and
# local_size < world_size is the only shape where table-row-wise differs from
# row-wise at all. The equal case is run too, as the boundary where an
# off-by-one between the two would show.

# Which half to run: "fwd_bwd", "load_dump", or empty for both. The central
# runner (test/unit_test.sh) sets DYNAMICEMB_TEST_GROUP, so this script is
# registered in both groups and each run covers the half it is about. A
# positional works too, for running one half by hand.
GROUP="${1:-${DYNAMICEMB_TEST_GROUP:-}}"
want() { [ -z "$GROUP" ] || [ "$GROUP" = "$1" ]; }

run() {  # run <gpus> <ranks-per-node> <args...>
  local gpus=$1 node_size=$2; shift 2
  echo "=== ${gpus} GPU(s), nodes of ${node_size}: $* ==="
  # --node-size, not the environment: torchrun writes LOCAL_WORLD_SIZE =
  # nproc_per_node over whatever the shell exports, so an exported value
  # silently gives one node and a suite that never tests table-row-wise.
  torchrun --nnodes 1 --nproc_per_node "${gpus}" \
    ./test/unit_tests/test_table_row_wise.py --node-size "${node_size}" "$@" || exit 1
}

if want fwd_bwd; then
# `plan` reads no key, so it is run once per topology rather than once per
# dist_type: check_plan asserts sharding types, shard counts, which node the
# table landed on and how many rows a rank got, and none of those is what the
# bucketizer decides.
#
# Two nodes of one rank, then one node of two. The second is the degenerate
# case -- local_size == world_size, where table-row-wise and row-wise should
# agree exactly, and where an off-by-one between them shows up.
run 2 1 --mode plan --dist-type hash_roundrobin
run 2 2 --mode plan --dist-type hash_roundrobin

# `parity` and `jagged` do read keys, so both rules are run. Bags of differing
# length, empty ones included, reach the bucketizer's edges, and the two rules
# bucket by different arithmetic.
for dist_type in roundrobin hash_roundrobin hash_bucket; do
  run 2 1 --mode parity --dist-type ${dist_type}
  run 2 1 --mode jagged --dist-type ${dist_type}
  run 2 2 --mode parity --dist-type ${dist_type}
done

# MEAN against SUM, the one absolute check here. Pooling happens after the
# lookup -- the kernel's downgrade to SUM and the EBC's own mean callback --
# so like `plan` it does not depend on where a key was sent, and a rule that
# sent one to the wrong rank would already have failed `parity` above.
run 2 1 --mode pooling --dist-type hash_roundrobin

# The backward: the gradient divisions depend on the node size and the node
# count, not on the rule that placed the key, so one rule and both topologies.
# The four-rank one below is the only place the two divisions are both > 1.
run 2 1 --mode grad --dist-type hash_roundrobin
run 2 1 --mode grad --dist-type hash_bucket

# Four ranks split two ways, so a node holds more than one rank. The only
# topology here where intra-node is itself a group.
if [ "$(nvidia-smi -L | wc -l)" -ge 4 ]; then
  run 4 2 --mode plan   --dist-type hash_roundrobin
  run 4 2 --mode parity --dist-type hash_roundrobin
  run 4 2 --mode jagged --dist-type hash_roundrobin
  run 4 2 --mode grad   --dist-type hash_roundrobin
fi
fi  # fwd_bwd

# Checkpoints, including across geometries: a checkpoint written by a
# table-row-wise table has to load into a row-wise one and back, which is the
# resharding path M4 added and the one §8.1's defect lived on.
CKPT=/tmp/dynamicemb_trw_ckpt
# The three combinations that involve a table-row-wise table, written out
# rather than looped over: row_wise to row_wise is the path every other
# checkpoint suite already covers, and both halves of it are reached by the
# two crossings below.
if want load_dump; then
for dist_type in roundrobin hash_roundrobin hash_bucket; do
  for pair in "table_row_wise table_row_wise" \
              "table_row_wise row_wise" \
              "row_wise table_row_wise"; do
    dump_sharding=${pair% *} load_sharding=${pair#* }
    rm -rf ${CKPT}
    run 2 1 --mode dump --trw-sharding ${dump_sharding} \
        --dist-type ${dist_type} --save-path ${CKPT}
    run 2 1 --mode load --trw-sharding ${load_sharding} \
        --dist-type ${dist_type} --save-path ${CKPT}
  done
done
rm -rf ${CKPT}
fi  # load_dump

echo "table-row-wise${GROUP:+ ($GROUP)}: all modes passed"
