#!/bin/bash
set -e

# A "node" here is LOCAL_WORLD_SIZE ranks, which TorchRec's
# intra_and_cross_node_pg and the planner's Topology both read. Two nodes of
# one rank each is a faithful table-row-wise topology on a two-GPU box -- and
# local_size < world_size is the only shape where table-row-wise differs from
# row-wise at all. The equal case is run too, as the boundary where an
# off-by-one between the two would show.

# Which half to run: "fwd_bwd", "load_dump", or empty for both. The central
# runner (test/unit_test.sh) passes its own group through, so this script is
# registered in both groups and each run covers the half it is about.
GROUP="${1:-}"
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
for dist_type in roundrobin hash_roundrobin; do
  # Two nodes of one rank: a table-row-wise table is on one of them.
  run 2 1 --mode plan   --dist-type ${dist_type}
  run 2 1 --mode parity --dist-type ${dist_type}
  run 2 1 --mode pooling --dist-type ${dist_type}
  run 2 1 --mode jagged  --dist-type ${dist_type}

  # One node: local_size == world_size, so table-row-wise is row-wise. The
  # degenerate case, which is where an off-by-one between the two shows up.
  run 2 2 --mode plan   --dist-type ${dist_type}
  run 2 2 --mode parity --dist-type ${dist_type}
done

# Four ranks split two ways, so a node holds more than one rank.
if [ "$(nvidia-smi -L | wc -l)" -ge 4 ]; then
  run 4 2 --mode plan   --dist-type hash_roundrobin
  run 4 2 --mode parity --dist-type hash_roundrobin
  run 4 2 --mode jagged --dist-type hash_roundrobin
fi
fi  # fwd_bwd

# Checkpoints, including across geometries: a checkpoint written by a
# table-row-wise table has to load into a row-wise one and back, which is the
# resharding path M4 added and the one §8.1's defect lived on.
CKPT=/tmp/dynamicemb_trw_ckpt
if want load_dump; then
for dist_type in roundrobin hash_roundrobin; do
  for dump_sharding in table_row_wise row_wise; do
    for load_sharding in table_row_wise row_wise; do
      rm -rf ${CKPT}
      run 2 1 --mode dump --trw-sharding ${dump_sharding} \
          --dist-type ${dist_type} --save-path ${CKPT}
      run 2 1 --mode load --trw-sharding ${load_sharding} \
          --dist-type ${dist_type} --save-path ${CKPT}
    done
  done
done
rm -rf ${CKPT}
fi  # load_dump

echo "table-row-wise${GROUP:+ ($GROUP)}: all modes passed"
