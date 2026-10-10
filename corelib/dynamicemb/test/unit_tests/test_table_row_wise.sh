#!/bin/bash
set -e

# A "node" here is LOCAL_WORLD_SIZE ranks. Both TorchRec's
# intra_and_cross_node_pg and the planner's Topology read it from the
# environment, so two nodes of one rank each is a faithful table-row-wise
# topology on a two-GPU box -- and the only shape where local_size <
# world_size, which is the whole point. With LOCAL_WORLD_SIZE unset,
# torchrun sets it to nproc_per_node and table-row-wise degenerates to
# row-wise on a single node, which is worth running as the boundary case.

run() {  # run <gpus> <local_world_size> <args...>
  local gpus=$1 lws=$2; shift 2
  echo "=== ${gpus} GPU(s), LOCAL_WORLD_SIZE=${lws}: $* ==="
  LOCAL_WORLD_SIZE=${lws} torchrun \
    --nnodes 1 --nproc_per_node "${gpus}" \
    ./test/unit_tests/test_table_row_wise.py "$@" || exit 1
}

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

# Checkpoints, including across geometries: a checkpoint written by a
# table-row-wise table has to load into a row-wise one and back, which is the
# resharding path M4 added and the one §8.1's defect lived on.
CKPT=/tmp/dynamicemb_trw_ckpt
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

echo "table-row-wise: all modes passed"
