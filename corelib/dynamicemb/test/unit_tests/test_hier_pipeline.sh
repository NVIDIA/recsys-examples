#!/bin/bash
# test_hier_pipeline.py with the JaggedMegatron pipelines on 8, 4 or 2 GPUs, then on one GPU.
set -e

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)/../../examples"
NUM_GPUS=$(nvidia-smi -L | wc -l)
for nproc in 8 4 2; do
  [ "${nproc}" -le "${NUM_GPUS}" ] || continue
  torchrun --nnodes 1 --nproc_per_node "${nproc}" \
    ./test/unit_tests/test_hier_pipeline.py --megatron "$@"
  break
done
torchrun --nnodes 1 --nproc_per_node 1 ./test/unit_tests/test_hier_pipeline.py
