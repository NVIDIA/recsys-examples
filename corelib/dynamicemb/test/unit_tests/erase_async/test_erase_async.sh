#!/bin/bash
# Erasing the keys whose score is below a threshold, in the background of
# training. The kernel and its Python wrapper on a hash table, what an erase leaves
# in a slot, the storage's scan of a table in chunks, and the gate, the worker thread
# and the handle, alone and on a module beside a prefetch, the calls a user makes, an erase end to end, and the call on a sharded model.
set -e

DIR=./test/unit_tests/erase_async

pytest -svv $DIR/test_erase_by_score.py \
             $DIR/test_storage_erase_by_score.py \
             $DIR/test_async_erase.py \
             $DIR/test_erase_api.py \
             $DIR/test_erase_end_to_end.py

# The call on a model TorchRec has sharded: a DistributedModelParallel, two ranks.
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  $DIR/test_erase_distributed.py
