#!/bin/bash
# Everything about a score per key. What needs no process group first -- the
# ScoredKeyedJaggedTensor, the bucketize that carries its scores, the options
# and what is refused -- then the two-GPU run, which builds sharded collections
# and checks the scores from the input distribution to the tables.
set -e

DIR=./test/unit_tests/per_key_score

pytest -svv $DIR/test_scored_kjt.py \
             $DIR/test_scored_bucketize.py \
             $DIR/test_per_key_score_options.py

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  $DIR/test_per_key_score_dist.py
