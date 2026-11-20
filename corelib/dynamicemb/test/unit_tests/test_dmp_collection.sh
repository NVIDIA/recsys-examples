#!/bin/bash
set -e

# Two ranks with a sharding group of one: two replicas of the same table, the
# smallest DMPCollection where a rank's group rank differs from its global rank.
torchrun --nnodes 1 --nproc_per_node 2 ./test/unit_tests/test_dmp_collection.py
