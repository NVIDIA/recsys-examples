#!/usr/bin/env bash
set -e

WORLD_SIZE=1 torchrun --nnodes 1 --nproc_per_node 1 -m pytest test/unit_tests/test_alignment.py -q
WORLD_SIZE=4 torchrun --nnodes 1 --nproc_per_node 4 -m pytest test/unit_tests/test_alignment.py -q
WORLD_SIZE=8 torchrun --nnodes 1 --nproc_per_node 8 -m pytest test/unit_tests/test_alignment.py -q