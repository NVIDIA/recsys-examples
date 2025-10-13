#!/bin/bash 
set -e

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  -m pytest -s \
  test/unit_tests/test_dynamicemb_table_dump_load.py 
