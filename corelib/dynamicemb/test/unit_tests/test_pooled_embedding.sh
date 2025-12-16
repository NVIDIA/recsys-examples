#!/bin/bash 

set -e

export PYHKV_DEBUG=1
export PYHKV_DEBUG_ITER=10

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 129

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --output_dtype=bf16

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --dynamicemb_num=4

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --multi_hot_sizes=20,1,101,49


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --batch_size=1024 --num_embeddings_per_feature=8388608,4194304,524288,1048576

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  ./test/unit_tests/test_pooled_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --batch_size=1024 --num_embeddings_per_feature=8388608,4194304,524288,1048576


# Test sequence embedding's backward on a single GPU with different ["use_index_dedup", "dim", "batch_size", "multi_hot_sizes"]
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_pooled_embedding_bw.py --batch_size 1024

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_pooled_embedding_bw.py --embedding_dim 129 --batch_size 1024

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_pooled_embedding_bw.py --embedding_dim 15 --batch_size 1023 --multi_hot_sizes=20,1,101,49

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_pooled_embedding_bw.py --embedding_dim 15 --batch_size 1023  --multi_hot_sizes=20,49,101,1
