#!/bin/bash 
set -e

# Test configurations
NUM_EMBEDDING_COLLECTIONS=2
NUM_EMBEDDINGS=10000,10000,10000,10000
MULTI_HOT_SIZES=5,5,5,5
EMBEDDING_DIM=16
NUM_GPUS=(1)
OPTIMIZER_TYPE=("sgd" "adam" "adagrad" "rowwise_adagrad")
BATCH_SIZE=32
NUM_ITERATIONS=10
THRESHOLD=5

# Cache configurations
CACHING_MODES=("False" "True")
CACHE_CAPACITY_RATIO=0.3  # 30% cache capacity to trigger evictions


for num_gpus in ${NUM_GPUS[@]}; do
  for optimizer_type in ${OPTIMIZER_TYPE[@]}; do
    echo ""
    echo "----------------------------------------"
    echo "Test: Storage-Only | GPUs: $num_gpus | Optimizer: $optimizer_type"
    echo "----------------------------------------"
    torchrun \
      --nnodes 1 \
      --nproc_per_node $num_gpus \
      ./test/unit_tests/test_embedding_admission.py \
      --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
      --num-embeddings $NUM_EMBEDDINGS \
      --multi-hot-sizes $MULTI_HOT_SIZES \
      --embedding-dim $EMBEDDING_DIM \
      --optimizer-type ${optimizer_type} \
      --batch-size $BATCH_SIZE \
      --num-iterations $NUM_ITERATIONS \
      --threshold $THRESHOLD || exit 1
  done
done