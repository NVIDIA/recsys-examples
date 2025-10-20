#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

optimizer_types=("adam" "sgd")
batch_sizes=(1048576 65536)
embedding_dims=(128)
alphas=(1.05)
use_index_dedups=("False")

capacities=("256")
gpu_ratio=0.1
declare -A hbm=(["sgd"]=12.8 ["adam"]=38.4)

rm benchmark_results.json
for batch_size in "${batch_sizes[@]}"; do
  echo "batch_size: $batch_size"
  for capacity in "${capacities[@]}"; do
    echo "capacity: $capacity"
    for optimizer_type in "${optimizer_types[@]}"; do
      echo "optimizer_type: $optimizer_type"
      for embedding_dim in "${embedding_dims[@]}"; do
        echo "embedding_dim: $embedding_dim"
        for alpha in "${alphas[@]}"; do
          echo "alpha: $alpha"

          torchrun --nnodes 1 --nproc_per_node 1 \
              ./benchmark/benchmark_batched_dynamicemb_tables.py  \
                --caching \
                --cache_algorithm "lru" \
                --gpu_ratio $gpu_ratio \
                --batch_size $batch_size \
                --num_embeddings_per_feature $capacity \
                --embedding_dim $embedding_dim \
                --hbm_for_embeddings ${hbm[$optimizer_type]} \
                --optimizer_type $optimizer_type \
                --feature_distribution "pow-law" \
                --alpha $alpha \
                --num_iterations 100 \

          torchrun --nnodes 1 --nproc_per_node 1 \
              ./benchmark/benchmark_batched_dynamicemb_tables.py  \
                --batch_size $batch_size \
                --num_embeddings_per_feature $capacity \
                --embedding_dim $embedding_dim \
                --hbm_for_embeddings ${hbm[$optimizer_type]} \
                --optimizer_type $optimizer_type \
                --feature_distribution "pow-law" \
                --alpha $alpha \
                --num_iterations 100 \
                --cache_algorithm "lru" \

        done
      done
    done
  done
done

capacities=("24")
gpu_ratio=1.0
declare -A hbm=(["sgd"]=12 ["adam"]=36)

for batch_size in "${batch_sizes[@]}"; do
  echo "batch_size: $batch_size"
  for capacity in "${capacities[@]}"; do
    echo "capacity: $capacity"
    for optimizer_type in "${optimizer_types[@]}"; do
      echo "optimizer_type: $optimizer_type"
      for embedding_dim in "${embedding_dims[@]}"; do
        echo "embedding_dim: $embedding_dim"
        for alpha in "${alphas[@]}"; do
          echo "alpha: $alpha"

          torchrun --nnodes 1 --nproc_per_node 1 \
              ./benchmark/benchmark_batched_dynamicemb_tables.py  \
                --batch_size $batch_size \
                --num_embeddings_per_feature $capacity \
                --embedding_dim $embedding_dim \
                --hbm_for_embeddings ${hbm[$optimizer_type]} \
                --optimizer_type $optimizer_type \
                --feature_distribution "pow-law" \
                --alpha $alpha \
                --num_iterations 100 \
                --cache_algorithm "lru" \
                --gpu_ratio $gpu_ratio
        done
      done
    done
  done
done