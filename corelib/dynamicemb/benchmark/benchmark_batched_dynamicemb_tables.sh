#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

declare -A hbm=(["sgd"]=4 ["adam"]=12)
use_index_dedups=("False")
batch_sizes=(65536 1048576)
capacities=("8" "64")
optimizer_types=("sgd" "adam")
embedding_dims=(128)
alphas=(1.05)
gpu_ratio=0.125

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
                --table_version 2

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
                --table_version 2
                

          # ncu -f --target-processes all --export de_and_tr-$batch_size-$capacity-$optimizer_type-rep.report --section SchedulerStats --section WarpStateStats --import-source=yes --page raw --set full --profile-from-start no -k regex:"load_or_initialize_" \
          # nsys profile  -s none -t cuda,nvtx,osrt,mpi,ucx -f true -o de_and_tr-$batch_size-$capacity-$optimizer_type.qdrep -c cudaProfilerApi  --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node \
        done
      done
    done
  done
done