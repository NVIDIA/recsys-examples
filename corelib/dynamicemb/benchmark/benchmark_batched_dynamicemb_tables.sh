#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# declare -A hbm=(["sgd"]=16 ["adam"]=48)
declare -A hbm=(["sgd"]=4 ["adam"]=12)

use_index_dedups=("True")
batch_sizes=(1048576) #(65536 131072 262144) # batch_size = 32, sequence_len=(2k, 4k, 8k)
capacities=( "64") # 64 * 1024 * 1024
optimizer_types=("sgd" "adam")
embedding_dims=(128)
gpu_ratios=0.125
# cache_algorithms=("lru" "lfu")

rm benchmark_results.json
for use_index_dedup in "${use_index_dedups[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for capacity in "${capacities[@]}"; do
      for optimizer_type in "${optimizer_types[@]}"; do
        for embedding_dim in "${embedding_dims[@]}"; do
          # for cache_algorithm in "${cache_algorithms[@]}"; do
  
            # echo "####" $use_index_dedup $batch_size $capacity ${hbm[$optimizer_type]} $optimizer_type

            # ncu -f --target-processes all --export dynamicemb-rep.report --section SchedulerStats --section WarpStateStats --import-source=yes --page raw --set full --profile-from-start no -k regex:"gpu_select_kvm_kernel" \
            nsys profile  -s none -t cuda,nvtx,osrt,mpi,ucx -f true -o de_and_tr$batch_size-$optimizer_type-cache.qdrep -c cudaProfilerApi  --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node \
            torchrun --nnodes 1 --nproc_per_node 1 \
              ./benchmark/benchmark_batched_dynamicemb_tables.py  \
                --use_index_dedup \
                --caching \
                --batch_size $batch_size \
                --num_embeddings_per_feature $capacity \
                --hbm_for_embeddings ${hbm[$optimizer_type]} \
                --optimizer_type $optimizer_type \
                --feature_distribution "pow-law" \
                --embedding_dim $embedding_dim 
                # --cache_algorithm $cache_algorithm
          # done
        done
      done
    done
  done
done

for use_index_dedup in "${use_index_dedups[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for capacity in "${capacities[@]}"; do
      for optimizer_type in "${optimizer_types[@]}"; do
        for embedding_dim in "${embedding_dims[@]}"; do
          # for cache_algorithm in "${cache_algorithms[@]}"; do
  
            # echo "####" $use_index_dedup $batch_size $capacity ${hbm[$optimizer_type]} $optimizer_type

            # ncu -f --target-processes all --export dynamicemb-rep.report --section SchedulerStats --section WarpStateStats --import-source=yes --page raw --set full --profile-from-start no -k regex:"gpu_select_kvm_kernel" \
            nsys profile  -s none -t cuda,nvtx,osrt,mpi,ucx -f true -o de_and_tr$batch_size-$optimizer_type-nc.qdrep -c cudaProfilerApi  --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node \
            torchrun --nnodes 1 --nproc_per_node 1 \
              ./benchmark/benchmark_batched_dynamicemb_tables.py  \
                --use_index_dedup \
                --batch_size $batch_size \
                --num_embeddings_per_feature $capacity \
                --hbm_for_embeddings ${hbm[$optimizer_type]} \
                --optimizer_type $optimizer_type \
                --feature_distribution "pow-law" \
                --embedding_dim $embedding_dim 
                # --cache_algorithm $cache_algorithm
          # done
        done
      done
    done
  done
done

for use_index_dedup in "${use_index_dedups[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for capacity in "${capacities[@]}"; do
      for optimizer_type in "${optimizer_types[@]}"; do
        for embedding_dim in "${embedding_dims[@]}"; do
          # for cache_algorithm in "${cache_algorithms[@]}"; do
  
            # echo "####" $use_index_dedup $batch_size $capacity ${hbm[$optimizer_type]} $optimizer_type

            # ncu -f --target-processes all --export dynamicemb-rep.report --section SchedulerStats --section WarpStateStats --import-source=yes --page raw --set full --profile-from-start no -k regex:"gpu_select_kvm_kernel" \
            nsys profile  -s none -t cuda,nvtx,osrt,mpi,ucx -f true -o de_and_tr$batch_size-$optimizer_type-nc-no-dedup.qdrep.qdrep -c cudaProfilerApi  --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node \
            torchrun --nnodes 1 --nproc_per_node 1 \
              ./benchmark/benchmark_batched_dynamicemb_tables.py  \
                --batch_size $batch_size \
                --num_embeddings_per_feature $capacity \
                --hbm_for_embeddings ${hbm[$optimizer_type]} \
                --optimizer_type $optimizer_type \
                --feature_distribution "pow-law" \
                --embedding_dim $embedding_dim 
                # --cache_algorithm $cache_algorithm
          # done
        done
      done
    done
  done
done
