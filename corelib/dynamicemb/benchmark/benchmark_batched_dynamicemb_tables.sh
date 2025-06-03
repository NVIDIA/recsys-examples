#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

declare -A hbm=(["sgd"]=4 ["adam"]=12)

use_index_dedups=("True")
batch_sizes=(65536 1048576)
capacities=("8" "64")
optimizer_types=("sgd" "adam")

rm benchmark_results.json
for use_index_dedup in "${use_index_dedups[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for capacity in "${capacities[@]}"; do
      for optimizer_type in "${optimizer_types[@]}"; do
  
        echo "####" $use_index_dedup $batch_size $capacity ${hbm[$optimizer_type]} $optimizer_type

        # ncu -f --target-processes all --export dynamicemb-rep.report --section SchedulerStats --section WarpStateStats --import-source=yes --page raw --set full --profile-from-start no -k regex:"fill_output_with_table_vectors_kernel|initialize_optimizer_state_kernel" \
        # nsys profile  -s none -t cuda,nvtx,osrt,mpi,ucx -f true -o dynamicemb$batch_size.qdrep -c cudaProfilerApi  --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node \
        torchrun --nnodes 1 --nproc_per_node 1 \
          ./benchmark/benchmark_batched_dynamicemb_tables.py  \
            --use_index_dedup $use_index_dedup \
            --batch_size $batch_size \
            --num_embeddings_per_feature $capacity \
            --hbm_for_embeddings ${hbm[$optimizer_type]} \
            --optimizer_type $optimizer_type
      done
    done
  done
done