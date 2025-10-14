#!/bin/bash 
set -e

NUM_EMBEDDING_COLLECTIONS=4
NUM_EMBEDDINGS=1000000,1000000,1000000,1000000,1000000,1000000
MULTI_HOT_SIZES=10,10,10,10,10,10
NUM_GPUS=(1 4)
OPTIMIZER_TYPE=("adam" "sgd" "adagrad" "rowwise_adagrad")
INCLUDE_OPTIM=("True" "False")
SCORE_STRATEGY=("timestamp" "lfu" "step")

for num_gpus in ${NUM_GPUS[@]}; do
  for optimizer_type in ${OPTIMIZER_TYPE[@]}; do  
    for include_optim in ${INCLUDE_OPTIM[@]}; do
      for score_strategy in ${SCORE_STRATEGY[@]}; do
        echo "num_gpus: $num_gpus, optimizer_type: $optimizer_type, include_optim: $include_optim, score_strategy: $score_strategy"
        torchrun \
          --nnodes 1 \
          --nproc_per_node $num_gpus \
          ./test/unit_tests/test_embedding_dump_load.py \
          --optimizer-type ${optimizer_type} \
          --score-strategy ${score_strategy} \
          --mode "dump" \
          --optim ${include_optim} \
          --save-path "debug_weight_${optimizer_type}_${num_gpus}_${include_optim}_${score_strategy}" \
          --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
          --num-embeddings $NUM_EMBEDDINGS \
          --multi-hot-sizes $MULTI_HOT_SIZES \
          --embedding-dim 16 || exit 1
      done
    done
  done
done

for num_load_gpus in ${NUM_GPUS[@]}; do
  for num_dump_gpus in ${NUM_GPUS[@]}; do
    for optimizer_type in ${OPTIMIZER_TYPE[@]}; do  
      for include_optim in ${INCLUDE_OPTIM[@]}; do
        for score_strategy in ${SCORE_STRATEGY[@]}; do
          echo "num_load_gpus: $num_load_gpus, num_dump_gpus: $num_dump_gpus, optimizer_type: $optimizer_type, include_optim: $include_optim, score_strategy: $score_strategy"
          torchrun \
            --nnodes 1 \
            --nproc_per_node $num_load_gpus \
            ./test/unit_tests/test_embedding_dump_load.py \
            --optimizer-type ${optimizer_type} \
            --score-strategy ${score_strategy} \
            --mode "load" \
            --optim ${include_optim} \
            --save-path "debug_weight_${optimizer_type}_${num_dump_gpus}_${include_optim}_${score_strategy}" \
            --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
            --num-embeddings $NUM_EMBEDDINGS \
            --multi-hot-sizes $MULTI_HOT_SIZES \
            --embedding-dim 16 || exit 1
        done
      done
    done
  done
done