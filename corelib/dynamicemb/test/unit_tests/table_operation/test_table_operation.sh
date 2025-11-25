#!/bin/bash 
set -e

pytest test/unit_tests/table_operation/test_table_operation.py -s

torchrun --nproc_per_node=1 -m pytest test/unit_tests/table_operation/test_dump_load_same_rank.py -s
torchrun --nproc_per_node=4 -m pytest test/unit_tests/table_operation/test_dump_load_same_rank.py -s

NUM_GPUS=(1 4)
SCORE_POLICY=("assign" "accumulate" "global_timer")
CAPACITY=1000000

for num_gpus in ${NUM_GPUS[@]}; do
  for score_policy in ${SCORE_POLICY[@]}; do
    echo "num_gpus: $num_gpus, score_policy: $score_policy"
    torchrun \
      --nnodes 1 \
      --nproc_per_node $num_gpus \
      ./test/unit_tests/table_operation/test_dump_load_different_rank.py \
      --score_policy ${score_policy} \
      --mode "dump" \
      --save-path "debug_table_${num_gpus}_${score_policy}" \
      --capacity $CAPACITY || exit 1
  done
done

for num_load_gpus in ${NUM_GPUS[@]}; do
  for num_dump_gpus in ${NUM_GPUS[@]}; do
    for score_policy in ${SCORE_POLICY[@]}; do
      echo "num_load_gpus: $num_load_gpus, num_dump_gpus: $num_dump_gpus, score_policy: $score_policy"
      torchrun \
        --nnodes 1 \
        --nproc_per_node $num_load_gpus \
        ./test/unit_tests/table_operation/test_dump_load_different_rank.py \
        --score_policy ${score_policy} \
        --mode "load" \
        --save-path "debug_table_${num_dump_gpus}_${score_policy}" \
        --capacity $CAPACITY || exit 1
    done
  done
done