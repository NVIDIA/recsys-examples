#!/bin/bash 
set -e

NUM_EMBEDDING_COLLECTIONS=4
NUM_EMBEDDINGS=1000000,1000000,1000000,1000000,1000000,1000000
MULTI_HOT_SIZES=10,10,10,10,10,10
NUM_GPUS=(1 8)
OPTIMIZER_TYPE=("adam" "sgd" "rowwise_adagrad")
INCLUDE_OPTIM=("True" "False")
SCORE_STRATEGY=("timestamp" "lfu" "step")
INCLUDE_COUNTER=("True" "False")

for num_gpus in ${NUM_GPUS[@]}; do
  for optimizer_type in ${OPTIMIZER_TYPE[@]}; do  
    for include_optim in ${INCLUDE_OPTIM[@]}; do
      for include_counter in ${INCLUDE_COUNTER[@]}; do
        for score_strategy in ${SCORE_STRATEGY[@]}; do
          echo "num_gpus: $num_gpus, optimizer_type: $optimizer_type, include_optim: $include_optim, include_counter: $include_counter, score_strategy: $score_strategy"
          torchrun \
            --nnodes 1 \
            --nproc_per_node $num_gpus \
            ./test/unit_tests/test_embedding_dump_load.py \
            --optimizer-type ${optimizer_type} \
            --score-strategy ${score_strategy} \
            --mode "dump" \
            --optim ${include_optim} \
            --counter ${include_counter} \
            --save-path "debug_weight_${optimizer_type}_${num_gpus}_${include_optim}_${include_counter}_${score_strategy}" \
            --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
            --num-embeddings $NUM_EMBEDDINGS \
            --multi-hot-sizes $MULTI_HOT_SIZES \
            --embedding-dim 16 || exit 1
        done
      done
    done
  done
done

# hash_roundrobin, dumped and reloaded at the same world size. Multi-rank is
# the point: a rank's checkpoint file holds the keys with murmur(k) % ws == r,
# so a loader applying the roundrobin rule instead keeps only the ~1/ws of them
# that also satisfy k % ws == r and silently drops the rest. Single rank cannot
# see that -- there is no filtering to get wrong.
HASH_RR_NUM_GPUS=(1 2)

for num_gpus in ${HASH_RR_NUM_GPUS[@]}; do
  for mode in "dump" "load"; do
    echo "hash_roundrobin: num_gpus: $num_gpus, mode: $mode"
    torchrun \
      --nnodes 1 \
      --nproc_per_node $num_gpus \
      ./test/unit_tests/test_embedding_dump_load.py \
      --optimizer-type sgd \
      --score-strategy step \
      --dist-type hash_roundrobin \
      --mode "${mode}" \
      --optim False \
      --counter False \
      --save-path "debug_weight_hash_roundrobin_${num_gpus}" \
      --num-embedding-collections 1 \
      --num-embeddings 1000000 \
      --multi-hot-sizes 10 \
      --embedding-dim 16 || exit 1
  done
done

# Tables named "<scope>/<name>" dump into, and load from, a subdirectory of
# the collection directory that nothing creates in advance.
for mode in "dump" "load"; do
  echo "table scope: mode: $mode"
  torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    ./test/unit_tests/test_embedding_dump_load.py \
    --optimizer-type adam \
    --score-strategy timestamp \
    --mode "${mode}" \
    --optim True \
    --counter False \
    --table-scope scope \
    --save-path "debug_weight_table_scope" \
    --num-embedding-collections 2 \
    --num-embeddings 1000000,1000000 \
    --multi-hot-sizes 10,10 \
    --embedding-dim 16 || exit 1
done

for num_load_gpus in ${NUM_GPUS[@]}; do
  for num_dump_gpus in ${NUM_GPUS[@]}; do
    for optimizer_type in ${OPTIMIZER_TYPE[@]}; do  
      for include_optim in ${INCLUDE_OPTIM[@]}; do
        for include_counter in ${INCLUDE_COUNTER[@]}; do
          for score_strategy in ${SCORE_STRATEGY[@]}; do
            echo "num_load_gpus: $num_load_gpus, num_dump_gpus: $num_dump_gpus, optimizer_type: $optimizer_type, include_optim: $include_optim, include_counter: $include_counter, score_strategy: $score_strategy"
            torchrun \
              --nnodes 1 \
              --nproc_per_node $num_load_gpus \
              ./test/unit_tests/test_embedding_dump_load.py \
              --optimizer-type ${optimizer_type} \
              --score-strategy ${score_strategy} \
              --mode "load" \
              --optim ${include_optim} \
              --counter ${include_counter} \
              --save-path "debug_weight_${optimizer_type}_${num_dump_gpus}_${include_optim}_${include_counter}_${score_strategy}" \
              --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
              --num-embeddings $NUM_EMBEDDINGS \
              --multi-hot-sizes $MULTI_HOT_SIZES \
              --embedding-dim 16 || exit 1
          done
        done
      done
    done
  done
done

# int32 keys, negative ones included, dumped and reloaded at the same world
# size. A table stores an int32 key widened to int64 with its sign, the
# checkpoint holds that int64 key, and the loader keeps each key on the rank
# the int64 key belongs to. The bucketize has to send the int32 key to that
# same rank, or after the load a key sits on one rank and is looked up on
# another. Read in 32 bits, many negative keys would get another rank: under
# hash_roundrobin at every world size from 2, under roundrobin where 2**32
# and 2**64 leave different remainders, as at 7 ranks and not at 8.
INT32_KEY_CASES=("hash_roundrobin 8" "hash_roundrobin 7" "roundrobin 7")

for int32_key_case in "${INT32_KEY_CASES[@]}"; do
  read -r dist_type num_gpus <<< "${int32_key_case}"
  for mode in "dump" "load"; do
    echo "int32 keys: dist_type: $dist_type, num_gpus: $num_gpus, mode: $mode"
    torchrun \
      --nnodes 1 \
      --nproc_per_node $num_gpus \
      ./test/unit_tests/test_embedding_dump_load.py \
      --optimizer-type sgd \
      --score-strategy step \
      --dist-type ${dist_type} \
      --key-dtype int32 \
      --batch-size $((128 / num_gpus * num_gpus)) \
      --mode "${mode}" \
      --optim False \
      --counter False \
      --save-path "debug_weight_int32_${dist_type}_${num_gpus}" \
      --num-embedding-collections 1 \
      --num-embeddings 1000000 \
      --multi-hot-sizes 10 \
      --embedding-dim 16 || exit 1
  done
done
