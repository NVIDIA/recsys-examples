#!/bin/bash
gpu_arch=$(nvidia-smi -L |head -n 1| cut -d' ' -f4)
num_layers=${1:-1}
PROFILE=${PROFILE:-0}
ASYNC_WGRAD=${ASYNC_WGRAD:-True}
nsys_profile_args='-f true -s none -t cuda,nvtx -c cudaProfilerApi --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node'

# dim_per_head=(64 128 256)
# num_heads=(1 4 8) 
# max_seqlen=(512 1024 2048 4096)
# batchsize=(32 64 128)
dim_per_heads=(128 )
num_heads=(8 )
max_seqlens=(1024 )
batchsizes=(32 )

profiler_start=20
profiler_end=40

for dim_per_head in ${dim_per_heads[@]}; do
    for num_head in ${num_heads[@]}; do
        for max_seqlen in ${max_seqlens[@]}; do
            for batchsize in ${batchsizes[@]}; do

                echo "==== dim_per_head: $dim_per_head, num_heads: $num_head, max_seqlen: $max_seqlen, batchsize: $batchsize ==== "
                native_output_profile_name=${gpu_arch}_native_bs${batchsize}_dim${dim_per_head}_heads${num_head}_seqlen${max_seqlen}_async${ASYNC_WGRAD}
                fused_output_profile_name=${gpu_arch}_fused_bs${batchsize}_dim${dim_per_head}_heads${num_head}_seqlen${max_seqlen}_async${ASYNC_WGRAD}

                if [ $PROFILE -eq 1 ]; then
                    fused_nsys_cmd="nsys profile -o ./profile/fused_op/${fused_output_profile_name} ${nsys_profile_args}"
                    native_nsys_cmd="nsys profile -o ./profile/fused_op/${native_output_profile_name} ${nsys_profile_args}"
                else
                    fused_nsys_cmd=""
                    native_nsys_cmd=""
                fi

                ${fused_nsys_cmd} \
                    python ./benchmark/fused_hstu_layer_benchmark.py run \
                    --iters 100 \
                    --warmup-iters 50 \
                    --layer-type fused \
                    --kernel-backend cutlass \
                    --dim-per-head $dim_per_head \
                    --num-heads $num_head \
                    --num-layers $num_layers \
                    --dtype bfloat16 \
                    --max-seqlen $max_seqlen \
                    --batchsize $batchsize \
                    --async-wgrad $ASYNC_WGRAD \
                    --profiler-start $profiler_start \
                    --profiler-end $profiler_end | tee ./profile/fused_op/${gpu_arch}_${fused_output_profile_name}.log

                sleep 1
                ${native_nsys_cmd} \
                    python ./benchmark/fused_hstu_layer_benchmark.py run \
                    --iters 100 \
                    --warmup-iters 50 \
                    --layer-type native \
                    --kernel-backend cutlass \
                    --dim-per-head $dim_per_head \
                    --num-heads $num_head \
                    --num-layers $num_layers \
                    --dtype bfloat16 \
                    --max-seqlen $max_seqlen \
                    --batchsize $batchsize \
                    --async-wgrad $ASYNC_WGRAD \
                    --profiler-start $profiler_start \
                    --profiler-end $profiler_end | tee ./profile/fused_op/${gpu_arch}_${native_output_profile_name}.log
                  sleep 1
            done
        done
    done
done
