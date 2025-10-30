#!/bin/bash
gpu_arch=$(nvidia-smi -L |head -n 1| cut -d' ' -f4)
num_layers=${1:-1}
PROFILE=${PROFILE:-0}
nsys_cmd='nsys profile -o ./profile/<placeholder> -f true -s none -t cuda,nvtx -c cudaProfilerApi --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node '

dim_per_heads=(256)
num_heads=(4)
max_seqlens=(1024 2048 4096 8192)
batchsizes=(32)
embedding_dims=(1024)
full_sequence=True

profiler_start=20
profiler_end=40

mkdir -p ./profile/
for dim_per_head in "${dim_per_heads[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for max_seqlen in "${max_seqlens[@]}"; do
            for batchsize in "${batchsizes[@]}"; do
                echo "==== dim_per_head: $dim_per_head, num_heads: $num_head, max_seqlen: $max_seqlen, batchsize: $batchsize, full_sequence: $full_sequence, num_layers: $num_layers ==== "
                baseline_profile_name="${gpu_arch}_baseline_bs${batchsize}_dim${dim_per_head}_heads${num_head}_seqlen${max_seqlen}"
                cutlass_profile_name="${gpu_arch}_cutlass_bs${batchsize}_dim${dim_per_head}_heads${num_head}_seqlen${max_seqlen}"
                fused_profile_name="${gpu_arch}_fused_bs${batchsize}_dim${dim_per_head}_heads${num_head}_seqlen${max_seqlen}"
                recompute_profile_name="${gpu_arch}_recompute_bs${batchsize}_dim${dim_per_head}_heads${num_head}_seqlen${max_seqlen}"

                if [ "$PROFILE" -eq 1 ]; then
                    nsys_profile_cmd=${nsys_cmd}
                else
                    nsys_profile_cmd=""
                fi
                echo -e "\n\033[32mbaseline hstu layer \033[0m:"
                ${nsys_profile_cmd/<placeholder>/${baseline_profile_name}} \
                    python ./training/benchmark/hstu_layer_benchmark.py run \
                    --iters 100 \
                    --warmup-iters 50 \
                    --kernel-backend triton \
                    --full-sequence "$full_sequence" \
                    --dim-per-head "$dim_per_head" \
                    --num-heads "$num_head" \
                    --num-layers "$num_layers" \
                    --dtype bfloat16 \
                    --max-seqlen "$max_seqlen" \
                    --batchsize "$batchsize" \
                    --async-wgrad False \
                    --profiler-start "$profiler_start" \
                    --profiler-end "$profiler_end" \
                    --fuse-norm-mul-dropout False \
                    --recompute-input-silu False \
                    --recompute-input-layernorm False | tee "./profile/${gpu_arch}_${baseline_profile_name}.log"


                echo -e "\n\033[32m +cutlass\033[0m:"
                ${nsys_profile_cmd/<placeholder>/${cutlass_profile_name}} \
                    python ./training/benchmark/hstu_layer_benchmark.py run \
                    --iters 100 \
                    --warmup-iters 50 \
                    --kernel-backend cutlass \
                    --full-sequence "$full_sequence" \
                    --dim-per-head "$dim_per_head" \
                    --num-heads "$num_head" \
                    --num-layers "$num_layers" \
                    --dtype bfloat16 \
                    --max-seqlen "$max_seqlen" \
                    --batchsize "$batchsize" \
                    --async-wgrad False \
                    --profiler-start "$profiler_start" \
                    --profiler-end "$profiler_end" \
                    --fuse-norm-mul-dropout False \
                    --recompute-input-silu False \
                    --recompute-input-layernorm False | tee "./profile/${gpu_arch}_${cutlass_profile_name}.log"

                echo -e "\n\033[32m +fused\033[0m:"
                ${nsys_profile_cmd/<placeholder>/${fused_profile_name}} \
                    python ./training/benchmark/hstu_layer_benchmark.py run \
                    --iters 100 \
                    --warmup-iters 50 \
                    --kernel-backend cutlass \
                    --full-sequence "$full_sequence" \
                    --dim-per-head "$dim_per_head" \
                    --num-heads "$num_head" \
                    --num-layers "$num_layers" \
                    --dtype bfloat16 \
                    --max-seqlen "$max_seqlen" \
                    --batchsize "$batchsize" \
                    --async-wgrad False \
                    --profiler-start "$profiler_start" \
                    --profiler-end "$profiler_end" \
                    --fuse-norm-mul-dropout True \
                    --recompute-input-silu False \
                    --recompute-input-layernorm False | tee "./profile/${gpu_arch}_${fused_profile_name}.log"

                echo -e "\n\033[32m + recompute\033[0m:"
                ${nsys_profile_cmd/<placeholder>/${recompute_profile_name}} \
                    python ./training/benchmark/hstu_layer_benchmark.py run \
                    --iters 100 \
                    --warmup-iters 50 \
                    --kernel-backend cutlass \
                    --full-sequence "$full_sequence" \
                    --dim-per-head "$dim_per_head" \
                    --num-heads "$num_head" \
                    --num-layers "$num_layers" \
                    --dtype bfloat16 \
                    --max-seqlen "$max_seqlen" \
                    --batchsize "$batchsize" \
                    --async-wgrad False \
                    --profiler-start "$profiler_start" \
                    --profiler-end "$profiler_end" \
                    --fuse-norm-mul-dropout True \
                    --recompute-input-silu True \
                    --recompute-input-layernorm True | tee "./profile/${gpu_arch}_${recompute_profile_name}.log"
            done
        done
    done
done
