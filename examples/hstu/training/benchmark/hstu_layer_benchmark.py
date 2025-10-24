# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
#! example:
# python ./benchmark/fused_hstu_layer_benchmark.py run \
# --iters 100 --warmup-iters 50 --layer-type fused \
# --kernel-backend cutlass --full-sequence True \
# --dim-per-head 128 --num-heads 4 --num-layers 3 \
# --dtype bfloat16 --max-seqlen 4096 --batchsize 32 \
# --async-wgrad False \
# --recompute-input-silu  True \
# --recompute-input-layernorm True


import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
from typing import Union

import click
import commons.utils.initialize as init
import nvtx
from commons.utils.gpu_timer import IGPUTimer
from configs.hstu_config import (
    HSTUConfig,
    HSTULayerType,
    KernelBackend,
    get_hstu_config,
)
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.fused_hstu_layer import FusedHSTULayer
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer as NativeHSTULayer
from ops.length_to_offsets import length_to_complete_offsets
from training.utils import cal_flops_single_rank

_backend_str_to_type = {
    "cutlass": KernelBackend.CUTLASS,
    "triton": KernelBackend.TRITON,
    "pytorch": KernelBackend.PYTORCH,
}

_layer_type_str_to_type = {
    "native": HSTULayerType.NATIVE,
    "fused": HSTULayerType.FUSED,
    "debug": HSTULayerType.DEBUG,
}

_dtype_str_to_type = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@click.group()
def cli() -> None:
    pass


def create_hstu_layer(
    hstu_config: HSTUConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> Union[DebugHSTULayer, NativeHSTULayer, FusedHSTULayer]:
    if hstu_config.hstu_layer_type == HSTULayerType.DEBUG:
        module = DebugHSTULayer(hstu_config).to(dtype).cuda()
    elif hstu_config.hstu_layer_type == HSTULayerType.NATIVE:
        module = NativeHSTULayer(hstu_config).to(dtype).cuda()
    else:
        module = FusedHSTULayer(hstu_config).to(dtype).cuda()

    return module


@cli.command()
@click.option("--iters", type=int, default=100, required=False)
@click.option("--warmup-iters", type=int, default=50, required=False)
@click.option(
    "--layer-type",
    type=click.Choice(_layer_type_str_to_type.keys()),
    default="native",
    required=False,
)
@click.option(
    "--async-wgrad",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--fuse-norm-mul-dropout",
    type=bool,
    default=False,
    required=False,
)
@click.option(
    "--recompute-input-layernorm",
    type=bool,
    default=False,
    required=False,
)
@click.option(
    "--recompute-input-silu",
    type=bool,
    default=False,
    required=False,
)
@click.option(
    "--kernel-backend",
    type=click.Choice(_backend_str_to_type.keys()),
    default="cutlass",
    required=False,
)
@click.option("--embedding-dim", type=int, default=0, required=True)
@click.option("--dim-per-head", type=int, default=128, required=True)
@click.option("--num-heads", type=int, default=8, required=True)
@click.option(
    "--dtype",
    type=click.Choice(_dtype_str_to_type.keys()),
    default="bfloat16",
    required=False,
)
@click.option("--max-seqlen", type=int, default=1024, required=True)
@click.option("--full-sequence", type=bool, default=False, required=True)
@click.option("--batchsize", type=int, default=32, required=True)
@click.option("--profiler-start", type=int, default=20, required=False)
@click.option("--profiler-end", type=int, default=40, required=False)
@click.option("--dump-memory-snapshot", type=bool, default=True, required=False)
@click.option("--num-layers", type=int, default=1, required=False)
def run(
    iters,
    warmup_iters,
    layer_type,
    embedding_dim,
    dim_per_head,
    num_heads,
    dtype,
    kernel_backend,
    max_seqlen,
    batchsize,
    profiler_start,
    profiler_end,
    full_sequence,
    async_wgrad,
    dump_memory_snapshot,
    num_layers,
    recompute_input_layernorm,
    recompute_input_silu,
    fuse_norm_mul_dropout,
):
    log_layer_type = layer_type.upper()
    layer_type = _layer_type_str_to_type[layer_type]
    kernel_backend = _backend_str_to_type[kernel_backend]
    dtype = _dtype_str_to_type[dtype]

    hidden_size = embedding_dim if embedding_dim > 0 else dim_per_head * num_heads
    hstu_config = get_hstu_config(
        hidden_size=hidden_size,
        kv_channels=dim_per_head,
        num_attention_heads=num_heads,
        num_layers=num_layers,
        dtype=dtype,
        kernel_backend=kernel_backend,
        hstu_layer_type=layer_type,
        learnable_input_layernorm=True,
        async_wgrad=async_wgrad,
        recompute_input_layernorm=recompute_input_layernorm,
        recompute_input_silu=recompute_input_silu,
        fuse_norm_mul_dropout=fuse_norm_mul_dropout,
    )
    hstu_blocks = [
        create_hstu_layer(
            hstu_config=hstu_config,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]
    # generate random input
    if full_sequence:
        lengths = torch.full((batchsize,), max_seqlen, dtype=torch.int32, device="cuda")
    else:
        lengths = torch.randint(
            low=1,
            high=max_seqlen + 1,
            size=(batchsize,),
            dtype=torch.int32,
            device="cuda",
        )
    seq_offsets = length_to_complete_offsets(lengths)
    L = int(seq_offsets[-1].item())
    input = torch.randn(L, hidden_size, dtype=dtype, device="cuda")
    # invoke backward
    input.requires_grad_()
    ctor_nograd_dict = {
        "seqlen": lengths,
        "seqlen_offsets": seq_offsets,
        "max_seqlen": max_seqlen,
        "max_num_candidates": 0,
        "num_candidates": None,
        "num_candidates_offsets": None,
        "contextual_max_seqlen": 0,
        "contextual_seqlen": None,
        "contextual_seqlen_offsets": None,
    }
    jagged_input = JaggedData(values=input, **ctor_nograd_dict)
    grad_output = torch.randn_like(input)
    # warmup
    if dump_memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=10000)
    for _ in range(warmup_iters):
        ret_jd = hstu_blocks[0](jagged_input)
        for hstu_layer in hstu_blocks[1:]:
            ret_jd = hstu_layer(ret_jd)
        ret_jd.values.backward(grad_output)
    if dump_memory_snapshot:
        torch.cuda.memory._dump_snapshot(
            f"{log_layer_type}x{num_layers}_bs{batchsize}_max_seqlen{max_seqlen}_dim{dim_per_head}_heads{num_heads}_memory_recomputeln{recompute_input_layernorm}_recomputesilu{recompute_input_silu}_snapshot.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)

    # benchmark
    igpu_timer = IGPUTimer(max_iters=iters)
    # fwd
    for iteration in range(iters):
        igpu_timer.start(iteration)
        ret_jd = hstu_blocks[0](jagged_input)
        for hstu_layer in hstu_blocks[1:]:
            ret_jd = hstu_layer(ret_jd)
        # ret_jd.values.backward(grad_output)
        igpu_timer.stop(iteration)

    fwd_median_time = igpu_timer.elapsed_time(reduction="median")
    fwd_flops = cal_flops_single_rank(
        hstu_config, lengths, num_contextuals=None, num_candidates=None, has_bwd=False
    )
    print(
        f"[{log_layer_type}] [fwd] tokens {L};time (median): {fwd_median_time:.4f} ms;achieved flops: {fwd_flops / fwd_median_time * 1e-9:.4f} TFLOPS"
    )
    # bwd
    for iteration in range(iters):
        ret_jd = hstu_blocks[0](jagged_input)
        for hstu_layer in hstu_blocks[1:]:
            ret_jd = hstu_layer(ret_jd)
        igpu_timer.start(iteration)
        ret_jd.values.backward(grad_output)
        igpu_timer.stop(iteration)

    bwd_median_time = igpu_timer.elapsed_time(reduction="median")
    bwd_flops = (
        cal_flops_single_rank(
            hstu_config,
            lengths,
            num_contextuals=None,
            num_candidates=None,
            has_bwd=True,
        )
        - fwd_flops
    )
    print(
        f"[{log_layer_type}] [bwd] tokens {L};time (median): {bwd_median_time:.4f} ms;achieved flops: {bwd_flops / bwd_median_time * 1e-9:.4f} TFLOPS"
    )
    print(
        f"[{log_layer_type}] [e2e] tokens {L};time: {fwd_median_time + bwd_median_time:.4f} ms;achieved flops: {(fwd_flops + bwd_flops) / (fwd_median_time + bwd_median_time) * 1e-9:.4f} TFLOPS"
    )
    # nsys
    for iteration in range(iters):
        if iteration == profiler_start or iteration == iters - 1:
            torch.cuda.profiler.start()

        with nvtx.annotate(f"hstu_layer_fwd {iteration}", color="ORANGE"):
            ret_jd = hstu_blocks[0](jagged_input)
            for hstu_layer in hstu_blocks[1:]:
                ret_jd = hstu_layer(ret_jd)

        with nvtx.annotate(f"hstu_layer_bwd {iteration}", color="PURPLE"):
            ret_jd.values.backward(grad_output)

        if iteration == profiler_end or iteration == iters - 1:
            torch.cuda.profiler.stop()


if __name__ == "__main__":
    init.initialize_single_rank()
    init.set_random_seed(1234)
    cli()
