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


import os
import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
from typing import Union

import click
import commons.utils.initialize as init
import nvtx
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.utils.gpu_timer import IGPUTimer
from commons.utils.perf import cal_hstu_flops_single_rank, get_current_device_spec
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
    default="fused",
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
@click.option("--profile", type=bool, default=False, required=False)
@click.option(
    "--output-dir",
    type=str,
    default=".",
    required=False,
    help="Directory to write the memory snapshot pickle. Defaults to cwd.",
)
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
    profile,
    output_dir,
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

    def _reset_grads():
        # prevent gradient accumulation across iterations from inflating bwd timings
        input.grad = None
        for block in hstu_blocks:
            block.zero_grad(set_to_none=True)

    def _fwd():
        ret_jd = hstu_blocks[0](jagged_input)
        for hstu_layer in hstu_blocks[1:]:
            ret_jd = hstu_layer(ret_jd)
        return ret_jd

    # warmup
    if dump_memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=10000)
    for _ in range(warmup_iters):
        _reset_grads()
        ret_jd = _fwd()
        ret_jd.values.backward(grad_output)
    _reset_grads()
    if dump_memory_snapshot:
        os.makedirs(output_dir, exist_ok=True)
        snapshot_filename = (
            f"{log_layer_type}x{num_layers}_bs{batchsize}_max_seqlen{max_seqlen}"
            f"_dim{dim_per_head}_heads{num_heads}"
            f"_memory_recomputeln{recompute_input_layernorm}"
            f"_recomputesilu{recompute_input_silu}_snapshot.pickle"
        )
        torch.cuda.memory._dump_snapshot(os.path.join(output_dir, snapshot_filename))
        torch.cuda.memory._record_memory_history(enabled=None)

    flops_kwargs = dict(
        num_layers=hstu_config.num_layers,
        hidden_size=hstu_config.hidden_size,
        num_heads=hstu_config.num_attention_heads,
        dim_per_head=hstu_config.kv_channels,
        seqlens=lengths,
        num_contextuals=None,
        num_candidates=None,
        is_causal=hstu_config.is_causal,
        residual=hstu_config.residual,
    )
    fwd_flops = cal_hstu_flops_single_rank(has_bwd=False, **flops_kwargs)
    total_flops = cal_hstu_flops_single_rank(has_bwd=True, **flops_kwargs)
    bwd_flops = total_flops - fwd_flops

    # Resolve peak TFLOPS for the current device + dtype so we can emit
    # MFU alongside TFLOPS. Mirrors the kernel benchmark's approach.
    device_spec = get_current_device_spec()
    dtype_key = "bf16" if dtype == torch.bfloat16 else "fp16"
    peak_tflops = device_spec.peak_tflops.get(
        dtype_key, device_spec.peak_tflops.get("fp16", 312.0)
    )

    def _mfu_pct(flops, time_ms):
        """flops: total FLOPs across all tokens; time_ms: step time in ms."""
        tflops = flops / time_ms * 1e-9
        return tflops, tflops / peak_tflops * 100.0

    # benchmark
    igpu_timer = IGPUTimer(max_iters=iters)
    # [train_fwd] — forward with autograd capture (matches the fwd portion of a training step)
    for iteration in range(iters):
        _reset_grads()
        igpu_timer.start(iteration)
        ret_jd = _fwd()
        igpu_timer.stop(iteration)

    fwd_median_time = igpu_timer.elapsed_time(reduction="median")
    fwd_tflops, fwd_mfu = _mfu_pct(fwd_flops, fwd_median_time)
    print(
        f"[{log_layer_type}] [train_fwd] tokens {L};time (median): {fwd_median_time:.4f} ms;"
        f"achieved flops: {fwd_tflops:.4f} TFLOPS;MFU: {fwd_mfu:.2f}%"
    )
    # bwd
    for iteration in range(iters):
        _reset_grads()
        ret_jd = _fwd()
        igpu_timer.start(iteration)
        ret_jd.values.backward(grad_output)
        igpu_timer.stop(iteration)

    bwd_median_time = igpu_timer.elapsed_time(reduction="median")
    bwd_tflops, bwd_mfu = _mfu_pct(bwd_flops, bwd_median_time)
    print(
        f"[{log_layer_type}] [bwd] tokens {L};time (median): {bwd_median_time:.4f} ms;"
        f"achieved flops: {bwd_tflops:.4f} TFLOPS;MFU: {bwd_mfu:.2f}%"
    )
    # [e2e] — real fwd+bwd step time (single timed region, not sum of medians)
    for iteration in range(iters):
        _reset_grads()
        igpu_timer.start(iteration)
        ret_jd = _fwd()
        ret_jd.values.backward(grad_output)
        igpu_timer.stop(iteration)

    e2e_median_time = igpu_timer.elapsed_time(reduction="median")
    e2e_tflops, e2e_mfu = _mfu_pct(total_flops, e2e_median_time)
    print(
        f"[{log_layer_type}] [e2e] tokens {L};time (median): {e2e_median_time:.4f} ms;"
        f"achieved flops: {e2e_tflops:.4f} TFLOPS;MFU: {e2e_mfu:.2f}%"
    )
    print(
        f"[{log_layer_type}] [peak] {dtype_key} peak TFLOPS: {peak_tflops:.1f} on {device_spec.device_name}"
    )
    # nsys — only when --profile True
    if profile:
        for iteration in range(iters):
            if iteration == profiler_start or iteration == iters - 1:
                torch.cuda.profiler.start()

            _reset_grads()
            # Outer `hstu_layer_step <i>` wraps both fwd and bwd so nsys
            # post-analysis can pin the full step window (including any GPU
            # idle gap between fwd and bwd enqueues — autograd graph setup).
            with nvtx.annotate(f"hstu_layer_step {iteration}", color="CYAN"):
                with nvtx.annotate(f"hstu_layer_fwd {iteration}", color="ORANGE"):
                    ret_jd = _fwd()

                with nvtx.annotate(f"hstu_layer_bwd {iteration}", color="PURPLE"):
                    ret_jd.values.backward(grad_output)

            if iteration == profiler_end or iteration == iters - 1:
                torch.cuda.profiler.stop()


if __name__ == "__main__":
    init.initialize_single_rank()
    init.set_random_seed(1234)
    cli()
