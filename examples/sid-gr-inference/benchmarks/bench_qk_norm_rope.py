# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the Qwen3 q/k norm + RoPE step.

Compares two ``GR_INFERENCE_KERNEL_QK_NORM_ROPE`` choices on the same packed
qkv tensor:

- ``flashinfer``: SGLang's in-place qk-norm and RoPE kernels when SGLang is
  installed, otherwise the FlashInfer/PyTorch fallbacks
- ``trtllm_aligned``: the fused TRT-LLM-aligned kernel

Outputs are checked before timing. Each call is timed as the GPU time of the
kernels it launches, from the CUDA profiler, with the L2 cache cleared before
every call; the median over all calls is reported.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from gr_inference.gr_kernels.selection import reset_default_kernel_selection_policy
from gr_inference.gr_models.qwen3.config import Qwen3GRConfig
from gr_inference.gr_models.qwen3.layers import TorchQwen3LayerOps

ENV_KEY = "GR_INFERENCE_KERNEL_QK_NORM_ROPE"
VARIANTS = ("flashinfer", "trtllm_aligned")


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def set_variant(variant: str) -> None:
    os.environ[ENV_KEY] = variant
    reset_default_kernel_selection_policy()


def short_name(kernel: str) -> str:
    kernel = kernel.removeprefix("void ").replace("(anonymous namespace)::", "")
    return re.split(r"[<(]|_object_at_", kernel)[0]


def cuda_kernels(fn, calls: int = 1) -> list:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    return sorted(
        (
            event
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ),
        key=lambda event: event.time_range.start,
    )


def kernel_times_us(qk_norm_rope, calls: int) -> tuple[list[float], list[str]]:
    kernels = [event.name for event in cuda_kernels(qk_norm_rope)]
    flush = torch.empty(64 * 2**20, dtype=torch.float32, device="cuda")

    def call() -> None:
        flush.zero_()
        qk_norm_rope()

    for _ in range(5):
        call()
    launches = [e for e in cuda_kernels(call, calls) if e.name in kernels]
    if len(launches) != len(kernels) * calls:
        raise RuntimeError(
            f"expected kernels {kernels} in every call, "
            f"got {len(launches)} launches in {calls} calls"
        )
    per_call = len(kernels)
    times = [
        sum(event.device_time for event in launches[i * per_call : (i + 1) * per_call])
        for i in range(calls)
    ]
    return times, kernels


def run_case(args, ops, tokens: int) -> dict:
    hidden = torch.randn(
        1, tokens, ops.config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    result = {}
    outputs = {}
    for variant in VARIANTS:
        set_variant(variant)
        out_q, out_k = ops.qk_norm_rope(*ops.qkv(hidden)[:2])
        outputs[variant] = torch.cat([out_q.flatten(), out_k.flatten()]).float()

        q, k, _ = ops.qkv(hidden)
        times, kernels = kernel_times_us(lambda: ops.qk_norm_rope(q, k), args.calls)
        result[variant] = statistics.median(times)
        result[f"{variant}_kernels"] = kernels
    torch.testing.assert_close(
        outputs["trtllm_aligned"], outputs["flashinfer"], atol=0.0625, rtol=0.016
    )
    result["max_diff"] = (
        (outputs["trtllm_aligned"] - outputs["flashinfer"]).abs().max().item()
    )
    os.environ.pop(ENV_KEY)
    reset_default_kernel_selection_policy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=parse_int_list,
        default=[1, 16, 256, 1024, 4096, 16384, 65536],
        help="Comma or space separated token counts.",
    )
    parser.add_argument("--q-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--calls", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_qk_norm_rope.py requires a CUDA GPU")
    torch.manual_seed(0)
    config = Qwen3GRConfig(
        num_attention_heads=args.q_heads, num_kv_heads=args.kv_heads, head_dim=128
    )
    ops = TorchQwen3LayerOps(config, dtype=torch.bfloat16).cuda()
    ops.q_norm.weight.data.normal_(1, 0.1)
    ops.k_norm.weight.data.normal_(1, 0.1)

    print(
        f"gpu={torch.cuda.get_device_name()} q_heads={args.q_heads} "
        f"kv_heads={args.kv_heads} head_dim=128 dtype=bf16 calls={args.calls}"
    )
    print(
        f"{'tokens':>7} {'flashinfer_us':>13} {'trtllm_aligned_us':>17} "
        f"{'speedup':>8} {'max_abs_diff':>12}"
    )
    for tokens in args.tokens:
        result = run_case(args, ops, tokens)
        print(
            f"{tokens:>7} {result['flashinfer']:>13.2f} "
            f"{result['trtllm_aligned']:>17.2f} "
            f"{result['flashinfer'] / result['trtllm_aligned']:>7.2f}x "
            f"{result['max_diff']:>12.4f}",
            flush=True,
        )
    for variant in VARIANTS:
        names = map(short_name, result[f"{variant}_kernels"])
        print(f"{variant} kernels: {', '.join(names)}")


if __name__ == "__main__":
    main()
