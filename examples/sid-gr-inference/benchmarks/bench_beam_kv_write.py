# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the decode beam-KV write.

Compares the three paths ``BeamKVWriter.write_layer_step`` can take:

- ``pytorch``: the PyTorch slice copy (``GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT=0``)
- ``scalar``: the scalar CUDA kernel (``GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT=1``
  with ``GR_INFERENCE_TRTLLM_BEAM_KV_WRITE_VECTORIZE=0``)
- ``vectorized``: the vectorized CUDA kernel (default, both unset)

Outputs are checked bit-identical before timing. Each call is timed as the GPU
time of its write kernels from the CUDA profiler, with the L2 cache cleared
before every call; the median over all calls is reported.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from gr_inference.gr_kv import BeamKV
from gr_inference.gr_runtime.decode_kv import BeamKVWriter

VARIANTS = {
    "pytorch": (
        {"GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT": "0"},
        ("Memcpy DtoD", "direct_copy_kernel_cuda"),
    ),
    "scalar": (
        {
            "GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT": "1",
            "GR_INFERENCE_TRTLLM_BEAM_KV_WRITE_VECTORIZE": "0",
        },
        ("write_beam_kv_step_kernel",),
    ),
    "vectorized": ({}, ("write_beam_kv_step_vec_kernel",)),
}
ENV_KEYS = (
    "GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT",
    "GR_INFERENCE_TRTLLM_BEAM_KV_WRITE_VECTORIZE",
)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def set_variant_env(variant: str) -> None:
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(VARIANTS[variant][0])


def kernel_times_us(write, kernel_names, calls: int) -> list[float]:
    flush = torch.empty(64 * 2**20, dtype=torch.float32, device="cuda")
    for _ in range(5):
        write()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            flush.zero_()
            write()
        torch.cuda.synchronize()
    launches = sorted(
        (
            event
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and any(name in event.name for name in kernel_names)
        ),
        key=lambda event: event.time_range.start,
    )
    per_call, extra = divmod(len(launches), calls)
    if not per_call or extra:
        names = sorted({event.name for event in prof.events()})
        raise RuntimeError(
            f"expected kernels {kernel_names} in every call, got {len(launches)} "
            f"launches in {calls} calls; kernels seen: {names}"
        )
    return [
        sum(event.device_time for event in launches[i * per_call : (i + 1) * per_call])
        for i in range(calls)
    ]


def run_case(args, *, batch: int, beam: int) -> dict[str, float]:
    shape = (1, batch, args.steps, beam, args.kv_heads, args.head_dim)
    key = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    writer = BeamKVWriter(BeamKV(key, value))
    k = torch.randn(
        batch, beam, args.kv_heads, args.head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    step = args.steps // 2

    def write() -> None:
        writer.write_layer_step(
            layer_idx=0, step=step, active_beam_width=beam, k=k, v=v
        )

    outputs = {}
    for variant in VARIANTS:
        set_variant_env(variant)
        key.zero_()
        value.zero_()
        write()
        outputs[variant] = (key.clone(), value.clone())
    reference = outputs["pytorch"]
    for variant, (out_key, out_value) in outputs.items():
        if not (
            torch.equal(out_key, reference[0]) and torch.equal(out_value, reference[1])
        ):
            raise AssertionError(
                f"{variant} output differs at batch={batch} beam={beam}"
            )

    medians = {}
    for variant, (_, kernel_names) in VARIANTS.items():
        set_variant_env(variant)
        medians[variant] = statistics.median(
            kernel_times_us(write, kernel_names, args.calls)
        )
    set_variant_env("vectorized")
    return medians


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=parse_int_list,
        default=[1, 8],
        help="Comma or space separated batch list.",
    )
    parser.add_argument(
        "--beams",
        type=parse_int_list,
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Comma or space separated beam width list.",
    )
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--calls", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_beam_kv_write.py requires a CUDA GPU")
    from gr_inference_trtllm_kernels.qwen3 import _cuda_extension

    if _cuda_extension() is None:
        raise SystemExit("the gr_inference_trtllm_kernels CUDA extension did not build")

    print(
        f"gpu={torch.cuda.get_device_name()} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} dtype=bf16 calls={args.calls}"
    )
    print(
        f"{'batch':>5} {'beam':>5} {'pytorch_us':>10} {'scalar_us':>9} "
        f"{'vectorized_us':>13} {'vs_pytorch':>10} {'vs_scalar':>9}"
    )
    for batch in args.batches:
        for beam in args.beams:
            us = run_case(args, batch=batch, beam=beam)
            print(
                f"{batch:>5} {beam:>5} {us['pytorch']:>10.2f} {us['scalar']:>9.2f} "
                f"{us['vectorized']:>13.2f} "
                f"{us['pytorch'] / us['vectorized']:>9.2f}x "
                f"{us['scalar'] / us['vectorized']:>8.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
