# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the fused beam attention + merge decode path.

Both variants call ExistingGRDecodeAttentionBackend on the same inputs:
``fused_beam_combine=False`` (context attention, beam attention, merge) and
``fused_beam_combine=True`` (context attention, then beam attention and merge
in one kernel). The fused kernel runs on SM10x when the context attention uses
one split; otherwise the backend falls back to the default path. Outputs are
checked bit-identical before timing.

Whole decode attention: CUDA graph replay, L2 flushed before every replay,
variants interleaved, median. Beam attention + merge vs the fused kernel:
per-call kernel time from the CUDA profiler, L2 flushed before every call,
median.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from gr_inference.gr_kernels.attention import (
    ExistingGRDecodeAttentionBackend,
    GRDecodeAttention,
    GRDecodeAttentionInputs,
)
from gr_inference.gr_kv import BeamKV, BeamPath, ContextKV

DEFAULT_KERNEL_ROOT = REPO_ROOT.parents[1] / "corelib" / "gr_decode_atten"
VARIANTS = ("default", "fused")
BEAM_MERGE_KERNELS = {
    "default": ("FlashAttentionForwardDecode", "FlashAttentionForwardCombine"),
    "fused": ("FillFunctor<int>", "BeamSparseCombineVec"),
}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def make_inputs(args, batch: int, width: int):
    q_heads = args.kv_heads * args.q_per_kv
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16)

    context_shape = (1, batch, args.context, args.kv_heads, args.head_dim)
    beam_shape = (1, batch, args.decode_depth, width, args.kv_heads, args.head_dim)
    q = randn(batch, width, q_heads, args.head_dim)
    context_kv = ContextKV(randn(*context_shape), randn(*context_shape))
    beam_kv = BeamKV(randn(*beam_shape), randn(*beam_shape))
    ancestors = torch.randint(
        0,
        width,
        (batch, 1, args.kv_heads, args.decode_depth, width),
        generator=gen,
        device="cuda",
    )
    ancestors[..., -1, :] = torch.arange(width, device="cuda")
    steps = torch.arange(args.decode_depth, device="cuda").view(1, 1, 1, -1, 1)
    topk = (steps * width + ancestors).to(torch.int32)
    topk = topk.repeat_interleave(args.q_per_kv, dim=2)

    def build(return_lse: bool) -> GRDecodeAttentionInputs:
        return GRDecodeAttentionInputs(
            q=q,
            context_kv=context_kv,
            beam_kv=beam_kv,
            beam_path=BeamPath(args.decode_depth, width),
            layer_idx=0,
            step=args.decode_depth - 1,
            active_beam_width=width,
            topk_indices=topk,
            decode_nums=args.decode_depth,
            return_lse=return_lse,
        )

    return build


def graph_replay(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def time_whole(replays, flush, iters: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        for replay in replays.values():
            replay()
    events = {name: [] for name in replays}
    order = list(replays)
    rng = random.Random(0)
    for _ in range(iters):
        rng.shuffle(order)
        for name in order:
            flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            replays[name]()
            end.record()
            events[name].append((start, end))
    torch.cuda.synchronize()
    return {
        name: statistics.median(s.elapsed_time(e) * 1000 for s, e in pairs)
        for name, pairs in events.items()
    }


def kernel_times(fn, flush, calls: int, warmup: int) -> list[tuple[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            flush.zero_()
            fn()
        torch.cuda.synchronize()
    kernels = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    kernels.sort(key=lambda event: event.time_range.start)
    return [(event.name, event.device_time) for event in kernels]


def per_call_us(kernels, patterns, calls: int) -> float:
    totals = [0.0] * calls
    for pattern in patterns:
        hits = [time for name, time in kernels if pattern in name]
        assert len(hits) == calls, f"{pattern}: {len(hits)} launches in {calls} calls"
        totals = [total + hit for total, hit in zip(totals, hits)]
    return statistics.median(totals)


def format_us(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-root",
        default=os.environ.get("GR_DECODE_ATTEN_ROOT", str(DEFAULT_KERNEL_ROOT)),
    )
    parser.add_argument("--batches", type=parse_int_list, default=[1, 8])
    parser.add_argument(
        "--widths", type=parse_int_list, default=[256, 1024, 4096, 8192]
    )
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--q-per-kv", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--decode-depth", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--profile-calls", type=int, default=100)
    args = parser.parse_args()

    attention = {
        name: GRDecodeAttention(
            ExistingGRDecodeAttentionBackend(
                args.kernel_root, fused_beam_combine=name == "fused"
            ).ensure_available()
        )
        for name in VARIANTS
    }
    from gr_inference.gr_kernels.attention.fused_beam_decode import (
        _context_num_splits,
    )

    flush = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    major, minor = torch.cuda.get_device_capability()
    print(
        f"gpu={torch.cuda.get_device_name()} sm={major}{minor} "
        f"h_q={args.kv_heads * args.q_per_kv} h_kv={args.kv_heads} "
        f"dim={args.head_dim} context={args.context} "
        f"decode_depth={args.decode_depth}"
    )
    if major != 10:
        print(
            "fused_beam_combine is used only on SM10x; "
            "here both variants run the same default path"
        )
    print(
        "whole decode attention and beam attention + merge in us, "
        "default/fused (speedup)"
    )
    for batch in args.batches:
        for width in args.widths:
            build = make_inputs(args, batch, width)
            inputs = build(False)
            splits = _context_num_splits(
                inputs.q.unsqueeze(1), inputs.context_kv.key[0], None, None
            )
            outputs = {
                name: (attention[name](build(True)), attention[name](build(False)))
                for name in VARIANTS
            }
            (out, lse), out_no_lse = outputs["default"]
            (fused_out, fused_lse), fused_out_no_lse = outputs["fused"]
            assert (
                torch.equal(out, fused_out)
                and torch.equal(lse, fused_lse)
                and torch.equal(out_no_lse, fused_out_no_lse)
            ), f"outputs differ at batch={batch} width={width}"
            del outputs, out, lse, out_no_lse, fused_out, fused_lse, fused_out_no_lse

            calls = {name: (lambda fn=attention[name]: fn(inputs)) for name in VARIANTS}
            whole = time_whole(
                {name: graph_replay(calls[name]) for name in VARIANTS},
                flush,
                args.iters,
                args.warmup,
            )
            kernels = {
                name: kernel_times(calls[name], flush, args.profile_calls, args.warmup)
                for name in VARIANTS
            }
            fused_ran = any(
                "BeamSparseCombineVec" in name for name, _ in kernels["fused"]
            )
            beam_merge = {
                name: (
                    per_call_us(kernels[name], patterns, args.profile_calls)
                    if fused_ran
                    else None
                )
                for name, patterns in BEAM_MERGE_KERNELS.items()
            }
            beam_merge_speedup = (
                f"{beam_merge['default'] / beam_merge['fused']:.2f}x"
                if fused_ran
                else "n/a"
            )
            print(
                f"batch={batch} beams={width} context_splits={splits} "
                f"fused_kernel_ran={fused_ran} bit_identical=True "
                f"whole={whole['default']:.1f}/{whole['fused']:.1f} "
                f"({whole['default'] / whole['fused']:.2f}x) "
                f"beam_merge={format_us(beam_merge['default'])}/"
                f"{format_us(beam_merge['fused'])} ({beam_merge_speedup})",
                flush=True,
            )
            del build, inputs, calls
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
