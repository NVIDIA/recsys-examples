# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark real-weight Qwen3 GR skeleton paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gr_inference import (  # noqa: E402
    GRDecodeAttention,
    GRDecodeEngine,
    GRGenerationState,
)
from gr_inference.gr_kernels.attention import (  # noqa: E402
    ExistingGRDecodeAttentionBackend,
)
from gr_inference.gr_models.qwen3 import (  # noqa: E402
    flashinfer_call_counts,
    reset_flashinfer_call_counts,
)
from gr_inference.gr_runtime import TimingRecorder  # noqa: E402
from run_qwen3_real_weight_tiny_gr import (  # noqa: E402
    build_identity_topk_indices,
    load_model,
)


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(torch, fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(torch)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(torch)
    return (time.perf_counter() - start) / iters * 1000


def make_input(torch, config, args, device):
    return torch.randint(0, config.vocab_size, (1, args.context_len), device=device)


def bench_prefill(torch, model, config, args, device) -> float:
    input_ids = make_input(torch, config, args, device)

    def run():
        with torch.no_grad():
            model.forward_prefill(input_ids, return_result=True)

    return time_call(torch, run, warmup=args.warmup, iters=args.iters)


def bench_fake_decode(torch, model, config, args, device) -> float:
    input_ids = make_input(torch, config, args, device)
    with torch.no_grad():
        prefill = model.forward_prefill(input_ids, return_result=True)

    def run():
        with torch.no_grad():
            generation = GRGenerationState.from_prefill(
                request_id="bench-fake",
                prefill=prefill,
                max_decode_steps=config.max_decode_steps,
                max_beam_width=config.max_beam_width,
                fixed_beam_width=args.beam_width,
            )
            decode_engine = GRDecodeEngine(
                attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
                fixed_beam_width=args.beam_width,
            )
            model.generate_fixed_beam(
                generation, decode_engine, max_steps=args.decode_steps
            )

    return time_call(torch, run, warmup=args.warmup, iters=args.iters)


def bench_real_decode(torch, model, config, args, device) -> float:
    if device != "cuda":
        raise RuntimeError("real decode benchmark requires CUDA")
    if args.decode_steps != 1:
        raise RuntimeError("real decode benchmark currently supports --decode-steps 1")

    input_ids = make_input(torch, config, args, device)
    with torch.no_grad():
        prefill = model.forward_prefill(input_ids, return_result=True)
        topk_indices = build_identity_topk_indices(
            torch,
            batch=1,
            head_q=config.num_attention_heads,
            decode_nums=1,
            beam_width=args.beam_width,
            device=device,
        )
        decode_engine = GRDecodeEngine(
            attention=GRDecodeAttention(backend=ExistingGRDecodeAttentionBackend()),
            fixed_beam_width=args.beam_width,
        )

    def run():
        with torch.no_grad():
            generation = GRGenerationState.from_prefill(
                request_id="bench-real",
                prefill=prefill,
                max_decode_steps=config.max_decode_steps,
                max_beam_width=config.max_beam_width,
                fixed_beam_width=args.beam_width,
            )
            selection = generation.initialize_beams()
            beam_token_ids = torch.tensor(
                [selection.token_ids], dtype=torch.long, device=device
            )
            model.forward_decode_step(
                beam_token_ids,
                generation,
                decode_engine,
                step=0,
                topk_indices=topk_indices,
                decode_nums=1,
                return_lse=True,
                backend_name=args.kernel_backend,
            )

    return time_call(torch, run, warmup=args.warmup, iters=args.iters)


def profile_real_decode(torch, model, config, args, device) -> dict:
    if device != "cuda":
        raise RuntimeError("decode profiling requires CUDA")
    if args.decode_steps != 1:
        raise RuntimeError("decode profiling currently supports --decode-steps 1")

    input_ids = make_input(torch, config, args, device)
    with torch.no_grad():
        prefill = model.forward_prefill(input_ids, return_result=True)
        generation = GRGenerationState.from_prefill(
            request_id="profile-real",
            prefill=prefill,
            max_decode_steps=config.max_decode_steps,
            max_beam_width=config.max_beam_width,
            fixed_beam_width=args.beam_width,
        )
        selection = generation.initialize_beams()
        beam_token_ids = torch.tensor(
            [selection.token_ids], dtype=torch.long, device=device
        )
        topk_indices = build_identity_topk_indices(
            torch,
            batch=1,
            head_q=config.num_attention_heads,
            decode_nums=1,
            beam_width=args.beam_width,
            device=device,
        )
        decode_engine = GRDecodeEngine(
            attention=GRDecodeAttention(backend=ExistingGRDecodeAttentionBackend()),
            fixed_beam_width=args.beam_width,
        )
        reset_flashinfer_call_counts()
        recorder = TimingRecorder(sync_module=torch, detail=args.profile_detail)
        with recorder.section("model.forward_decode_step"):
            model.forward_decode_step(
                beam_token_ids,
                generation,
                decode_engine,
                step=0,
                topk_indices=topk_indices,
                decode_nums=1,
                return_lse=True,
                backend_name=args.kernel_backend,
                timing_recorder=recorder,
            )
    summary = recorder.summary()
    summary["flashinfer.calls.rmsnorm"] = {
        "total_ms": float(flashinfer_call_counts().get("rmsnorm", 0)),
        "count": 1,
        "avg_ms": float(flashinfer_call_counts().get("rmsnorm", 0)),
    }
    summary["flashinfer.calls.rope"] = {
        "total_ms": float(flashinfer_call_counts().get("rope", 0)),
        "count": 1,
        "avg_ms": float(flashinfer_call_counts().get("rope", 0)),
    }
    return summary


def aggregate_profile(summary: dict) -> dict[str, float]:
    groups = {
        "qkv_ms": ".qkv",
        "input_norm_ms": ".input_norm",
        "qkv_proj_ms": ".qkv_proj",
        "qk_norm_rope_ms": ".qk_norm_rope",
        "q_norm_ms": ".q_norm",
        "k_norm_ms": ".k_norm",
        "rope_ms": ".rope",
        "beam_kv_write_ms": ".beam_kv_write",
        "decode_attention_ms": ".decode_attention",
        "post_attention_mlp_ms": ".post_attention",
        "o_proj_ms": ".o_proj",
        "post_norm_ms": ".post_norm",
        "mlp_ms": ".mlp",
        "gate_up_proj_ms": ".gate_up_proj",
        "silu_mul_ms": ".silu_mul",
        "down_proj_ms": ".down_proj",
        "layer_total_ms": ".decode_total",
    }
    aggregate = {
        output_name: sum(
            stats["total_ms"]
            for name, stats in summary.items()
            if name.startswith("layer") and name.endswith(suffix)
        )
        for output_name, suffix in groups.items()
    }
    aggregate["model_forward_decode_step_ms"] = summary.get(
        "model.forward_decode_step",
        {},
    ).get("total_ms", 0.0)
    aggregate["flashinfer_rmsnorm_calls"] = summary.get(
        "flashinfer.calls.rmsnorm",
        {},
    ).get("total_ms", 0.0)
    aggregate["flashinfer_rope_calls"] = summary.get(
        "flashinfer.calls.rope",
        {},
    ).get("total_ms", 0.0)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=["prefill", "fake-decode", "real-decode", "all"],
        default="all",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--kernel-backend", choices=["dsl", "3kernel"], default="dsl")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile-decode", action="store_true")
    parser.add_argument(
        "--profile-detail",
        choices=["coarse", "fine"],
        default="coarse",
        help="coarse avoids extra syncs; fine reports per-operator sub-stages",
    )
    args = parser.parse_args()

    import torch

    model, config, device = load_model(args, torch)
    print("Real-weight Qwen3 benchmark")
    print("=" * 72)
    print(
        f"context_len={args.context_len} decode_steps={args.decode_steps} "
        f"beam_width={args.beam_width} device={device}"
    )

    if args.mode in {"prefill", "all"}:
        print(f"prefill_ms={bench_prefill(torch, model, config, args, device):.3f}")
    if args.mode in {"fake-decode", "all"}:
        print(
            f"fake_decode_loop_ms={bench_fake_decode(torch, model, config, args, device):.3f}"
        )
    if args.mode == "real-decode":
        print(
            f"real_decode_step_ms={bench_real_decode(torch, model, config, args, device):.3f}"
        )
    elif args.mode == "all":
        if args.decode_steps == 1 and device == "cuda":
            print(
                f"real_decode_step_ms={bench_real_decode(torch, model, config, args, device):.3f}"
            )
        else:
            print("real_decode_step_ms=SKIP (requires --decode-steps 1 and CUDA)")

    if args.profile_decode:
        print("\nDecode profile:")
        summary = profile_real_decode(torch, model, config, args, device)
        aggregate = aggregate_profile(summary)
        for name, value in aggregate.items():
            print(f"{name}={value:.3f}")
        print("\nDecode profile by layer:")
        for name, stats in summary.items():
            print(
                f"{name}: total_ms={stats['total_ms']:.3f} "
                f"count={int(stats['count'])} avg_ms={stats['avg_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
