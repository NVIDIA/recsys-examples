"""Microbenchmark prefill attention backends.

This benchmark measures the framework prefill path, including ContextKV layer
write validation. It is intentionally backend-oriented rather than a vLLM or
SGLang end-to-end serving benchmark.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gr_inference.gr_kernels.prefill import (
    AutoPrefillBackend,
    FlashAttentionPrefillBackend,
    PrefillAttention,
    TorchSDPAPrefillBackend,
)
from gr_inference.gr_kv import ContextKV
from gr_inference.gr_runtime import GRPrefillRunner


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_backend_list(value: str) -> list[str]:
    return value.replace(",", " ").split()


def make_backend(name: str):
    if name == "auto":
        return AutoPrefillBackend()
    if name == "torch_sdpa":
        return TorchSDPAPrefillBackend()
    if name == "flash_attn":
        return FlashAttentionPrefillBackend()
    raise ValueError(f"unknown backend: {name}")


def synchronize(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def run_one(args, torch, *, backend: str, batch: int, seq: int) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    runner = GRPrefillRunner(PrefillAttention(make_backend(backend)))

    q = torch.randn(batch, seq, args.h_q, args.dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, args.h_kv, args.dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    context_k = torch.empty(
        args.layers, batch, seq, args.h_kv, args.dim, device=device, dtype=dtype
    )
    context_v = torch.empty_like(context_k)
    context_kv = ContextKV(context_k, context_v)

    for _ in range(args.warmup):
        runner.run_layer(q=q, k=k, v=v, context_kv=context_kv, layer_idx=0)
    synchronize(torch)

    start = time.perf_counter()
    for _ in range(args.iters):
        runner.run_layer(q=q, k=k, v=v, context_kv=context_kv, layer_idx=0)
    synchronize(torch)
    elapsed = time.perf_counter() - start
    return elapsed / args.iters * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["auto", "torch_sdpa", "flash_attn"],
        default="auto",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4700)
    parser.add_argument("--h-q", type=int, default=16)
    parser.add_argument("--h-kv", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run a backend x batch x sequence matrix.",
    )
    parser.add_argument(
        "--backends",
        type=parse_backend_list,
        default=["torch_sdpa", "flash_attn"],
        help="Comma or space separated backend list for --matrix.",
    )
    parser.add_argument(
        "--batches",
        type=parse_int_list,
        default=[1, 2, 4],
        help="Comma or space separated batch list for --matrix.",
    )
    parser.add_argument(
        "--seqs",
        type=parse_int_list,
        default=[1024, 2048, 4700, 8192],
        help="Comma or space separated sequence list for --matrix.",
    )
    args = parser.parse_args()

    import torch

    if args.matrix:
        results: dict[tuple[int, int, str], float] = {}
        for batch in args.batches:
            for seq in args.seqs:
                for backend in args.backends:
                    latency_ms = run_one(args, torch, backend=backend, batch=batch, seq=seq)
                    results[(batch, seq, backend)] = latency_ms
                    print(
                        f"backend={backend} batch={batch} seq={seq} "
                        f"h_q={args.h_q} h_kv={args.h_kv} dim={args.dim} "
                        f"latency_ms={latency_ms:.3f}"
                    )

        if "torch_sdpa" in args.backends and "flash_attn" in args.backends:
            print("\nComparison: flash_attn vs torch_sdpa")
            for batch in args.batches:
                for seq in args.seqs:
                    sdpa = results[(batch, seq, "torch_sdpa")]
                    flash = results[(batch, seq, "flash_attn")]
                    speedup = sdpa / flash
                    print(
                        f"batch={batch} seq={seq} "
                        f"torch_sdpa={sdpa:.3f}ms flash_attn={flash:.3f}ms "
                        f"speedup={speedup:.3f}x"
                    )
        return

    latency_ms = run_one(args, torch, backend=args.backend, batch=args.batch, seq=args.seq)

    print(
        f"backend={args.backend} batch={args.batch} seq={args.seq} "
        f"h_q={args.h_q} h_kv={args.h_kv} dim={args.dim} "
        f"latency_ms={latency_ms:.3f}"
    )


if __name__ == "__main__":
    main()
