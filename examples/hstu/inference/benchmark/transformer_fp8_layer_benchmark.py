# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: time the part of a Transformer layer that FP8 changes, attention left out.

Compares the bf16 ``TransformerInferLayer`` with its FP8 variant
``Fp8FusedTransformerInferLayer`` (same weights) on the work outside attention: the two
LayerNorms, the four projections, the GELU and the residual adds. Also reports the cosine
similarity of the two whole layers' outputs.

This part is timed with the script's own CUDA-graph harness: calls are captured back to
back into one ``torch.cuda.CUDAGraph`` and replayed. That is a diagnostic, not the repo's
method. The numbers to quote are the ones of transformer_e2e_benchmark.py, which run
through the repo's per-layer graphs and its timing loop. The helpers below are shared
with that script. Run from examples/hstu in the repository's Docker image, which builds
DeepGEMM from third_party/DeepGEMM:

    PYTHONPATH=$PWD python inference/benchmark/transformer_fp8_layer_benchmark.py
"""

import argparse
import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.transformer_infer_layer import TransformerInferLayer
from modules.transformer_infer_layer_fp8 import Fp8FusedTransformerInferLayer

DIMS = {
    "pr": dict(label="H=512", hidden=512, heads=4, head_dim=128, ffn=2048),
    "large": dict(label="H=2048", hidden=2048, heads=16, head_dim=128, ffn=8192),
}
# (users, tokens per user); the packed token count M is their product.
SCENARIOS = [(1, 1024), (1, 4096), (4, 4096)]
NUM_CANDIDATES = 100
VARIANTS = ("bf16", "fp8")


def elapsed_ms(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def eager_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    return elapsed_ms(fn, iters)


def timed_loop_ms(fn, warmup=10, iters=10, target_ms=1000.0):
    """Average ms per call, with the timing loop of paged_hstu_with_kvcache_benchmark.py.

    Warm up, time ``iters`` back-to-back calls between two CUDA events, then repeat with
    the number of calls raised until the loop lasts about ``target_ms``.
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    total_ms = elapsed_ms(fn, iters) * iters
    if total_ms < target_ms:
        iters *= math.ceil(target_ms / total_ms)
        total_ms = elapsed_ms(fn, iters) * iters
    return total_ms / iters


def graph_ms(fn, target_ms=30.0, replays=10):
    """GPU time per call: capture enough calls to fill ``target_ms`` into one graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    reps = max(2, min(50, int(target_ms / max(eager_ms(fn, 2, 5), 1e-3))))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return elapsed_ms(graph.replay, replays) / reps


def make_config(dims, users, length):
    return SimpleNamespace(
        num_heads=dims["heads"],
        head_dim=dims["head_dim"],
        hidden_size=dims["hidden"],
        max_seq_len=length,
        max_batch_size=users,
        export_mode=False,
        residual=True,
        bf16=True,
        fp16=False,
        layernorm_epsilon=1e-5,
        transformer_ffn_dim=dims["ffn"],
    )


def build_layers(dims, users, length):
    """The bf16 layer and its FP8 layer with the same random weights."""
    cfg = make_config(dims, users, length)
    base = TransformerInferLayer(cfg, 0, "cuda")
    with torch.no_grad():
        for module in base.modules():
            if isinstance(module, nn.Linear):
                module.weight.normal_(0, module.in_features**-0.5)
                module.bias.normal_(0, 0.1)
    fp8 = Fp8FusedTransformerInferLayer(cfg, 0, "cuda")
    fp8.load_state_dict(base.state_dict())
    return base, fp8


def projections(layer):
    return [layer.qkv, layer.proj, layer.ffn[0], layer.ffn[2]]


def measure(timer, make, models):
    """Best of two rounds per model, the second round in reverse order."""
    names = list(models)
    times = {name: [] for name in names}
    for order in (names, names[::-1]):
        for name in order:
            times[name].append(timer(make(models[name])))
    return [min(times[name]) for name in names]


def print_table(title, results, key, names, scale=1):
    """One column per model; each after the first also shows its speedup over it."""
    print(f"\n{title}")
    header = f"{'sizes':<8}{'M':>7} | {names[0]:>9}"
    print(header + "".join(f" | {name:>9} {'x':>6}" for name in names[1:]))
    for row in results:
        first, *others = (t * scale for t in row[key])
        line = f"{row['label']:<8}{row['m']:>7} | {first:9.2f}"
        print(line + "".join(f" | {t:9.2f} {first / t:5.2f}x" for t in others))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", choices=[*DIMS, "all"], default="all")
    args = parser.parse_args()
    names = list(DIMS) if args.dims == "all" else [args.dims]

    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    results = []
    for name, (users, length) in itertools.product(names, SCENARIOS):
        dims = DIMS[name]
        layers = dict(zip(VARIANTS, build_layers(dims, users, length)))
        m = users * length
        x = torch.randn(m, dims["hidden"], device="cuda", dtype=torch.bfloat16)
        attention_out = torch.randn(
            m, dims["heads"] * dims["head_dim"], device="cuda", dtype=torch.bfloat16
        )
        offsets = torch.arange(users + 1, device="cuda", dtype=torch.int32) * length
        candidates = torch.full((users,), NUM_CANDIDATES, device="cuda")

        def outside_attention(layer):
            return lambda: (layer.project_qkv(x), layer.finish(x, attention_out))

        with torch.inference_mode():
            cos = F.cosine_similarity(
                layers["fp8"](x, offsets, candidates).float().flatten(),
                layers["bf16"](x, offsets, candidates).float().flatten(),
                dim=0,
            ).item()
            outside = measure(graph_ms, outside_attention, layers)
        results.append(dict(label=dims["label"], m=m, cos=cos, outside=outside))
        print(f"done {dims['label']} {users} x {length}", file=sys.stderr, flush=True)

    print("M = packed tokens; x = bf16 time / FP8 time, above 1 means FP8 is faster")
    print_table(
        "DIAGNOSTIC (own CUDA-graph harness, not the repo's method): one layer without "
        "attention (2 LayerNorm + 4 projections + GELU + residual adds), us per call",
        results,
        "outside",
        VARIANTS,
        1e3,
    )
    print("\ncosine similarity of the whole layer's output, FP8 vs bf16")
    for row in results:
        print(f"{row['label']:<8}{row['m']:>7} | {row['cos']:.4f}")


if __name__ == "__main__":
    main()
