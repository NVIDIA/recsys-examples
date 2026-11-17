# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Transformer inference variants with the repo's timing method.

The baseline is the bf16 Transformer with its reference SDPA attention. Against it run
FP8 projections (``transformer_fp8_linear``), FA4 attention
(``transformer_attention="fa4"``) and both. Two levels are timed, each with eager layers
and with the repo's per-layer CUDA graphs (``InferenceDenseModule.setup_for_cudagraph``):

* block: ``HSTUBlockInference.predict`` over all layers, the call that
  paged_hstu_with_kvcache_benchmark.py times, including its copy of the inputs into the
  static graph buffers;
* dense module: ``InferenceDenseModule.forward``, that is the shared recommendation
  preprocessor, the layers, the postprocessor and the prediction head, as the Triton
  serving model runs it (no KV cache).

The timing loop is the one of that benchmark: 10 warmup calls, CUDA events around
back-to-back calls, the number of calls raised until the loop lasts about a second. The
best of two rounds per model is reported. The batch is synthetic, the weights are random
and loaded from the baseline into every variant, and the sparse embedding lookup is not
part of the dense module and not timed. Run from examples/hstu in the repository's Docker
image, which installs DeepGEMM and FA4:

    PYTHONPATH=$PWD/..:$PWD python inference/benchmark/transformer_e2e_benchmark.py
"""

import argparse
import itertools
import sys
from pathlib import Path

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
HSTU_ROOT = BENCHMARK_DIR.parents[1]
sys.path[:0] = [str(BENCHMARK_DIR), str(HSTU_ROOT), str(HSTU_ROOT.parent)]

from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceEmbeddingConfig, RankingConfig, get_inference_hstu_config
from modules.inference_dense_module import InferenceDenseModule, copy_jagged_metadata
from torch.nn import functional as F
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from transformer_fp8_layer_benchmark import (
    DIMS,
    NUM_CANDIDATES,
    SCENARIOS,
    measure,
    print_table,
    timed_loop_ms,
)

# Inference config options per variant; the first one is the baseline.
MODELS = {
    "bf16": {},
    "fp8": dict(transformer_fp8_linear=True),
    "fa4": dict(transformer_attention="fa4"),
    "fp8+fa4": dict(transformer_fp8_linear=True, transformer_attention="fa4"),
}


def build_dense(dims, users, length, num_layers, options):
    cfg = get_inference_hstu_config(
        hidden_size=dims["hidden"],
        num_layers=num_layers,
        num_attention_heads=dims["heads"],
        head_dim=dims["head_dim"],
        max_batch_size=users,
        max_seq_len=length,
        dtype=torch.bfloat16,
        contextual_max_seqlen=1,
        backbone="transformer",
        transformer_ffn_dim=dims["ffn"],
        **options,
    )
    task = RankingConfig(
        embedding_configs=[
            InferenceEmbeddingConfig(["item"], "item", 16, dims["hidden"], False)
        ],
        prediction_head_arch=[dims["hidden"], 2],
        num_tasks=2,
    )
    return cfg, InferenceDenseModule(cfg, None, task).bfloat16().eval()


def make_batch(users, history, candidates, hidden):
    """Per user: one context token, ``history`` items with actions and ``candidates``."""
    item_len = history + candidates
    rows = {
        "context": [1] * users,
        "item": [item_len] * users,
        "action": [item_len] * users,
    }
    embeddings = {
        name: JaggedTensor(
            values=torch.randn(sum(lens), hidden, device="cuda", dtype=torch.bfloat16),
            lengths=torch.tensor(lens, device="cuda"),
        )
        for name, lens in rows.items()
    }
    lengths = sum(rows.values(), [])
    features = KeyedJaggedTensor(
        keys=list(rows),
        values=torch.arange(sum(lengths), device="cuda"),
        lengths=torch.tensor(lengths, device="cuda"),
    )
    batch = HSTUBatch(
        features=features,
        batch_size=users,
        feature_to_max_seqlen={"context": 1, "item": item_len, "action": item_len},
        contextual_feature_names=["context"],
        item_feature_name="item",
        action_feature_name="action",
        max_num_candidates=candidates,
        num_candidates=torch.full((users,), candidates, device="cuda"),
    )
    return batch, embeddings


def block_call(model, jagged, num_tokens, users, graphs):
    """``HSTUBlockInference.predict`` as paged_hstu_with_kvcache_benchmark.py calls it."""
    block = model._hstu_block
    if not graphs:
        return lambda: block.predict(users, num_tokens, jagged.values, jagged, None)

    def call():
        model._hidden_states[:num_tokens].copy_(jagged.values, non_blocking=True)
        copy_jagged_metadata(model._jagged_metadata, jagged)
        return block.predict(
            users, num_tokens, model._hidden_states, model._jagged_metadata, None
        )

    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", choices=[*DIMS, "all"], default="all")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS)[1:],
        default=list(MODELS)[1:],
        help="variants to time against the bf16 baseline",
    )
    args = parser.parse_args()
    names = list(DIMS) if args.dims == "all" else [args.dims]
    variants = ["bf16", *args.models]

    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    results = []
    for name, (users, length) in itertools.product(names, SCENARIOS):
        dims = DIMS[name]
        torch.manual_seed(0)
        built = {
            v: build_dense(dims, users, length, args.layers, MODELS[v])
            for v in variants
        }
        models = {v: model for v, (_, model) in built.items()}
        for variant in variants[1:]:
            # FP8 layers quantize the loaded bf16 weights in the checkpoint loader.
            models[variant].load_state_dict(models["bf16"].state_dict(), strict=False)

        # One context token, two tokens per history item (item and action) and the
        # candidates make up a user's sequence, which must fit in ``length``.
        history = (length - 1 - NUM_CANDIDATES) // 2
        batch, embeddings = make_batch(users, history, NUM_CANDIDATES, dims["hidden"])

        with torch.inference_mode():
            jagged = {}
            for variant, model in models.items():
                jagged[variant] = model._hstu_block._preprocessor(embeddings, batch)
                jagged[variant].scaling_seqlen = model._scaling_seqlen
            tokens = jagged["bf16"].values.shape[0]
            assert tokens <= users * length, (tokens, users * length)
            jagged_of = {id(models[v]): jagged[v] for v in variants}

            def block(graphs):
                return lambda model: block_call(
                    model, jagged_of[id(model)], tokens, users, graphs
                )

            def dense(model):
                return lambda: model(batch, embeddings)

            logits = {v: dense(models[v])() for v in variants}
            cos = [
                F.cosine_similarity(
                    logits[v].float().flatten(), logits["bf16"].float().flatten(), dim=0
                ).item()
                for v in variants[1:]
            ]
            row = dict(label=dims["label"], m=tokens, cos=cos)
            row["block_eager"] = measure(timed_loop_ms, block(False), models)
            row["dense_eager"] = measure(timed_loop_ms, dense, models)
            graph_cfg = {"batch_size": [users], "length_per_sequence": [length]}
            for cfg, model in built.values():
                model.setup_for_cudagraph(cfg, None, True, graph_cfg)
            for model in models.values():
                block(True)(model)()
                dense(model)()
            row["block_graph"] = measure(timed_loop_ms, block(True), models)
            row["dense_graph"] = measure(timed_loop_ms, dense, models)
        results.append(row)
        print(f"done {dims['label']} {users} x {length}", file=sys.stderr, flush=True)
        del built, models

    n = f"{args.layers} layer" + ("" if args.layers == 1 else "s")
    print(f"{n} per model, {NUM_CANDIDATES} candidates per user")
    print("M = packed tokens after preprocessing; x = bf16 time / variant time")
    block_part = f"BLOCK predict ({n})"
    dense_part = f"DENSE MODULE forward (preprocessor + {n} + postprocessor + head)"
    for part, prefix in ((block_part, "block"), (dense_part, "dense")):
        for mode, suffix in (
            ("EAGER layers (called one after another from Python)", "eager"),
            ("repo's per-layer CUDA GRAPHS (set_cudagraph)", "graph"),
        ):
            title = f"{part}, {mode}, ms per call"
            print_table(title, results, f"{prefix}_{suffix}", variants)
    print("\ncosine similarity of the DENSE MODULE logits against bf16")
    print(f"{'sizes':<8}{'M':>7} | " + " | ".join(f"{v:>7}" for v in variants[1:]))
    for row in results:
        cells = " | ".join(f"{c:7.4f}" for c in row["cos"])
        print(f"{row['label']:<8}{row['m']:>7} | {cells}")


if __name__ == "__main__":
    main()
