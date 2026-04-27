# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Branch-agnostic standalone profiling script for HSTU KVCache timeline.

Goals:
- Run the same workload on before/after branches.
- Avoid pytest dependency and test-file availability differences.
- Support warmup + steady phases for more stable timing.
- Emit NVTX ranges for nsys timeline inspection.
"""

import argparse
import itertools
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from typing import Callable, List, Tuple

import torch

# Ensure inference-only path for branches/images without training deps.
os.environ.setdefault("HSTU_INFERENCE_ONLY", "1")

CUR_DIR = os.path.dirname(__file__)
HSTU_DIR = os.path.abspath(CUR_DIR)
sys.path.append(HSTU_DIR)
sys.path.append(os.path.join(HSTU_DIR, "model"))

from commons.datasets.hstu_batch import FeatureConfig  # noqa: E402
from commons.datasets.random_inference_dataset import RandomInferenceDataset  # noqa: E402
from configs import (  # noqa: E402
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from inference_ranking_gr import get_inference_ranking_gr  # noqa: E402


def _patch_torch_dynamic_shapes_intwrapper_compat() -> None:
    """
    Some branches reference torch.export.dynamic_shapes._IntWrapper directly.
    On newer/other torch versions this symbol may not exist, which breaks
    dataset/HSTUBatch construction. Patch it at runtime for compatibility.
    """
    export_mod = getattr(torch, "export", None)
    dynamic_shapes_mod = getattr(export_mod, "dynamic_shapes", None)
    if dynamic_shapes_mod is not None and not hasattr(dynamic_shapes_mod, "_IntWrapper"):
        setattr(dynamic_shapes_mod, "_IntWrapper", int)


def _apply_dynamic_split_fix_if_supported(model) -> bool:
    """
    Some branches/config combinations require forcing dynamic split to map
    item feature correctly when feature order differs (item->act vs act->item).
    Return True when patch is applied.
    """
    dynamic_collection = getattr(model.sparse_module, "_dynamic_embedding_collection", None)
    if dynamic_collection is not None and hasattr(dynamic_collection, "set_feature_splits"):
        dynamic_collection.set_feature_splits([1, 1], [0])
        return True
    return False


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _cuda_elapsed_ms(fn: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    _ = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _shutdown_model_kvcache_threads(model) -> None:
    mgr = model.dense_module.async_kvcache
    if hasattr(mgr, "executor"):
        mgr.executor.shutdown(wait=False)
    if hasattr(mgr, "onload_worker"):
        mgr.onload_worker.shutdown(wait=False)


def _build_model_and_dataset(num_batches: int):
    max_batch_size = 4
    max_history_length = 128
    max_num_candidates = 16
    max_incremental_seqlen = 16
    max_seq_len = max_history_length * 2 + max_num_candidates
    item_fea_name, item_vocab_size = "item_feat", 10000
    action_fea_name, action_vocab_size = "act_feat", 128

    feature_configs = [
        FeatureConfig(
            feature_names=[item_fea_name, action_fea_name],
            max_item_ids=[item_vocab_size - 1, action_vocab_size - 1],
            max_sequence_length=max_seq_len,
            is_jagged=False,
        ),
    ]

    hidden_dim_size = 128
    num_heads = 2
    head_dim = 64
    num_layers = 2
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
    )
    # Use only common args so this works on both before/after branches.
    kvcache_config = get_kvcache_config(
        blocks_in_primary_pool=512,
        page_size=32,
        offload_chunksize=128,
    )

    # Keep business order: act first, item second.
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=[action_fea_name],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=[item_fea_name],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=True,
        ),
    ]
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[64, 16, 1],
        num_tasks=1,
    )
    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kvcache_config,
        task_config=task_config,
        use_cudagraph=False,
    )
    model.bfloat16()
    model.eval()

    dataset = RandomInferenceDataset(
        feature_configs=feature_configs,
        item_feature_name=item_fea_name,
        contextual_feature_names=[],
        action_feature_name=action_fea_name,
        max_num_users=16,
        max_batch_size=max_batch_size,
        max_history_length=max_history_length,
        max_num_candidates=max_num_candidates,
        max_incremental_seqlen=max_incremental_seqlen,
        max_num_cached_batches=num_batches,
        full_mode=True,
    )
    return model, dataset


def _run_profile(
    model,
    batches: List[Tuple[object, torch.Tensor, torch.Tensor]],
    warmup_iters: int,
    steady_iters: int,
    enable_nvtx: bool,
):
    available_batches = len(batches)
    if available_batches <= 0:
        raise RuntimeError("No batches available for profiling.")

    valid_warmup_iters = max(min(warmup_iters, available_batches), 0)
    remaining_batches = available_batches - valid_warmup_iters
    valid_steady_iters = min(steady_iters, remaining_batches)
    if valid_steady_iters <= 0:
        raise RuntimeError(
            "No steady batches left after warmup. Increase --num-batches."
        )

    with _nvtx_range("warmup_phase", enable_nvtx):
        for idx in range(valid_warmup_iters):
            batch, user_ids, total_history_lengths = batches[idx]
            _ = model.forward_with_kvcache(batch, user_ids, total_history_lengths)
            _ = model.forward_nokvcache(batch)
    torch.cuda.synchronize()

    kvcache_times_ms = []
    nokvcache_times_ms = []
    with _nvtx_range("steady_phase", enable_nvtx):
        for idx in range(valid_steady_iters):
            batch, user_ids, total_history_lengths = batches[valid_warmup_iters + idx]
            with _nvtx_range(f"forward_with_kvcache_iter_{idx}", enable_nvtx):
                kvcache_times_ms.append(
                    _cuda_elapsed_ms(
                        lambda: model.forward_with_kvcache(
                            batch,
                            user_ids,
                            total_history_lengths,
                        )
                    )
                )
            with _nvtx_range(f"forward_nokvcache_iter_{idx}", enable_nvtx):
                nokvcache_times_ms.append(
                    _cuda_elapsed_ms(lambda: model.forward_nokvcache(batch))
                )

    avg_kvcache_ms = sum(kvcache_times_ms) / len(kvcache_times_ms)
    avg_nokvcache_ms = sum(nokvcache_times_ms) / len(nokvcache_times_ms)
    ratio = (
        avg_kvcache_ms / avg_nokvcache_ms if avg_nokvcache_ms > 0.0 else float("inf")
    )
    return {
        "warmup_iters": valid_warmup_iters,
        "steady_iters": valid_steady_iters,
        "avg_forward_with_kvcache_ms": avg_kvcache_ms,
        "avg_forward_nokvcache_ms": avg_nokvcache_ms,
        "kvcache_over_nokv": ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone HSTU KVCache profiler for before/after comparison."
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--steady-iters", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument(
        "--force-dynamic-split-fix",
        action="store_true",
        help=(
            "Apply dynamic split override for item/act mismatch. "
            "Useful when branch has known feature order issue."
        ),
    )
    parser.add_argument(
        "--disable-dynamic-split-fix",
        action="store_true",
        help=(
            "Disable dynamic split compatibility fix. "
            "By default, the fix is enabled for branch compatibility."
        ),
    )
    parser.add_argument("--summary-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script.")

    _patch_torch_dynamic_shapes_intwrapper_compat()

    model, dataset = _build_model_and_dataset(num_batches=max(args.num_batches, 2))
    try:
        # Default-on for branch compatibility; can be disabled explicitly.
        apply_dynamic_split_fix = (
            args.force_dynamic_split_fix or not args.disable_dynamic_split_fix
        )
        if apply_dynamic_split_fix:
            applied = _apply_dynamic_split_fix_if_supported(model)
            if applied:
                print("[standalone-timeline] Applied dynamic split compatibility fix.")

        batches = list(itertools.islice(iter(dataset), max(args.num_batches, 2)))
        with torch.inference_mode():
            result = _run_profile(
                model=model,
                batches=batches,
                warmup_iters=max(args.warmup_iters, 0),
                steady_iters=max(args.steady_iters, 1),
                enable_nvtx=args.enable_nvtx,
            )

        print(
            "[standalone-timeline] "
            f"avg_forward_with_kvcache_ms={result['avg_forward_with_kvcache_ms']:.3f}, "
            f"avg_forward_nokvcache_ms={result['avg_forward_nokvcache_ms']:.3f}, "
            f"kvcache_over_nokv={result['kvcache_over_nokv']:.3f}, "
            f"warmup_iters={result['warmup_iters']}, "
            f"steady_iters={result['steady_iters']}"
        )

        if args.summary_json:
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"[standalone-timeline] Wrote summary to {args.summary_json}")
    finally:
        _shutdown_model_kvcache_threads(model)


if __name__ == "__main__":
    main()
