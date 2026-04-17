# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# end-to-end test of HSTU block inference with KVcache
import itertools
import os
import sys
from contextlib import contextmanager, nullcontext
from typing import Callable, List, Tuple
import pytest
import torch
from commons.datasets.hstu_batch import FeatureConfig
from commons.datasets.random_inference_dataset import RandomInferenceDataset
from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for inference kvcache smoke test.",
)
_CUR_DIR = os.path.dirname(__file__)
_HSTU_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
sys.path.append(os.path.join(_HSTU_DIR, "model"))
from inference_ranking_gr import get_inference_ranking_gr  # noqa: E402


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
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


def _profile_forward_timeline(
    model,
    batches: List[Tuple[object, torch.Tensor, torch.Tensor]],
    iters: int,
    enable_nvtx: bool,
) -> None:
    kvcache_times_ms = []
    nokvcache_times_ms = []
    valid_iters = min(iters, len(batches))
    if valid_iters <= 0:
        raise RuntimeError("No profiling batches available for timeline capture.")
    if valid_iters < iters:
        print(
            "[timeline] Requested iterations exceed unique batches. "
            f"Capping iters from {iters} to {valid_iters} to avoid replaying stale KV states."
        )

    for idx in range(valid_iters):
        batch, user_ids, total_history_lengths = batches[idx]
        nvtx_ctx = _nvtx_range if enable_nvtx else nullcontext
        with nvtx_ctx(f"forward_with_kvcache_iter_{idx}"):
            kvcache_times_ms.append(
                _cuda_elapsed_ms(
                    lambda: model.forward_with_kvcache(
                        batch,
                        user_ids,
                        total_history_lengths,
                    )
                )
            )
        with nvtx_ctx(f"forward_nokvcache_iter_{idx}"):
            nokvcache_times_ms.append(
                _cuda_elapsed_ms(lambda: model.forward_nokvcache(batch))
            )

    avg_kvcache_ms = sum(kvcache_times_ms) / len(kvcache_times_ms)
    avg_nokvcache_ms = sum(nokvcache_times_ms) / len(nokvcache_times_ms)
    ratio = (
        avg_kvcache_ms / avg_nokvcache_ms if avg_nokvcache_ms > 0.0 else float("inf")
    )
    print(
        "[timeline] avg_forward_with_kvcache_ms="
        f"{avg_kvcache_ms:.3f}, avg_forward_nokvcache_ms={avg_nokvcache_ms:.3f}, "
        f"kvcache_over_nokv={ratio:.3f}, iters={valid_iters}"
    )


def _shutdown_model_kvcache_threads(model) -> None:
    mgr = model.dense_module.async_kvcache
    if hasattr(mgr, "executor"):
        mgr.executor.shutdown(wait=False)
    if hasattr(mgr, "onload_worker"):
        mgr.onload_worker.shutdown(wait=False)
def _build_model_and_dataset(max_num_cached_batches: int = 2):
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
    kvcache_config = get_kvcache_config(
        blocks_in_primary_pool=512,
        page_size=32,
        offload_chunksize=128,
        secondary_backend="nop",
        namespace_mode="uid",
        namespace_base="recsys_hstu_test",
    )
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
    model.sparse_module._dynamic_embedding_collection.set_feature_splits([1, 1], [0])
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
        max_num_cached_batches=max_num_cached_batches,
        full_mode=True,
    )
    return model, dataset
def test_inference_forward_with_kvcache_smoke():
    model, dataset = _build_model_and_dataset(max_num_cached_batches=2)
    try:
        enable_timeline = _env_flag("HSTU_KVCACHE_PROFILE_TIMELINE", default=False)
        enable_nvtx = _env_flag("HSTU_KVCACHE_PROFILE_NVTX", default=False)
        profile_iters = max(_env_int("HSTU_KVCACHE_PROFILE_ITERS", default=1), 1)
        batches = list(itertools.islice(iter(dataset), 2))
        assert len(batches) > 0
        with torch.inference_mode():
            for batch, user_ids, total_history_lengths in batches:
                logits = model.forward_with_kvcache(
                    batch,
                    user_ids,
                    total_history_lengths,
                )
                assert torch.is_tensor(logits)
                assert logits.is_cuda
                assert logits.numel() > 0
                assert logits.shape[-1] == 1
            # quick sanity: no-kvcache path also runs
            batch, _, _ = batches[0]
            logits_nokv = model.forward_nokvcache(batch)
            assert torch.is_tensor(logits_nokv)
            assert logits_nokv.is_cuda
            assert logits_nokv.numel() > 0
            assert logits_nokv.shape[-1] == 1
        if enable_timeline:
            profile_num_batches = max(
                _env_int(
                    "HSTU_KVCACHE_PROFILE_NUM_BATCHES",
                    default=max(profile_iters, 2),
                ),
                2,
            )
            profile_model, profile_dataset = _build_model_and_dataset(
                max_num_cached_batches=profile_num_batches
            )
            try:
                profile_batches = list(
                    itertools.islice(iter(profile_dataset), profile_num_batches)
                )
                with torch.inference_mode():
                    _profile_forward_timeline(
                        model=profile_model,
                        batches=profile_batches,
                        iters=profile_iters,
                        enable_nvtx=enable_nvtx,
                    )
            finally:
                _shutdown_model_kvcache_threads(profile_model)
    finally:
        _shutdown_model_kvcache_threads(model)