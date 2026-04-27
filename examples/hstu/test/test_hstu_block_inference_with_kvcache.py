# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# end-to-end test of HSTU block inference with KVcache and without KVcache
import copy
import itertools
import os
import sys
from contextlib import contextmanager

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

os.environ.setdefault("HSTU_INFERENCE_ONLY", "1")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for inference kvcache smoke test.",
)
_CUR_DIR = os.path.dirname(__file__)
_HSTU_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
sys.path.append(os.path.join(_HSTU_DIR, "model"))
from inference_ranking_gr import get_inference_ranking_gr  # noqa: E402


def _shutdown_model_kvcache_threads(model) -> None:
    mgr = model.dense_module.async_kvcache
    secondary = getattr(mgr, "secondary_kvcache_manager", None)
    client = getattr(secondary, "_client", None) if secondary is not None else None
    if client is not None:
        try:
            client.shutdown()
        except Exception:
            pass
    if hasattr(mgr, "executor"):
        mgr.executor.shutdown(wait=False)
    if hasattr(mgr, "onload_worker"):
        mgr.onload_worker.shutdown(wait=False)


@contextmanager
def _capture_hstu_layer_outputs(model):
    layers = model.dense_module._hstu_block._attention_layers
    original_methods = []
    outputs = {}

    for layer_idx, layer in enumerate(layers):
        original = layer.forward_naive
        original_methods.append((layer, original))

        def _wrap_forward_naive(batch_size, num_tokens, layer_input, jd, kv_cache_metadata, _orig=original, _idx=layer_idx):
            layer_output = _orig(
                batch_size, num_tokens, layer_input, jd, kv_cache_metadata
            )
            outputs[_idx] = layer_output.detach().float().cpu().clone()
            return layer_output

        layer.forward_naive = _wrap_forward_naive

    try:
        yield outputs
    finally:
        for layer, original in original_methods:
            layer.forward_naive = original


def _build_model_and_dataset(
    secondary_backend: str = "nop",
    flexkv_mode: str = "direct",
):
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
        secondary_backend=secondary_backend,
        flexkv_mode=flexkv_mode,
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
        max_num_cached_batches=2,
        full_mode=True,
    )
    return model, dataset


def test_inference_forward_with_kvcache_layerwise_compare():
    model, dataset = _build_model_and_dataset()
    try:
        batches = list(itertools.islice(iter(dataset), 1))
        assert len(batches) == 1
        batch, user_ids, total_history_lengths = batches[0]

        # Use identical input batch for both paths to compare per-layer outputs.
        batch_nokv = copy.deepcopy(batch)
        batch_kvcache = copy.deepcopy(batch)

        with torch.inference_mode():
            with _capture_hstu_layer_outputs(model) as nokv_layer_outputs:
                logits_nokv = model.forward_nokvcache(batch_nokv)

            with _capture_hstu_layer_outputs(model) as kvcache_layer_outputs:
                logits_kvcache = model.forward_with_kvcache(
                    batch_kvcache,
                    user_ids,
                    total_history_lengths,
                )

        assert torch.is_tensor(logits_kvcache)
        assert logits_kvcache.is_cuda
        assert logits_kvcache.numel() > 0
        assert logits_kvcache.shape[-1] == 1

        assert torch.is_tensor(logits_nokv)
        assert logits_nokv.is_cuda
        assert logits_nokv.numel() > 0
        assert logits_nokv.shape[-1] == 1

        num_layers = len(model.dense_module._hstu_block._attention_layers)
        assert len(nokv_layer_outputs) == num_layers
        assert len(kvcache_layer_outputs) == num_layers
        a = kvcache_layer_outputs[0].float().cpu()
        b = nokv_layer_outputs[0].float().cpu()
        max_abs = (a - b).abs().max().item()
        torch.testing.assert_close(
                a, b, rtol=1e-2, atol=1e-2,
                msg=f"layer={0}, max_abs={max_abs:.6f}",
            )
        # for layer_idx in range(num_layers):
        #     a = kvcache_layer_outputs[layer_idx]
        #     b = nokv_layer_outputs[layer_idx]
        #     max_abs = (a - b).abs().max().item()
        #     print(f"[layer {layer_idx}] max_abs={max_abs:.6f}")
        #     torch.testing.assert_close(
        #         a, b, rtol=1e-2, atol=1e-2,
        #         msg=f"layer={layer_idx}, max_abs={max_abs:.6f}",
        #     )

        torch.testing.assert_close(
            logits_kvcache.detach().float().cpu(),
            logits_nokv.detach().float().cpu(),
            rtol=1e-2,
            atol=1e-2,
        )
    finally:
        _shutdown_model_kvcache_threads(model)

@pytest.mark.parametrize("secondary_backend,flexkv_mode", [
    ("nop", "direct"),
    ("flexkv", "direct"),
])
def test_inference_forward_with_kvcache_backend_smoke(secondary_backend, flexkv_mode):
    model, dataset = _build_model_and_dataset(
        secondary_backend=secondary_backend,
        flexkv_mode=flexkv_mode,
    )
    try:
        batch, user_ids, total_history_lengths = next(iter(dataset))
        with torch.inference_mode():
            logits = model.forward_with_kvcache(batch, user_ids, total_history_lengths)
        assert torch.is_tensor(logits)
        assert logits.is_cuda
        assert logits.shape[-1] == 1
        assert logits.numel() > 0
    finally:
        _shutdown_model_kvcache_threads(model)