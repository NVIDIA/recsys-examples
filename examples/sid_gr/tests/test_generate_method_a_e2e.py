# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
End-to-end test for generate_method_a().

Constructs a full SIDGRModel, runs generate_method_a() with a random batch,
and verifies:
  1. It runs without errors (smoke test)
  2. Output shapes are correct
  3. Generated SIDs are within valid codebook ranges
"""
from typing import List

import pytest
import torch

try:
    import commons.utils as init
    from commons.checkpoint import get_unwrapped_module
    from commons.datasets.gpt_sid_batch import FeatureConfig, GPTSIDBatch
    from commons.modules.embedding import ShardedEmbeddingConfig
    from commons.ops.length_to_offsets import length_to_complete_offsets
    from tests.test_utils import create_sid_gr_model_and_optimizer
    HAS_SIDGR_DEPS = True
except ImportError as e:
    HAS_SIDGR_DEPS = False
    _SIDGR_IMPORT_ERR = str(e)

try:
    from flash_attn.cute.interface import flash_attn_func  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

_SKIP_REASON = (
    "flash_attn not installed" if not HAS_FLASH_ATTN
    else f"SID-GR deps unavailable: {_SIDGR_IMPORT_ERR}" if not HAS_SIDGR_DEPS
    else None
)
_SHOULD_SKIP = not (HAS_FLASH_ATTN and HAS_SIDGR_DEPS)


def _generate_batch(
    batchsize: int,
    max_history_length: int,
    codebook_sizes: List[int],
    history_feature_name: str,
    candidate_feature_name: str,
) -> GPTSIDBatch:
    num_hierarchies = len(codebook_sizes)
    codebook_sizes_t = torch.tensor(codebook_sizes)
    cum_sum = length_to_complete_offsets(codebook_sizes_t)
    max_item_ids = cum_sum[1:]
    min_item_ids = cum_sum[:-1]
    raw_hist_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
    feature_configs = [
        FeatureConfig(
            feature_names=raw_hist_names,
            max_item_ids=max_item_ids,
            min_item_ids=min_item_ids,
            max_sequence_length=max_history_length,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=raw_cand_names,
            max_item_ids=max_item_ids,
            min_item_ids=min_item_ids,
            max_sequence_length=1,
            is_jagged=False,
        ),
    ]
    return GPTSIDBatch.random(
        batch_size=batchsize,
        feature_configs=feature_configs,
        raw_hist_sid_names=raw_hist_names,
        raw_cand_sid_names=raw_cand_names,
        combined_history_feature_name=history_feature_name,
        combined_candidate_feature_name=candidate_feature_name,
        contextual_feature_names=[],
        device=torch.cuda.current_device(),
    )


@pytest.mark.skipif(_SHOULD_SKIP, reason=_SKIP_REASON or "")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("kv_channels", [64])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("max_history_length", [64])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
@pytest.mark.parametrize("batchsize", [4])
def test_generate_method_a_smoke(
    dtype,
    hidden_size,
    num_attention_heads,
    kv_channels,
    num_layers,
    max_history_length,
    codebook_sizes,
    batchsize,
):
    """generate_method_a() runs end-to-end without errors."""
    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(42)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=hidden_size,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_batch(
            batchsize=batchsize,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        generated_sids, log_probs = model_unwrapped.generate_method_a(batch)

        # Shape checks
        actual_bs = batch.actual_batch_size
        top_k = model_unwrapped.top_k_for_generation
        assert generated_sids.shape == (
            actual_bs,
            top_k,
            num_hierarchies,
        ), f"Expected ({actual_bs}, {top_k}, {num_hierarchies}), got {generated_sids.shape}"

        assert log_probs.shape == (
            actual_bs,
            top_k,
        ), f"Expected ({actual_bs}, {top_k}), got {log_probs.shape}"

        # SIDs should be within codebook ranges
        for h in range(num_hierarchies):
            assert torch.all(generated_sids[:, :, h] >= 0)
            assert torch.all(generated_sids[:, :, h] < codebook_sizes[h])

        # Log probs should be negative (log of probabilities)
        assert torch.all(log_probs <= 0)


@pytest.mark.skipif(_SHOULD_SKIP, reason=_SKIP_REASON or "")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("kv_channels", [64])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("max_history_length", [64])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
def test_generate_method_a_vs_original(
    dtype,
    hidden_size,
    num_attention_heads,
    kv_channels,
    num_layers,
    max_history_length,
    codebook_sizes,
):
    """
    generate_method_a() and generate() should produce valid outputs for the
    same input. They use different decoders (JaggedFlashAttnBlock vs Megatron)
    with different weights, so outputs won't be identical, but both should
    produce valid SIDs and log_probs.
    """
    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(42)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=hidden_size,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_batch(
            batchsize=4,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        # Run original generate
        orig_sids, orig_probs = model_unwrapped.generate(batch)

        # Run Method A generate
        method_a_sids, method_a_probs = model_unwrapped.generate_method_a(batch)

        # Both should have valid shapes
        actual_bs = batch.actual_batch_size
        top_k = model_unwrapped.top_k_for_generation
        assert orig_sids.shape == method_a_sids.shape == (
            actual_bs, top_k, num_hierarchies
        )
        assert orig_probs.shape == method_a_probs.shape == (actual_bs, top_k)

        # Both should produce valid SIDs
        for h in range(num_hierarchies):
            assert torch.all(orig_sids[:, :, h] >= 0)
            assert torch.all(orig_sids[:, :, h] < codebook_sizes[h])
            assert torch.all(method_a_sids[:, :, h] >= 0)
            assert torch.all(method_a_sids[:, :, h] < codebook_sizes[h])
