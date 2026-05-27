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
Tests for beam_decode_attn kernel integration:
  1. BeamSearch.build_beam_topk_indices correctness.
  2. JaggedGPTLayer prefill/decode_beam smoke test.
  3. Variable-length history correctness (seqused_k masking).
  4. End-to-end generate_beam_decode through SIDGRModel.
  5. Regression guards comparing generate() / dense / jagged paths.
  6. Entry-contract tests (backend whitelist, fallback rejection, etc).
  7. padded_target_aware_causal_mask beam-isolation geometry.
"""

import argparse
import os
import sys
from typing import List

import pytest
import torch
from beam_search.beam_search import BeamSearch

# Import jagged_flash_attn_block directly to avoid model/__init__.py
# which pulls in heavy dependencies (dynamicemb, megatron, torchrec).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from jagged_flash_attn_block import (  # noqa: E402
    JaggedGPTLayer,
    _beam_decode_attn_reference,
    _get_beam_decode_attn,
)

sys.path.pop(0)


# Heavy imports (require the full distributed/torchrec stack: commons,
# dynamicemb, megatron, torchrec) are deferred inside fixtures/tests so the
# lightweight unit tests above can run without them. Tests using these are
# marked with `_REQUIRES_E2E_STACK` and skip gracefully when imports fail.
def _try_import_e2e_deps():
    try:
        import commons.utils as init  # noqa
        from commons.checkpoint import get_unwrapped_module  # noqa
        from commons.datasets.gpt_sid_batch import FeatureConfig, GPTSIDBatch  # noqa
        from commons.modules.embedding import ShardedEmbeddingConfig  # noqa
        from commons.ops.length_to_offsets import length_to_complete_offsets  # noqa
        from tests.test_utils import create_sid_gr_model_and_optimizer  # noqa

        return {
            "init": init,
            "get_unwrapped_module": get_unwrapped_module,
            "FeatureConfig": FeatureConfig,
            "GPTSIDBatch": GPTSIDBatch,
            "ShardedEmbeddingConfig": ShardedEmbeddingConfig,
            "length_to_complete_offsets": length_to_complete_offsets,
            "create_sid_gr_model_and_optimizer": create_sid_gr_model_and_optimizer,
        }
    except (ImportError, ModuleNotFoundError) as e:
        return {"_error": str(e)}


_E2E_DEPS = _try_import_e2e_deps()
_E2E_AVAILABLE = "_error" not in _E2E_DEPS
_E2E_SKIP_REASON = (
    f"E2E deps unavailable (need full Docker build): {_E2E_DEPS.get('_error', '')}"
)


# Some paths still go through `flash_attn.cute` (target-aware arbitrary
# mask in baseline `generate()` and the Mode-3 dense prefill fallback),
# which asserts SM90+. Tests that exercise either are gated on this.
def _is_sm90_or_above() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


_SM90_AVAILABLE = _is_sm90_or_above()
_SM90_SKIP_REASON = (
    "requires SM90+ (cute FA arbitrary-mask path / dense Mode-3 prefill); "
    "current device compute capability < 9.0"
)


# ---------------------------------------------------------------------------
# Test: BeamSearch.build_beam_topk_indices
# ---------------------------------------------------------------------------
class TestBuildBeamTopkIndices:
    def test_step0_self_only(self):
        """At decode step 0, topk should be self-pointers: [0..W-1]."""
        W = 4
        bs = BeamSearch(W, 3, [10, 10, 10], record_history=True)
        log_probs = torch.randn(2, 1, 10, device="cuda")
        bs.propagate(log_probs)

        topk = bs.build_beam_topk_indices(decode_step=0, num_heads=4)
        assert topk.shape == (2, 1, 4, 1, W)
        # At step 0, each beam w should point to 0*W + w = w
        for w in range(W):
            assert (topk[:, 0, :, 0, w] == w).all()

    def test_step1_parent_traced(self):
        """At decode step 1, ancestors at step 0 should match parent_indices[1]."""
        W = 3
        B = 2
        bs = BeamSearch(W, 3, [10, 10, 10], record_history=True)

        # Step 0
        log_probs0 = torch.randn(B, 1, 10, device="cuda")
        bs.propagate(log_probs0)

        # Step 1
        log_probs1 = torch.randn(B, W, 10, device="cuda")
        bs.propagate(log_probs1)

        topk = bs.build_beam_topk_indices(decode_step=1, num_heads=4)
        assert topk.shape == (B, 1, 4, 2, W)

        for b in range(B):
            for w in range(W):
                # Self at step 1
                assert topk[b, 0, 0, 1, w].item() == 1 * W + w
                # Ancestor at step 0
                parent = bs.parent_indices[1][b, w].item()
                assert topk[b, 0, 0, 0, w].item() == 0 * W + parent

    def test_list_equal_widths_matches_int_path(self):
        """Pass beam_width as list [W,W,W] vs int W; topk_indices must match."""
        W = 4
        B = 2
        torch.manual_seed(123)
        bs_int = BeamSearch(W, 3, [10, 10, 10], record_history=True)
        bs_list = BeamSearch([W, W, W], 3, [10, 10, 10], record_history=True)

        topk_prev = 1
        for s in range(3):
            log_probs = torch.randn(B, topk_prev, 10, device="cuda")
            bs_int.propagate(log_probs.clone())
            bs_list.propagate(log_probs.clone())
            topk_prev = W

        for d in range(3):
            t_int = bs_int.build_beam_topk_indices(decode_step=d, num_heads=4)
            t_list = bs_list.build_beam_topk_indices(decode_step=d, num_heads=4)
            assert torch.equal(t_int, t_list), (
                f"step {d}: int and list-with-equal-values disagree:\n"
                f"  int:  {t_int.tolist()}\n"
                f"  list: {t_list.tolist()}"
            )

    def test_nonuniform_widths_accepted_at_init(self):
        """BeamSearch itself accepts non-uniform widths; the kernel-using
        path (SIDGRModel.generate_beam_decode) is what enforces uniformity.
        """
        bs = BeamSearch([3, 5, 7], 3, [10, 10, 10])
        assert bs.beam_widths == [3, 5, 7]

    def test_cumsum_offsets_with_nonuniform_widths(self):
        """Verify build_beam_topk_indices uses cumulative offsets, not s*W_d.

        BeamSearch itself accepts non-uniform widths; the kernel-using
        path (SIDGRModel.generate_beam_decode) is what enforces uniform
        widths. This is a math check on the indexing logic — non-uniform
        widths can't currently be passed end-to-end through the kernel.
        """
        beam_widths = [3, 5, 7]
        codebook_sizes = [50, 50, 50]
        bs = BeamSearch(beam_widths, 3, codebook_sizes, record_history=True)

        B = 2
        topk_prev = 1
        for s, W in enumerate(beam_widths):
            # Simulate propagate with the right shapes by direct injection.
            log_probs = torch.randn(B, topk_prev, codebook_sizes[s], device="cuda")
            # We need to temporarily set beam_widths[s] correctly for the
            # propagate call — propagate uses beam_widths[step] for topk_this_step.
            bs.propagate(log_probs)
            topk_prev = W

        # Expected step offsets: cumsum([3, 5, 7]) = [0, 3, 8]
        expected_offsets = [0, 3, 8]
        for d in range(3):
            topk = bs.build_beam_topk_indices(decode_step=d, num_heads=2)
            W_d = beam_widths[d]
            # Self entries at step d should be expected_offsets[d] + w.
            for w in range(W_d):
                assert (topk[:, 0, :, d, w] == expected_offsets[d] + w).all(), (
                    f"step d={d}, w={w}: self index should be "
                    f"{expected_offsets[d] + w}, got {topk[:, 0, 0, d, w].tolist()}"
                )
            # Ancestors at step s < d should be expected_offsets[s] + parent_idx
            for s in range(d):
                pos = torch.arange(W_d, device="cuda").unsqueeze(0).expand(B, -1)
                for ss in range(d, s, -1):
                    pos = torch.gather(bs.parent_indices[ss], dim=1, index=pos)
                expected = expected_offsets[s] + pos
                got = topk[:, 0, 0, s, :]
                assert torch.equal(got, expected.to(torch.int32)), (
                    f"step d={d} s={s}: ancestors mismatch.\n"
                    f"  expected: {expected.tolist()}\n"
                    f"  got:      {got.tolist()}"
                )

    @pytest.mark.parametrize("num_hierarchies", [2, 3, 4])
    def test_shape_and_range(self, num_hierarchies):
        """Verify topk_indices shape and value range."""
        W = 3
        B = 2
        num_heads = 4
        bs = BeamSearch(W, num_hierarchies, [10] * num_hierarchies, record_history=True)

        topk_prev = 1
        for s in range(num_hierarchies):
            log_probs = torch.randn(B, topk_prev, 10, device="cuda")
            bs.propagate(log_probs)
            topk_prev = W

        for d in range(num_hierarchies):
            topk = bs.build_beam_topk_indices(decode_step=d, num_heads=num_heads)
            decode_nums = d + 1
            assert topk.shape == (B, 1, num_heads, decode_nums, W)
            # All indices should be in [0, decode_nums * W)
            assert (topk >= 0).all()
            assert (topk < decode_nums * W).all()
            # Self at last position should be d*W + w
            for w in range(W):
                assert (topk[:, 0, :, d, w] == d * W + w).all()


# ---------------------------------------------------------------------------
# Test: BeamSearch parent_indices bug fix
# ---------------------------------------------------------------------------
class TestBeamSearchParentIndices:
    def test_parent_indices_stored(self):
        """parent_indices should be recorded at each propagate step."""
        bs = BeamSearch(2, 3, [10, 10, 10])
        topk_prev = 1
        for _ in range(3):
            log_probs = torch.randn(1, topk_prev, 10, device="cuda")
            bs.propagate(log_probs)
            topk_prev = 2
        assert len(bs.parent_indices) == 3

    def test_reset_clears_parent_indices(self):
        bs = BeamSearch(2, 2, [10, 10])
        topk_prev = 1
        for _ in range(2):
            log_probs = torch.randn(1, topk_prev, 10, device="cuda")
            bs.propagate(log_probs)
            topk_prev = 2
        bs.reset()
        assert len(bs.parent_indices) == 0

    def test_beam_width_uniform_list_accepted(self):
        """beam_width as a uniform list should be accepted (matches int path)."""
        bs = BeamSearch([2, 2, 2], 3, [10, 10, 10])
        assert bs.beam_widths == [2, 2, 2]


# ---------------------------------------------------------------------------
# Test: JaggedGPTLayer prefill & decode_beam (smoke)
# ---------------------------------------------------------------------------
class TestJaggedGPTLayerPrefillDecode:
    @pytest.fixture
    def layer(self):
        return (
            JaggedGPTLayer(
                hidden_size=64,
                num_attention_heads=4,
                ffn_hidden_size=128,
            )
            .cuda()
            .bfloat16()
        )

    @pytest.mark.skipif(not _SM90_AVAILABLE, reason=_SM90_SKIP_REASON)
    def test_prefill_returns_kv_cache(self, layer):
        B, S, D = 2, 16, 64
        x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        out, (k_cache, v_cache) = layer.prefill(x)
        assert out.shape == (B, S, D)
        assert k_cache.shape == (B, S, 4, 16)  # [B, S, H, D_head]
        assert v_cache.shape == k_cache.shape

    def test_decode_beam_smoke(self, layer):
        """decode_beam should run without errors (requires beam_decode_attn)."""
        # Uses reference fallback if CuTe kernel not available

        B, W, S_ctx, H, D_head = 2, 3, 16, 4, 16
        D = H * D_head

        x = torch.randn(B, W, D, device="cuda", dtype=torch.bfloat16)
        k_ctx = torch.randn(B, S_ctx, H, D_head, device="cuda", dtype=torch.bfloat16)
        v_ctx = torch.randn(B, S_ctx, H, D_head, device="cuda", dtype=torch.bfloat16)

        # First decode step: no previous beam KV
        topk = (
            torch.arange(W, device="cuda")
            .view(1, 1, 1, 1, W)
            .expand(B, 1, H, 1, W)
            .to(torch.int32)
        )

        out, (k_new, v_new) = layer.decode_beam(
            x,
            k_ctx,
            v_ctx,
            k_beam=None,
            v_beam=None,
            topk_indices=topk,
            decode_nums=1,
        )
        assert out.shape == (B, W, D)
        assert k_new.shape == (B, W, H, D_head)


# ---------------------------------------------------------------------------
# Test: variable-length history padding (P0 #1: padding KV must be masked)
# ---------------------------------------------------------------------------
class TestVariableLengthHistory:
    """Verify padding K/V positions don't contaminate attention output.

    Build two scenarios with the SAME valid history content but different
    padding lengths; outputs should match within tolerance.
    """

    def test_padding_does_not_affect_output(self):
        """Same valid hist + different padding length → same valid attention output.

        Verifies the kernel's seqused_k arg correctly masks padding K positions
        so they don't contaminate the context attention output.
        """
        B, W, H, D_head = 2, 3, 4, 64
        valid_seqlen = 8
        torch.manual_seed(123)

        valid_k = torch.randn(
            B, valid_seqlen, H, D_head, device="cuda", dtype=torch.bfloat16
        )
        valid_v = torch.randn(
            B, valid_seqlen, H, D_head, device="cuda", dtype=torch.bfloat16
        )
        q = torch.randn(B, 1, W, H, D_head, device="cuda", dtype=torch.bfloat16)

        decode_nums = 1
        k_beam = torch.randn(
            B, decode_nums * W, H, D_head, device="cuda", dtype=torch.bfloat16
        )
        v_beam = torch.randn(
            B, decode_nums * W, H, D_head, device="cuda", dtype=torch.bfloat16
        )
        topk = (
            torch.arange(W, device="cuda")
            .view(1, 1, 1, 1, W)
            .expand(B, 1, H, 1, W)
            .to(torch.int32)
        )

        # Run 1: exactly valid_seqlen tokens (no padding)
        seqused_a = torch.tensor(
            [valid_seqlen, valid_seqlen], device="cuda", dtype=torch.int32
        )

        # Run 2: padded to 16, extra positions contain garbage values
        max_padded = 16
        k_padded = torch.randn(
            B, max_padded, H, D_head, device="cuda", dtype=torch.bfloat16
        )
        v_padded = torch.randn(
            B, max_padded, H, D_head, device="cuda", dtype=torch.bfloat16
        )
        k_padded[:, :valid_seqlen] = valid_k
        v_padded[:, :valid_seqlen] = valid_v
        seqused_b = torch.tensor(
            [valid_seqlen, valid_seqlen], device="cuda", dtype=torch.int32
        )

        kernel_fn = _get_beam_decode_attn()
        if kernel_fn is _beam_decode_attn_reference:
            pytest.skip("CuTe kernel needed for seqused_k masking")
        out_a, _ = kernel_fn(
            q,
            valid_k,
            valid_v,
            k_beam,
            v_beam,
            topk,
            decode_nums,
            backend="3kernel",
            seqused_k=seqused_a,
        )
        out_b, _ = kernel_fn(
            q,
            k_padded,
            v_padded,
            k_beam,
            v_beam,
            topk,
            decode_nums,
            backend="3kernel",
            seqused_k=seqused_b,
        )

        diff = (out_a.float() - out_b.float()).abs().max().item()
        assert diff < 0.05, (
            f"Padding contaminates output: diff={diff:.4f}. "
            f"seqused_k masking is broken."
        )


# ---------------------------------------------------------------------------
# Test: end-to-end generate_beam_decode through SIDGRModel
# ---------------------------------------------------------------------------
def _generate_random_batch(
    batchsize: int,
    max_history_length: int,
    codebook_sizes: List[int],
    history_feature_name: str,
    candidate_feature_name: str,
):
    """Build a random GPTSIDBatch (history-only, candidate length 1)."""
    FeatureConfig = _E2E_DEPS["FeatureConfig"]
    GPTSIDBatch = _E2E_DEPS["GPTSIDBatch"]
    length_to_complete_offsets = _E2E_DEPS["length_to_complete_offsets"]

    num_hierarchies = len(codebook_sizes)
    cum_sum = length_to_complete_offsets(torch.tensor(codebook_sizes))
    raw_hist_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
    feature_configs = [
        FeatureConfig(
            feature_names=raw_hist_names,
            max_item_ids=cum_sum[1:],
            min_item_ids=cum_sum[:-1],
            max_sequence_length=max_history_length,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=raw_cand_names,
            max_item_ids=cum_sum[1:],
            min_item_ids=cum_sum[:-1],
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


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("kv_channels", [64])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("max_history_length", [64])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
@pytest.mark.parametrize("batchsize", [4])
def test_generate_beam_decode_e2e(
    dtype,
    hidden_size,
    num_attention_heads,
    kv_channels,
    num_layers,
    max_history_length,
    codebook_sizes,
    batchsize,
):
    """End-to-end: generate_beam_decode runs through full SIDGRModel."""
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

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
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=batchsize,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        with torch.no_grad():
            generated_sids, log_probs = model_unwrapped.generate_beam_decode(batch)

        actual_bs = batch.actual_batch_size
        top_k = model_unwrapped.top_k_for_generation

        # Shape checks
        assert generated_sids.shape == (
            actual_bs,
            top_k,
            num_hierarchies,
        ), f"got {generated_sids.shape}"
        assert log_probs.shape == (actual_bs, top_k)

        # SIDs must be within their codebook ranges
        for h in range(num_hierarchies):
            assert (generated_sids[:, :, h] >= 0).all()
            assert (generated_sids[:, :, h] < codebook_sizes[h]).all()

        # log_probs are accumulated log_softmax → should be ≤ 0
        assert (log_probs <= 1e-3).all()  # small slack for fp32 rounding


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
@pytest.mark.skipif(not _SM90_AVAILABLE, reason=_SM90_SKIP_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("max_history_length", [32, 128])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
@pytest.mark.parametrize("batchsize", [4])
def test_generate_vs_generate_beam_decode_regression_guard(
    dtype,
    max_history_length,
    num_layers,
    codebook_sizes,
    batchsize,
):
    """Regression guard between generate() and generate_beam_decode().

    Both paths implement the same beam-isolated attention semantics by
    different means (full-sequence rerun with arbitrary mask vs
    prefill+incremental decode with topk_indices). They should produce
    very close results, but bf16 attention's per-layer rounding plus
    beam-search's topk decision boundary make bit-exact match impossible.

    This is a regression GUARD — it catches a path going significantly off
    spec. It is NOT a mathematical equivalence proof; for that, see the
    kernel-level reference oracle in
    `corelib/gr_decode_atten/tests/test_fwd.py`, which compares the CuTe
    kernel against a fp32 PyTorch reference at the attention-call level.

    Thresholds:
      - top-1 SID per sample matches exactly (small bf16 noise should not
        flip the argmax of accumulated log-probs)
      - per-position |log_prob delta| < 0.15
      - top-K beam SID set overlap >= 70%
    """
    hidden_size = 256
    num_attention_heads = 4
    kv_channels = 64

    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(7)

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
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=batchsize,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        with torch.no_grad():
            sids_a, lp_a = model_unwrapped.generate(batch)
            sids_b, lp_b = model_unwrapped.generate_beam_decode(batch)

        assert sids_a.shape == sids_b.shape
        assert lp_a.shape == lp_b.shape

        # Both paths use beam-isolating attention (generate() builds
        # padded_target_aware_causal_mask which is then converted to the
        # jagged arbitrary_func; generate_beam_decode uses topk_indices for
        # the same isolation). With both paths mathematically equivalent,
        # the only differences are bf16 layout/order rounding.

        # 1. Top-1 beam's full SID tuple must match (this is THE selection
        #    that downstream consumers use; tiny rounding usually cannot
        #    flip the argmax of accumulated log-probs).
        top1_a = sids_a[:, 0, :]
        top1_b = sids_b[:, 0, :]
        per_sample_match = (top1_a == top1_b).all(dim=-1)
        assert per_sample_match.all(), (
            f"Top-1 SIDs differ on samples "
            f"{(~per_sample_match).nonzero(as_tuple=True)[0].tolist()}:\n"
            f"  generate: {top1_a.tolist()}\n"
            f"  decode:   {top1_b.tolist()}"
        )

        # 2. Per-position log_prob differences are within bf16 rounding —
        #    a few times the typical attention rounding error.
        lp_diff = (lp_a - lp_b).abs().max().item()
        assert lp_diff < 0.15, (
            f"log_probs differ by {lp_diff:.4f} (limit 0.15):\n"
            f"  generate: {lp_a.tolist()}\n"
            f"  decode:   {lp_b.tolist()}"
        )

        # 3. The top-K SIDs as sets must overlap by ≥ 70% per sample
        #    (some beams at lower ranks may swap positions due to bf16
        #    rounding when their log-probs are within ~lp_diff of each other).
        top_k = sids_a.shape[1]

        def _to_set_per_sample(sids):
            return [
                {tuple(sids[b, k].tolist()) for k in range(top_k)}
                for b in range(sids.shape[0])
            ]

        sets_a = _to_set_per_sample(sids_a)
        sets_b = _to_set_per_sample(sids_b)
        for b, (sa, sb) in enumerate(zip(sets_a, sets_b)):
            overlap = len(sa & sb) / len(sa)
            assert overlap >= 0.7, (
                f"Sample {b}: top-{top_k} beam overlap {overlap*100:.0f}% "
                f"is below 70% threshold.\n"
                f"  generate beams: {sorted(sa)}\n"
                f"  decode beams:   {sorted(sb)}"
            )


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
@pytest.mark.skipif(not _SM90_AVAILABLE, reason=_SM90_SKIP_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("max_history_length", [32, 128])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
@pytest.mark.parametrize("batchsize", [4])
def test_generate_beam_decode_jagged_kv_matches_dense(
    dtype,
    max_history_length,
    num_layers,
    codebook_sizes,
    batchsize,
):
    """Cross-check use_jagged_kv=True vs the default dense+seqused_k path.

    Both should produce the same top-1 SID tuple per sample. Lower-ranked
    beams may swap due to bf16 reduction-order differences (jagged prefill
    runs FA with arbitrary_func instead of causal=True), so we use the same
    relaxed thresholds as the generate() regression guard.
    """
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

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
        dim=256,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=256,
            num_attention_heads=4,
            kv_channels=64,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=batchsize,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        with torch.no_grad():
            sids_dense, lp_dense = model_unwrapped.generate_beam_decode(
                batch, use_jagged_kv=False
            )
            sids_jagged, lp_jagged = model_unwrapped.generate_beam_decode(
                batch, use_jagged_kv=True
            )

        # Top-1 must match exactly.
        top1_dense = sids_dense[:, 0, :]
        top1_jagged = sids_jagged[:, 0, :]
        per_sample_match = (top1_dense == top1_jagged).all(dim=-1)
        assert per_sample_match.all(), (
            f"Top-1 SIDs differ on samples "
            f"{(~per_sample_match).nonzero(as_tuple=True)[0].tolist()}:\n"
            f"  dense:  {top1_dense.tolist()}\n"
            f"  jagged: {top1_jagged.tolist()}"
        )

        # Log-probs within bf16 reduction-order noise.
        lp_diff = (lp_dense - lp_jagged).abs().max().item()
        assert lp_diff < 0.15, (
            f"log_probs differ by {lp_diff:.4f} (limit 0.15):\n"
            f"  dense:  {lp_dense.tolist()}\n"
            f"  jagged: {lp_jagged.tolist()}"
        )


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
def test_use_jagged_kv_with_dsl_backend_rejected_at_entry():
    """``use_jagged_kv=True`` is meaningless under ``backend="dsl"`` (the
    fused path doesn't accept cu_seqlens_k). The combination must be
    rejected at entry, not silently downgraded to the dense path.
    """
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(42)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_sizes = [128, 128, 128]
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=256,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=torch.bfloat16,
            hidden_size=256,
            num_attention_heads=4,
            kv_channels=64,
            num_layers=2,
            num_hierarchies=3,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=4,
            max_history_length=64,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        with pytest.raises(
            ValueError, match="use_jagged_kv=True requires backend='3kernel'"
        ):
            with torch.no_grad():
                model_unwrapped.generate_beam_decode(
                    batch,
                    backend="dsl",
                    use_jagged_kv=True,
                )


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
def test_use_jagged_kv_with_reference_fallback_rejected_at_entry(monkeypatch):
    """When the real CuTe kernel isn't installed, the resolver returns
    ``_beam_decode_attn_reference`` — which has ``cu_seqlens_k`` in its
    signature for the explicit-raise contract but cannot actually run
    jagged. The capability probe must catch this at entry, not later.
    """
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    # gpt_model imports via `from .jagged_flash_attn_block import ...`,
    # which registers the module under the package-qualified name. The
    # standalone `import jagged_flash_attn_block` would land a *separate*
    # module object in sys.modules, so patching that wouldn't affect the
    # one the probe actually reads. Pull the right one out of sys.modules.
    import sys as _sys

    jfab = _sys.modules.get("model.jagged_flash_attn_block")
    assert jfab is not None, (
        "model.jagged_flash_attn_block not loaded; the test's "
        "create_sid_gr_model_and_optimizer call should have triggered the "
        "import chain"
    )

    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(42)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_sizes = [128, 128, 128]
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=256,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=torch.bfloat16,
            hidden_size=256,
            num_attention_heads=4,
            kv_channels=64,
            num_layers=2,
            num_hierarchies=3,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=4,
            max_history_length=64,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        # Force the cached resolver result to be the PyTorch reference
        # fallback, mimicking an environment without the CuTe kernel
        # installed. _get_beam_decode_attn caches at the module level on
        # the _beam_decode_attn name, so patch that directly.
        monkeypatch.setattr(
            jfab,
            "_beam_decode_attn",
            jfab._beam_decode_attn_reference,
        )

        with pytest.raises(RuntimeError, match="reference fallback does not implement"):
            with torch.no_grad():
                model_unwrapped.generate_beam_decode(
                    batch,
                    backend="3kernel",
                    use_jagged_kv=True,
                )


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
def test_use_jagged_kv_uninspectable_signature_rejected_at_entry(monkeypatch):
    """If the resolved kernel's signature cannot be inspected (e.g. a C
    extension), the cu_seqlens_k capability probe must fail closed: emit
    a clear entry-time RuntimeError instead of silently letting the call
    proceed and surfacing a deep "unexpected keyword argument" later.
    """
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    import inspect as _inspect
    import sys as _sys

    jfab = _sys.modules.get("model.jagged_flash_attn_block")
    assert jfab is not None

    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(42)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_sizes = [128, 128, 128]
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=256,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=torch.bfloat16,
            hidden_size=256,
            num_attention_heads=4,
            kv_channels=64,
            num_layers=2,
            num_hierarchies=3,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=4,
            max_history_length=64,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        # Install a kernel stub that is NOT the reference fallback (so the
        # earlier identity-based reject doesn't fire) but whose signature
        # is unreachable: builtin_function_or_method-style objects raise
        # TypeError from inspect.signature.
        def _stub_kernel(*args, **kwargs):
            raise AssertionError(
                "kernel must not be called — the probe should reject earlier"
            )

        monkeypatch.setattr(jfab, "_beam_decode_attn", _stub_kernel)

        # Force inspect.signature to raise TypeError for this stub,
        # simulating a C-extension callable. gpt_model imports inspect
        # at function-scope inside generate_beam_decode, so patch the
        # module-level binding on `inspect` itself; the filter ensures
        # other callers (pytest internals, etc.) are unaffected.
        orig_signature = _inspect.signature

        def _raising_signature(obj, *args, **kwargs):
            if obj is _stub_kernel:
                raise TypeError("simulated uninspectable signature")
            return orig_signature(obj, *args, **kwargs)

        monkeypatch.setattr(_inspect, "signature", _raising_signature)

        with pytest.raises(RuntimeError, match="signature is not inspectable"):
            with torch.no_grad():
                model_unwrapped.generate_beam_decode(
                    batch,
                    backend="3kernel",
                    use_jagged_kv=True,
                )


def test_compare_kv_modes_default_raises_on_validation_failure():
    """``run_compare_kv_modes`` must raise by default when validation
    fails. Pass ``--allow_validation_fail`` to suppress.

    We exercise the shared test_utils validator directly with mismatched
    sids/lp tensors, then replicate the post-sweep raise block without
    importing the benchmark module.
    """
    from tests.test_utils import validate_compare_outputs  # noqa: E402

    B, K, H = 2, 5, 3
    torch.manual_seed(0)
    sids_a = torch.randint(0, 100, (B, K, H), dtype=torch.int64)
    # Validator uses top-K set overlap >= 70%. Shift every beam of
    # sample 0 so the resulting overlap is 0% regardless of K — the
    # synthetic mismatch must keep failing if someone tunes K later.
    sids_b = sids_a.clone()
    sids_b[0, :, 0] = (sids_a[0, :, 0] + 1) % 100
    sids_c = sids_a.clone()

    passed, summary, overlap = validate_compare_outputs(sids_a, sids_b, sids_c)
    assert not passed, "synthetic mismatch should fail validation"
    assert "overlap" in summary
    assert overlap < 0.7

    # Drive the strict-mode raise path.
    args_strict = argparse.Namespace(allow_validation_fail=False)
    args_lenient = argparse.Namespace(allow_validation_fail=True)

    # Replicate the small post-sweep raise block from run_compare_kv_modes
    # so we test the contract without needing to spin up a model.
    def _post_sweep(args, val_fails):
        total = 1
        if val_fails:
            if not getattr(args, "allow_validation_fail", False):
                raise RuntimeError(
                    f"--compare_kv_modes validation failed on "
                    f"{len(val_fails)}/{total} configs."
                )

    with pytest.raises(RuntimeError, match="validation failed"):
        _post_sweep(args_strict, ["fake config failure"])

    # Lenient mode must NOT raise.
    _post_sweep(args_lenient, ["fake config failure"])


# ---------------------------------------------------------------------------
# Test: beam isolation (verify the attention mask actually isolates beams)
# ---------------------------------------------------------------------------
class TestBeamIsolationMask:
    """Direct unit tests on padded_target_aware_causal_mask construction.

    Verifies that for sequence layout [history, beam0_target, beam1_target, ...]:
      - Each beam region attends only to history + its own (causal) region.
      - Different beam regions are mutually invisible.
    """

    def test_target_regions_are_isolated(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
        try:
            from attention_mask import padded_target_aware_causal_mask
        finally:
            sys.path.pop(0)

        history_seqlen = torch.tensor([4, 4], device="cuda")
        max_history_seqlen = 6
        num_target_region = 3  # 3 beams
        target_max_seqlen_per_region = 2
        mask = padded_target_aware_causal_mask(
            history_seqlen,
            max_history_seqlen,
            num_target_region,
            target_max_seqlen_per_region,
        )
        # padded_target_aware_causal_mask returns ~mask (invalid mask convention)
        valid = ~mask  # True = can attend
        # Layout: [history(0..max_history_seqlen-1), region0(6..7), region1(8..9), region2(10..11)]
        for b_a in range(num_target_region):
            for b_b in range(num_target_region):
                if b_a == b_b:
                    continue
                a_start = max_history_seqlen + b_a * target_max_seqlen_per_region
                a_end = a_start + target_max_seqlen_per_region
                b_start = max_history_seqlen + b_b * target_max_seqlen_per_region
                b_end = b_start + target_max_seqlen_per_region
                # No row in beam-a should be able to attend to any column in beam-b
                cross = valid[:, 0, a_start:a_end, b_start:b_end]
                assert (
                    not cross.any()
                ), f"beam {b_a} can see beam {b_b}: {cross.any(dim=-1)}"

    def test_target_regions_attend_history(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
        try:
            from attention_mask import padded_target_aware_causal_mask
        finally:
            sys.path.pop(0)

        history_seqlen = torch.tensor([4], device="cuda")
        max_history_seqlen = 4
        num_target_region = 2
        target_max_seqlen_per_region = 2
        mask = padded_target_aware_causal_mask(
            history_seqlen,
            max_history_seqlen,
            num_target_region,
            target_max_seqlen_per_region,
        )
        valid = ~mask
        # All target tokens (positions max_history_seqlen..) should attend to all history (0..3)
        for region in range(num_target_region):
            for offset in range(target_max_seqlen_per_region):
                target_pos = (
                    max_history_seqlen + region * target_max_seqlen_per_region + offset
                )
                hist_visible = valid[0, 0, target_pos, :max_history_seqlen]
                assert (
                    hist_visible.all()
                ), f"target at pos {target_pos} can't see all history: {hist_visible.tolist()}"


def _arbitrary_func_to_dense_jagged(
    af: torch.Tensor, total_tokens: int
) -> torch.Tensor:
    """Expand a flattened (B=1) arbitrary_func tensor to a dense
    [total_tokens, total_tokens] bool mask using the interval semantics:
        valid(q, k) = (k < F0[q]) OR (F1[q] <= k < F2[q]) OR ...
    """
    assert af.dim() == 4 and af.shape[0] == 1 and af.shape[1] == 1
    n_func = af.shape[2]
    device = af.device
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
    kv_idx = torch.arange(total_tokens, device=device)
    af2d = af[0, 0, :, :total_tokens]  # [n_func, total_tokens]
    for q in range(total_tokens):
        f0 = af2d[0, q].item()
        row = kv_idx < f0
        for iv in range((n_func - 1) // 2):
            f_start = af2d[2 * iv + 1, q].item()
            f_end = af2d[2 * iv + 2, q].item()
            row = row | ((kv_idx >= f_start) & (kv_idx < f_end))
        mask[q] = row
    return mask


@pytest.mark.parametrize("history_seqlens", [[4, 4], [3, 5], [1, 6, 4]])
@pytest.mark.parametrize("num_target_region", [0, 2, 3])
@pytest.mark.parametrize("candidate_length", [1, 3])
def test_jagged_target_aware_builder_matches_dense(
    history_seqlens, num_target_region, candidate_length
):
    """Vectorised builder must produce the same flattened mask as the
    dense path (padded_target_aware_causal_mask + dense_to_jagged).

    Covers variable-length history, beam_width 0/2/3, candidate_length 1/3.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
    try:
        from attention_mask import (
            build_jagged_target_aware_arbitrary_func,
            dense_mask_to_jagged_arbitrary_func,
            padded_target_aware_causal_mask,
        )
    finally:
        sys.path.pop(0)

    device = "cuda"
    len(history_seqlens)
    cl = candidate_length
    W = num_target_region

    # Per-sample concatenated layout: [hist_b, beam0(cl), ..., beam_{W-1}(cl)]
    # Total per sample = hist_seqlen[b] + W * cl.
    sample_lens = [hl + W * cl for hl in history_seqlens]
    offsets = torch.tensor(
        [0] + list(torch.tensor(sample_lens).cumsum(0).tolist()),
        device=device,
        dtype=torch.long,
    )
    total_tokens = int(offsets[-1].item())
    history_t = torch.tensor(history_seqlens, device=device, dtype=torch.long)

    # Vectorised path
    af_fast = build_jagged_target_aware_arbitrary_func(
        history_seqlen=history_t,
        num_target_region=W,
        target_max_seqlen_per_region=cl,
        offsets=offsets,
        total_tokens=total_tokens,
    )

    # Reference: build dense mask, then convert via dense_to_jagged.
    # padded_target_aware_causal_mask uses a uniform max_history_seqlen,
    # so we pad each sample to max(history_seqlens) + W*cl.
    max_hist = max(history_seqlens)
    dense_mask = padded_target_aware_causal_mask(
        history_t,
        max_hist,
        W,
        cl,
    )
    valid_dense = ~dense_mask  # [B, 1, max_hist+W*cl, max_hist+W*cl]
    af_ref = dense_mask_to_jagged_arbitrary_func(
        valid_dense,
        offsets,
        total_tokens,
    )

    fast_dense = _arbitrary_func_to_dense_jagged(af_fast, total_tokens)
    ref_dense = _arbitrary_func_to_dense_jagged(af_ref, total_tokens)
    assert torch.equal(fast_dense, ref_dense), (
        f"vectorised builder mask differs from dense oracle.\n"
        f"  diff at: {(fast_dense != ref_dense).nonzero(as_tuple=False).tolist()}"
    )


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
@pytest.mark.skipif(not _SM90_AVAILABLE, reason=_SM90_SKIP_REASON)
def test_generate_is_deterministic():
    """generate() called twice on the same batch must produce identical SIDs.

    A determinism prerequisite for any beam-level perturbation argument
    (which would otherwise see noise from non-determinism). This is NOT
    an actual perturbation invariance test — for that, see the
    TestBeamIsolationMask geometric checks plus the equivalence-vs-decode
    regression guard.
    """
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(99)

    cs = [128, 128, 128]
    hidden = 256
    cfg = ShardedEmbeddingConfig(
        feature_names=["hist_sids", "cand_sids"],
        table_name="codebook",
        vocab_size=sum(cs),
        dim=hidden,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, opt = create_sid_gr_model_and_optimizer(
            dtype=torch.bfloat16,
            hidden_size=hidden,
            num_attention_heads=4,
            kv_channels=64,
            num_layers=2,
            num_hierarchies=3,
            codebook_embedding_config=cfg,
            codebook_sizes=cs,
            use_jagged_flash_attn=True,
        )
        opt.reload_model_params()
        m = get_unwrapped_module(model)
        m.eval()
        batch = _generate_random_batch(2, 32, cs, "hist_sids", "cand_sids")
        batch.to(torch.cuda.current_device())

        # Run generate twice on the same batch — should be deterministic.
        with torch.no_grad():
            sids_a, _ = m.generate(batch)
            sids_b, _ = m.generate(batch)
        assert torch.equal(
            sids_a, sids_b
        ), "generate is non-deterministic, perturbation test would be invalid"

        # Test: replace each beam's first-step code with a different code and
        # verify the other beams' final SIDs don't change. We simulate this
        # by re-running generate but injecting a different selection at step 0
        # for ONE specific beam, then comparing the OTHER beams' outputs.
        # For simplicity here we do a structural check: the mask used in
        # generate() must be padded_target_aware_causal_mask (verified by the
        # mask unit test above), and we already test that
        # generate_beam_decode produces equivalent results — which only works
        # under proper beam isolation. So this test is a smoke check that
        # generate() itself is deterministic, which is required for the
        # perturbation argument to even make sense.
        # If beam isolation is broken, generate is still deterministic but
        # the per-beam outputs would be cross-contaminated; that case is
        # caught by the structural mask unit test above and by the
        # equivalence test (which compares to the truly isolated decode path).
