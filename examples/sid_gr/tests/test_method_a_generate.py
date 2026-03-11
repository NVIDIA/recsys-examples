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
Tests for Method A (Incremental Append) beam search:
  1. Beam search ancestor tracking.
  2. Dense mask structure for Method A.
  3. arbitrary_func matches the dense reference mask.
"""

import os
import sys

import pytest
import torch

from beam_search.beam_search import BeamSearch

# Import attention_mask directly to avoid model/__init__.py which pulls in
# heavy dependencies (dynamicemb, megatron, torchrec).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from attention_mask import (
    build_incremental_append_arbitrary_func,
    build_incremental_append_dense_mask,
)
sys.path.pop(0)


# ---------------------------------------------------------------------------
# Helper: expand arbitrary_func to dense bool mask for comparison
# ---------------------------------------------------------------------------
def arbitrary_func_to_dense(
    af: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
) -> torch.Tensor:
    """
    Expand an arbitrary_func tensor [B, 1, n_func, seqlen_q+256] into a dense
    [B, seqlen_q, seqlen_k] bool mask using the interval semantics:
      valid(q, k) = (k < F0[q]) OR (F1[q] <= k < F2[q]) OR ...
    """
    B = af.shape[0]
    n_func = af.shape[2]
    mask = torch.zeros(B, seqlen_q, seqlen_k, dtype=torch.bool, device=af.device)

    kv_idx = torch.arange(seqlen_k, device=af.device)

    for b in range(B):
        for q in range(seqlen_q):
            f0 = af[b, 0, 0, q].item()
            row_mask = kv_idx < f0
            for interval in range(n_func // 2):
                f_start = af[b, 0, 2 * interval + 1, q].item()
                f_end = af[b, 0, 2 * interval + 2, q].item()
                row_mask = row_mask | ((kv_idx >= f_start) & (kv_idx < f_end))
            mask[b, q] = row_mask

    return mask


def _run_beam_search(batch_size, beam_width, num_hierarchies, codebook_sizes):
    """Run a full beam search with random logits and return the BeamSearch object."""
    bs = BeamSearch(beam_width, num_hierarchies, codebook_sizes, record_history=True)
    topk_prev = 1
    for step in range(num_hierarchies):
        log_probs = torch.randn(
            batch_size, topk_prev, codebook_sizes[step], device="cuda"
        )
        bs.propagate(log_probs)
        topk_prev = beam_width
    return bs


# ---------------------------------------------------------------------------
# Test: beam search ancestor tracking
# ---------------------------------------------------------------------------
class TestBeamSearchAncestorTracking:
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("beam_width", [2, 4])
    def test_ancestor_positions_shape(self, batch_size, beam_width):
        """Verify shape and range of ancestor positions."""
        num_h = 3
        codebook_sizes = [10] * num_h
        bs = _run_beam_search(batch_size, beam_width, num_h, codebook_sizes)

        hist_len = torch.full((batch_size,), 6, device="cuda", dtype=torch.long)
        anc_pos = bs.get_ancestor_positions(hist_len)

        assert anc_pos is not None
        assert anc_pos.shape == (batch_size, beam_width, num_h)

        for s in range(num_h):
            offset = sum(bs.beam_widths[:s])
            for b_idx in range(batch_size):
                hl = hist_len[b_idx].item()
                positions = anc_pos[b_idx, :, s]
                assert torch.all(positions >= hl + offset)
                assert torch.all(positions < hl + offset + bs.beam_widths[s])

    def test_ancestor_positions_step_zero(self):
        """At step 0, get_ancestor_positions should return None."""
        bs = BeamSearch(2, 3, [10, 10, 10], record_history=True)
        assert bs.get_ancestor_positions(torch.tensor([6], device="cuda")) is None

    def test_parent_indices_stored(self):
        """Verify parent_indices are stored during propagate."""
        bs = _run_beam_search(1, 2, 3, [10, 10, 10])
        assert len(bs.parent_indices) == 3
        for pi in bs.parent_indices:
            assert pi.shape[0] == 1
            assert pi.shape[1] == 2

    def test_reset_clears_parent_indices(self):
        """Verify reset() clears parent_indices."""
        bs = _run_beam_search(1, 2, 2, [10, 10])
        assert len(bs.parent_indices) == 2
        bs.reset()
        assert len(bs.parent_indices) == 0

    def test_ancestor_self_position(self):
        """The last entry in ancestor_positions should be the token's own position."""
        bs = _run_beam_search(1, 2, 3, [10, 10, 10])
        hist_len = torch.tensor([6], device="cuda", dtype=torch.long)
        anc_pos = bs.get_ancestor_positions(hist_len)

        last_step_offset = sum(bs.beam_widths[:2])
        for b in range(2):
            expected_self_pos = 6 + last_step_offset + b
            assert anc_pos[0, b, 2].item() == expected_self_pos


# ---------------------------------------------------------------------------
# Test: dense mask structure for Method A
# ---------------------------------------------------------------------------
class TestMethodADenseMask:
    def test_step0_pure_causal(self):
        """At step 0, mask is pure causal over history."""
        B, hist_len, max_hist = 1, 4, 4
        history_seqlens = torch.tensor([hist_len], device="cuda")

        mask = build_incremental_append_dense_mask(
            history_seqlens, max_hist, current_step=0,
            beam_widths=[2], ancestor_positions=None,
        )
        assert mask.shape == (B, max_hist, max_hist)
        expected = torch.tril(torch.ones(hist_len, hist_len, dtype=torch.bool, device="cuda"))
        assert torch.equal(mask[0, :hist_len, :hist_len], expected)

    def test_step1_beam_isolation(self):
        """At step 1, step-0 beams should not see each other."""
        B, hist_len, max_hist = 1, 4, 4
        beam_width = 2
        history_seqlens = torch.tensor([hist_len], device="cuda")

        bs = _run_beam_search(B, beam_width, 1, [10])
        anc_pos = bs.get_ancestor_positions(history_seqlens)

        mask = build_incremental_append_dense_mask(
            history_seqlens, max_hist, current_step=1,
            beam_widths=[beam_width, beam_width],
            ancestor_positions=anc_pos,
        )
        N = max_hist + beam_width
        assert mask.shape == (B, N, N)

        beam_a_pos = max_hist
        beam_b_pos = max_hist + 1

        # Beams should NOT see each other
        assert not mask[0, beam_a_pos, beam_b_pos].item()
        assert not mask[0, beam_b_pos, beam_a_pos].item()

        # Both should see self
        assert mask[0, beam_a_pos, beam_a_pos].item()
        assert mask[0, beam_b_pos, beam_b_pos].item()

        # Both should see all history
        assert torch.all(mask[0, beam_a_pos, :hist_len])
        assert torch.all(mask[0, beam_b_pos, :hist_len])

    def test_tree_ancestry_step2(self):
        """At step 2, each beam sees its ancestor chain but not other branches."""
        B, hist_len, max_hist = 1, 4, 4
        beam_width = 2
        history_seqlens = torch.tensor([hist_len], device="cuda")

        bs = _run_beam_search(B, beam_width, 2, [10, 10])
        anc_pos = bs.get_ancestor_positions(history_seqlens)

        mask = build_incremental_append_dense_mask(
            history_seqlens, max_hist, current_step=2,
            beam_widths=[beam_width, beam_width, beam_width],
            ancestor_positions=anc_pos,
        )
        N = max_hist + beam_width * 2
        assert mask.shape == (B, N, N)

        step1_start = max_hist + beam_width
        for b in range(beam_width):
            token_pos = step1_start + b
            # Should see all history
            assert torch.all(mask[0, token_pos, :hist_len])
            # Should see self
            assert mask[0, token_pos, token_pos].item()
            # Should see its step-0 ancestor
            step0_ancestor = anc_pos[0, b, 0].item()
            assert mask[0, token_pos, step0_ancestor].item()

    def test_padding_handled(self):
        """Samples with different history lengths should have correct padding."""
        B = 2
        history_seqlens = torch.tensor([4, 2], device="cuda")
        max_hist = 4

        mask = build_incremental_append_dense_mask(
            history_seqlens, max_hist, current_step=0,
            beam_widths=[2], ancestor_positions=None,
        )
        # Sample 0: 4 history tokens
        assert mask[0, 3, 3].item()  # pos 3 sees self
        # Sample 1: only 2 history tokens
        assert mask[1, 1, 1].item()  # pos 1 sees self
        assert not mask[1, 2, 2].item()  # pos 2 is padding, all zeros
        assert not mask[1, 3, 3].item()  # pos 3 is padding


# ---------------------------------------------------------------------------
# Test: arbitrary_func matches dense mask for last-step tokens
# ---------------------------------------------------------------------------
class TestArbitraryFuncMatchesDense:
    @pytest.mark.parametrize("beam_width", [2, 3])
    @pytest.mark.parametrize("num_hierarchies", [1, 2, 3])
    def test_last_step_tokens_match(self, beam_width, num_hierarchies):
        """
        The arbitrary_func encoding should produce the same mask as the dense
        reference for the last step's tokens (the logit-producing tokens).
        """
        B = 2
        hist_len = 6
        max_hist = hist_len
        codebook_sizes = [10] * num_hierarchies
        history_seqlens = torch.tensor([hist_len] * B, device="cuda")
        beam_widths = [beam_width] * num_hierarchies

        bs = _run_beam_search(B, beam_width, num_hierarchies, codebook_sizes)

        # Test mask at the point of predicting the LAST hierarchy
        test_step = num_hierarchies
        anc_pos = bs.get_ancestor_positions(history_seqlens)

        dense = build_incremental_append_dense_mask(
            history_seqlens, max_hist, current_step=test_step,
            beam_widths=beam_widths, ancestor_positions=anc_pos,
        )

        af = build_incremental_append_arbitrary_func(
            history_seqlens, max_hist, current_step=test_step,
            beam_widths=beam_widths, ancestor_positions=anc_pos,
        )

        total_gen = sum(beam_widths[:test_step])
        N = max_hist + total_gen
        af_dense = arbitrary_func_to_dense(af, N, N)

        # Check that last-step tokens match exactly
        step_start = max_hist + sum(beam_widths[: test_step - 1])
        for b_idx in range(B):
            for pos in range(step_start, step_start + beam_width):
                dense_row = dense[b_idx, pos, :N]
                af_row = af_dense[b_idx, pos, :N]
                assert torch.equal(dense_row, af_row), (
                    f"Mismatch at batch={b_idx}, pos={pos}\n"
                    f"  dense: {dense_row.int().tolist()}\n"
                    f"  af:    {af_row.int().tolist()}"
                )

    def test_step0_causal_match(self):
        """At step 0, arbitrary_func should produce a pure causal mask."""
        B = 1
        hist_len = 6
        history_seqlens = torch.tensor([hist_len], device="cuda")

        af = build_incremental_append_arbitrary_func(
            history_seqlens, hist_len, current_step=0,
            beam_widths=[2], ancestor_positions=None,
        )

        af_dense = arbitrary_func_to_dense(af, hist_len, hist_len)
        expected = torch.tril(torch.ones(hist_len, hist_len, dtype=torch.bool, device="cuda"))
        assert torch.equal(af_dense[0], expected)

    def test_history_region_matches(self):
        """History region should be identical between dense and arbitrary_func."""
        B = 1
        hist_len = 6
        max_hist = hist_len
        beam_widths = [2, 2]
        history_seqlens = torch.tensor([hist_len], device="cuda")

        bs = _run_beam_search(B, 2, 2, [10, 10])
        anc_pos = bs.get_ancestor_positions(history_seqlens)

        dense = build_incremental_append_dense_mask(
            history_seqlens, max_hist, current_step=2,
            beam_widths=beam_widths, ancestor_positions=anc_pos,
        )
        af = build_incremental_append_arbitrary_func(
            history_seqlens, max_hist, current_step=2,
            beam_widths=beam_widths, ancestor_positions=anc_pos,
        )

        N = max_hist + sum(beam_widths[:2])
        af_dense = arbitrary_func_to_dense(af, N, N)

        # History region [0:hist_len, 0:hist_len] should match
        assert torch.equal(
            dense[0, :hist_len, :hist_len],
            af_dense[0, :hist_len, :hist_len],
        )
