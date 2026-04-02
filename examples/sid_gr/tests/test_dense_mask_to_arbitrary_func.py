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
Tests for dense_mask_to_arbitrary_func(): verifies that converting a dense
[B,N,N] mask to arbitrary_func interval encoding preserves mask semantics.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from attention_mask import (
    build_jagged_causal_arbitrary_func,
    dense_mask_to_arbitrary_func,
    padded_causal_mask_with_optional_bos,
    padded_target_aware_causal_mask,
)
sys.path.pop(0)


def dense_mask_to_jagged_arbitrary_func(
    valid_mask: torch.Tensor,
    offsets: torch.Tensor,
    total_tokens: int,
    padding: int = 256,
) -> torch.Tensor:
    """
    Test utility: convert per-batch dense mask [B, N, N] to a flattened
    (B=1) arbitrary_func [1, 1, n_func, total_tokens + padding].

    Maps each row from per-batch padded coordinates to global (flattened)
    coordinates.  Used in tests to build arbitrary_func from dense masks
    so the FA path can be compared against the mcore/reference path.
    """
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    assert valid_mask.dim() == 3, f"Expected [B, N, N], got {valid_mask.shape}"

    B, N, _ = valid_mask.shape
    device = valid_mask.device

    shifted = torch.zeros_like(valid_mask)
    shifted[:, :, 1:] = valid_mask[:, :, :-1]
    starts = valid_mask & ~shifted

    ends_shifted = torch.zeros_like(valid_mask)
    ends_shifted[:, :, :-1] = valid_mask[:, :, 1:]
    ends = valid_mask & ~ends_shifted

    max_intervals = int(starts.sum(dim=-1).max().item())
    n_func = max(2 * max_intervals + 1, 3)
    if n_func % 2 == 0:
        n_func += 1

    af = torch.zeros(
        1, 1, n_func, total_tokens + padding, dtype=torch.int32, device=device
    )

    for b in range(B):
        batch_start = offsets[b].item()
        batch_end = offsets[b + 1].item()
        seq_len = batch_end - batch_start

        for local_q in range(seq_len):
            global_q = batch_start + local_q
            row = valid_mask[b, local_q, :seq_len]
            if not row.any():
                continue
            start_pos = starts[b, local_q, :seq_len].nonzero(as_tuple=False).squeeze(-1)
            end_pos = ends[b, local_q, :seq_len].nonzero(as_tuple=False).squeeze(-1) + 1
            for iv in range(len(start_pos)):
                s = start_pos[iv].item() + batch_start
                e = end_pos[iv].item() + batch_start
                af[0, 0, 2 * iv + 1, global_q] = s
                af[0, 0, 2 * iv + 2, global_q] = e

    return af


def arbitrary_func_to_dense(af, seqlen_q, seqlen_k):
    """Expand arbitrary_func back to dense [B, seqlen_q, seqlen_k] bool mask."""
    B, n_func = af.shape[0], af.shape[2]
    mask = torch.zeros(B, seqlen_q, seqlen_k, dtype=torch.bool, device=af.device)
    kv_idx = torch.arange(seqlen_k, device=af.device)
    for b in range(B):
        for q in range(seqlen_q):
            f0 = af[b, 0, 0, q].item()
            row_mask = kv_idx < f0
            for iv in range(n_func // 2):
                f_start = af[b, 0, 2 * iv + 1, q].item()
                f_end = af[b, 0, 2 * iv + 2, q].item()
                row_mask = row_mask | ((kv_idx >= f_start) & (kv_idx < f_end))
            mask[b, q] = row_mask
    return mask


class TestDenseMaskToArbitraryFunc:
    def test_causal_mask(self):
        N, B = 16, 1
        valid = torch.tril(torch.ones(B, N, N, dtype=torch.bool, device="cuda"))
        af = dense_mask_to_arbitrary_func(valid, N)
        assert torch.equal(valid, arbitrary_func_to_dense(af, N, N))

    def test_full_attention(self):
        N, B = 8, 2
        valid = torch.ones(B, N, N, dtype=torch.bool, device="cuda")
        af = dense_mask_to_arbitrary_func(valid, N)
        assert torch.equal(valid, arbitrary_func_to_dense(af, N, N))

    def test_empty_mask(self):
        N, B = 8, 1
        valid = torch.zeros(B, N, N, dtype=torch.bool, device="cuda")
        af = dense_mask_to_arbitrary_func(valid, N)
        assert torch.equal(valid, arbitrary_func_to_dense(af, N, N))

    def test_block_diagonal(self):
        N, B = 8, 1
        valid = torch.zeros(B, N, N, dtype=torch.bool, device="cuda")
        valid[0, :4, :4] = True
        valid[0, 4:, 4:] = True
        af = dense_mask_to_arbitrary_func(valid, N)
        assert torch.equal(valid, arbitrary_func_to_dense(af, N, N))

    @pytest.mark.parametrize("beam_width", [2, 3])
    @pytest.mark.parametrize("candidate_len", [1, 3])
    def test_target_aware_causal_mask(self, beam_width, candidate_len):
        B = 2
        hist_lens = torch.tensor([6, 4], device="cuda")
        inverted = padded_target_aware_causal_mask(hist_lens, 6, beam_width, candidate_len)
        valid = ~inverted
        N = valid.shape[-1]
        af = dense_mask_to_arbitrary_func(valid, N)
        assert torch.equal(valid.squeeze(1), arbitrary_func_to_dense(af, N, N))

    def test_mask_with_gaps(self):
        N, B = 10, 1
        valid = torch.zeros(B, N, N, dtype=torch.bool, device="cuda")
        valid[0, 5, 0:3] = True
        valid[0, 5, 5:7] = True
        valid[0, 5, 9] = True
        af = dense_mask_to_arbitrary_func(valid, N)
        assert torch.equal(valid, arbitrary_func_to_dense(af, N, N))

    def test_4d_input(self):
        N, B = 8, 1
        valid_4d = torch.tril(torch.ones(B, 1, N, N, dtype=torch.bool, device="cuda"))
        af = dense_mask_to_arbitrary_func(valid_4d, N)
        assert torch.equal(valid_4d.squeeze(1), arbitrary_func_to_dense(af, N, N))

    def test_batch_independence(self):
        N, B = 8, 2
        valid = torch.zeros(B, N, N, dtype=torch.bool, device="cuda")
        valid[0] = torch.tril(torch.ones(N, N, dtype=torch.bool, device="cuda"))
        valid[1] = torch.ones(N, N, dtype=torch.bool, device="cuda")
        af = dense_mask_to_arbitrary_func(valid, N)
        recon = arbitrary_func_to_dense(af, N, N)
        assert torch.equal(valid[0], recon[0])
        assert torch.equal(valid[1], recon[1])


class TestJaggedFlattenedArbitraryFunc:
    """Tests for the B=1 flattened arbitrary_func builders."""

    @staticmethod
    def _build_expected_jagged_causal(offsets):
        """Build the expected [1, total, total] causal block-diagonal mask."""
        total = offsets[-1].item()
        device = offsets.device
        expected = torch.zeros(total, total, dtype=torch.bool, device=device)
        B = offsets.size(0) - 1
        for b in range(B):
            s, e = offsets[b].item(), offsets[b + 1].item()
            block = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool, device=device))
            expected[s:e, s:e] = block
        return expected

    def test_jagged_causal_basic(self):
        offsets = torch.tensor([0, 4, 7, 10], device="cuda")
        total = 10
        af = build_jagged_causal_arbitrary_func(offsets, total)
        recon = arbitrary_func_to_dense(af, total, total).squeeze(0)
        expected = self._build_expected_jagged_causal(offsets)
        assert torch.equal(expected, recon)

    def test_jagged_causal_single_batch(self):
        offsets = torch.tensor([0, 6], device="cuda")
        total = 6
        af = build_jagged_causal_arbitrary_func(offsets, total)
        recon = arbitrary_func_to_dense(af, total, total).squeeze(0)
        expected = torch.tril(torch.ones(6, 6, dtype=torch.bool, device="cuda"))
        assert torch.equal(expected, recon)

    def test_jagged_causal_uneven_lengths(self):
        offsets = torch.tensor([0, 2, 8, 9], device="cuda")
        total = 9
        af = build_jagged_causal_arbitrary_func(offsets, total)
        recon = arbitrary_func_to_dense(af, total, total).squeeze(0)
        expected = self._build_expected_jagged_causal(offsets)
        assert torch.equal(expected, recon)

    def test_dense_to_jagged_causal(self):
        """dense_mask_to_jagged_arbitrary_func should match build_jagged_causal for causal masks."""
        offsets = torch.tensor([0, 3, 7], device="cuda")
        B, total = 2, 7
        max_seqlen = 4
        per_batch = torch.zeros(B, max_seqlen, max_seqlen, dtype=torch.bool, device="cuda")
        for b in range(B):
            sl = (offsets[b + 1] - offsets[b]).item()
            per_batch[b, :sl, :sl] = torch.tril(
                torch.ones(sl, sl, dtype=torch.bool, device="cuda")
            )
        af = dense_mask_to_jagged_arbitrary_func(per_batch, offsets, total)
        recon = arbitrary_func_to_dense(af, total, total).squeeze(0)
        expected = self._build_expected_jagged_causal(offsets)
        assert torch.equal(expected, recon)

    @pytest.mark.parametrize("beam_width", [2, 3])
    @pytest.mark.parametrize("candidate_len", [1, 3])
    def test_dense_to_jagged_target_grouped(self, beam_width, candidate_len):
        """Verify target-grouped masks survive the jagged conversion roundtrip."""
        B = 2
        hist_lens = torch.tensor([5, 3], device="cuda")
        max_hist = 5
        inverted = padded_target_aware_causal_mask(hist_lens, max_hist, beam_width, candidate_len)
        valid = ~inverted  # [B, 1, N, N]
        N = valid.shape[-1]
        total_per_batch = (hist_lens + beam_width * candidate_len).tolist()
        offsets = torch.tensor(
            [0] + [sum(total_per_batch[:i + 1]) for i in range(B)],
            device="cuda",
        )
        total = offsets[-1].item()
        af = dense_mask_to_jagged_arbitrary_func(valid, offsets, total)
        recon = arbitrary_func_to_dense(af, total, total).squeeze(0)

        # Build expected flattened mask from per-batch dense mask
        expected = torch.zeros(total, total, dtype=torch.bool, device="cuda")
        valid_3d = valid.squeeze(1)
        for b in range(B):
            s = offsets[b].item()
            sl = total_per_batch[b]
            expected[s:s + sl, s:s + sl] = valid_3d[b, :sl, :sl]

        assert torch.equal(expected, recon)
