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
    dense_mask_to_arbitrary_func,
    padded_target_aware_causal_mask,
)
sys.path.pop(0)


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
