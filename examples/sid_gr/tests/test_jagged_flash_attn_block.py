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
Tests for JaggedFlashAttnBlock / JaggedGPTLayer.

Test strategy (same pattern as HSTU's test_hstu_layer.py):
  1. Build a reference implementation using PyTorch's scaled_dot_product_attention
  2. Build JaggedGPTLayer with the same weights
  3. Run the same input, compare outputs

Tests:
  - Smoke: forward pass runs without error, output shape correct
  - Causal: causal mask produces correct output vs PyTorch reference
  - Backward: gradients flow correctly
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from jagged_flash_attn_block import (
    JaggedFlashAttnBlock,
    JaggedGPTLayer,
)
sys.path.pop(0)

try:
    from flash_attn.cute.interface import flash_attn_func  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# PyTorch reference: single GPT layer with standard attention
# ---------------------------------------------------------------------------
class ReferenceGPTLayer(nn.Module):
    """Minimal GPT layer using PyTorch's SDPA for correctness comparison."""

    def __init__(self, hidden_size, num_heads, ffn_hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.linear_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.linear_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pre_mlp_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp_fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.mlp_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attn_mask=None, is_causal=False):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        B, S, _ = x.shape

        qkv = self.linear_qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.hidden_size)
        attn_out = self.linear_proj(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        x = self.pre_mlp_layernorm(hidden_states)
        x = self.mlp_fc1(x)
        x = F.gelu(x)
        x = self.mlp_fc2(x)
        hidden_states = residual + x

        return hidden_states


def _copy_weights(src: nn.Module, dst: nn.Module):
    """Copy all matching weights from src to dst."""
    dst.load_state_dict(src.state_dict(), strict=False)
    src_dict = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        if name in src_dict:
            param.data.copy_(src_dict[name].data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
class TestJaggedGPTLayerSmoke:
    """Basic smoke tests that don't require flash_attn."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seqlen", [16, 64])
    @pytest.mark.parametrize("hidden_size,num_heads", [(256, 4), (512, 8)])
    def test_forward_shape(self, batch_size, seqlen, hidden_size, num_heads):
        """Output shape should match input shape."""
        ffn_size = hidden_size * 4
        layer = JaggedGPTLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            ffn_hidden_size=ffn_size,
        ).cuda()

        x = torch.randn(batch_size, seqlen, hidden_size, device="cuda")
        out = layer(x)
        assert out.shape == (batch_size, seqlen, hidden_size)

    def test_block_forward_shape(self):
        """JaggedFlashAttnBlock stacks layers correctly."""
        block = JaggedFlashAttnBlock(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
        ).cuda().bfloat16()

        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
        out = block(x)
        assert out.shape == (2, 32, 256)


@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
class TestJaggedGPTLayerCorrectness:
    """Compare JaggedGPTLayer (FA) against PyTorch reference."""

    @pytest.mark.parametrize("hidden_size,num_heads", [(256, 4)])
    @pytest.mark.parametrize("seqlen", [16, 32])
    def test_causal_matches_reference(self, hidden_size, num_heads, seqlen):
        """
        With causal=True (no arbitrary_func), JaggedGPTLayer should produce
        the same output as the PyTorch reference (within bf16 precision).
        """
        B = 2
        ffn_size = hidden_size * 4
        torch.manual_seed(42)

        ref_layer = ReferenceGPTLayer(hidden_size, num_heads, ffn_size).cuda().bfloat16()
        test_layer = JaggedGPTLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            ffn_hidden_size=ffn_size,
            hidden_dropout=0.0,
        ).cuda().bfloat16()

        test_layer.load_state_dict(ref_layer.state_dict())

        x = torch.randn(B, seqlen, hidden_size, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = ref_layer(x, is_causal=True)
            test_out = test_layer(x, arbitrary_func=None)

        torch.testing.assert_close(test_out, ref_out, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("hidden_size,num_heads", [(256, 4)])
    def test_arbitrary_causal_matches_standard_causal(self, hidden_size, num_heads):
        """
        An arbitrary_func encoding a causal mask should produce the same
        result as the built-in causal=True path.
        """
        B, S = 1, 32
        ffn_size = hidden_size * 4
        torch.manual_seed(42)

        layer = JaggedGPTLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            ffn_hidden_size=ffn_size,
            hidden_dropout=0.0,
        ).cuda().bfloat16()

        x = torch.randn(B, S, hidden_size, device="cuda", dtype=torch.bfloat16)

        # Build causal arbitrary_func: F0[i] = i+1
        n_func = 1
        af = torch.zeros(B, 1, n_func, S + 256, dtype=torch.int32, device="cuda")
        for i in range(S):
            af[:, :, 0, i] = i + 1

        with torch.no_grad():
            out_causal = layer(x, arbitrary_func=None)
            out_arb = layer(x, arbitrary_func=af)

        torch.testing.assert_close(out_arb, out_causal, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
class TestJaggedGPTLayerBackward:
    """Verify gradients flow correctly."""

    def test_backward_runs(self):
        """Forward + backward should not error."""
        layer = JaggedGPTLayer(
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            hidden_dropout=0.0,
        ).cuda().bfloat16()

        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_gradient_correctness(self):
        """
        Gradient of JaggedGPTLayer (causal) should match the PyTorch reference.
        """
        hidden_size, num_heads = 256, 4
        ffn_size = 1024
        B, S = 2, 16
        torch.manual_seed(42)

        ref_layer = ReferenceGPTLayer(hidden_size, num_heads, ffn_size).cuda().bfloat16()
        test_layer = JaggedGPTLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            ffn_hidden_size=ffn_size,
            hidden_dropout=0.0,
        ).cuda().bfloat16()
        test_layer.load_state_dict(ref_layer.state_dict())

        x_ref = torch.randn(B, S, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_(True)

        ref_out = ref_layer(x_ref, is_causal=True)
        test_out = test_layer(x_test, arbitrary_func=None)

        dout = torch.randn_like(ref_out)
        ref_out.backward(dout)
        test_out.backward(dout)

        torch.testing.assert_close(
            x_test.grad, x_ref.grad, atol=5e-2, rtol=5e-2
        )
