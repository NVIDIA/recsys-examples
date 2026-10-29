# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A compact PyTorch HSTU-style sequential recommender.

This is intentionally dependency-light. It follows the public Meta HSTU shape:
normalized input -> joint U/V/Q/K projection -> causal SiLU attention -> gated
output projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # @torch_optimize:begin
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight
        # @torch_optimize:end


class HSTUBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attn_alpha = 1.0 / math.sqrt(attention_dim)

        projected_dim = 2 * num_heads * (hidden_dim + attention_dim)
        self.input_norm = RMSNorm(embedding_dim)
        self.uvqk = nn.Linear(embedding_dim, projected_dim)
        self.output_norm = RMSNorm(num_heads * hidden_dim)
        self.output = nn.Linear(
            num_heads * hidden_dim * 2 + embedding_dim, embedding_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        uvqk = self.uvqk(self.input_norm(x))
        u, v, q, k = torch.split(
            uvqk,
            [
                self.num_heads * self.hidden_dim,
                self.num_heads * self.hidden_dim,
                self.num_heads * self.attention_dim,
                self.num_heads * self.attention_dim,
            ],
            dim=-1,
        )

        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # @torch_optimize:begin
        scores = F.silu(scores).masked_fill(~causal, 0.0)
        scores = scores / float(seq_len)
        # @torch_optimize:end

        attn = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)

        # @torch_optimize:begin
        gated = self.output_norm(attn) * F.silu(u)
        y = torch.cat([gated, u, residual], dim=-1)
        # @torch_optimize:end

        return residual + self.dropout(self.output(y))


class HSTUSequentialRecommender(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 32,
        attention_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_user_groups: int = 32,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_user_groups + 1, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    attention_dim=attention_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output.weight = self.item_embedding.weight

    def forward(
        self, input_ids: torch.Tensor, user_group: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = x + self.user_embedding(user_group).unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        return self.output(self.norm(x))
