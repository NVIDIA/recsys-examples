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

"""A compact OneRec-style model for talos performance experiments."""

from __future__ import annotations

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


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        up = self.up(x)
        # @torch_optimize:begin
        x = F.silu(gate) * up
        # @torch_optimize:end
        return self.dropout(self.down(x))


class ValueDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4, dropout)
        self.cross_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q_len = x.size(1)
        causal = torch.triu(
            torch.ones(q_len, q_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.self_norm(x)
        self_out, _ = self.self_attn(
            h,
            h,
            h,
            attn_mask=causal,
            need_weights=False,
        )
        x = x + self_out
        cross_out, _ = self.cross_attn(
            self.cross_norm(x),
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )

        # @torch_optimize:begin
        x = x + torch.sigmoid(self.cross_gate) * cross_out
        # @torch_optimize:end

        return x + self.ffn(self.ffn_norm(x))


class ValueDecoder(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_layers: int, sid_depth: int, dropout: float
    ) -> None:
        super().__init__()
        self.boc = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Embedding(sid_depth + 2, dim)
        self.memory_norm = RMSNorm(dim)
        self.blocks = nn.ModuleList(
            [ValueDecoderBlock(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.head_norm = RMSNorm(dim)
        self.head = SwiGLU(dim, dim * 2, dropout)
        self.reward_head = nn.Linear(dim, 1)
        self.ltv_head = nn.Linear(dim, 1)
        nn.init.normal_(self.boc, mean=0.0, std=0.02)

    def forward(
        self,
        memory: torch.Tensor,
        decoder_features: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.size(0)
        memory = self.memory_norm(memory)
        if memory_key_padding_mask is None:
            pooled = memory.mean(dim=1)
        else:
            valid = (~memory_key_padding_mask).float().unsqueeze(-1)
            pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        x = torch.cat(
            [
                self.boc.expand(batch_size, -1, -1),
                pooled.unsqueeze(1),
                decoder_features.detach(),
            ],
            dim=1,
        )
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position(pos)

        for block in self.blocks:
            x = block(x, memory, memory_key_padding_mask)

        features = self.head(self.head_norm(x[:, 2:, :]))
        reward = self.reward_head(features).squeeze(-1)
        ltv = self.ltv_head(features).squeeze(-1)
        return reward, ltv


class OneRecSIDWithValue(nn.Module):
    """Encoder-decoder generative recommender over hierarchical semantic IDs."""

    def __init__(
        self,
        num_classes: int,
        sid_depth: int,
        max_hist_len: int,
        user_feat_dim: int,
        dim: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        value_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sid_depth = sid_depth
        self.dim = dim

        self.sid_embedding = nn.Embedding(num_classes, dim, padding_idx=0)
        self.sid_position = nn.Embedding(sid_depth, dim)
        self.hist_position = nn.Embedding(max_hist_len, dim)
        self.sid_merge = nn.Linear(dim * sid_depth, dim)
        self.user_proj = nn.Linear(user_feat_dim, dim)
        self.bos = nn.Parameter(torch.zeros(dim))
        self.decoder_position = nn.Embedding(sid_depth, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(dim, num_classes)
        self.value_decoder = ValueDecoder(
            dim, num_heads, value_layers, sid_depth, dropout
        )

        nn.init.normal_(self.bos, mean=0.0, std=0.02)

    def encode_history(
        self,
        hist_sid: torch.Tensor,
        hist_len: torch.Tensor,
        user_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, hist_size, sid_depth = hist_sid.shape
        depth_pos = torch.arange(sid_depth, device=hist_sid.device).view(
            1, 1, sid_depth
        )
        sid_emb = self.sid_embedding(hist_sid) + self.sid_position(depth_pos)
        item_vec = self.sid_merge(
            sid_emb.reshape(batch_size, hist_size, sid_depth * self.dim)
        )
        hist_pos = torch.arange(hist_size, device=hist_sid.device).view(1, hist_size)
        user_vec = self.user_proj(user_feat).unsqueeze(1)

        # @torch_optimize:begin
        item_vec = item_vec + self.hist_position(hist_pos) + user_vec
        # @torch_optimize:end

        mask = (hist_sid == 0).all(dim=-1)
        len_mask = torch.arange(hist_size, device=hist_sid.device).view(
            1, hist_size
        ) >= hist_len.view(batch_size, 1)
        mask = mask | len_mask
        all_pad = mask.all(dim=1)
        if all_pad.any():
            mask[all_pad, -1] = False
        return self.history_encoder(item_vec, src_key_padding_mask=mask), mask

    def build_decoder_input(self, target_sid: torch.Tensor) -> torch.Tensor:
        batch_size, sid_depth = target_sid.shape
        shifted = target_sid.new_zeros(batch_size, sid_depth)
        if sid_depth > 1:
            shifted[:, 1:] = target_sid[:, :-1]
        x = self.sid_embedding(shifted)
        x[:, 0, :] = self.bos
        pos = torch.arange(sid_depth, device=target_sid.device).unsqueeze(0)
        return x + self.decoder_position(pos)

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        target_sid: torch.Tensor,
        user_feat: torch.Tensor,
        hist_sid: torch.Tensor,
        hist_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, memory_mask = self.encode_history(hist_sid, hist_len, user_feat)
        x = self.build_decoder_input(target_sid)
        dec = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.causal_mask(x.size(1), x.device),
            memory_key_padding_mask=memory_mask,
        )
        logits = self.output(dec)
        return logits, dec, memory, memory_mask

    def compute_loss(
        self,
        target_sid: torch.Tensor,
        user_feat: torch.Tensor,
        hist_sid: torch.Tensor,
        hist_len: torch.Tensor,
        immediate_reward: torch.Tensor,
        ltv: torch.Tensor,
        cls_weight: float = 1.0,
        reward_weight: float = 0.2,
        ltv_weight: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits, dec, memory, memory_mask = self(
            target_sid, user_feat, hist_sid, hist_len
        )
        cls_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_sid.reshape(-1)
        )
        reward_pred, ltv_pred = self.value_decoder(memory, dec, memory_mask)
        reward_loss = F.mse_loss(reward_pred, immediate_reward)
        ltv_loss = F.mse_loss(ltv_pred.mean(dim=1), ltv)
        loss = (
            cls_weight * cls_loss + reward_weight * reward_loss + ltv_weight * ltv_loss
        )
        logs = {
            "loss": float(loss.detach().cpu()),
            "cls": float(cls_loss.detach().cpu()),
            "reward": float(reward_loss.detach().cpu()),
            "ltv": float(ltv_loss.detach().cpu()),
        }
        return loss, logs
