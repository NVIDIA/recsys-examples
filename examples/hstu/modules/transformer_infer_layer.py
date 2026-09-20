# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Recommendation Transformer with the paged HSTU inference layer contract.

The attention implementation materializes bounded, padded K/V tensors and uses
PyTorch SDPA. It is a correctness-oriented backend, not a fused paged-attention
kernel. Position embeddings are supplied by the shared recommendation processor.
This module deliberately imports only PyTorch so its numerical path also runs
on CPU. CUDA cache writes use the existing paged_kvcache_ops operator.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _pad(values, offsets, width):
    lengths = offsets[1:] - offsets[:-1]
    positions = torch.arange(width, device=values.device)
    valid = positions[None, :] < lengths[:, None]
    indices = torch.where(valid, offsets[:-1, None] + positions, values.shape[0])
    # A zero sentinel also handles an empty packed tensor without a negative index.
    extended = torch.cat((values, values.new_zeros((1,) + values.shape[1:])))
    return extended[indices.long()], valid


def _unpad(values, offsets, token_count):
    tokens = torch.arange(token_count, device=values.device)
    users = torch.searchsorted(offsets[1:].contiguous(), tokens, right=True)
    users = users.clamp(max=offsets.shape[0] - 2)
    positions = (tokens - offsets[users]).clamp(max=values.shape[1] - 1)
    result = values[users, positions]
    # CUDA graph buckets may contain unused packed-token slots.
    return torch.where((tokens < offsets[-1])[:, None], result, 0)


def read_paged_history(table, page_ids, page_indptr, history_lengths, width):
    """Gather NHD pages, masking every inactive slot before using its value."""
    positions = torch.arange(width, device=table.device)
    valid = positions[None, :] < history_lengths[:, None]
    logical_pages = page_indptr[:-1, None] + positions // table.shape[2]
    # Zero-history users need no pages; never dereference their inactive metadata.
    ids = torch.cat((page_ids, page_ids.new_zeros(1)))
    logical_pages = torch.where(valid, logical_pages, page_ids.shape[0])
    physical_pages = ids[logical_pages.long()].long()
    page_offsets = positions % table.shape[2]
    key = table[physical_pages, 0, page_offsets]
    value = table[physical_pages, 1, page_offsets]
    live = valid[:, :, None, None]
    return torch.where(live, key, 0), torch.where(live, value, 0)


class TransformerInferLayer(nn.Module):
    """Pre-LN MHA + GELU FFN, with independent recommendation candidates."""

    def __init__(self, config, layer_idx, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.width = config.max_seq_len
        self._export_mode = config.export_mode
        self._residual = config.residual
        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        kwargs = dict(device=device, dtype=dtype)
        inner = self.num_heads * self.head_dim
        self.input_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=getattr(config, "learnable_input_layernorm", True),
            **kwargs,
        )
        self.qkv = nn.Linear(config.hidden_size, 3 * inner, **kwargs)
        self.proj = nn.Linear(inner, config.hidden_size, **kwargs)
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon, **kwargs
        )
        ffn_size = config.transformer_ffn_dim or 4 * config.hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, ffn_size, **kwargs),
            nn.GELU(),
            nn.Linear(ffn_size, config.hidden_size, **kwargs),
        )
        capacity = config.max_batch_size * config.max_seq_len
        self.register_buffer(
            "qkv_buffer", torch.empty(capacity, 3 * inner, **kwargs), persistent=False
        )
        self.register_buffer(
            "output_buffer_",
            torch.empty(capacity, config.hidden_size, **kwargs),
            persistent=False,
        )
        self.requires_grad_(False)

    def project_qkv(self, hidden):
        mixed = self.qkv(self.input_norm(hidden))
        return tuple(
            x.reshape(-1, self.num_heads, self.head_dim) for x in mixed.chunk(3, dim=-1)
        )

    def attention(
        self,
        query,
        key,
        value,
        offsets,
        candidates,
        cache_table=None,
        page_ids=None,
        page_indptr=None,
        history_lengths=None,
    ):
        if query.shape[0] == 0:
            return query.new_empty((0, self.num_heads * self.head_dim))
        # A cached suffix often contains far fewer tokens than the configured
        # maximum. Shape bounds avoid a GPU-to-host maximum-length reduction.
        # A fixed export bound permits token dimensions to cross max_seq_len
        # (the packed batch can exceed one user's limit) without shape guards.
        query_width = (
            self.width
            if torch.compiler.is_compiling()
            else min(self.width, query.shape[0])
        )
        q, query_valid = _pad(query, offsets, query_width)
        k, _ = _pad(key, offsets, query_width)
        v, _ = _pad(value, offsets, query_width)
        lengths = offsets[1:] - offsets[:-1]
        torch._assert_async(
            torch.all((lengths >= 0) & (lengths <= self.width)),
            "sequence length exceeds Transformer max_seq_len",
        )
        torch._assert_async(
            torch.all((candidates >= 0) & (candidates <= lengths)),
            "invalid Transformer candidate count",
        )
        new_history = lengths - candidates
        query_local_positions = torch.arange(query_width, device=query.device)
        key_width = self.width if cache_table is not None else query_width
        positions = torch.arange(key_width, device=query.device)
        if cache_table is None:
            history_lengths = new_history
            cached_lengths = torch.zeros_like(lengths)
        else:
            cached_lengths = history_lengths - new_history
            torch._assert_async(
                torch.all(cached_lengths >= 0), "negative cached prefix length"
            )
            torch._assert_async(
                torch.all(history_lengths + candidates <= self.width),
                "cached sequence exceeds Transformer max_seq_len",
            )
            history_k, history_v = read_paged_history(
                cache_table, page_ids, page_indptr, history_lengths, self.width
            )
            # Candidates are not persisted in the cache. Read them from this call.
            local_positions = (
                (positions[None, :] - cached_lengths[:, None])
                .clamp(min=0, max=query_width - 1)
                .long()
            )
            users = torch.arange(lengths.shape[0], device=query.device)[:, None]
            use_history = (positions[None, :] < history_lengths[:, None])[
                :, :, None, None
            ]
            k = torch.where(use_history, history_k, k[users, local_positions])
            v = torch.where(use_history, history_v, v[users, local_positions])
        query_positions = query_local_positions[None, :] + cached_lengths[:, None]
        keys = positions[None, None, :]
        queries = query_positions[:, :, None]
        # Histories are causal, candidates see history and themselves only.
        allowed = (
            (keys <= queries)
            & ((keys < history_lengths[:, None, None]) | (keys == queries))
            & (keys < (history_lengths + candidates)[:, None, None])
            & query_valid[:, :, None]
        )
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed[:, None],
            dropout_p=0.0,
        )
        output = output.transpose(1, 2).flatten(2)
        return _unpad(output, offsets, query.shape[0])

    def finish(self, hidden, attention):
        projected = self.proj(attention)
        hidden = hidden + projected if self._residual else projected
        output = self.ffn(self.ffn_norm(hidden))
        return hidden + output if self._residual else output

    def forward(self, hidden, offsets, candidates):
        """Tensor-only no-cache entry point, also usable with torch.export."""
        q, k, v = self.project_qkv(hidden)
        return self.finish(hidden, self.attention(q, k, v, offsets, candidates))

    def _append(self, key, value, offsets, candidates, metadata, batch_size):
        table = metadata.kv_cache_table[self.layer_idx]
        if table.is_cuda:
            return torch.ops.paged_kvcache_ops.append_kvcache(
                key,
                value,
                metadata.batch_indices,
                metadata.position,
                torch.cat((candidates.new_zeros(1), candidates.cumsum(0))).to(
                    offsets.dtype
                ),
                # A positive upper bound avoids the operator's nnz.item() path
                # during CUDA graph capture. The kernel reads the live count.
                metadata.new_history_nnz_cuda,
                0 if self._export_mode else key.shape[0],
                table,
                metadata.kv_indices,
                metadata.kv_indptr,
                metadata.kv_last_page_len,
                0,
            )
        # CPU reference backend: same persistent NHD format, history tokens only.
        tokens = torch.arange(key.shape[0], device=key.device)
        users = torch.searchsorted(offsets[1:].contiguous(), tokens, right=True).clamp(
            max=batch_size - 1
        )
        local = tokens - offsets[users]
        lengths = offsets[1:] - offsets[:-1]
        new_history = lengths - candidates
        valid = (local < new_history[users]) & (tokens < offsets[-1])
        users, local = users[valid], local[valid]
        positions = metadata.total_history_lengths[users] - new_history[users] + local
        pages = metadata.kv_indices[
            (metadata.kv_indptr[users] + positions // table.shape[2]).long()
        ].long()
        table[pages, 0, positions % table.shape[2]] = key[valid]
        table[pages, 1, positions % table.shape[2]] = value[valid]
        return table

    def _compute(self, hidden, q, k, v, jd, metadata, batch_size, append):
        offsets = jd.seqlen_offsets[: batch_size + 1]
        candidates = (
            jd.num_candidates[:batch_size]
            if jd.num_candidates is not None
            else torch.zeros_like(offsets[:-1])
        )
        cache = {}
        if metadata is not None:
            table = (
                self._append(k, v, offsets, candidates, metadata, batch_size)
                if append
                else metadata.kv_cache_table[self.layer_idx]
            )
            handle = metadata.kv_onload_handle
            if not self._export_mode and handle is not None:
                handle.stream_wait_layer(self.layer_idx)
            cache = dict(
                cache_table=table,
                page_ids=metadata.kv_indices,
                page_indptr=metadata.kv_indptr[: batch_size + 1],
                history_lengths=metadata.total_history_lengths[:batch_size],
            )
        attended = self.attention(q, k, v, offsets, candidates, **cache)
        return self.finish(hidden, attended)

    def forward_naive(self, batch_size, num_tokens, hidden, jd, metadata):
        hidden = hidden[:num_tokens]
        q, k, v = self.project_qkv(hidden)
        return self._compute(hidden, q, k, v, jd, metadata, batch_size, append=True)

    def forward_input(self, batch_size, num_tokens, hidden, jd, metadata):
        mixed = self.qkv(self.input_norm(hidden[:num_tokens]))
        self.qkv_buffer[:num_tokens].copy_(mixed)
        if metadata is not None:
            _, k, v = (
                x.reshape(-1, self.num_heads, self.head_dim) for x in mixed.chunk(3, -1)
            )
            offsets = jd.seqlen_offsets[: batch_size + 1]
            candidates = (
                jd.num_candidates[:batch_size]
                if jd.num_candidates is not None
                else torch.zeros_like(offsets[:-1])
            )
            self._append(k, v, offsets, candidates, metadata, batch_size)
        return self.qkv_buffer[:num_tokens]

    def forward_output(self, batch_size, num_tokens, hidden, jd, metadata):
        q, k, v = (
            x.reshape(-1, self.num_heads, self.head_dim)
            for x in self.qkv_buffer[:num_tokens].chunk(3, -1)
        )
        output = self._compute(
            hidden[:num_tokens], q, k, v, jd, metadata, batch_size, append=False
        )
        self.output_buffer_[:num_tokens].copy_(output)
        return self.output_buffer_[:num_tokens]
