# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention 4 (CuTe DSL) attention for the Transformer inference layers.

``Fa4AttentionMixin`` replaces the reference attention of ``TransformerInferLayer``, which
pads every user and feeds a dense mask to SDPA, by one varlen FA4 kernel over the packed
tokens. The recommendation mask is FA4's causal mask, aligned to the end of each user's
keys, plus a ``score_mod`` that hides the other candidates from a candidate: counting key
positions over the cached prefix, the new history and the candidates, a query sees key
``k`` when ``k`` is history or ``k`` is its own position. A paged KV cache is first
gathered into packed K/V, the history from the pages followed by this call's candidates,
because the candidates are never cached and FA4 cannot read pages on SM120.

``HSTUBlockInference`` builds these layers when the inference config sets
``transformer_attention="fa4"`` (gin: ``NetworkArgs.transformer_attention``), with bf16 or
FP8 projections. FA4 compiles each kernel configuration at its first call, which the CUDA
graph warmup does before capture, and it cannot be exported.
"""

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute import utils as fa4_utils
from flash_attn.cute.interface import flash_attn_varlen_func
from modules.transformer_infer_layer import TransformerInferLayer
from modules.transformer_infer_layer_fp8 import Fp8FusedTransformerInferLayer


@cute.jit
def _candidate_score_mod(scores, batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    # FA4 passes the indices by keyword, counted per user. With causal=True it already
    # hides later keys; aux_tensors[0] holds the candidate count of each user.
    history = fa4_utils.scalar_to_ssa(
        seqlen_info.seqlen_k - aux_tensors[0][batch[0]], cutlass.Int32
    )
    query = q_idx + fa4_utils.scalar_to_ssa(
        seqlen_info.seqlen_k - seqlen_info.seqlen_q, cutlass.Int32
    )
    keep = (kv_idx < history) | (kv_idx == query)
    return cute.where(keep, scores, cute.full_like(scores, float("-inf")))


class Fa4AttentionMixin:
    """Run ``TransformerInferLayer.attention`` as one FA4 varlen kernel; see the module."""

    def __init__(self, config, layer_idx, device=None):
        if config.export_mode:
            raise ValueError("FA4 attention cannot be exported")
        super().__init__(config, layer_idx, device)

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
        """``TransformerInferLayer.attention`` as one FA4 kernel.

        Rows past the last user, the padding of a CUDA graph bucket, must be finite. They
        never get weight, but FA4's SM90 kernel loads whole K/V tiles past a user's end and
        weighs those rows by zero, which a NaN survives (0 * NaN). The dense module's
        static input buffer starts random and then only holds finite activations.
        """
        tokens = query.shape[0]
        if tokens == 0:
            return query.new_empty((0, self.num_heads * self.head_dim))
        # The reference's checks: the bounds below and the packed K/V assume them.
        self._checked_lengths(
            offsets, candidates, None if cache_table is None else history_lengths
        )
        offsets = _with_end_sentinel(offsets.int())
        candidates = _with_end_sentinel(candidates.int())
        # Static bounds keep the call free of GPU-to-host reads, as CUDA graphs need.
        query_width = min(self._max_seq_len, tokens)
        key_offsets, key_width = offsets, query_width
        if cache_table is not None:
            key, value, key_offsets = self._packed_kv(
                key,
                value,
                offsets,
                candidates,
                cache_table,
                page_ids,
                page_indptr,
                history_lengths,
            )
            key_width = self._max_seq_len
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=offsets,
            cu_seqlens_k=key_offsets,
            max_seqlen_q=query_width,
            max_seqlen_k=key_width,
            causal=True,
            score_mod=_candidate_score_mod,
            aux_tensors=[candidates],
        )
        output = output[0] if isinstance(output, tuple) else output
        # CUDA graph buckets pack more rows than the request; zero them as the reference.
        live = torch.arange(tokens, device=query.device) < offsets[-1]
        return torch.where(live[:, None], output.flatten(1), 0)

    def _packed_kv(
        self, key, value, offsets, candidates, table, page_ids, page_indptr, history
    ):
        """Pack each user's history from the pages, then this call's candidates."""
        users = candidates.shape[0]
        key_lengths = (history + candidates).int()
        key_offsets = torch.cat((key_lengths.new_zeros(1), key_lengths.cumsum(0).int()))
        key_offsets = _with_end_sentinel(key_offsets)
        # Every user fits in max_seq_len, so this bound needs no GPU-to-host read.
        slots = torch.arange(users * self._max_seq_len, device=key.device)
        owner = torch.searchsorted(key_offsets[1:].long(), slots, right=True)
        owner = owner.clamp(max=users - 1)
        local = slots - key_offsets[owner]
        cached = local < history[owner]
        # Slots outside the history read a sentinel page instead of inactive metadata.
        ids = torch.cat((page_ids, page_ids.new_zeros(1)))
        logical = page_indptr[owner] + local // table.shape[2]
        pages = ids[torch.where(cached, logical, page_ids.shape[0]).long()].long()
        within = local % table.shape[2]
        rows = offsets[owner + 1] - candidates[owner] + local - history[owner]
        rows = rows.clamp(0, key.shape[0] - 1)
        cached = cached[:, None, None]
        packed_key = torch.where(cached, table[pages, 0, within], key[rows])
        packed_value = torch.where(cached, table[pages, 1, within], value[rows])
        return packed_key, packed_value, key_offsets


def _with_end_sentinel(values):
    """``values`` as a view of a copy one element longer that repeats the last one.

    FA4's varlen scheduler (4.0.0b17 through 4.0.0b21) also runs spare CTAs for the
    batch index past the last user: they read ``cu_seqlens[batch + 1]`` and the score_mod
    reads ``candidates[batch]``, one element out of bounds (compute-sanitizer on SM120; not on
    SM90).
    Out of bounds memory can give such a CTA a nonzero length, so it reads and writes
    outside the tensors (illegal addresses, overwritten rows). The repeated end makes it
    empty.
    """
    return torch.cat((values, values[-1:]))[:-1]


class Fa4TransformerInferLayer(Fa4AttentionMixin, TransformerInferLayer):
    """``TransformerInferLayer`` with FA4 attention."""


class Fa4Fp8TransformerInferLayer(Fa4AttentionMixin, Fp8FusedTransformerInferLayer):
    """``Fp8FusedTransformerInferLayer`` (FP8 projections) with FA4 attention."""
