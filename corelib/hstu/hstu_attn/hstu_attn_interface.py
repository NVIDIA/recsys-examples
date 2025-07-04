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
# Copyright (c) 2023, Tri Dao.
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

from typing import Optional, Union

import torch

import hstu_attn_2_cuda as hstu_attn_cuda
import torch


class HstuAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,  # need grad
        k,  # need grad
        v,  # need grad
        seq_offsets_q,
        seq_offsets_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size=(-1, -1),
        alpha=1.0,
        rab=None,  # need grad
        has_drab=False,
        is_delta_q=False,
        kv_cache=None,
        page_offsets=None,
        page_ids=None,
        last_page_lens=None,
        seq_offsets_t=None,
    ):
        assert q.dim() == 3, "q shape should be (L, num_heads, head_dim)"
        assert k.dim() == 3, "k shape should be (L, num_heads, head_dim)"
        assert v.dim() == 3, "v shape should be (L, num_heads, hidden_dim)"
        num_heads = q.size(1)
        head_dim = q.size(2)
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            out, rab_padded = hstu_attn_cuda.varlen_fwd(
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                is_delta_q,
                kv_cache,
                page_offsets,
                page_ids,
                last_page_lens,
                seq_offsets_t
            )
        P = out[:, :, :head_dim].reshape(-1, num_heads * head_dim)

        ctx.save_for_backward(
            q,
            k,
            v,
            seq_offsets_q,
            seq_offsets_k,
            num_contexts,
            num_targets,
            rab_padded,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.alpha = alpha
        ctx.window_size_left = window_size[0]
        ctx.window_size_right = window_size[1]
        ctx.has_drab = has_drab
        ctx.is_delta_q = is_delta_q
        return P.view(-1, num_heads, head_dim)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            seq_offsets_q,
            seq_offsets_k,
            num_contexts,
            num_targets,
            rab_padded,
        ) = ctx.saved_tensors

        num_heads, head_dim = (ctx.num_heads, ctx.head_dim)
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        target_group_size = ctx.target_group_size
        window_size_left = ctx.window_size_left
        window_size_right = ctx.window_size_right
        alpha = ctx.alpha
        has_drab = ctx.has_drab
        is_delta_q = ctx.is_delta_q
        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            dq, dk, dv, dRab = hstu_attn_cuda.varlen_bwd(
                dout.view(-1, num_heads, head_dim),
                q,
                k,
                v,
                None,
                None,
                None,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                is_delta_q,
                False,  # deterministic
            )

        if has_drab:
            rab_head = rab_padded.size(1) # TODO: need discuss with customer, casue we have padded rab
            qkv_head = q.size(1)
            dRab = dRab.view(-1, num_heads, max_seqlen_k, max_seqlen_k)

        # q & k grad shape
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dRab if ctx.has_drab else None,
            None,
            None,
            None,
            None,
            None,
            None
        )


def hstu_attn_varlen_func(
    q,
    k,
    v,
    seq_offsets_q,
    seq_offsets_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts=None,
    num_targets=None,
    target_group_size=1,
    window_size=(-1, -1),
    alpha=1.0,
    rab=None,
    has_drab=False,
    is_delta_q=False,
    kv_cache=None,
    page_offsets=None,
    page_ids=None,
    last_page_lens=None,
    seq_offsets_t=None,
):
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        seq_offsets_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        seq_offsets_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        num_contexts: (batch_size,). Number of context tokens in each batch.
        num_targets: (batch_size,). Number of target tokens in each batch.
        target_group_size: int. Number of target tokens in each group.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. If (-1, 0), implements causal attention.
        alpha: float. Scaling factor between add rab and silu.
        rab: (batch_size, max_seqlen_k, max_seqlen_k). Random access bias for the key.
        has_drab: bool. Whether to apply random access bias for the key.
        is_delta_q: bool. Whether to apply delta query.
        kv_cache: (page_num, 2, page_size, nheads, headdim). Key and value paged cache.
        page_offsets: (batch_size + 1,). The cumulative sequence lengths of the page_ptr in the batch, used to index into kv_cache.
        page_ids: (page_offsets[-1],). The ids of the pages in the batch.
        last_page_lens: (batch_size,). The lengths of the last pages in the batch.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError("AssertError: rab is None, but has_drab is True, is not allowed in backward")
    if num_contexts != None and window_size != (-1, 0):
        raise ValueError("AssertError: context is True and causal is not True, this is undefined behavior")
    if num_targets != None and window_size != (-1, 0):
        raise ValueError("AssertError: target is True and causal is not True, this is undefined behavior")
    # if (num_contexts != None and is_delta_q is True) or (num_targets != None and is_delta_q is True):
    #     raise ValueError("AssertError: delta_q is True, but num_contexts or num_targets is not None, this is undefined behavior")
    if num_targets is None and target_group_size < 1:
        raise ValueError("AssertError: target_group_size should be greater than 0 when target is True")
    if max_seqlen_q < max_seqlen_k and window_size != (-1, -1) and is_delta_q is False:
        raise ValueError("AssertError: seq_len_q < seq_len_k, is_delta_q should be True, as is_delta_q represents mask behavior under the case")
    if max_seqlen_q > max_seqlen_k:
        raise ValueError("AssertError: seq_len_q >= seq_len_k, this is undefined behavior")

    return HstuAttnVarlenFunc.apply(
        q,
        k,
        v,
        seq_offsets_q,
        seq_offsets_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        is_delta_q,
        kv_cache,
        page_offsets,
        page_ids,
        last_page_lens,
        seq_offsets_t
    )

class HstuAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        seq_offsets_q,
        seq_offsets_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size=(-1, -1),
        alpha=1.0,
        rab=None,  # need grad
        has_drab=False,
        is_delta_q=False
    ):
        q = qkv[:, 0, :, :].detach()
        k = qkv[:, 1, :, :].detach()
        v = qkv[:, 2, :, :].detach()
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            out, rab_padded = hstu_attn_cuda.varlen_fwd(
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                is_delta_q,
                None,
                None,
                None,
                None,
                None
            )
        num_heads = q.size(1)
        head_dim = q.size(2)
        P = out[:, :, :head_dim].reshape(-1, num_heads * head_dim)

        ctx.save_for_backward(
            q,
            k,
            v,
            seq_offsets_q,
            seq_offsets_k,
            num_contexts,
            num_targets,
            rab_padded,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.alpha = alpha
        ctx.window_size_left = window_size[0]
        ctx.window_size_right = window_size[1]
        ctx.has_drab = has_drab
        ctx.is_delta_q = is_delta_q
        return P.view(-1, num_heads, head_dim)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            seq_offsets_q,
            seq_offsets_k,
            num_contexts,
            num_targets,
            rab_padded,
        ) = ctx.saved_tensors

        num_heads, head_dim = (ctx.num_heads, ctx.head_dim)
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        target_group_size = ctx.target_group_size
        window_size_left = ctx.window_size_left
        window_size_right = ctx.window_size_right
        alpha = ctx.alpha
        has_drab = ctx.has_drab
        is_delta_q = ctx.is_delta_q
        qkv_shape = (q.shape[0], 3, q.shape[1], q.shape[2])
        dqkv = torch.empty(qkv_shape, device=q.device, dtype=q.dtype)
        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            dq, dk, dv, dRab = hstu_attn_cuda.varlen_bwd(
                dout.view(-1, num_heads, head_dim),
                q,
                k,
                v,
                dqkv[:,0,:,:], # dq
                dqkv[:,1,:,:], # dk
                dqkv[:,2,:,:], # dv
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                is_delta_q,
                False,  # deterministic
            )

        if has_drab:
            rab_head = rab_padded.size(1) # TODO: need discuss with customer, casue we have padded rab
            qkv_head = q.size(1)
            dRab = dRab.view(-1, num_heads, max_seqlen_k, max_seqlen_k)

        # q & k grad shape
        return (
            dqkv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dRab if ctx.has_drab else None,
            None,
            None
        )

def hstu_attn_qkvpacked_func(
    qkv,
    seq_offsets_q,
    seq_offsets_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts=None,
    num_targets=None,
    target_group_size=1,
    window_size=(-1, -1),
    alpha=1.0,
    rab=None,
    has_drab=False,
    is_delta_q=False
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        seq_offsets_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        seq_offsets_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        num_contexts: (batch_size,). Number of context tokens in each batch.
        num_targets: (batch_size,). Number of target tokens in each batch.
        target_group_size: int. Number of target tokens in each group.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. If (-1, 0), implements causal attention.
        alpha: float. Scaling factor between add rab and silu.
        rab: (batch_size, max_seqlen_k, max_seqlen_k). Random access bias for the key.
        has_drab: bool. Whether to apply random access bias for the key.
        is_delta_q: bool. Whether to apply delta query.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError("AssertError: rab is None, but has_drab is True, is not allowed in backward")
    if num_contexts != None and window_size != (-1, 0):
        raise ValueError("AssertError: context is True and causal is not True, this is undefined behavior")
    if num_targets != None and window_size != (-1, 0):
        raise ValueError("AssertError: target is True and causal is not True, this is undefined behavior")
    if (num_contexts != None and is_delta_q is True) or (num_targets != None and is_delta_q is True):
        raise ValueError("AssertError: delta_q is True, but num_contexts or num_targets is not None, this is undefined behavior")
    if num_targets is None and target_group_size < 1:
        raise ValueError("AssertError: target_group_size should be greater than 0 when target is True")
    if max_seqlen_q < max_seqlen_k and window_size != (-1, -1) and is_delta_q is False:
        raise ValueError("AssertError: seq_len_q < seq_len_k, is_delta_q should be True, as is_delta_q represents mask behavior under the case")
    if max_seqlen_q > max_seqlen_k:
        raise ValueError("AssertError: seq_len_q >= seq_len_k, this is undefined behavior")

    return HstuAttnQKVPackedFunc.apply(
        qkv,
        seq_offsets_q,
        seq_offsets_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        is_delta_q
    )


if __name__ == "__main__":
    print("main")
