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


import torch

# isort: off
# We need to import the CUDA kernels after importing torch
import hstu_hopper_cuda

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _hstu_attn_varlen_forward(
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
    window_size=(-1, -1),
    alpha=1.0,
    rab=None,
    is_delta_q=False,
    descale_q=None,
    descale_k=None,
    descale_v=None,
):
    has_rab = False
    if rab is not None:
        has_rab = True
        rab = maybe_contiguous(rab)
    out, rab = hstu_hopper_cuda.varlen_fwd(
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
        descale_q,
        descale_k,
        descale_v,
    )
    return out, rab if has_rab else None


def _hstu_attn_varlen_backward(
    dout,
    q,
    k,
    v,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts,
    num_targets,
    target_group_size,
    window_size=(-1, -1),
    alpha=1.0,
    rab=None,
    has_drab=False,
    is_delta_q=False,
    descale_q=None,
    descale_k=None,
    descale_v=None,
    descale_do=None,
    deterministic=False,
):
    if rab is not None:
        rab = maybe_contiguous(rab)
    (
        dq,
        dk,
        dv,
        drab,
    ) = hstu_hopper_cuda.varlen_bwd(
        dout,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size[0],
        window_size[1],
        alpha,
        rab,
        has_drab,
        is_delta_q,
        # descale_q,
        # descale_k,
        # descale_v,
        # descale_do,
        deterministic,
    )
    return dq, dk, dv, drab if has_drab else None


class HSTUAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        seq_offsets_q,
        seq_offsets_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=1.0,
        rab=None,
        has_drab=False,
        is_delta_q=False,
        descale_q=None,
        descale_k=None,
        descale_v=None,
        descale_do=None,
    ):
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            out, rab = _hstu_attn_varlen_forward(
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
                is_delta_q,
                descale_q,
                descale_k,
                descale_v,
            )
        ctx.save_for_backward(
            q, k, v, rab, seq_offsets_q, seq_offsets_k, num_contexts, num_targets
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.window_size = window_size
        ctx.descale_q = descale_q
        ctx.descale_k = descale_k
        ctx.descale_v = descale_v
        ctx.descale_do = descale_do
        ctx.has_drab = has_drab
        ctx.alpha = alpha
        ctx.is_delta_q = is_delta_q
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            rab,
            cu_seqlens_q,
            cu_seqlens_k,
            num_contexts,
            num_targets,
        ) = ctx.saved_tensors
        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            dq, dk, dv, drab = _hstu_attn_varlen_backward(
                dout,
                q,
                k,
                v,
                None,
                None,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                num_contexts,
                num_targets,
                ctx.target_group_size,
                ctx.window_size,
                ctx.alpha,
                rab,
                ctx.has_drab,
                ctx.is_delta_q,
                ctx.descale_q,
                ctx.descale_k,
                ctx.descale_v,
                ctx.descale_do,
                False,  # deterministic
            )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        drab = drab[..., : ctx.max_seqlen_k] if ctx.has_drab else None
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
            drab,
            None,
            None,
            None,
            None,
            None,
            None,
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
    descale_q=None,
    descale_k=None,
    descale_v=None,
    descale_do=None,
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
        descale_q: (1,). Descaling factor for the query.
        descale_k: (1,). Descaling factor for the key.
        descale_v: (1,). Descaling factor for the value.
        descale_do: (1,). Descaling factor for the do.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError(
            "AssertError: rab is None, but has_drab is True, is not allowed in backward"
        )
    if num_contexts != None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: context is True and causal is not True, this is undefined behavior"
        )
    if num_targets != None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: target is True and causal is not True, this is undefined behavior"
        )
    if (num_contexts != None and is_delta_q is True) or (
        num_targets != None and is_delta_q is True
    ):
        raise ValueError(
            "AssertError: delta_q is True, but num_contexts or num_targets is not None, this is undefined behavior"
        )
    if num_targets is None and target_group_size < 1:
        raise ValueError(
            "AssertError: target_group_size should be greater than 0 when target is True"
        )
    if max_seqlen_q < max_seqlen_k and window_size != (-1, -1) and is_delta_q is False:
        raise ValueError(
            "AssertError: seq_len_q < seq_len_k, is_delta_q should be True, as is_delta_q represents mask behavior under the case"
        )
    if max_seqlen_q > max_seqlen_k:
        raise ValueError(
            "AssertError: seq_len_q >= seq_len_k, this is undefined behavior"
        )

    return HSTUAttnVarlenFunc.apply(
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
        descale_q,
        descale_k,
        descale_v,
        descale_do,
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
        target_group_size=1,
        window_size=(-1, -1),
        alpha=1.0,
        rab=None,
        has_drab=False,
        is_delta_q=False,
        descale_q=None,
        descale_k=None,
        descale_v=None,
        descale_do=None,
    ):
        q = qkv[:, 0, :, :].detach()
        k = qkv[:, 1, :, :].detach()
        v = qkv[:, 2, :, :].detach()
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            out, rab = _hstu_attn_varlen_forward(
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
                is_delta_q,
                descale_q,
                descale_k,
                descale_v,
            )
        ctx.save_for_backward(
            q, k, v, rab, seq_offsets_q, seq_offsets_k, num_contexts, num_targets
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.window_size = window_size
        ctx.descale_q = descale_q
        ctx.descale_k = descale_k
        ctx.descale_v = descale_v
        ctx.descale_do = descale_do
        ctx.has_drab = has_drab
        ctx.alpha = alpha
        ctx.is_delta_q = is_delta_q
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            rab,
            cu_seqlens_q,
            cu_seqlens_k,
            num_contexts,
            num_targets,
        ) = ctx.saved_tensors
        qkv_shape = (q.shape[0], 3, q.shape[1], q.shape[2])
        dqkv = torch.empty(qkv_shape, device=q.device, dtype=q.dtype)
        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            dq, dk, dv, drab = _hstu_attn_varlen_backward(
                dout,
                q,
                k,
                v,
                dqkv[:, 0, :, :],  # dq
                dqkv[:, 1, :, :],  # dk
                dqkv[:, 2, :, :],  # dv
                cu_seqlens_q,
                cu_seqlens_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                num_contexts,
                num_targets,
                ctx.target_group_size,
                ctx.window_size,
                ctx.alpha,
                rab,
                ctx.has_drab,
                ctx.is_delta_q,
                ctx.descale_q,
                ctx.descale_k,
                ctx.descale_v,
                ctx.descale_do,
                False,  # deterministic
            )
        drab = drab[..., : ctx.max_seqlen_k] if ctx.has_drab else None
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
            drab,
            None,
            None,
            None,
            None,
            None,
            None,
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
    is_delta_q=False,
    descale_q=None,
    descale_k=None,
    descale_v=None,
    descale_do=None,
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
        descale_q: (1,). Descaling factor for the query.
        descale_k: (1,). Descaling factor for the key.
        descale_v: (1,). Descaling factor for the value.
        descale_do: (1,). Descaling factor for the do.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError(
            "AssertError: rab is None, but has_drab is True, is not allowed in backward"
        )
    if num_contexts != None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: context is True and causal is not True, this is undefined behavior"
        )
    if num_targets != None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: target is True and causal is not True, this is undefined behavior"
        )
    if (num_contexts != None and is_delta_q is True) or (
        num_targets != None and is_delta_q is True
    ):
        raise ValueError(
            "AssertError: delta_q is True, but num_contexts or num_targets is not None, this is undefined behavior"
        )
    if num_targets is None and target_group_size < 1:
        raise ValueError(
            "AssertError: target_group_size should be greater than 0 when target is True"
        )
    if max_seqlen_q < max_seqlen_k and window_size != (-1, -1) and is_delta_q is False:
        raise ValueError(
            "AssertError: seq_len_q < seq_len_k, is_delta_q should be True, as is_delta_q represents mask behavior under the case"
        )
    if max_seqlen_q > max_seqlen_k:
        raise ValueError(
            "AssertError: seq_len_q >= seq_len_k, this is undefined behavior"
        )

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
        is_delta_q,
        descale_q,
        descale_k,
        descale_v,
        descale_do,
    )
