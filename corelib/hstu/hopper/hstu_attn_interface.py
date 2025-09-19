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
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import hstu_hopper_cuda

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def quantize_for_two_directions(x, seq_offsets, fp8_type=torch.float8_e4m3fn):
    B = seq_offsets.size(0) - 1
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0
    # x: (total_seq, head, dim)
    if x.dim() != 3:
        raise ValueError(
            "AssertError: x in quantize_for_two_directions should be three dimensions"
        )

    with torch.no_grad():
        x_descale = (
            torch.amax(x.abs(), dim=-1, keepdim=True).to(torch.float32) / fp8_max
        )
        x_descale = torch.max(
            x_descale, torch.tensor([1e-6], dtype=torch.float32, device="cuda")
        )
        x_quantized = (x / x_descale).to(fp8_type)
        x_descale = x_descale.squeeze(-1)
        x_descale = nn.functional.pad(x_descale, (0, 0, 0, 128)).to(torch.float32)
        x_descale = x_descale.transpose(1, 0).contiguous()

        cu_seqlens_xt_descale = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
        for i in range(B):
            actual_len = seq_offsets[i + 1] - seq_offsets[i]
            xt_descale_len = (actual_len + 127) // 128
            cu_seqlens_xt_descale[i + 1] = cu_seqlens_xt_descale[i] + xt_descale_len

        xt_descale = torch.zeros(
            cu_seqlens_xt_descale[-1],
            x.shape[1],
            x.shape[2],
            dtype=torch.float32,
            device="cuda",
        )
        xt_quantized = x.to(fp8_type)
        for i in range(B):
            xt_descale_len = cu_seqlens_xt_descale[i + 1] - cu_seqlens_xt_descale[i]
            for j in range(xt_descale_len - 1):
                xt_descale[cu_seqlens_xt_descale[i] + j] = (
                    torch.amax(
                        x[
                            seq_offsets[i] + j * 128 : seq_offsets[i] + (j + 1) * 128
                        ].abs(),
                        dim=0,
                        keepdim=True,
                    )
                    / fp8_max
                )
                xt_descale[cu_seqlens_xt_descale[i] + j] = torch.max(
                    xt_descale[cu_seqlens_xt_descale[i] + j],
                    torch.tensor([1e-6], dtype=torch.float32, device="cuda"),
                )
                xt_quantized[
                    seq_offsets[i] + j * 128 : seq_offsets[i] + (j + 1) * 128
                ] = (
                    x[seq_offsets[i] + j * 128 : seq_offsets[i] + (j + 1) * 128]
                    / xt_descale[cu_seqlens_xt_descale[i] + j]
                ).to(
                    fp8_type
                )

            xt_descale[cu_seqlens_xt_descale[i] + xt_descale_len - 1] = (
                torch.amax(
                    x[
                        seq_offsets[i] + (xt_descale_len - 1) * 128 : seq_offsets[i + 1]
                    ].abs(),
                    dim=0,
                    keepdim=True,
                )
                / fp8_max
            )
            xt_descale[cu_seqlens_xt_descale[i] + xt_descale_len - 1] = torch.max(
                xt_descale[cu_seqlens_xt_descale[i] + xt_descale_len - 1],
                torch.tensor([1e-6], dtype=torch.float32, device="cuda"),
            )
            xt_quantized[
                seq_offsets[i] + (xt_descale_len - 1) * 128 : seq_offsets[i + 1]
            ] = (
                x[seq_offsets[i] + (xt_descale_len - 1) * 128 : seq_offsets[i + 1]]
                / xt_descale[cu_seqlens_xt_descale[i] + xt_descale_len - 1]
            ).to(
                fp8_type
            )

    return x_quantized, x_descale, xt_quantized, xt_descale, cu_seqlens_xt_descale


def quantize_for_block_scale(
    x, seq_offsets, block_size=128, fp8_type=torch.float8_e4m3fn
):
    # x: (total_seq, head, dim)
    # q and kv might have diffrent block_size
    if x.dim() != 3:
        raise ValueError(
            "AssertError: x in quantize_for_block_scale should be three dimensions"
        )
    B = seq_offsets.size(0) - 1
    head = x.size(1)
    dim = x.size(2)
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0

    cu_seqlens_x_descale = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    x_quantized_list = []
    x_descale_list = []

    with torch.no_grad():
        for i in range(B):
            actual_len = seq_offsets[i + 1] - seq_offsets[i]
            cur_bs_tensor = x[seq_offsets[i] : (seq_offsets[i] + actual_len)]
            actual_len_padding_block_num = (actual_len + block_size - 1) // block_size
            cu_seqlens_x_descale[i + 1] = (
                cu_seqlens_x_descale[i] + actual_len_padding_block_num
            )

            cur_padding_len = actual_len_padding_block_num * block_size - actual_len
            if cur_padding_len > 0:
                pad_tensor = torch.zeros(
                    cur_padding_len,
                    cur_bs_tensor.shape[1],
                    cur_bs_tensor.shape[2],
                    device=cur_bs_tensor.device,
                    dtype=cur_bs_tensor.dtype,
                )
                cur_bs_tensor = torch.cat([cur_bs_tensor, pad_tensor], dim=0)
            else:
                cur_bs_tensor = cur_bs_tensor
            cur_bs_tensor = cur_bs_tensor.view(
                actual_len_padding_block_num, block_size, head, dim
            )
            cur_bs_scale_tensor = (
                torch.amax(cur_bs_tensor.abs(), dim=(1, 3), keepdim=True).to(
                    torch.float32
                )
                / fp8_max
            )
            cur_bs_scale_tensor = torch.max(
                cur_bs_scale_tensor,
                torch.tensor([1e-6], dtype=torch.float32, device="cuda"),
            )
            x_descale_list.append(cur_bs_scale_tensor)
            cur_bs_tensor_quantized = (
                (cur_bs_tensor / cur_bs_scale_tensor)
                .to(fp8_type)
                .view(actual_len_padding_block_num * block_size, head, dim)[
                    0:actual_len
                ]
            )  # [actual_len_padding_block_num * cur_block_size, head, dim] - > [actual_len, head, dim]
            x_quantized_list.append(cur_bs_tensor_quantized)

        x_quantized = torch.cat(x_quantized_list, dim=0)
        x_descale = torch.cat(x_descale_list, dim=0)  # [total_seq, head]

    assert (
        x_quantized.shape == x.shape
    ), "assert x_quantized shape must equal to x shape"
    return (
        x_quantized,
        x_descale.squeeze(1).squeeze(-1).transpose(1, 0).contiguous(),
        cu_seqlens_x_descale,
    )  # For x_descale, the original layout is ([sum(cur_bs_len/bm), head]: (head, 1)), and we transform into ([head, sum(cur_bs_len/bm): (sum(cur_bs_len/bm), 1)])


def get_bm_and_bn_block_size_fwd(rab, dim):
    """
    Design for fp8, Returns the block size for BM and BN. Need to be the same as the "get_tile_size_fwd" function.
    BM: Block size for the first dimension of the input tensor.
    BN: Block size for the second dimension of the input tensor.
    """
    if rab is not None:
        if dim == 64:
            return 128, 128
        else:
            return 128, 64
    else:
        if dim == 64:
            return 128, 128
        elif dim == 128:
            return 128, 128
        else:
            return 128, 64


def get_bm_and_bn_block_size_bwd():
    """
    Design for fp8, Returns the block size for BM and BN. Need to be the same as the "get_tile_size_bwd" function.
    BM: Block size for the first dimension of the input tensor.
    BN: Block size for the second dimension of the input tensor.
    """
    return 64, 128


def quantize_for_head_batch_tensor(
    x, seq_offsets, quant_mode=3, fp8_type=torch.float8_e4m3fn
):
    B = seq_offsets.size(0) - 1
    head = x.size(1)
    fp8_max = 448.0 if fp8_type == torch.float8_e4m3fn else 57344.0
    # x: (total_seq, head, dim)
    if x.dim() != 3:
        raise ValueError(
            "AssertError: x in quantize_for_head_batch_tensor should be three dimensions"
        )
    if quant_mode != 3 and quant_mode != 4 and quant_mode != 5:
        raise ValueError(
            "AssertError: quant_mode in quantize_for_head_batch_tensor should be 3, 4 or 5"
        )

    if quant_mode == 3:
        with torch.no_grad():
            x_descale = torch.zeros(B, head, dtype=torch.float32, device="cuda")
            x_quantized = torch.zeros_like(x, dtype=fp8_type, device="cuda")
            for i in range(B):
                x_descale[i, :] = (
                    torch.amax(
                        x[seq_offsets[i] : seq_offsets[i + 1], :, :].abs(),
                        dim=(0, 2),
                        keepdim=True,
                    )
                    .squeeze(0)
                    .squeeze(-1)
                    / fp8_max
                )
                x_descale[i, :] = torch.max(
                    x_descale[i, :],
                    torch.tensor([1e-6], dtype=torch.float32, device="cuda"),
                )
                x_quantized[seq_offsets[i] : seq_offsets[i + 1], :, :] = (
                    x[seq_offsets[i] : seq_offsets[i + 1], :, :]
                    / x_descale[i, :].unsqueeze(0).unsqueeze(-1)
                ).to(fp8_type)
        return x_quantized, x_descale
    elif quant_mode == 4:
        with torch.no_grad():
            x_descale = torch.zeros(B, dtype=torch.float32, device="cuda")
            x_quantized = torch.zeros_like(x, dtype=fp8_type, device="cuda")
            for i in range(B):
                x_descale[i] = (
                    torch.amax(
                        x[seq_offsets[i] : seq_offsets[i + 1], :, :].abs(), keepdim=True
                    )
                    / fp8_max
                )
                x_descale[i] = torch.max(
                    x_descale[i],
                    torch.tensor([1e-6], dtype=torch.float32, device="cuda"),
                )
                x_quantized[seq_offsets[i] : seq_offsets[i + 1], :, :] = (
                    x[seq_offsets[i] : seq_offsets[i + 1], :, :] / x_descale[i]
                ).to(fp8_type)
        return x_quantized, x_descale
    else:
        with torch.no_grad():
            x_descale = (
                torch.amax(x.abs(), keepdim=True).squeeze(0).squeeze(-1) / fp8_max
            )
            x_descale = torch.max(
                x_descale, torch.tensor([1e-6], dtype=torch.float32, device="cuda")
            )
            x_quantized = (x / x_descale).to(fp8_type)
        return x_quantized, x_descale


def _hstu_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts,
    num_targets,
    target_group_size,
    window_size=(-1, -1),
    alpha=1.0,
    quant_mode=0,
    rab=None,
    func=None,
    vt=None,
    cu_seqlens_descale_vt=None,
    descale_q=None,
    descale_k=None,
    descale_v=None,
    descale_vt=None,
    cu_seqlens_block_descale_q=None,
    cu_seqlens_block_descale_kv=None,
):
    if rab is not None:
        rab = maybe_contiguous(rab)
    out, rab = hstu_hopper_cuda.varlen_fwd(
        q,
        k,
        v,
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
        func,
        quant_mode,
        vt,
        cu_seqlens_descale_vt,
        descale_q,
        descale_k,
        descale_v,
        descale_vt,
        cu_seqlens_block_descale_q,
        cu_seqlens_block_descale_kv,
    )
    return out, rab


def _hstu_attn_varlen_backward(
    dout,
    dout_t,
    q,
    q_t,
    k,
    k_t,
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
    quant_mode=0,
    rab=None,
    has_drab=False,
    func=None,
    descale_q=None,
    descale_qt=None,
    descale_k=None,
    descale_kt=None,
    descale_v=None,
    descale_do=None,
    descale_dot=None,
    cu_seqlens_descale_qt=None,
    cu_seqlens_descale_kt=None,
    cu_seqlens_q_block_descale=None,
    cu_seqlens_kv_block_descale=None,
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
        dout_t,
        q,
        q_t,
        k,
        k_t,
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
        quant_mode,
        rab,
        has_drab,
        func,
        descale_q,
        descale_qt,
        descale_k,
        descale_kt,
        descale_v,
        descale_do,
        descale_dot,
        cu_seqlens_descale_qt,
        cu_seqlens_descale_kt,
        cu_seqlens_q_block_descale,
        cu_seqlens_kv_block_descale,
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
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=1.0,
        rab=None,
        has_drab=False,
        func=None,
        quant_mode=-1,
    ):
        vt = None
        descale_q = None
        descale_k = None
        descale_v = None
        descale_vt = None
        cu_seqlens_descale_vt = None
        cu_seqlens_block_descale_q = None
        cu_seqlens_block_descale_k = None
        ctx.q_fp16 = q
        ctx.k_fp16 = k
        ctx.v_fp16 = v
        if quant_mode == 0:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)
            descale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            descale_k = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            descale_v = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        elif quant_mode == 1:
            q, descale_q, _, _, _ = quantize_for_two_directions(
                q, cu_seqlens_q, fp8_type=torch.float8_e4m3fn
            )
            k, descale_k, _, _, _ = quantize_for_two_directions(
                k, cu_seqlens_k, fp8_type=torch.float8_e4m3fn
            )
            (
                v,
                descale_v,
                vt,
                descale_vt,
                cu_seqlens_descale_vt,
            ) = quantize_for_two_directions(
                v, cu_seqlens_k, fp8_type=torch.float8_e4m3fn
            )
            vt = vt.transpose(0, 2).contiguous().transpose(0, 2).detach()
        elif quant_mode == 2:  # block_scale
            dim = q.shape[-1]
            bm, bn = get_bm_and_bn_block_size_fwd(rab, dim)
            q, descale_q, cu_seqlens_block_descale_q = quantize_for_block_scale(
                q, cu_seqlens_q, block_size=bm, fp8_type=torch.float8_e4m3fn
            )
            k, descale_k, cu_seqlens_block_descale_k = quantize_for_block_scale(
                k, cu_seqlens_k, block_size=bn, fp8_type=torch.float8_e4m3fn
            )
            v, descale_v, cu_seqlens_block_descale_k = quantize_for_block_scale(
                v, cu_seqlens_k, block_size=bn, fp8_type=torch.float8_e4m3fn
            )
        elif quant_mode == 3 or quant_mode == 4 or quant_mode == 5:
            q, descale_q = quantize_for_head_batch_tensor(
                q, cu_seqlens_q, quant_mode=quant_mode, fp8_type=torch.float8_e4m3fn
            )
            k, descale_k = quantize_for_head_batch_tensor(
                k, cu_seqlens_k, quant_mode=quant_mode, fp8_type=torch.float8_e4m3fn
            )
            v, descale_v = quantize_for_head_batch_tensor(
                v, cu_seqlens_k, quant_mode=quant_mode, fp8_type=torch.float8_e4m3fn
            )

        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            out, rab = _hstu_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size,
                alpha,
                quant_mode,
                rab,
                func,
                vt,
                cu_seqlens_descale_vt,
                descale_q,
                descale_k,
                descale_v,
                descale_vt,
                cu_seqlens_block_descale_q,
                cu_seqlens_block_descale_k,
            )
        ctx.save_for_backward(
            q, k, v, rab, cu_seqlens_q, cu_seqlens_k, num_contexts, num_targets
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.window_size = window_size
        ctx.has_drab = has_drab
        ctx.alpha = alpha
        ctx.quant_mode = quant_mode
        ctx.func = func
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
        dout_t = None
        qt = None
        kt = None
        descale_q = None
        descale_qt = None
        descale_k = None
        descale_kt = None
        descale_v = None
        descale_do = None
        descale_dot = None
        cu_seqlens_descale_qt = None
        cu_seqlens_descale_kt = None
        cu_seqlens_block_descale_q = None
        cu_seqlens_block_descale_k = None
        bwd_fp8_type = torch.float8_e4m3fn
        if ctx.quant_mode == 0:
            q = q.to(bwd_fp8_type)
            k = k.to(bwd_fp8_type)
            v = v.to(bwd_fp8_type)
            dout = dout.to(bwd_fp8_type)
            descale_q = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            descale_k = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            descale_v = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            descale_do = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        elif ctx.quant_mode == 1:
            (
                q,
                descale_q,
                qt,
                descale_qt,
                cu_seqlens_descale_qt,
            ) = quantize_for_two_directions(
                ctx.q_fp16, cu_seqlens_q, fp8_type=bwd_fp8_type
            )
            qt = qt.transpose(0, 2).contiguous().transpose(0, 2).detach()
            (
                k,
                descale_k,
                kt,
                descale_kt,
                cu_seqlens_descale_kt,
            ) = quantize_for_two_directions(
                ctx.k_fp16, cu_seqlens_k, fp8_type=bwd_fp8_type
            )
            kt = kt.transpose(0, 2).contiguous().transpose(0, 2).detach()
            v, descale_v, _, _, _ = quantize_for_two_directions(
                ctx.v_fp16, cu_seqlens_k, fp8_type=bwd_fp8_type
            )
            dout, descale_do, dout_t, descale_dot, _ = quantize_for_two_directions(
                dout, cu_seqlens_q, fp8_type=bwd_fp8_type
            )
            dout_t = dout_t.transpose(0, 2).contiguous().transpose(0, 2).detach()
        elif ctx.quant_mode == 2:
            q.shape[-1]
            bm, bn = get_bm_and_bn_block_size_bwd()
            q, descale_q, cu_seqlens_block_descale_q = quantize_for_block_scale(
                ctx.q_fp16, cu_seqlens_q, block_size=bm, fp8_type=bwd_fp8_type
            )
            k, descale_k, cu_seqlens_block_descale_k = quantize_for_block_scale(
                ctx.k_fp16, cu_seqlens_k, block_size=bn, fp8_type=bwd_fp8_type
            )
            v, descale_v, _ = quantize_for_block_scale(
                ctx.v_fp16, cu_seqlens_k, block_size=bn, fp8_type=bwd_fp8_type
            )
            dout, descale_do, _ = quantize_for_block_scale(
                dout, cu_seqlens_q, block_size=bm, fp8_type=bwd_fp8_type
            )
        elif ctx.quant_mode == 3 or ctx.quant_mode == 4 or ctx.quant_mode == 5:
            q, descale_q = quantize_for_head_batch_tensor(
                ctx.q_fp16,
                cu_seqlens_q,
                quant_mode=ctx.quant_mode,
                fp8_type=bwd_fp8_type,
            )
            k, descale_k = quantize_for_head_batch_tensor(
                ctx.k_fp16,
                cu_seqlens_k,
                quant_mode=ctx.quant_mode,
                fp8_type=bwd_fp8_type,
            )
            v, descale_v = quantize_for_head_batch_tensor(
                ctx.v_fp16,
                cu_seqlens_k,
                quant_mode=ctx.quant_mode,
                fp8_type=bwd_fp8_type,
            )
            dout, descale_do = quantize_for_head_batch_tensor(
                dout, cu_seqlens_q, quant_mode=ctx.quant_mode, fp8_type=bwd_fp8_type
            )

        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            dq, dk, dv, drab = _hstu_attn_varlen_backward(
                dout,
                dout_t,
                q,
                qt,
                k,
                kt,
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
                ctx.quant_mode,
                rab,
                ctx.has_drab,
                ctx.func,
                descale_q,
                descale_qt,
                descale_k,
                descale_kt,
                descale_v,
                descale_do,
                descale_dot,
                cu_seqlens_descale_qt,
                cu_seqlens_descale_kt,
                cu_seqlens_block_descale_q,
                cu_seqlens_block_descale_k,
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
        )


def hstu_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts=None,
    num_targets=None,
    target_group_size=1,
    window_size=(-1, -1),
    alpha=1.0,
    rab=None,
    has_drab=False,
    func=None,
    quant_mode=-1,
):
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
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
        func: (nheads, total_q + 256). Function for describe the mask shape.
        quant_mode: int. -1: no quantization, 0: cast to fp8, 1: 1xDIM&128x1 quantization, 2: per-block quantization, 3: per-head quantization, 4: per-batch quantization, 5: per-tensor quantization
    Return:
        out: (total_q, nheads, headdim).
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
    if num_targets is None and target_group_size < 1:
        raise ValueError(
            "AssertError: target_group_size should be greater than 0 when target is True"
        )
    if max_seqlen_q > max_seqlen_k:
        raise ValueError(
            "AssertError: seq_len_q >= seq_len_k, this is undefined behavior"
        )

    return HSTUAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        func,
        quant_mode,
    )


class HstuAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=1.0,
        rab=None,
        has_drab=False,
        func=None,
        quant_mode=-1,
    ):
        q = qkv[:, 0, :, :].detach()
        k = qkv[:, 1, :, :].detach()
        v = qkv[:, 2, :, :].detach()
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            out, rab = _hstu_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size,
                alpha,
                quant_mode,
                rab,
                func,
            )
        ctx.save_for_backward(
            q, k, v, rab, cu_seqlens_q, cu_seqlens_k, num_contexts, num_targets
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.window_size = window_size
        ctx.has_drab = has_drab
        ctx.alpha = alpha
        ctx.quant_mode = quant_mode
        ctx.func = func
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
                None,
                q,
                None,
                k,
                None,
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
                ctx.quant_mode,
                rab,
                ctx.has_drab,
                ctx.func,
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
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts=None,
    num_targets=None,
    target_group_size=1,
    window_size=(-1, -1),
    alpha=1.0,
    rab=None,
    has_drab=False,
    func=None,
    quant_mode=-1,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
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
        func: (nheads, total_q + 256). Function for describe the mask shape.
        quant_mode: int. -1: no quantization, 0: cast to fp8, 1: 1xDIM&128x1 quantization, 2: per-block quantization, 3: per-head quantization, 4: per-batch quantization, 5: per-tensor quantization
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
    if num_targets is None and target_group_size < 1:
        raise ValueError(
            "AssertError: target_group_size should be greater than 0 when target is True"
        )
    if max_seqlen_q > max_seqlen_k:
        raise ValueError(
            "AssertError: seq_len_q >= seq_len_k, this is undefined behavior"
        )

    return HstuAttnQKVPackedFunc.apply(
        qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        func,
        quant_mode,
    )
