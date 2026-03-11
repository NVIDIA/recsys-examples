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
Smoke test for the FBGEMM HSTU attention (hstu package) against PyTorch reference.

New package:
  - hstu.hstu_attn_varlen_func  (unified, auto-dispatch by GPU arch)
"""

import fbgemm_gpu  # noqa: F401
import pytest
import torch
from hstu import hstu_attn_varlen_func as new_hstu_attn_func


def get_arch_sm():
    major = torch.cuda.get_device_properties(0).major
    minor = torch.cuda.get_device_properties(0).minor
    return f"{major}{minor}"


@pytest.mark.parametrize("batchsize", [32])
@pytest.mark.parametrize("max_seqlen", [200])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("max_num_targets", [10])
@pytest.mark.parametrize("max_num_contextuals", [0, 4])
def test_fbgemm_hstu_fwd_bwd(
    batchsize,
    max_seqlen,
    head_dim,
    num_heads,
    max_num_targets,
    max_num_contextuals,
):
    """Test forward + backward of FBGEMM HSTU against PyTorch reference."""
    arch_sm = get_arch_sm()
    if arch_sm[0] not in ("8", "9"):
        pytest.skip(f"Unsupported SM major version: {arch_sm}")

    from commons.utils.hstu_assert_close import assert_hstu_close
    from ops.pt_ops.pt_hstu_attention import pytorch_hstu_mha

    device = torch.device("cuda")
    lengths = torch.randint(
        low=2,
        high=max_seqlen + 1,
        size=(batchsize,),
        device=device,
        dtype=torch.int,
    )
    num_targets = torch.randint(
        low=0,
        high=max_num_targets + 1,
        size=(batchsize,),
        device=device,
        dtype=torch.int32,
    )
    num_targets = torch.clamp(
        num_targets, max=lengths - 1, min=torch.zeros_like(num_targets)
    )

    num_contextuals = torch.randint(
        low=0,
        high=max_num_contextuals + 1,
        size=(batchsize,),
        device=device,
        dtype=torch.int32,
    )
    num_contextuals = torch.clamp(
        num_contextuals,
        max=lengths - 1 - num_targets,
        min=torch.zeros_like(num_contextuals),
    )

    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    L = int(seq_offsets[-1].item())
    alpha = 1.0 / (head_dim**0.5)

    x = torch.empty(
        (L, num_heads * head_dim), dtype=torch.bfloat16, device=device
    ).uniform_(-0.1, 0.1)

    head_dim_total = num_heads * head_dim
    normed = torch.nn.functional.layer_norm(x, (head_dim_total,))
    linear = (
        torch.nn.Linear(head_dim_total, head_dim_total * 3, bias=False)
        .cuda()
        .bfloat16()
    )
    mixed = linear(normed)
    mixed = torch.nn.functional.layer_norm(mixed, (mixed.shape[-1],))
    tq, tk, tv = torch.split(
        mixed, [head_dim_total, head_dim_total, head_dim_total], dim=-1
    )
    tq, tk, tv = tq.contiguous(), tk.contiguous(), tv.contiguous()

    cu_seqlens = seq_offsets.to(torch.int32)

    # New FBGEMM HSTU
    nq = tq.view(-1, num_heads, head_dim).detach().clone().requires_grad_(True)
    nk = tk.view(-1, num_heads, head_dim).detach().clone().requires_grad_(True)
    nv = tv.view(-1, num_heads, head_dim).detach().clone().requires_grad_(True)

    new_out = new_hstu_attn_func(
        nq,
        nk,
        nv,
        cu_seqlens,
        cu_seqlens,
        None,
        None,  # seqused_q, seqused_k
        max_seqlen,
        max_seqlen,
        max_seqlen,  # scaling_seqlen
        num_contextuals,
        num_targets,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
    )

    # PyTorch reference (bf16)
    ref_q = tq.view(-1, num_heads, head_dim).detach().clone().requires_grad_(True)
    ref_k = tk.view(-1, num_heads, head_dim).detach().clone().requires_grad_(True)
    ref_v = tv.view(-1, num_heads, head_dim).detach().clone().requires_grad_(True)
    ref_out = pytorch_hstu_mha(
        max_seq_len=max_seqlen,
        alpha=alpha,
        q=ref_q,
        k=ref_k,
        v=ref_v,
        seq_offsets=seq_offsets,
        num_contextuals=num_contextuals,
        num_targets=num_targets,
        causal=True,
        dropout_pr=0.0,
        training=True,
        target_group_size=1,
        scaling_seqlen=max_seqlen,
    )

    # PyTorch reference (fp32)
    ref_q_fp32 = ref_q.detach().clone().float().requires_grad_(True)
    ref_k_fp32 = ref_k.detach().clone().float().requires_grad_(True)
    ref_v_fp32 = ref_v.detach().clone().float().requires_grad_(True)
    ref_out_fp32 = pytorch_hstu_mha(
        max_seq_len=max_seqlen,
        alpha=alpha,
        q=ref_q_fp32,
        k=ref_k_fp32,
        v=ref_v_fp32,
        seq_offsets=seq_offsets,
        num_contextuals=num_contextuals,
        num_targets=num_targets,
        causal=True,
        dropout_pr=0.0,
        training=True,
        target_group_size=1,
        scaling_seqlen=max_seqlen,
    )

    torch.cuda.synchronize()
    assert_hstu_close(new_out, ref_out, ref_out_fp32, fwd=True)
    print(f"[FWD] sm{arch_sm} head_dim={head_dim} ctx={max_num_contextuals} PASS")

    dout = torch.rand_like(new_out)
    new_out.backward(dout)
    ref_out.backward(dout)
    ref_out_fp32.backward(dout.float())
    torch.cuda.synchronize()

    assert_hstu_close(nq.grad, ref_q.grad, ref_q_fp32.grad, fwd=False)
    assert_hstu_close(nk.grad, ref_k.grad, ref_k_fp32.grad, fwd=False)
    assert_hstu_close(nv.grad, ref_v.grad, ref_v_fp32.grad, fwd=False)
    print(f"[BWD] sm{arch_sm} head_dim={head_dim} ctx={max_num_contextuals} PASS")
