import fbgemm_gpu  # noqa: F401
import pytest  # noqa: F401
import torch
from commons.utils.hstu_assert_close import assert_hstu_close
from hopper.hstu_attn_interface import hstu_attn_varlen_func as hopper_attn_func
from hstu_attn import hstu_attn_varlen_func as ampere_attn_func
from ops.pt_ops.pt_hstu_attention import pytorch_hstu_mha as pytorch_hstu_mha


def get_arch_sm():
    sm_major_version = torch.cuda.get_device_properties(0).major
    sm_minor_version = torch.cuda.get_device_properties(0).minor
    return f"{sm_major_version}{sm_minor_version}"


def preprocess_input(input_tensor):
    head_dim = input_tensor.shape[-1]
    normed_x = torch.nn.functional.layer_norm(input_tensor, (head_dim,))
    linear_module = (
        torch.nn.Linear(head_dim, head_dim * 3, bias=False).cuda().bfloat16()
    )
    mixed_qkv = linear_module(normed_x)
    tq, tk, tv = torch.split(mixed_qkv, [head_dim, head_dim, head_dim], dim=-1)
    return tq.contiguous(), tk.contiguous(), tv.contiguous()


@pytest.mark.parametrize("batchsize", [32])
@pytest.mark.parametrize("max_seqlen", [200])
@pytest.mark.parametrize("head_dim", [32])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("max_num_targets", [10])
@pytest.mark.parametrize("max_num_contextuals", [4])
def test_hstn_fwd_bwd(
    batchsize,
    max_seqlen,
    head_dim,
    num_heads,
    max_num_targets,
    max_num_contextuals,
):
    arch_sm = get_arch_sm()
    if arch_sm[0] == "8":
        hstu_attn_func = ampere_attn_func
    elif arch_sm[0] == "9":
        hstu_attn_func = hopper_attn_func
    else:
        raise ValueError(f"Unsupported SM major version: {arch_sm}")
    device = torch.device("cuda")
    max_seqlen = torch.randint(10, 100, (1,)).item()
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
    )  # at least 1 history

    num_contextuals = torch.randint(
        low=0,
        high=max_num_contextuals + 1,
        size=(batchsize,),
        device=device,
        dtype=torch.int32,
    )
    num_contextuals = torch.clamp(
        num_contextuals,
        max=lengths - 1 - num_targets if num_targets is not None else lengths - 1,
        min=torch.zeros_like(num_contextuals),
    )  # at least 1 history!!
    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    L = int(seq_offsets[-1].item())
    x = (
        torch.empty(
            (L, num_heads * head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        .uniform_(-0.1, 0.1)
        .requires_grad_(False)
    )

    q, k, v = preprocess_input(x)

    q = q.view(-1, num_heads, head_dim)
    k = k.view(-1, num_heads, head_dim)
    v = v.view(-1, num_heads, head_dim)
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    cu_seqlens_q = seq_offsets.clone()
    cu_seqlens_k = seq_offsets.clone()

    max_seqlen_q = max_seqlen
    max_seqlen_k = max_seqlen

    out = hstu_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contextuals,
        num_targets,
        rab=None,
        alpha=1.0 / (head_dim**0.5),
        target_group_size=1,
        window_size=(-1, 0),
    )

    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)
    ref_out = pytorch_hstu_mha(
        max_seq_len=max_seqlen,
        alpha=1.0 / (head_dim**0.5),
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

    ref_q_fp32 = ref_q.detach().clone().float().requires_grad_(True)
    ref_k_fp32 = ref_k.detach().clone().float().requires_grad_(True)
    ref_v_fp32 = ref_v.detach().clone().float().requires_grad_(True)
    ref_out_fp32 = pytorch_hstu_mha(
        max_seq_len=max_seqlen,
        alpha=1.0 / (head_dim**0.5),
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
    assert_hstu_close(out, ref_out, ref_out_fp32, fwd=True)
    print(f"sm {arch_sm} fwd pass")
    dout = torch.rand_like(out)
    # torch.testing.assert_close(out, ref_out)
    out.backward(dout)
    ref_out.backward(dout)
    ref_out_fp32.backward(dout.float())
    torch.cuda.synchronize()

    print(f"sm {arch_sm} bwd pass")

    assert_hstu_close(q.grad, ref_q.grad, ref_q_fp32.grad, fwd=False)
    assert_hstu_close(k.grad, ref_k.grad, ref_k_fp32.grad, fwd=False)
    assert_hstu_close(v.grad, ref_v.grad, ref_v_fp32.grad, fwd=False)
