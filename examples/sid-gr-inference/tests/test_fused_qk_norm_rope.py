# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest


def _qwen3_kernels():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from gr_inference_trtllm_kernels import qwen3

    if qwen3._cuda_extension() is None:
        pytest.skip("gr_trtllm CUDA extension is not available")
    return torch, qwen3


def _inputs(torch, tokens, q_heads, kv_heads, head_dim, dtype, extra_columns=0):
    generator = torch.Generator(device="cuda").manual_seed(7)
    packed = (q_heads + 2 * kv_heads) * head_dim + extra_columns
    qkv = torch.randn(tokens, packed, device="cuda", generator=generator).mul_(3)
    weights = 1 + 0.1 * torch.randn(2, head_dim, device="cuda", generator=generator)
    position_ids = torch.randint(
        0, 2048, (tokens,), device="cuda", generator=generator, dtype=torch.int32
    )
    return qkv.to(dtype), weights[0].to(dtype), weights[1].to(dtype), position_ids


def _run_op(torch, qwen3, qkv, q_heads, kv_heads, head_dim, q_weight, k_weight, **kw):
    qwen3.reset_call_counts()
    torch.ops.gr_trtllm.fused_qk_norm_rope(
        qkv,
        q_heads,
        kv_heads,
        kv_heads,
        head_dim,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        1e6,
        kw.get("is_neox", True),
        kw["position_ids"],
        1.0,
        0,
        0,
        1.0,
        kw.get("is_qk_norm", True),
    )
    torch.cuda.synchronize()
    calls = qwen3.call_counts()
    assert calls.get("fused_qk_norm_rope_cuda") == 1
    return calls.get("fused_qk_norm_rope_cuda_fast", 0) == 1


@pytest.mark.parametrize(
    ("dtype", "head_dim", "is_neox", "is_qk_norm", "fast"),
    [
        ("bfloat16", 64, True, True, True),
        ("bfloat16", 128, True, True, True),
        ("bfloat16", 128, False, True, True),
        ("bfloat16", 256, True, True, True),
        ("bfloat16", 96, True, True, False),  # not a multiple of 64
        ("bfloat16", 192, True, True, False),  # no fast-kernel instantiation
        ("bfloat16", 128, True, False, False),
        ("float16", 128, True, True, False),
        ("float32", 128, True, True, False),
    ],
)
def test_cuda_qk_norm_rope_takes_the_fast_kernel_only_where_it_applies(
    dtype, head_dim, is_neox, is_qk_norm, fast
) -> None:
    torch, qwen3 = _qwen3_kernels()

    qkv, q_weight, k_weight, position_ids = _inputs(
        torch, 5, 2, 1, head_dim, getattr(torch, dtype)
    )
    assert (
        _run_op(
            torch,
            qwen3,
            qkv,
            2,
            1,
            head_dim,
            q_weight,
            k_weight,
            position_ids=position_ids,
            is_neox=is_neox,
            is_qk_norm=is_qk_norm,
        )
        is fast
    )


@pytest.mark.parametrize(
    "layout",
    [
        "padded_rows",
        "no_v_heads",
        "long_position_ids",
        "strided_qkv",
        "strided_q_weight",
        "strided_k_weight",
        "strided_position_ids",
        "misaligned_qkv",
    ],
)
def test_cuda_fast_qk_norm_rope_declines_unsupported_layouts(layout) -> None:
    torch, qwen3 = _qwen3_kernels()

    tokens, q_heads, kv_heads, head_dim = 6, 2, 1, 128
    v_heads = 0 if layout == "no_v_heads" else kv_heads
    extra = 2 if layout == "padded_rows" else 0
    qkv, q_weight, k_weight, position_ids = _inputs(
        torch, tokens, q_heads, kv_heads, head_dim, torch.bfloat16, extra
    )
    qkv = qkv[:, : (q_heads + kv_heads + v_heads) * head_dim + extra]
    if layout == "strided_qkv":
        qkv = torch.cat([qkv, qkv], dim=1)[:, : qkv.shape[1]]
    else:
        qkv = qkv.contiguous()
    if layout == "strided_q_weight":
        q_weight = q_weight.repeat_interleave(2)[::2]
    if layout == "strided_k_weight":
        k_weight = k_weight.repeat_interleave(2)[::2]
    if layout == "strided_position_ids":
        position_ids = position_ids.repeat_interleave(2)[::2]
    if layout == "long_position_ids":
        position_ids = torch.cat([position_ids, position_ids[:1]])
    if layout == "misaligned_qkv":
        qkv = torch.cat([qkv.new_empty(1), qkv.flatten()])[1:].view(qkv.shape)

    extension = qwen3._cuda_extension()
    fast = extension.fused_qk_norm_rope_cuda(
        qkv,
        q_heads,
        kv_heads,
        head_dim,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        1e6,
        True,
        position_ids,
        1.0,
        True,
    )
    torch.cuda.synchronize()

    assert fast is False


@pytest.mark.parametrize(
    ("tensor", "dtype"),
    [("qkv", "float16"), ("q_weight", "float32"), ("k_weight", "float32")],
)
def test_cuda_fast_qk_norm_rope_leaves_mixed_dtypes_to_the_per_head_kernel(
    tensor, dtype
) -> None:
    torch, qwen3 = _qwen3_kernels()

    qkv, q_weight, k_weight, position_ids = _inputs(torch, 4, 2, 1, 128, torch.bfloat16)
    tensors = {"qkv": qkv, "q_weight": q_weight, "k_weight": k_weight}
    tensors[tensor] = tensors[tensor].to(getattr(torch, dtype))

    # The per-head kernel reads both weights as the qkv dtype and rejects a mismatch.
    with pytest.raises(RuntimeError, match="BFloat16"):
        qwen3._cuda_extension().fused_qk_norm_rope_cuda(
            tensors["qkv"],
            2,
            1,
            128,
            128,
            1e-6,
            tensors["q_weight"],
            tensors["k_weight"],
            1e6,
            True,
            position_ids,
            1.0,
            True,
        )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize(
    ("tokens", "q_heads", "kv_heads"),
    [(7, 2, 1), (5, 6, 1), (9, 24, 8), (3, 32, 8)],  # the last two share a rotary table
)
def test_cuda_fast_qk_norm_rope_matches_the_per_head_kernel(
    head_dim, is_neox, tokens, q_heads, kv_heads
) -> None:
    torch, qwen3 = _qwen3_kernels()

    # Two extra columns per row make the fast kernel decline, so the padded
    # copy runs the per-head kernel on the same rows.
    padded, q_weight, k_weight, position_ids = _inputs(
        torch, tokens, q_heads, kv_heads, head_dim, torch.bfloat16, extra_columns=2
    )
    qkv = padded[:, :-2].contiguous()
    args = (q_heads, kv_heads, head_dim, q_weight, k_weight)
    kw = {"position_ids": position_ids, "is_neox": is_neox}

    assert _run_op(torch, qwen3, qkv, *args, **kw)
    assert not _run_op(torch, qwen3, padded, *args, **kw)

    # The per-head kernel rounds to bf16 after RMSNorm, so outputs differ by up to one bf16 ulp.
    q_k = (q_heads + kv_heads) * head_dim
    torch.testing.assert_close(
        qkv[:, :q_k].float(), padded[:, :q_k].float(), atol=0.04, rtol=0.008
    )
    assert torch.equal(qkv[:, q_k:], padded[:, q_k:-2])


def test_cuda_fast_qk_norm_rope_declines_more_elements_than_int32_indexes() -> None:
    torch, qwen3 = _qwen3_kernels()
    if torch.cuda.mem_get_info()[0] < 6 * 2**30:
        pytest.skip("needs 6 GiB of free GPU memory")

    tokens = 2**31 // 256 + 1
    qkv = torch.zeros(tokens, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.zeros(tokens, device="cuda", dtype=torch.int32)

    assert not _run_op(
        torch, qwen3, qkv, 2, 1, 64, weight, weight, position_ids=position_ids
    )


def test_cuda_fast_qk_norm_rope_leaves_rows_past_the_last_token_untouched() -> None:
    torch, qwen3 = _qwen3_kernels()

    # 7 tokens x 3 q/k heads leave the last block with warps past the last token.
    buffer, q_weight, k_weight, position_ids = _inputs(
        torch, 8, 2, 1, 128, torch.bfloat16
    )
    original = buffer.clone()

    assert _run_op(
        torch,
        qwen3,
        buffer[:7],
        2,
        1,
        128,
        q_weight,
        k_weight,
        position_ids=position_ids[:7],
    )
    assert torch.equal(buffer[7:], original[7:])


TOKEN_KERNEL = "GR_INFERENCE_TRTLLM_QK_NORM_ROPE_TOKEN_KERNEL"


def _launched_kernel(
    torch,
    qwen3,
    qkv,
    q_heads,
    kv_heads,
    head_dim,
    q_weight,
    k_weight,
    position_ids,
    is_neox=True,
):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        qwen3._cuda_extension().fused_qk_norm_rope_cuda(
            qkv,
            q_heads,
            kv_heads,
            head_dim,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            1e6,
            is_neox,
            position_ids,
            1.0,
            True,
        )
        torch.cuda.synchronize()
    launched = [
        event.name for event in prof.events() if "fused_qk_norm_rope" in event.name
    ]
    assert len(launched) == 1
    return launched[0]


@pytest.mark.parametrize(
    ("tokens", "q_heads", "kv_heads", "head_dim", "is_neox", "token_kernel"),
    [
        (1023, 6, 1, 128, True, False),
        (1024, 2, 1, 128, True, True),
        (1024, 6, 1, 128, True, True),
        (1031, 6, 1, 128, True, True),
        (1024, 24, 8, 128, True, True),
        (1024, 32, 8, 128, True, False),  # more q/k heads than 4 warps x 8 heads
        (1024, 6, 1, 64, True, False),
        (1024, 6, 1, 256, True, False),
        (1024, 6, 1, 128, False, False),
    ],
)
def test_cuda_qk_norm_rope_takes_the_token_kernel_only_where_it_applies(
    monkeypatch: pytest.MonkeyPatch,
    tokens,
    q_heads,
    kv_heads,
    head_dim,
    is_neox,
    token_kernel,
) -> None:
    monkeypatch.setenv(TOKEN_KERNEL, "1")
    torch, qwen3 = _qwen3_kernels()

    qkv, q_weight, k_weight, position_ids = _inputs(
        torch, tokens, q_heads, kv_heads, head_dim, torch.bfloat16
    )
    kernel = _launched_kernel(
        torch,
        qwen3,
        qkv,
        q_heads,
        kv_heads,
        head_dim,
        q_weight,
        k_weight,
        position_ids,
        is_neox,
    )

    expected = (
        "fused_qk_norm_rope_token_kernel"
        if token_kernel
        else "fused_qk_norm_rope_fast_kernel"
    )
    assert expected in kernel


@pytest.mark.parametrize("tokens", [1024, 1031])
@pytest.mark.parametrize(
    ("q_heads", "kv_heads"),
    [(2, 1), (6, 1), (24, 8)],  # 1, 2 (with a partial last row) and 8 heads per warp
)
def test_cuda_token_qk_norm_rope_matches_the_fast_kernel_exactly(
    monkeypatch: pytest.MonkeyPatch, tokens, q_heads, kv_heads
) -> None:
    torch, qwen3 = _qwen3_kernels()

    qkv, q_weight, k_weight, position_ids = _inputs(
        torch, tokens, q_heads, kv_heads, 128, torch.bfloat16
    )
    # No profiler here, so the test also runs under compute-sanitizer.
    outputs = {}
    for setting in ("0", "1"):
        monkeypatch.setenv(TOKEN_KERNEL, setting)
        outputs[setting] = qkv.clone()
        assert _run_op(
            torch,
            qwen3,
            outputs[setting],
            q_heads,
            kv_heads,
            128,
            q_weight,
            k_weight,
            position_ids=position_ids,
        )

    assert torch.equal(outputs["1"], outputs["0"])


@pytest.mark.parametrize("setting", [None, "1", "0"])
def test_cuda_token_qk_norm_rope_runs_only_on_sm10x_unless_forced(
    monkeypatch: pytest.MonkeyPatch, setting
) -> None:
    if setting is None:
        monkeypatch.delenv(TOKEN_KERNEL, raising=False)
    else:
        monkeypatch.setenv(TOKEN_KERNEL, setting)
    torch, qwen3 = _qwen3_kernels()

    qkv, q_weight, k_weight, position_ids = _inputs(
        torch, 1024, 6, 1, 128, torch.bfloat16
    )
    kernel = _launched_kernel(
        torch, qwen3, qkv, 6, 1, 128, q_weight, k_weight, position_ids
    )

    enabled = setting == "1" or (
        setting is None and torch.cuda.get_device_capability()[0] == 10
    )
    assert ("fused_qk_norm_rope_token_kernel" in kernel) is enabled


def _two_gpu_kernels():
    torch, qwen3 = _qwen3_kernels()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    return torch, qwen3


@pytest.mark.parametrize("extra_columns, fast", [(0, True), (2, False)])
def test_cuda_qk_norm_rope_runs_on_the_tensors_device(extra_columns, fast) -> None:
    torch, qwen3 = _two_gpu_kernels()

    with torch.cuda.device(1):
        qkv, q_weight, k_weight, position_ids = _inputs(
            torch, 6, 2, 1, 128, torch.bfloat16, extra_columns
        )
        expected = qkv.clone()
        args = (2, 1, 128, q_weight, k_weight)
        assert _run_op(torch, qwen3, expected, *args, position_ids=position_ids) is fast

    assert _run_op(torch, qwen3, qkv, *args, position_ids=position_ids) is fast
    assert torch.equal(qkv, expected)


def test_cuda_qk_norm_rope_rejects_weights_on_another_device() -> None:
    torch, qwen3 = _two_gpu_kernels()

    with torch.cuda.device(1):
        qkv, q_weight, k_weight, position_ids = _inputs(
            torch, 6, 2, 1, 128, torch.bfloat16
        )
    with pytest.raises(RuntimeError):
        _run_op(
            torch,
            qwen3,
            qkv,
            2,
            1,
            128,
            q_weight.to("cuda:0"),
            k_weight,
            position_ids=position_ids,
        )
    assert qwen3.call_counts().get("fused_qk_norm_rope_cuda_failed") == 1
