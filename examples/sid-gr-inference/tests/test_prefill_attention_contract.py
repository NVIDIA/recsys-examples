# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest
from gr_inference.gr_kernels.prefill import (
    AutoPrefillBackend,
    MissingPrefillBackend,
    PrefillAttention,
    PrefillAttentionInputs,
    TorchSDPAPrefillBackend,
)
from gr_inference.gr_kv import ContextKV, TensorSpec
from gr_inference.gr_runtime import ContextKVWriter, GRPrefillRunner


def test_prefill_attention_validates_shapes() -> None:
    attention = PrefillAttention(backend=lambda inputs: inputs.q)

    out = attention(
        PrefillAttentionInputs(
            q=TensorSpec("q", (1, 128, 16, 64)),
            k=TensorSpec("k", (1, 128, 4, 64)),
            v=TensorSpec("v", (1, 128, 4, 64)),
            layer_idx=0,
        )
    )

    assert out.hidden_shape == (1, 128, 16, 64)


def test_prefill_attention_rejects_bad_gqa() -> None:
    attention = PrefillAttention(backend=lambda inputs: inputs.q)

    with pytest.raises(ValueError, match="must be divisible"):
        attention(
            PrefillAttentionInputs(
                q=TensorSpec("q", (1, 128, 10, 64)),
                k=TensorSpec("k", (1, 128, 4, 64)),
                v=TensorSpec("v", (1, 128, 4, 64)),
                layer_idx=0,
            )
        )


def test_prefill_attention_requires_backend() -> None:
    attention = PrefillAttention()

    with pytest.raises(MissingPrefillBackend):
        attention(
            PrefillAttentionInputs(
                q=TensorSpec("q", (1, 128, 16, 64)),
                k=TensorSpec("k", (1, 128, 4, 64)),
                v=TensorSpec("v", (1, 128, 4, 64)),
                layer_idx=0,
            )
        )


def test_context_kv_writer_validates_layer_shape() -> None:
    context_kv = ContextKV(
        TensorSpec("context_k", (28, 1, 4700, 8, 128)),
        TensorSpec("context_v", (28, 1, 4700, 8, 128)),
    )
    writer = ContextKVWriter(context_kv)

    write = writer.validate_layer(
        0,
        TensorSpec("k_layer", (1, 4700, 8, 128)),
        TensorSpec("v_layer", (1, 4700, 8, 128)),
    )

    assert write.expected_layer_shape == (1, 4700, 8, 128)


def test_context_kv_writer_rejects_bad_layer_shape() -> None:
    context_kv = ContextKV(
        TensorSpec("context_k", (28, 1, 4700, 8, 128)),
        TensorSpec("context_v", (28, 1, 4700, 8, 128)),
    )
    writer = ContextKVWriter(context_kv)

    with pytest.raises(ValueError, match="k layer shape"):
        writer.validate_layer(
            0,
            TensorSpec("k_layer", (1, 4096, 8, 128)),
            TensorSpec("v_layer", (1, 4700, 8, 128)),
        )


def test_torch_sdpa_prefill_backend_optional_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    q = torch.randn(1, 16, 4, 32, device=device, dtype=dtype)
    k = torch.randn(1, 16, 2, 32, device=device, dtype=dtype)
    v = torch.randn_like(k)
    context_k = torch.empty(1, 1, 16, 2, 32, device=device, dtype=dtype)
    context_v = torch.empty_like(context_k)

    runner = GRPrefillRunner(PrefillAttention(TorchSDPAPrefillBackend()))
    hidden = runner.run_layer(
        q=q,
        k=k,
        v=v,
        context_kv=ContextKV(context_k, context_v),
        layer_idx=0,
    )

    assert tuple(hidden.shape) == (1, 16, 4, 32)
    assert torch.equal(context_k[0], k)
    assert torch.equal(context_v[0], v)


def test_auto_prefill_backend_falls_back_to_sdpa() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    q = torch.randn(1, 16, 4, 32, device=device, dtype=dtype)
    k = torch.randn(1, 16, 2, 32, device=device, dtype=dtype)
    v = torch.randn_like(k)

    backend = AutoPrefillBackend(prefer=("torch_sdpa",))
    attention = PrefillAttention(backend)
    output = attention(
        PrefillAttentionInputs(
            q=q,
            k=k,
            v=v,
            layer_idx=0,
        )
    )

    assert tuple(output.hidden.shape) == (1, 16, 4, 32)
    assert backend.selected_backend == "torch_sdpa"
