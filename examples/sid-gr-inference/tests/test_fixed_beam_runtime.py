# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gr_inference.gr_kernels.attention import (
    ExistingGRDecodeAttentionBackend,
    GRDecodeAttention,
    MissingKernelBackend,
)
from gr_inference.gr_kv import BeamKV, BeamPath, ContextKV, TensorSpec
from gr_inference.gr_runtime import GRDecodeEngine, GRRequestState


def make_request() -> GRRequestState:
    context_kv = ContextKV(
        TensorSpec("context_key", (28, 1, 4700, 8, 128)),
        TensorSpec("context_value", (28, 1, 4700, 8, 128)),
    )
    beam_kv = BeamKV(
        TensorSpec("beam_key", (28, 1, 3, 128, 8, 128)),
        TensorSpec("beam_value", (28, 1, 3, 128, 8, 128)),
    )
    return GRRequestState(
        request_id="req-1",
        context_kv=context_kv,
        beam_kv=beam_kv,
        beam_path=BeamPath(max_decode_steps=3, max_beam_width=128),
    )


def test_runtime_dispatches_to_existing_kernel_backend_contract() -> None:
    calls = []

    def fake_backend(inputs):
        calls.append(inputs)
        return TensorSpec("attention_out", (1, 128, 16, 128))

    engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=fake_backend),
        fixed_beam_width=128,
    )
    request = make_request()
    q = TensorSpec("q", (1, 128, 16, 128))

    output = engine.decode_attention_step(request, q, layer_idx=0, step=0)

    assert output.request_id == "req-1"
    assert output.step == 0
    assert output.active_beam_width == 128
    assert output.attention_output.shape == (1, 128, 16, 128)
    assert calls[0].beam_kv.flattened_beam_shape() == (1, 384, 8, 128)


def test_runtime_passes_existing_kernel_metadata() -> None:
    calls = []

    def fake_backend(inputs):
        calls.append(inputs)
        return "ok"

    engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=fake_backend),
        fixed_beam_width=128,
    )
    request = make_request()
    q = TensorSpec("q", (1, 128, 16, 128))
    topk = TensorSpec("topk_indices", (1, 1, 16, 3, 128), dtype="int32")

    output = engine.decode_attention_step(
        request,
        q,
        layer_idx=0,
        step=1,
        topk_indices=topk,
        decode_nums=1,
        return_lse=True,
        backend_name="3kernel",
    )

    assert output.attention_output == "ok"
    assert calls[0].topk_indices is topk
    assert calls[0].decode_nums == 1
    assert calls[0].return_lse is True
    assert calls[0].backend_name == "3kernel"


def test_runtime_rejects_missing_kernel_backend() -> None:
    engine = GRDecodeEngine(attention=GRDecodeAttention(), fixed_beam_width=128)
    request = make_request()
    q = TensorSpec("q", (1, 128, 16, 128))

    with pytest.raises(MissingKernelBackend):
        engine.decode_attention_step(request, q, layer_idx=0, step=0)


def test_attention_wrapper_rejects_wrong_q_width() -> None:
    engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=128,
    )
    request = make_request()
    q = TensorSpec("q", (1, 64, 16, 128))

    with pytest.raises(ValueError, match="active_beam_width"):
        engine.decode_attention_step(request, q, layer_idx=0, step=0)


def test_attention_wrapper_rejects_wrong_topk_shape() -> None:
    engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=128,
    )
    request = make_request()
    q = TensorSpec("q", (1, 128, 16, 128))
    topk = TensorSpec("topk_indices", (1, 1, 8, 3, 128), dtype="int32")

    with pytest.raises(ValueError, match="batch/head dimensions"):
        engine.decode_attention_step(
            request,
            q,
            layer_idx=0,
            step=1,
            topk_indices=topk,
            decode_nums=1,
        )


def test_existing_kernel_backend_resolves_downloaded_kernel_path() -> None:
    backend = ExistingGRDecodeAttentionBackend()

    assert backend.kernel_root.name == "gr-decode_atten"
