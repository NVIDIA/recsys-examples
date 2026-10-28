# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
from gr_inference.gr_kernels.attention import (
    ExistingGRDecodeAttentionBackend,
    GRDecodeAttention,
    GRDecodeAttentionInputs,
    MissingKernelBackend,
    existing_kernel_backend,
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


def _kernels_called(monkeypatch, tmp_path, *, fused, capability):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        existing_kernel_backend, "_cuda_capability_major", lambda _device: capability
    )
    called = []
    monkeypatch.setitem(
        sys.modules,
        "gr_inference.gr_kernels.attention.fused_beam_decode",
        SimpleNamespace(
            beam_decode_fused=lambda q, *args, **kwargs: called.append("fused") or q
        ),
    )
    (tmp_path / "interface.py").touch()
    kwargs = {} if fused is None else {"fused_beam_combine": fused}
    backend = ExistingGRDecodeAttentionBackend(tmp_path, **kwargs)
    backend._beam_decode_attn = lambda q, *args, **kwargs: called.append("public") or q
    context_key = torch.zeros(1, 2, 8, 1, 4)
    beam_key = torch.zeros(1, 2, 2, 2, 1, 4)

    backend(
        GRDecodeAttentionInputs(
            q=torch.zeros(2, 2, 2, 4),
            context_kv=ContextKV(context_key, torch.zeros_like(context_key)),
            beam_kv=BeamKV(beam_key, torch.zeros_like(beam_key)),
            beam_path=BeamPath(max_decode_steps=2, max_beam_width=2),
            layer_idx=0,
            step=1,
            active_beam_width=2,
            topk_indices=torch.zeros(2, 1, 2, 2, 2, dtype=torch.int32),
            decode_nums=1,
        )
    )
    return called


@pytest.mark.parametrize(
    ("fused", "expected"), [(None, "fused"), (True, "fused"), (False, "public")]
)
def test_existing_kernel_backend_calls_the_fused_kernel_unless_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fused, expected: str
) -> None:
    called = _kernels_called(monkeypatch, tmp_path, fused=fused, capability=10)

    assert called == [expected]


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (10, "fused"),  # B200, B300
        (9, "public"),  # Hopper
        (8, "public"),  # A100, L40S
        (12, "public"),  # consumer Blackwell
        (None, "public"),  # no GPU
    ],
)
def test_existing_kernel_backend_calls_the_fused_kernel_only_on_sm10x(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capability, expected: str
) -> None:
    called = _kernels_called(monkeypatch, tmp_path, fused=True, capability=capability)

    assert called == [expected]


def _valid_fused_inputs(torch):
    return {
        "q": torch.zeros(2, 1, 2, 2, 4, dtype=torch.bfloat16),
        "topk_indices": torch.zeros(2, 1, 2, 2, 2, dtype=torch.int32),
        "decode_nums": 1,
        "seqused_k": torch.tensor((5, 8), dtype=torch.int32),
        "cu_seqlens_k": None,
        "backend": "3kernel",
    }


def test_fused_beam_combine_rejects_inputs_the_public_function_would_reject() -> None:
    torch = pytest.importorskip("torch")
    from gr_inference.gr_kernels.attention.fused_beam_inputs import check_fused_inputs

    check_fused_inputs(**_valid_fused_inputs(torch))

    def rejected(message, **changes):
        with pytest.raises(ValueError, match=message):
            check_fused_inputs(**{**_valid_fused_inputs(torch), **changes})

    bf16, i32 = torch.bfloat16, torch.int32
    rejected("q must be", q=torch.zeros(2, 2, 2, 2, 4, dtype=bf16))
    rejected("host int", decode_nums=torch.tensor(1, dtype=i32))
    rejected("host int", decode_nums=-1)
    rejected("seqused_k", seqused_k=torch.tensor((5, 8)))
    rejected("seqused_k", seqused_k=torch.tensor((5, 8, 8), dtype=i32))
    offsets = torch.tensor((0, 5, 13), dtype=i32)
    rejected("cu_seqlens_k", seqused_k=None, cu_seqlens_k=offsets.long())
    rejected("cu_seqlens_k", seqused_k=None, cu_seqlens_k=offsets[:2])
    rejected("mutually exclusive", cu_seqlens_k=offsets)
    rejected("backend='3kernel'", backend="dsl")
    rejected("backend='3kernel'", seqused_k=None, cu_seqlens_k=offsets, backend="dsl")
    rejected("same device", topk_indices=torch.zeros(2, 1, 2, 2, 2, device="meta"))
    rejected("same device", seqused_k=torch.tensor((5, 8), dtype=i32, device="meta"))
    rejected(
        "same device",
        seqused_k=None,
        cu_seqlens_k=torch.tensor((0, 5, 13), dtype=i32, device="meta"),
    )


def test_fused_kernel_takes_only_the_rows_it_is_validated_for() -> None:
    torch = pytest.importorskip("torch")
    from gr_inference.gr_kernels.attention.fused_beam_inputs import (
        supports_vector_rows,
    )

    def rows(*shape, dtype=torch.bfloat16):
        return torch.zeros(*shape, dtype=dtype)

    q, k, v = rows(2, 1, 4, 8, 128), rows(2, 16, 2, 128), rows(2, 16, 2, 128)
    assert supports_vector_rows(q, k, v)

    head_dim_64 = rows(2, 16, 2, 64)
    assert not supports_vector_rows(rows(2, 1, 4, 8, 64), head_dim_64, head_dim_64)
    assert not supports_vector_rows(rows(2, 1, 4, 8, 128, dtype=torch.float32), k, v)

    shape = (2, 16, 2, 128)
    bad_rows = {
        "misaligned": torch.zeros(2 * 16 * 2 * 128 + 1, dtype=torch.bfloat16)[1:].view(
            shape
        ),
        "strided": torch.zeros(2, 16, 2, 256, dtype=torch.bfloat16)[..., ::2],
        "odd_pitch": torch.zeros(2, 16, 2, 129, dtype=torch.bfloat16)[..., :128],
    }
    for bad in bad_rows.values():
        assert not supports_vector_rows(q, bad, v)
        assert not supports_vector_rows(q, k, bad)
        assert not supports_vector_rows(bad.new_zeros(2, 1, 4, 8, 129)[..., :128], k, v)
