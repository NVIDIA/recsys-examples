"""Smoke test for the real gr-decode_atten backend.

This test is intentionally small and optional. It is skipped when torch, CUDA,
or the external CuTe DSL kernel dependencies are unavailable.
"""

from __future__ import annotations

import importlib.util

import pytest
from gr_inference.gr_kernels.attention import (
    ExistingGRDecodeAttentionBackend,
    GRDecodeAttention,
    MissingKernelBackend,
)
from gr_inference.gr_kv import BeamKV, BeamPath, ContextKV
from gr_inference.gr_runtime import GRDecodeEngine, GRRequestState


def _existing_backend_or_skip() -> ExistingGRDecodeAttentionBackend:
    try:
        return ExistingGRDecodeAttentionBackend().ensure_available()
    except MissingKernelBackend as exc:
        pytest.skip(str(exc))


def test_real_decode_attention_backend_smoke() -> None:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    backend = _existing_backend_or_skip()

    batch = 1
    layers = 1
    context_len = 256
    max_decode_steps = 3
    decode_nums = 1
    beam_width = 128
    head_q = 16
    head_kv = 4
    head_dim = 64
    dtype = torch.bfloat16
    device = "cuda"

    torch.manual_seed(0)

    q = torch.randn(batch, beam_width, head_q, head_dim, dtype=dtype, device=device)
    context_k = torch.randn(
        layers, batch, context_len, head_kv, head_dim, dtype=dtype, device=device
    )
    context_v = torch.randn_like(context_k)
    beam_k = torch.randn(
        layers,
        batch,
        max_decode_steps,
        beam_width,
        head_kv,
        head_dim,
        dtype=dtype,
        device=device,
    )
    beam_v = torch.randn_like(beam_k)

    qhead_per_kv = head_q // head_kv
    topk_kv = torch.randint(
        0,
        decode_nums * beam_width,
        (batch, 1, head_kv, max_decode_steps, beam_width),
        dtype=torch.int32,
        device=device,
    )
    topk_indices = topk_kv.repeat_interleave(qhead_per_kv, dim=2)

    request = GRRequestState(
        request_id="real-kernel-smoke",
        context_kv=ContextKV(context_k, context_v),
        beam_kv=BeamKV(beam_k, beam_v),
        beam_path=BeamPath(
            max_decode_steps=max_decode_steps,
            max_beam_width=beam_width,
        ),
    )
    engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=backend),
        fixed_beam_width=beam_width,
    )

    try:
        output = engine.decode_attention_step(
            request,
            q,
            layer_idx=0,
            step=1,
            topk_indices=topk_indices,
            decode_nums=decode_nums,
            return_lse=True,
            backend_name="dsl",
        )
    except MissingKernelBackend as exc:
        pytest.skip(str(exc))

    out, lse = output.attention_output
    assert out.shape == (batch, 1, beam_width, head_q, head_dim)
    assert lse.shape == (batch, 1, beam_width, head_q)
    assert out.dtype == dtype
    assert lse.dtype == torch.float32
