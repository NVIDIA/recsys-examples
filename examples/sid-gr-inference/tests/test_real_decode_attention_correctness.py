# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B correctness test for the real gr-decode_atten backend."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def _load_reference(kernel_root: Path):
    ref_path = kernel_root / "tests" / "reference.py"
    if not ref_path.is_file():
        pytest.skip(f"reference.py not found under {kernel_root}")

    spec = importlib.util.spec_from_file_location("gr_decode_atten_reference", ref_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load reference module from {ref_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_decode_attention_matches_reference() -> None:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    backend = _existing_backend_or_skip()

    reference = _load_reference(backend.kernel_root)

    batch = 1
    context_len = 256
    max_decode_steps = 3
    decode_nums = 3
    beam_width = 128
    head_q = 16
    head_kv = 4
    head_dim = 64
    dtype = torch.bfloat16
    device = "cuda"

    torch.manual_seed(123)

    q_decode = torch.randn(
        batch, 1, beam_width, head_q, head_dim, dtype=dtype, device=device
    )
    q_framework = q_decode[:, 0]
    k_context = torch.randn(
        batch, context_len, head_kv, head_dim, dtype=dtype, device=device
    )
    v_context = torch.randn_like(k_context)
    k_beam_flat = torch.randn(
        batch,
        decode_nums * beam_width,
        head_kv,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_beam_flat = torch.randn_like(k_beam_flat)

    qhead_per_kv = head_q // head_kv
    topk_kv = torch.randint(
        0,
        decode_nums * beam_width,
        (batch, 1, head_kv, max_decode_steps, beam_width),
        dtype=torch.int32,
        device=device,
    )
    topk_indices = topk_kv.repeat_interleave(qhead_per_kv, dim=2)

    context_kv = ContextKV(
        k_context.unsqueeze(0),
        v_context.unsqueeze(0),
    )
    beam_kv = BeamKV(
        k_beam_flat.reshape(
            batch, max_decode_steps, beam_width, head_kv, head_dim
        ).unsqueeze(0),
        v_beam_flat.reshape(
            batch, max_decode_steps, beam_width, head_kv, head_dim
        ).unsqueeze(0),
    )
    request = GRRequestState(
        request_id="real-kernel-correctness",
        context_kv=context_kv,
        beam_kv=beam_kv,
        beam_path=BeamPath(
            max_decode_steps=max_decode_steps,
            max_beam_width=beam_width,
        ),
    )
    engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=backend),
        fixed_beam_width=beam_width,
    )

    output = engine.decode_attention_step(
        request,
        q_framework,
        layer_idx=0,
        step=decode_nums - 1,
        topk_indices=topk_indices,
        decode_nums=decode_nums,
        return_lse=True,
        backend_name="dsl",
    )
    out, lse = output.attention_output

    out_ref, lse_ref = reference.beam_attention_ref(
        q_decode,
        k_context,
        v_context,
        k_beam_flat,
        v_beam_flat,
        topk_indices,
        decode_nums,
    )
    out_pt, _ = reference.beam_attention_ref(
        q_decode,
        k_context,
        v_context,
        k_beam_flat,
        v_beam_flat,
        topk_indices,
        decode_nums,
        upcast=False,
    )

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    kernel_diff = (out.float() - out_ref).abs().max().item()
    pt_diff = (out_pt.float() - out_ref).abs().max().item()
    assert kernel_diff <= 2 * pt_diff + fwd_atol

    finite_mask = lse.isfinite() & lse_ref.isfinite()
    lse_diff = (
        (lse - lse_ref).abs()[finite_mask].max().item() if finite_mask.any() else 0.0
    )
    assert lse_diff <= 1e-3
