# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-level smoke test through the real gr-decode_atten backend."""

from __future__ import annotations

import importlib.util

import pytest
from gr_inference.gr_kernels.attention import (
    ExistingGRDecodeAttentionBackend,
    GRDecodeAttention,
    GRDecodeAttentionInputs,
    MissingKernelBackend,
)


def _existing_backend_or_skip() -> ExistingGRDecodeAttentionBackend:
    try:
        return ExistingGRDecodeAttentionBackend().ensure_available()
    except MissingKernelBackend as exc:
        pytest.skip(str(exc))


def test_qwen3_model_real_decode_attention_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    backend = _existing_backend_or_skip()

    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import GRDecodeEngine, GRGenerationState

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-real-decode",
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_context_len=256,
        max_seq_len=260,
        max_decode_steps=3,
        max_beam_width=128,
        intermediate_size=1024,
        vocab_size=256,
    )
    dtype = torch.bfloat16
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        dtype=dtype,
    ).cuda()

    input_ids = torch.randint(0, config.vocab_size, (1, 256), device="cuda")
    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="real-model-decode",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
        fixed_beam_width=config.max_beam_width,
    )
    selection = generation.initialize_beams()
    beam_token_ids = torch.tensor(
        [selection.token_ids], dtype=torch.long, device="cuda"
    )

    qhead_per_kv = config.num_attention_heads // config.num_kv_heads
    topk_kv = torch.arange(config.max_beam_width, device="cuda", dtype=torch.int32)
    topk_kv = topk_kv.view(1, 1, 1, 1, config.max_beam_width)
    topk_kv = topk_kv.expand(
        1,
        1,
        config.num_kv_heads,
        config.max_decode_steps,
        config.max_beam_width,
    )
    topk_indices = topk_kv.repeat_interleave(qhead_per_kv, dim=2).contiguous()

    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=backend),
        fixed_beam_width=config.max_beam_width,
    )
    logits = model.forward_decode_step(
        beam_token_ids,
        generation,
        decode_engine,
        step=0,
        topk_indices=topk_indices,
        decode_nums=1,
        return_lse=True,
        backend_name="dsl",
    )

    assert tuple(logits.shape) == (1, config.max_beam_width, config.vocab_size)
    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()


def test_continuous_serving_real_decode_attention_multi_step_smoke() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    backend = _existing_backend_or_skip()

    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import GRDecodeEngine
    from gr_inference.gr_serving import (
        GRContinuousBatchingPolicy,
        GRContinuousScheduler,
        GRContinuousServingExecutor,
        GRServingConfig,
        GRServingEngine,
        GRServingRequest,
    )

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-real-continuous-decode",
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_context_len=16,
        max_seq_len=18,
        max_decode_steps=2,
        max_beam_width=128,
        intermediate_size=1024,
        vocab_size=256,
    )
    dtype = torch.bfloat16
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        dtype=dtype,
    ).cuda()
    engine = GRServingEngine(
        model=model,
        decode_engine=GRDecodeEngine(
            attention=GRDecodeAttention(backend=backend),
            fixed_beam_width=config.max_beam_width,
        ),
        config=GRServingConfig(
            max_decode_steps=2,
            max_beam_width=config.max_beam_width,
            enable_batched_decode=True,
            return_beam_details=True,
        ),
    )
    executor = GRContinuousServingExecutor(
        engine=engine,
        scheduler=GRContinuousScheduler(
            policy=GRContinuousBatchingPolicy(
                max_prefill_batch_size=2,
                max_decode_batch_size=2,
            )
        ),
        synchronize=torch.cuda.synchronize,
    )
    with torch.no_grad():
        for idx in range(2):
            executor.submit(
                GRServingRequest(
                    request_id=f"req-{idx}",
                    input_ids=torch.randint(
                        0, config.vocab_size, (1, 16), device="cuda"
                    ),
                    max_decode_steps=2,
                    beam_width=config.max_beam_width,
                )
            )
        responses = executor.run_until_empty()

    assert len(responses) == 2
    assert executor.scheduler.status()["ticks"] == 2
    assert all(response.metadata["decode_steps"] == 2 for response in responses)
    assert all(
        response.metadata["stop_reason"] == "max_decode_steps" for response in responses
    )
    assert all(
        tuple(response.scores) and torch.isfinite(torch.tensor(response.scores)).all()
        for response in responses
    )
    assert all(
        len(response.metadata["beam_details"]) == config.max_beam_width
        for response in responses
    )
    assert len(responses[0].metadata["beam_details"][0]["token_ids"]) == 3


@pytest.mark.parametrize("beam_width", [8, 16, 32, 64])
def test_existing_gr_decode_attention_backend_small_beam_width_smoke(
    beam_width: int,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    backend = _existing_backend_or_skip()

    from gr_inference.gr_kv import BeamKV, BeamPath, ContextKV

    batch = 1
    context_len = 128
    decode_nums = 1
    num_layers = 1
    head_q = 16
    head_kv = 4
    head_dim = 64
    dtype = torch.bfloat16
    q = torch.randn(batch, beam_width, head_q, head_dim, device="cuda", dtype=dtype)
    context_kv = ContextKV(
        torch.randn(
            num_layers,
            batch,
            context_len,
            head_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
        ),
        torch.randn(
            num_layers,
            batch,
            context_len,
            head_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
        ),
    )
    beam_kv = BeamKV(
        torch.randn(
            num_layers,
            batch,
            decode_nums,
            beam_width,
            head_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
        ),
        torch.randn(
            num_layers,
            batch,
            decode_nums,
            beam_width,
            head_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
        ),
    )
    beam_path = BeamPath(max_decode_steps=decode_nums, max_beam_width=beam_width)
    beam_path.append(
        parent_beams=tuple(0 for _ in range(beam_width)),
        token_ids=tuple(range(beam_width)),
        scores=tuple(float(-idx) for idx in range(beam_width)),
    )
    topk_indices = torch.arange(beam_width, device="cuda", dtype=torch.int32).view(
        1,
        1,
        1,
        1,
        beam_width,
    )
    topk_indices = topk_indices.expand(
        batch, 1, head_q, decode_nums, beam_width
    ).contiguous()

    out = backend(
        GRDecodeAttentionInputs(
            q=q,
            context_kv=context_kv,
            beam_kv=beam_kv,
            beam_path=beam_path,
            layer_idx=0,
            step=0,
            active_beam_width=beam_width,
            topk_indices=topk_indices,
            decode_nums=decode_nums,
            backend_name="dsl",
        )
    )

    assert tuple(out.shape) == (batch, 1, beam_width, head_q, head_dim)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
