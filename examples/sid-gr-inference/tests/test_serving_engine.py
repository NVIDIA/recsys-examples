# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest


def _torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    import torch

    return torch


def _tiny_config(*, max_decode_steps: int = 1, max_beam_width: int = 2):
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig

    return Qwen3GRConfig(
        model_name="tiny-serving-qwen3-gr",
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=8,
        max_seq_len=10,
        max_decode_steps=max_decode_steps,
        max_beam_width=max_beam_width,
        intermediate_size=64,
        vocab_size=32,
    )


def _serving_engine(
    *,
    max_decode_steps: int = 1,
    max_beam_width: int = 2,
    backend=None,
    **serving_config_kwargs,
):
    from gr_inference.gr_kernels.attention import GRDecodeAttention
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRModel
    from gr_inference.gr_runtime import GRDecodeEngine
    from gr_inference.gr_serving import GRServingConfig, GRServingEngine

    config = _tiny_config(
        max_decode_steps=max_decode_steps,
        max_beam_width=max_beam_width,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    engine = GRServingEngine(
        model=model,
        decode_engine=GRDecodeEngine(
            attention=GRDecodeAttention(backend=backend or (lambda inputs: inputs.q)),
            fixed_beam_width=max_beam_width,
        ),
        config=GRServingConfig(
            max_decode_steps=max_decode_steps,
            max_beam_width=max_beam_width,
            **serving_config_kwargs,
        ),
    )
    return engine, config


def _request(
    torch,
    config,
    *,
    request_id: str = "req-1",
    max_decode_steps: int = 1,
    beam_width: int = 2,
    **kwargs,
):
    from gr_inference.gr_serving import GRServingRequest

    return GRServingRequest(
        request_id=request_id,
        input_ids=torch.randint(0, config.vocab_size, (1, 8)),
        max_decode_steps=max_decode_steps,
        beam_width=beam_width,
        **kwargs,
    )


def _batch(torch, config, *, count: int = 2, max_decode_steps: int = 1, **kwargs):
    from gr_inference.gr_serving import GRRequestBatch

    return GRRequestBatch(
        tuple(
            _request(
                torch,
                config,
                request_id=f"req-{idx}",
                max_decode_steps=max_decode_steps,
                **kwargs,
            )
            for idx in range(count)
        )
    )


def _item_provider(items, *, vocab_size: int):
    from gr_inference.gr_runtime import TokenTrie, TrieItemMaskProvider

    return TrieItemMaskProvider(TokenTrie.from_items(items), vocab_size=vocab_size)


def test_serving_request_validation() -> None:
    from gr_inference.gr_serving import GRServingRequest

    GRServingRequest(
        request_id="req-1",
        input_ids=object(),
        max_decode_steps=1,
        beam_width=2,
    ).validate()


def test_serving_request_rejects_invalid_logits_processor() -> None:
    from gr_inference.gr_serving import GRServingRequest

    request = GRServingRequest(
        request_id="req-1",
        input_ids=object(),
        max_decode_steps=1,
        beam_width=2,
        logits_processors=(object(),),
    )

    with pytest.raises(ValueError, match="logits_processors"):
        request.validate()


def test_serving_engine_generates_with_tiny_model() -> None:
    torch = _torch()
    engine, config = _serving_engine()
    assert engine.status()["max_beam_width"] == 2
    provider = _item_provider(
        [("item-a", (1, 10)), ("item-b", (2, 11))],
        vocab_size=config.vocab_size,
    )

    request = _request(torch, config, item_mask_provider=provider)
    response = engine.generate(request)

    assert response.request_id == "req-1"
    assert len(response.token_ids) == 2
    assert len(response.scores) == 2
    assert response.metadata["total_ms"] >= 0.0
    assert response.metadata["prefill_ms"] >= 0.0
    assert response.metadata["decode_ms"] >= 0.0
    assert response.metadata.get("batched_prefill") is None
    assert len(response.metadata["item_results"]) == 2
    assert all(result["is_complete"] for result in response.metadata["item_results"])
    assert {result["item_id"] for result in response.metadata["item_results"]} == {
        "item-a",
        "item-b",
    }
    assert engine.warmup(request).metadata["warmup"] is True


def test_serving_engine_applies_initial_logits_processor_before_selection() -> None:
    torch = _torch()

    def force_initial_stop_tokens(logits, context):
        assert context.request_id == "req-processor"
        assert context.phase == "prefill"
        forced = torch.full_like(logits, -torch.inf)
        forced[:, -1, 7] = 3.0
        forced[:, -1, 8] = 2.0
        return forced

    def fail_if_decode_runs(inputs):
        raise AssertionError(
            "decode should not run after processor selects stop tokens"
        )

    engine, config = _serving_engine(backend=fail_if_decode_runs)
    request = _request(
        torch,
        config,
        request_id="req-processor",
        stop_token_ids=(7, 8),
        logits_processors=(force_initial_stop_tokens,),
    )

    response = engine.generate(request)

    assert response.token_ids == (7, 8)
    assert response.metadata["decode_steps"] == 0
    assert response.metadata["stop_reason"] == "stop_token"


def test_serving_engine_applies_decode_logits_processor_before_selection() -> None:
    torch = _torch()
    seen = []

    def force_decode_stop_tokens(logits, context):
        seen.append((context.phase, context.step, context.beam_width))
        forced = torch.full_like(logits, -torch.inf)
        if context.phase == "prefill":
            forced[:, -1, 11] = 3.0
            forced[:, -1, 12] = 2.0
        else:
            forced[:, :, 5] = 100.0
        return forced

    engine, config = _serving_engine()
    request = _request(
        torch,
        config,
        request_id="req-decode-processor",
        stop_token_ids=(5,),
        logits_processors=(force_decode_stop_tokens,),
    )

    response = engine.generate(request)

    assert response.token_ids == (5, 5)
    assert response.metadata["decode_steps"] == 1
    assert response.metadata["stop_reason"] == "stop_token"
    assert seen == [("prefill", 0, 2), ("decode", 0, 2)]


def test_serving_engine_stops_when_initial_item_beams_are_complete() -> None:
    torch = _torch()

    def fail_if_decode_runs(inputs):
        raise AssertionError("decode should not run after all initial beams complete")

    engine, config = _serving_engine(max_decode_steps=2, backend=fail_if_decode_runs)
    provider = _item_provider(
        [("item-a", (1,)), ("item-b", (2,))], vocab_size=config.vocab_size
    )
    request = _request(
        torch,
        config,
        request_id="req-complete",
        max_decode_steps=2,
        item_mask_provider=provider,
    )

    response = engine.generate(request)

    assert response.metadata["stop_reason"] == "item_complete"
    assert response.metadata["decode_steps"] == 0
    assert set(response.token_ids) == {1, 2}
    assert all(result["is_complete"] for result in response.metadata["item_results"])


def test_sync_serving_scheduler_runs_until_empty() -> None:
    torch = _torch()
    from gr_inference.gr_serving import SchedulerPolicy, SyncGRScheduler

    engine, config = _serving_engine()
    scheduler = SyncGRScheduler(engine, policy=SchedulerPolicy(max_batch_size=2))
    scheduler.submit(_request(torch, config))

    responses = scheduler.run_until_empty()

    assert len(responses) == 1
    assert len(scheduler.queue) == 0
    assert scheduler.status()["waiting"] == 0
    assert scheduler.status()["running"] == 0
    assert scheduler.status()["finished"] == 1
    assert scheduler.metrics()["submitted_requests"] == 1
    assert scheduler.metrics()["processed_requests"] == 1
    assert scheduler.metrics()["finished_requests"] == 1
    assert scheduler.metrics()["assembled_batches"] == 1
    assert scheduler.metrics()["max_batch_size"] == 2
    assert scheduler.metrics()["avg_batch_size"] == 1.0
    assert scheduler.metrics()["total_scheduler_ms"] >= 0.0


def test_serving_engine_uses_batched_prefill_for_compatible_batch() -> None:
    torch = _torch()
    engine, config = _serving_engine()

    responses = engine.generate_batch(_batch(torch, config))

    assert len(responses) == 2
    assert all(response.metadata["batched_prefill"] is True for response in responses)
    assert all(response.metadata["batch_size"] == 2 for response in responses)


def test_serving_engine_uses_batched_decode_when_enabled() -> None:
    torch = _torch()
    engine, config = _serving_engine(
        enable_batched_decode=True,
        return_beam_details=True,
    )

    responses = engine.generate_batch(_batch(torch, config))

    assert engine.status()["enable_batched_decode"] is True
    assert len(responses) == 2
    assert all(response.metadata["batched_prefill"] is True for response in responses)
    assert all(response.metadata["batched_decode"] is True for response in responses)
    assert all(
        response.metadata["batched_beam_path_steps"] == 2 for response in responses
    )
    assert all(len(response.metadata["beam_details"]) == 2 for response in responses)
    assert all(
        len(response.metadata["beam_details"][0]["token_ids"]) == 2
        for response in responses
    )
    assert all(
        len(response.metadata["beam_details"][0]["token_logprobs"]) == 2
        for response in responses
    )
    assert responses[0].metadata["beam_details"][0]["logprob_sum"] == pytest.approx(
        sum(responses[0].metadata["beam_details"][0]["token_logprobs"])
    )
    assert (
        responses[0].metadata["beam_details"][0]["logprob_type"] == "token_logsoftmax"
    )
    assert all(response.metadata["batch_size"] == 2 for response in responses)
    assert all(response.metadata["batched_decode_ms"] >= 0.0 for response in responses)


def test_serving_engine_batched_decode_stops_when_initial_items_complete() -> None:
    torch = _torch()

    def fail_if_decode_runs(inputs):
        raise AssertionError(
            "batched decode should not run after all initial beams complete"
        )

    engine, config = _serving_engine(
        max_decode_steps=2,
        backend=fail_if_decode_runs,
        enable_batched_decode=True,
        return_beam_details=True,
    )
    provider = _item_provider(
        [("item-a", (1,)), ("item-b", (2,))], vocab_size=config.vocab_size
    )
    responses = engine.generate_batch(
        _batch(torch, config, max_decode_steps=2, item_mask_provider=provider)
    )

    assert len(responses) == 2
    assert all(response.metadata["batched_decode"] is True for response in responses)
    assert all(
        response.metadata["stop_reason"] == "item_complete" for response in responses
    )
    assert all(response.metadata["batched_decode_steps"] == 0 for response in responses)
    assert all(
        response.metadata["batched_beam_path_steps"] == 1 for response in responses
    )
    assert all(set(response.token_ids) == {1, 2} for response in responses)
    assert all(
        len(response.metadata["beam_details"][0]["token_ids"]) == 1
        for response in responses
    )


def test_serving_engine_uses_multi_step_batched_decode_when_enabled() -> None:
    torch = _torch()
    seen_topk_shapes = []

    def capture_topk(inputs):
        seen_topk_shapes.append(tuple(inputs.topk_indices.shape))
        return inputs.q

    engine, config = _serving_engine(
        max_decode_steps=2,
        backend=capture_topk,
        enable_batched_decode=True,
        return_beam_details=True,
    )

    responses = engine.generate_batch(_batch(torch, config, max_decode_steps=2))

    assert len(responses) == 2
    assert all(response.metadata["batched_decode"] is True for response in responses)
    assert all(response.metadata["batched_decode_steps"] == 2 for response in responses)
    assert all(
        response.metadata["batched_beam_path_steps"] == 3 for response in responses
    )
    assert all(
        len(response.metadata["beam_details"][0]["token_ids"]) == 3
        for response in responses
    )
    assert all(
        len(response.metadata["beam_details"][0]["token_logprobs"]) == 3
        for response in responses
    )
    assert seen_topk_shapes == [(2, 1, 4, 1, 2), (2, 1, 4, 2, 2)]
    assert responses[0].metadata["batched_decode_step_plan"][0][
        "topk_indices_shape"
    ] == (
        2,
        1,
        4,
        1,
        2,
    )
    assert all(
        len(response.metadata["batched_decode_step_plan"]) == 2
        for response in responses
    )
    assert responses[0].metadata["decode_batch_plan"][0]["step"] == 0
    assert responses[0].metadata["decode_batch_plan"][1]["step"] == 1


def test_serving_engine_omits_beam_details_by_default() -> None:
    torch = _torch()
    engine, config = _serving_engine(enable_batched_decode=True)

    responses = engine.generate_batch(_batch(torch, config))

    assert responses[0].metadata["batched_decode"] is True
    assert "beam_details" not in responses[0].metadata
    assert "token_logprobs" not in str(responses[0].metadata)
    assert "logprob_sum" not in str(responses[0].metadata)


def test_serving_engine_can_use_raw_logits_score_mode() -> None:
    torch = _torch()
    engine, config = _serving_engine(
        enable_batched_decode=True,
        return_beam_details=True,
        beam_score_mode="raw_logits",
    )

    responses = engine.generate_batch(_batch(torch, config))

    assert engine.status()["beam_score_mode"] == "raw_logits"
    assert responses[0].metadata["beam_details"][0]["score_type"] == (
        "beam_score_raw_logits_cumulative"
    )


def test_serving_engine_records_batched_decode_fallback_reason() -> None:
    torch = _torch()

    def batch_only_failure(inputs):
        if inputs.q.shape[0] > 1:
            raise RuntimeError("batched backend unavailable")
        return inputs.q

    engine, config = _serving_engine(
        backend=batch_only_failure,
        enable_batched_decode=True,
    )
    responses = engine.generate_batch(_batch(torch, config))

    assert len(responses) == 2
    assert all(response.metadata["batched_prefill"] is True for response in responses)
    assert all(response.metadata["batched_decode"] is False for response in responses)
    assert all(
        "batched backend unavailable"
        in response.metadata["batched_decode_fallback_reason"]
        for response in responses
    )
