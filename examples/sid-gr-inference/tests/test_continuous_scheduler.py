# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from types import SimpleNamespace

import pytest
from gr_inference.gr_scheduler import ScheduledBeamPolicy
from gr_inference.gr_serving import (
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRContinuousServingExecutor,
    GRDenseBeamKVPool,
    GRKVLeaseAllocator,
    GRMemoryBudget,
    GRPagedKVLeaseAllocator,
    GRServingRequest,
)
from gr_inference.gr_serving.continuous import GRContinuousRequestState
from gr_inference.gr_serving.prefix_cache import GRPromptPrefixCache


class FakeInputIds:
    def __init__(self, shape):
        self.shape = shape


def torch_or_skip():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    return torch


def make_request(
    idx: int,
    *,
    decode_steps: int = 2,
    beam_width: int = 2,
    context_len: int = 8,
    beam_width_policy=None,
    metadata=None,
) -> GRServingRequest:
    return GRServingRequest(
        request_id=f"req-{idx}",
        input_ids=FakeInputIds((1, context_len)),
        max_decode_steps=decode_steps,
        beam_width=beam_width,
        metadata=dict(metadata or {}),
        beam_width_policy=beam_width_policy,
    )


def serving_engine(model=None, decode_engine=None, **config_overrides):
    config = SimpleNamespace(
        max_beam_width=2,
        beam_score_mode="logprob",
        return_beam_details=False,
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return SimpleNamespace(
        model=model if model is not None else SimpleNamespace(),
        decode_engine=decode_engine,
        config=config,
    )


def context_kv(
    torch, context_len: int, *, batch: int = 1, heads: int = 1, head_dim: int = 4
):
    from gr_inference.gr_kv import ContextKV

    return ContextKV(
        torch.zeros(1, batch, context_len, heads, head_dim),
        torch.zeros(1, batch, context_len, heads, head_dim),
    )


def zero_prefill(torch, context_len: int, *, vocab_size: int = 8):
    from gr_inference.gr_runtime import PrefillResult

    return PrefillResult(
        logits=torch.zeros(1, vocab_size),
        context_kv=context_kv(torch, context_len),
    )


def beam_path_with_two_steps():
    from gr_inference.gr_kv import BatchedBeamPath
    from gr_inference.gr_runtime import BatchedBeamSelection

    path = BatchedBeamPath.create(batch_size=1, max_decode_steps=2, max_beam_width=2)
    path.append(
        BatchedBeamSelection(
            token_ids=((1, 2),),
            scores=((1.0, 0.5),),
            parent_beams=((0, 0),),
        )
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((3, 4),),
            scores=((1.2, 0.9),),
            parent_beams=((1, 0),),
        )
    )
    return path


def counting_prefill_model(torch):
    from gr_inference.gr_runtime import PrefillResult

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def forward_prefill(self, input_ids, **_kwargs):
            self.calls += 1
            batch, context_len = input_ids.shape
            logits = (
                torch.arange(16, dtype=torch.float32)
                .view(1, 1, 16)
                .repeat(
                    batch,
                    1,
                    1,
                )
            )
            return PrefillResult(
                logits=logits,
                context_kv=context_kv(torch, context_len, batch=batch),
            )

    return Model()


def run_prefill_cache_requests(
    torch, engine, *, max_prefill_cache_entries=None
) -> None:
    executor_kwargs = {}
    if max_prefill_cache_entries is not None:
        executor_kwargs["max_prefill_cache_entries"] = max_prefill_cache_entries

    for request_id in ("first", "second"):
        executor = GRContinuousServingExecutor(
            engine=engine,
            scheduler=GRContinuousScheduler(
                policy=GRContinuousBatchingPolicy(
                    max_prefill_batch_size=1,
                    max_decode_batch_size=1,
                )
            ),
            prefill_cache_enabled=True,
            **executor_kwargs,
        )
        executor.submit(
            GRServingRequest(
                request_id=request_id,
                input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                max_decode_steps=1,
                beam_width=2,
            )
        )
        executor._run_prefill(executor.scheduler._admit_prefill_batch())


def cuda_graph_engine(kind: str):
    if kind == "decode":
        return SimpleNamespace(model=SimpleNamespace(), decode_engine=SimpleNamespace())

    def forward_prefill(*args, **kwargs):
        raise AssertionError("not called")

    return SimpleNamespace(model=SimpleNamespace(forward_prefill=forward_prefill))


def test_continuous_scheduler_admits_prefill_while_decoding() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=1,
            max_decode_batch_size=4,
        )
    )
    scheduler.submit(make_request(0, decode_steps=2))

    first = scheduler.tick()
    assert first.prefill_request_ids == ("req-0",)
    assert first.decode_batches[0].metadata()["request_ids"] == ["req-0"]
    assert first.finished_request_ids == ()

    scheduler.submit(make_request(1, decode_steps=2))
    second = scheduler.tick()

    assert second.prefill_request_ids == ("req-1",)
    assert [batch.step for batch in second.decode_batches] == [0, 1]
    assert second.finished_request_ids == ("req-0",)
    assert scheduler.status()["decoding"] == 1

    final = scheduler.tick()
    assert final.finished_request_ids == ("req-1",)
    assert scheduler.status()["finished"] == 2


def test_continuous_scheduler_evicts_old_finished_responses() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=2,
            max_decode_batch_size=2,
            max_finished_requests=1,
        )
    )
    scheduler.submit(make_request(0, decode_steps=1))
    scheduler.submit(make_request(1, decode_steps=1))

    tick = scheduler.tick()
    metrics = scheduler.metrics()

    assert tick.finished_request_ids == ("req-0", "req-1")
    assert tuple(scheduler.finished) == ("req-1",)
    assert scheduler.status()["succeeded_requests"] == 2
    assert metrics["retained_finished_requests"] == 1
    assert metrics["evicted_finished_requests"] == 1


def test_continuous_executor_caches_batched_prefill(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous
    from gr_inference.gr_runtime import GRGenerationState, PrefillResult

    calls = 0
    original = continuous._make_batched_prefill

    def wrapped_make_batched_prefill(generations):
        nonlocal calls
        calls += 1
        return original(generations)

    monkeypatch.setattr(
        continuous, "_make_batched_prefill", wrapped_make_batched_prefill
    )
    executor = GRContinuousServingExecutor(engine=object())
    generations = tuple(
        GRGenerationState.from_prefill(
            request_id=f"req-{idx}",
            prefill=PrefillResult(
                logits=torch.randn(1, 1, 16),
                context_kv=context_kv(torch, 4, head_dim=8),
            ),
            max_decode_steps=1,
            max_beam_width=2,
            fixed_beam_width=2,
        )
        for idx in range(2)
    )
    states = tuple(
        GRContinuousRequestState(request=make_request(idx)) for idx in range(2)
    )

    first = executor._batched_prefill_for_states(states, generations)
    second = executor._batched_prefill_for_states(states, generations)

    assert first is second
    assert calls == 1

    executor._drop_batched_prefill_cache_for(("req-0",))

    assert executor.batched_prefill_cache == {}


@pytest.mark.parametrize(
    ("max_prefill_cache_entries", "expected_calls", "expected_entries"),
    [
        (None, 1, 1),
        (0, 2, 0),
    ],
    ids=["reuse-exact", "zero-capacity"],
)
def test_continuous_executor_prefill_cache_storage_limits(
    max_prefill_cache_entries,
    expected_calls,
    expected_entries,
) -> None:
    torch = torch_or_skip()
    model = counting_prefill_model(torch)
    engine = serving_engine(model=model)

    run_prefill_cache_requests(
        torch,
        engine,
        max_prefill_cache_entries=max_prefill_cache_entries,
    )

    assert model.calls == expected_calls
    assert (
        len(engine._continuous_decode_template_caches["prefill_cache"])
        == expected_entries
    )


def test_prompt_prefix_cache_matches_divergent_prefix() -> None:
    torch = torch_or_skip()

    cache = GRPromptPrefixCache(max_entries=2)
    cache.insert(torch.tensor([[1, 2, 3, 4]]), zero_prefill(torch, 4))
    cache.insert(torch.tensor([[1, 2, 3, 5]]), zero_prefill(torch, 4))

    match = cache.match(torch.tensor([[1, 2, 3, 9]]))

    assert match is not None
    assert match.prefix_len == 3
    assert not match.exact
    assert match.source_token_count == 4
    assert cache.status()["entries"] == 2
    assert cache.status()["nodes"] >= 3


def test_prompt_prefix_cache_isolates_namespace_and_token_budget() -> None:
    torch = torch_or_skip()

    cache = GRPromptPrefixCache(max_entries=8, max_tokens=6)
    cache.insert([1, 2, 3], zero_prefill(torch, 3), extra_key="tenant-a")
    assert cache.match([1, 2, 3], extra_key="tenant-b") is None
    assert cache.match([1, 2, 3], extra_key="tenant-a") is not None

    cache.insert([9, 8, 7, 6], zero_prefill(torch, 4), extra_key="tenant-a")
    assert cache.match([1, 2, 3], extra_key="tenant-a") is None
    assert cache.match([9, 8, 7, 6], extra_key="tenant-a") is not None
    assert cache.status()["entries"] == 1
    assert cache.status()["tokens"] == 4
    assert cache.status()["evicted_tokens"] == 3


def test_prompt_prefix_cache_page_aligns_partial_matches() -> None:
    torch = torch_or_skip()

    cache = GRPromptPrefixCache(max_entries=4, page_size=4)
    cache.insert([1, 2, 3, 4, 5, 6], zero_prefill(torch, 6))

    exact = cache.match([1, 2, 3, 4, 5, 6])
    partial = cache.match([1, 2, 3, 4, 9, 9])
    page_mismatch = cache.match([1, 2, 3, 9, 5, 6])

    assert exact is not None
    assert exact.exact
    assert exact.prefix_len == 6
    assert partial is not None
    assert partial.prefix_len == 4
    assert not partial.exact
    assert page_mismatch is None
    assert cache.status()["tree_tokens"] == 4


def test_continuous_executor_extends_cached_prefix_prefill() -> None:
    torch = torch_or_skip()

    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel

    torch.manual_seed(7)
    config = Qwen3GRConfig(
        model_name="tiny-prefix-cache-qwen3",
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_context_len=8,
        max_seq_len=10,
        max_decode_steps=2,
        max_beam_width=4,
        intermediate_size=32,
        vocab_size=64,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    engine = serving_engine(model=model)
    executor = GRContinuousServingExecutor(
        engine=engine,
        scheduler=GRContinuousScheduler(
            policy=GRContinuousBatchingPolicy(
                max_prefill_batch_size=1,
                max_decode_batch_size=1,
            )
        ),
        prefill_cache_enabled=True,
        min_prefill_cache_prefix_tokens=2,
        max_prefill_cache_decode_extend_tokens=8,
    )

    full_ids = torch.tensor([[3, 5, 7, 9, 11, 13]], dtype=torch.long)
    prefix_ids = full_ids[:, :4]
    reference = model.forward_prefill(
        full_ids,
        return_result=True,
        last_token_logits_only=True,
    )
    original_forward_prefill = model.forward_prefill
    original_forward_prefill_extend = model.forward_prefill_extend
    original_forward_decode_step = model.forward_decode_step
    prefill_calls = 0
    extend_calls = 0
    decode_calls = 0

    def counted_forward_prefill(*args, **kwargs):
        nonlocal prefill_calls
        prefill_calls += 1
        return original_forward_prefill(*args, **kwargs)

    def counted_forward_prefill_extend(*args, **kwargs):
        nonlocal extend_calls
        extend_calls += 1
        return original_forward_prefill_extend(*args, **kwargs)

    def counted_forward_decode_step(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_forward_decode_step(*args, **kwargs)

    model.forward_prefill = counted_forward_prefill
    model.forward_prefill_extend = counted_forward_prefill_extend
    model.forward_decode_step = counted_forward_decode_step

    executor.submit(
        GRServingRequest(
            request_id="prefix",
            input_ids=prefix_ids,
            max_decode_steps=1,
            beam_width=2,
        )
    )
    executor._run_prefill(executor.scheduler._admit_prefill_batch())
    executor.submit(
        GRServingRequest(
            request_id="full",
            input_ids=full_ids,
            max_decode_steps=1,
            beam_width=2,
        )
    )
    executor._run_prefill(executor.scheduler._admit_prefill_batch())

    cached = executor.scheduler.states["full"].generation.prefill

    assert executor.prefill_cache_prefix_hits == 1
    assert executor.prefill_cache_prefix_tokens == 4
    assert executor.prefill_cache_extend_tokens == 2
    assert prefill_calls == 1
    assert extend_calls == 1
    assert decode_calls == 0
    assert torch.allclose(
        cached.logits.squeeze(1), reference.logits, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        cached.context_kv.key, reference.context_kv.key, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        cached.context_kv.value, reference.context_kv.value, atol=2e-5, rtol=2e-5
    )


def test_continuous_executor_caches_topk_indices(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous

    calls = 0
    original = continuous.make_batched_topk_indices

    def wrapped_make_topk(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(continuous, "make_batched_topk_indices", wrapped_make_topk)
    executor = GRContinuousServingExecutor(engine=object())
    path = beam_path_with_two_steps()

    first = executor._topk_indices_for_decode_batch(
        path,
        request_ids=("req-0",),
        num_q_heads=4,
        decode_nums=2,
        beam_width=2,
        device=torch.device("cpu"),
        needs_history_compaction=False,
    )
    second = executor._topk_indices_for_decode_batch(
        path,
        request_ids=("req-0",),
        num_q_heads=4,
        decode_nums=2,
        beam_width=2,
        device=torch.device("cpu"),
        needs_history_compaction=False,
    )

    assert first is second
    assert calls == 1
    assert executor.topk_indices_cache_hits == 1
    assert executor.topk_indices_cache_misses == 1

    executor._drop_batched_prefill_cache_for(("req-0",))

    assert executor.topk_indices_cache == {}


def test_continuous_executor_caches_decode_inputs(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous
    from gr_inference.gr_runtime import BatchedBeamSelection

    calls = 0
    original = continuous.make_batched_beam_token_ids

    def wrapped_make_decode_inputs(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        continuous,
        "make_batched_beam_token_ids",
        wrapped_make_decode_inputs,
    )
    executor = GRContinuousServingExecutor(engine=object())
    selection = BatchedBeamSelection(
        token_ids=((1, 2),),
        scores=((1.0, 0.5),),
        parent_beams=((0, 0),),
    )

    first = executor._decode_inputs_for_selection(
        selection,
        request_ids=("req-0",),
        device=torch.device("cpu"),
    )
    second = executor._decode_inputs_for_selection(
        selection,
        request_ids=("req-0",),
        device=torch.device("cpu"),
    )

    assert first is second
    assert calls == 1
    assert executor.decode_inputs_cache_hits == 1
    assert executor.decode_inputs_cache_misses == 1
    assert executor.metrics()["decode_inputs_cache_entries"] == 1

    executor._drop_batched_prefill_cache_for(("req-0",))

    assert executor.decode_inputs_cache == {}


def test_continuous_executor_caches_compacted_topk_indices(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous
    from gr_inference.gr_kv import BatchedBeamPath

    calls = 0
    original = continuous.make_compacted_batched_topk_indices

    def wrapped_make_compacted_topk(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        continuous,
        "make_compacted_batched_topk_indices",
        wrapped_make_compacted_topk,
    )
    executor = GRContinuousServingExecutor(engine=object())
    path = BatchedBeamPath.create(batch_size=2, max_decode_steps=2, max_beam_width=2)

    first = executor._topk_indices_for_decode_batch(
        path,
        request_ids=("req-0", "req-1"),
        num_q_heads=4,
        decode_nums=2,
        beam_width=2,
        device=torch.device("cpu"),
        needs_history_compaction=True,
    )
    second = executor._topk_indices_for_decode_batch(
        path,
        request_ids=("other-0", "other-1"),
        num_q_heads=4,
        decode_nums=2,
        beam_width=2,
        device=torch.device("cpu"),
        needs_history_compaction=True,
    )

    assert first is second
    assert calls == 1
    assert executor.metrics()["topk_indices_cache_hits"] == 1
    assert executor.metrics()["topk_indices_cache_entries"] == 1


def test_make_batched_generation_reuses_compaction_decision(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous
    from gr_inference.gr_kv import BatchedBeamPath, ContextKV
    from gr_inference.gr_runtime import GRGenerationState, PrefillResult

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("compaction decision should be supplied by caller")

    monkeypatch.setattr(
        continuous,
        "needs_batched_beam_kv_history_compaction",
        fail_if_recomputed,
    )
    generation = GRGenerationState.from_prefill(
        request_id="req-0",
        prefill=PrefillResult(
            logits=torch.randn(1, 1, 16),
            context_kv=ContextKV(
                torch.empty(1, 1, 4, 1, 8),
                torch.empty(1, 1, 4, 1, 8),
            ),
        ),
        max_decode_steps=1,
        max_beam_width=2,
        fixed_beam_width=2,
    )

    batched = continuous._make_batched_generation(
        request_id="batched",
        generations=(generation,),
        beam_score_mode="logprob",
        batched_beam_path=BatchedBeamPath((generation.beam_path,)),
        decode_nums=1,
        active_beam_width=2,
        needs_history_compaction=False,
    )

    assert batched.request_id == "batched"


def test_continuous_scheduler_groups_decode_by_step_and_beam_width() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=3,
            max_decode_batch_size=2,
        )
    )
    scheduler.submit(make_request(0, beam_width=2))
    scheduler.submit(make_request(1, beam_width=2))
    scheduler.submit(make_request(2, beam_width=4))

    tick = scheduler.tick()
    metadata = [batch.metadata() for batch in tick.decode_batches]

    assert [
        {
            key: row[key]
            for key in (
                "step",
                "beam_width",
                "active_beam_width",
                "next_beam_width",
                "context_len",
                "size",
                "request_ids",
            )
        }
        for row in metadata
    ] == [
        {
            "step": 0,
            "beam_width": 2,
            "active_beam_width": 2,
            "next_beam_width": 2,
            "context_len": 8,
            "size": 2,
            "request_ids": ["req-0", "req-1"],
        },
        {
            "step": 0,
            "beam_width": 4,
            "active_beam_width": 4,
            "next_beam_width": 4,
            "context_len": 8,
            "size": 1,
            "request_ids": ["req-2"],
        },
    ]
    assert metadata[0]["group_key"] == {
        "step": 0,
        "active_beam_width": 2,
        "next_beam_width": 2,
        "context_len": 8,
    }
    assert scheduler.metrics()["avg_decode_batch_size"] == 1.5


def test_continuous_scheduler_splits_decode_by_context_len() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=2,
            max_decode_batch_size=2,
        )
    )
    scheduler.submit(make_request(0, beam_width=2, context_len=8))
    scheduler.submit(make_request(1, beam_width=2, context_len=4))

    tick = scheduler.tick()

    assert [batch.context_len for batch in tick.decode_batches] == [4, 8]
    assert [batch.size for batch in tick.decode_batches] == [1, 1]


def test_continuous_scheduler_groups_dynamic_beam_by_current_and_next_width() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=3,
            max_decode_batch_size=4,
        )
    )
    scheduler.submit(
        make_request(
            0,
            beam_width=4,
            decode_steps=3,
            beam_width_policy=ScheduledBeamPolicy({0: 4, 1: 2}),
        )
    )
    scheduler.submit(
        make_request(
            1,
            beam_width=4,
            decode_steps=3,
            beam_width_policy=ScheduledBeamPolicy({0: 4, 1: 3}),
        )
    )

    first = scheduler.tick()
    assert [
        (batch.beam_width, batch.next_beam_width, batch.request_ids)
        for batch in first.decode_batches
    ] == [
        (4, 2, ("req-0",)),
        (4, 3, ("req-1",)),
    ]
    assert first.decode_batches[0].metadata()["active_beam_width"] == 4
    assert first.decode_batches[0].metadata()["group_key"] == {
        "step": 0,
        "active_beam_width": 4,
        "next_beam_width": 2,
        "context_len": 8,
    }
    assert scheduler.status()["policy"]["decode_batch_grouping"] == (
        "step,active_beam_width,next_beam_width,context_len"
    )

    second = scheduler.tick()
    assert [
        (batch.beam_width, batch.next_beam_width, batch.request_ids)
        for batch in second.decode_batches
    ] == [
        (2, 2, ("req-0",)),
        (3, 3, ("req-1",)),
    ]


def test_continuous_scheduler_respects_memory_budget() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=2,
            max_decode_batch_size=2,
            max_running_requests=1,
        )
    )
    scheduler.submit(make_request(0, decode_steps=2))
    scheduler.submit(make_request(1, decode_steps=2))

    first = scheduler.tick()
    assert first.prefill_request_ids == ("req-0",)
    assert scheduler.status()["waiting_prefill"] == 1

    second = scheduler.tick()
    assert second.prefill_request_ids == ()
    assert second.finished_request_ids == ("req-0",)

    third = scheduler.tick()
    assert third.prefill_request_ids == ("req-1",)


def test_memory_budget_counts_context_tokens_and_beam_slots() -> None:
    budget = GRMemoryBudget(max_context_tokens=12, max_beam_slots=8)
    state = scheduler_state(
        make_request(0, decode_steps=2, beam_width=2, context_len=8)
    )

    assert budget.can_admit((state,), make_request(1, context_len=4))
    assert not budget.can_admit((state,), make_request(2, context_len=5))
    assert not budget.can_admit((state,), make_request(3, decode_steps=3, beam_width=2))


def test_gr_kv_memory_estimator_quantifies_shared_context_savings() -> None:
    from gr_inference.gr_serving import estimate_gr_kv_memory

    estimate = estimate_gr_kv_memory(
        batch_size=2,
        num_layers=4,
        context_len=10,
        max_decode_steps=3,
        max_beam_width=5,
        active_beam_width=2,
        num_kv_heads=2,
        head_dim=8,
        bytes_per_element=2,
        vocab_size=100,
    )
    metadata = estimate.metadata()

    assert metadata["context_kv_bytes"] == 2 * 10 * 2 * 4 * 2 * 8 * 2
    assert metadata["beam_kv_bytes"] == 2 * 3 * 5 * 2 * 4 * 2 * 8 * 2
    assert metadata["dense_per_beam_context_kv_bytes"] == (
        metadata["context_kv_bytes"] * 5
    )
    assert metadata["context_kv_sharing_savings_bytes"] == (
        metadata["context_kv_bytes"] * 4
    )
    assert metadata["logits_workspace_bytes"] == 2 * 2 * 100 * 4
    assert metadata["topk_scores_workspace_bytes"] == 2 * 2 * 4


def test_kv_lease_allocator_tracks_usage_and_release() -> None:
    allocator = GRKVLeaseAllocator(
        max_running_requests=2, max_context_tokens=12, max_beam_slots=8
    )

    lease = allocator.allocate(request_id="req-0", context_tokens=8, beam_slots=4)

    assert lease.metadata() == {
        "request_id": "req-0",
        "context_tokens": 8,
        "beam_slots": 4,
    }
    assert allocator.usage() == {
        "running_requests": 1,
        "context_tokens": 8,
        "beam_slots": 4,
    }
    assert allocator.can_allocate(request_id="req-1", context_tokens=4, beam_slots=4)
    assert not allocator.can_allocate(
        request_id="req-2", context_tokens=5, beam_slots=1
    )
    status = allocator.status()
    assert status["available_running_requests"] == 1
    assert status["available_context_tokens"] == 4
    assert status["available_beam_slots"] == 4
    assert status["running_request_utilization"] == 0.5
    assert status["context_token_utilization"] == 8 / 12
    assert status["beam_slot_utilization"] == 0.5

    assert allocator.release("req-0") is lease
    assert allocator.usage() == {
        "running_requests": 0,
        "context_tokens": 0,
        "beam_slots": 0,
    }


def test_paged_kv_lease_allocator_assigns_and_reuses_pages() -> None:
    allocator = GRPagedKVLeaseAllocator(
        context_page_size=4,
        beam_page_size=2,
        max_context_pages=3,
        max_beam_pages=4,
    )

    lease = allocator.allocate(request_id="req-0", context_tokens=5, beam_slots=3)

    assert lease.context_pages == (0, 1)
    assert lease.beam_pages == (0, 1)
    assert lease.metadata()["context_capacity_tokens"] == 8
    assert lease.metadata()["beam_capacity_slots"] == 4
    assert allocator.usage()["context_pages"] == 2
    assert allocator.status()["free_context_pages"] == 1
    assert allocator.status()["context_page_capacity_tokens"] == 8
    assert allocator.status()["beam_page_capacity_slots"] == 4
    assert allocator.status()["context_internal_fragmentation_tokens"] == 3
    assert allocator.status()["beam_internal_fragmentation_slots"] == 1
    assert allocator.status()["max_used_context_pages"] == 2
    assert allocator.status()["max_used_beam_pages"] == 2
    assert allocator.status()["context_free_page_runs"] == ((2, 2),)
    assert allocator.status()["largest_free_context_page_run"] == 1
    assert allocator.status()["context_page_utilization"] == 2 / 3
    assert allocator.status()["beam_page_utilization"] == 0.5

    assert not allocator.can_allocate(
        request_id="req-1", context_tokens=8, beam_slots=2
    )
    assert allocator.release("req-0") is lease
    assert allocator.status()["free_context_pages"] == 3

    next_lease = allocator.allocate(request_id="req-1", context_tokens=4, beam_slots=2)
    assert next_lease.context_pages == (0,)
    assert next_lease.beam_pages == (0,)


def test_dense_beam_kv_pool_leases_dense_views() -> None:
    torch = torch_or_skip()

    from gr_inference.gr_kv import ContextKV

    context_kv = ContextKV(
        torch.empty(1, 1, 8, 2, 4),
        torch.empty(1, 1, 8, 2, 4),
    )
    pool = GRDenseBeamKVPool.like_context(
        context_kv,
        max_requests=2,
        max_decode_steps=3,
        max_beam_width=4,
    )

    lease = pool.allocate("req-0")
    lease.beam_kv.key.fill_(7)

    assert lease.slot == 0
    assert tuple(lease.beam_kv.key_shape) == (1, 1, 3, 4, 2, 4)
    assert torch.equal(pool.key[:, 0:1], lease.beam_kv.key)
    assert pool.usage()["beam_kv_pool_used"] == 1
    assert pool.status()["beam_kv_pool_max_used"] == 1
    assert pool.status()["beam_kv_pool_utilization"] == 0.5
    assert pool.status()["beam_kv_pool_high_watermark_utilization"] == 0.5
    assert pool.status()["beam_kv_pool_slot_shape"] == (1, 2, 3, 4, 2, 4)
    assert pool.status()["beam_kv_pool_slot_allocation_policy"] == "lowest_free_slot"
    assert pool.status()["beam_kv_pool_release_policy"] == "immediate_reuse"
    assert pool.status()["beam_kv_pool_allocation_count"] == 1

    assert pool.release("req-0") is lease
    assert pool.usage()["beam_kv_pool_free"] == 2
    assert pool.status()["beam_kv_pool_release_count"] == 1
    assert pool.status()["free_slot_runs"] == ((0, 1),)
    assert pool.allocate("req-1").slot == 0
    assert pool.status()["beam_kv_pool_max_used"] == 1


def test_scatter_batched_beam_kv_skips_pool_view_self_copy(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous
    from gr_inference.gr_kv import BeamKV

    copy_calls = 0

    def counting_copy(destination, source):
        nonlocal copy_calls
        copy_calls += 1
        destination.copy_(source)

    monkeypatch.setattr(continuous, "_copy_tensor", counting_copy)

    pool_key = torch.randn(1, 2, 3, 4, 1, 2)
    pool_value = torch.randn_like(pool_key)
    batched = BeamKV(pool_key, pool_value)
    generations = (
        SimpleNamespace(beam_kv=BeamKV(pool_key[:, 0:1], pool_value[:, 0:1])),
        SimpleNamespace(beam_kv=BeamKV(pool_key[:, 1:2], pool_value[:, 1:2])),
    )

    continuous._scatter_batched_beam_kv(
        batched,
        generations,
        step=1,
        active_beam_width=3,
    )

    assert copy_calls == 0


def test_scatter_batched_beam_kv_copies_non_alias_views(monkeypatch) -> None:
    torch = torch_or_skip()
    import gr_inference.gr_serving.continuous as continuous
    from gr_inference.gr_kv import BeamKV

    copy_calls = 0

    def counting_copy(destination, source):
        nonlocal copy_calls
        copy_calls += 1
        destination.copy_(source)

    monkeypatch.setattr(continuous, "_copy_tensor", counting_copy)

    source_key = torch.arange(1 * 2 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
        1,
        2,
        3,
        4,
        1,
        2,
    )
    source_value = source_key + 1000
    batched = BeamKV(source_key, source_value)
    generations = (
        SimpleNamespace(
            beam_kv=BeamKV(
                torch.zeros(1, 1, 3, 4, 1, 2),
                torch.zeros(1, 1, 3, 4, 1, 2),
            )
        ),
        SimpleNamespace(
            beam_kv=BeamKV(
                torch.zeros(1, 1, 3, 4, 1, 2),
                torch.zeros(1, 1, 3, 4, 1, 2),
            )
        ),
    )

    continuous._scatter_batched_beam_kv(
        batched,
        generations,
        step=1,
        active_beam_width=3,
    )

    assert copy_calls == 4
    assert torch.equal(generations[0].beam_kv.key[:, 0, 1, :3], source_key[:, 0, 1, :3])
    assert torch.equal(
        generations[1].beam_kv.value[:, 0, 1, :3],
        source_value[:, 1, 1, :3],
    )


def test_continuous_scheduler_rejects_request_that_exceeds_budget() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(max_context_tokens=4)
    )
    scheduler.submit(make_request(0, context_len=8))

    try:
        scheduler.tick()
    except ValueError as exc:
        assert "memory budget" in str(exc)
    else:
        raise AssertionError("expected memory budget validation error")


def test_continuous_scheduler_cancels_waiting_request() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=1,
            max_decode_batch_size=1,
        )
    )
    scheduler.submit(make_request(0, decode_steps=1))
    scheduler.submit(make_request(1, decode_steps=1))

    response = scheduler.cancel("req-1")

    assert response.request_id == "req-1"
    assert response.metadata["cancelled"] is True
    assert response.metadata["stop_reason"] == "cancelled"
    assert response.metadata["admitted_tick"] is None
    assert scheduler.status()["waiting_prefill"] == 1
    assert scheduler.status()["cancelled_requests"] == 1

    responses = scheduler.run_until_empty()
    assert [response.request_id for response in responses] == ["req-1", "req-0"]


def test_continuous_scheduler_cancels_decoding_request() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=1,
            max_decode_batch_size=1,
        )
    )
    scheduler.submit(make_request(0, decode_steps=3))
    scheduler.tick()

    response = scheduler.cancel("req-0", reason="client_cancelled")

    assert response.metadata["cancelled"] is True
    assert response.metadata["stop_reason"] == "client_cancelled"
    assert response.metadata["decode_steps"] == 1
    assert response.metadata["admitted_tick"] == 1
    assert scheduler.status()["decoding"] == 0
    assert scheduler.metrics()["cancelled_requests"] == 1
    assert scheduler.metrics()["kv_running_requests"] == 0
    assert scheduler.run_until_empty() == (response,)


def test_continuous_scheduler_cancel_finished_is_idempotent() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=1))
    (response,) = scheduler.run_until_empty()

    cancelled = scheduler.cancel("req-0")

    assert cancelled is response
    assert cancelled.metadata["stop_reason"] == "max_decode_steps"
    assert scheduler.status()["cancelled_requests"] == 0


def test_continuous_scheduler_fails_waiting_request() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=1))

    response = scheduler.fail("req-0", reason="validation_failed", error="bad request")

    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "validation_failed"
    assert response.metadata["error_type"] == "Error"
    assert response.metadata["error_message"] == "bad request"
    assert scheduler.status()["waiting_prefill"] == 0
    assert scheduler.status()["failed_requests"] == 1


def test_continuous_scheduler_converts_prefill_executor_exception_to_failure() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=1))

    def fail_prefill(_request_ids):
        raise RuntimeError("prefill boom")

    tick = scheduler.tick(prefill_executor=fail_prefill)

    assert tick.prefill_request_ids == ("req-0",)
    assert tick.decode_batches == ()
    assert tick.finished_request_ids == ("req-0",)
    response = scheduler.finished["req-0"]
    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "prefill_failed"
    assert response.metadata["error_type"] == "RuntimeError"
    assert response.metadata["error_message"] == "prefill boom"
    assert scheduler.status()["decoding"] == 0


def test_continuous_scheduler_converts_decode_executor_exception_to_failure() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=2))

    def fail_decode(_decode_batches):
        raise RuntimeError("decode boom")

    tick = scheduler.tick(decode_executor=fail_decode)

    assert len(tick.decode_batches) == 1
    assert tick.finished_request_ids == ("req-0",)
    response = scheduler.finished["req-0"]
    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "decode_failed"
    assert response.metadata["error_type"] == "RuntimeError"
    assert response.metadata["error_message"] == "decode boom"
    assert scheduler.status()["decoding"] == 0


def test_continuous_scheduler_times_out_unfinished_requests_at_max_ticks() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=3))

    responses = scheduler.run_until_empty(max_ticks=1, timeout_unfinished=True)

    assert len(responses) == 1
    response = responses[0]
    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "timeout"
    assert response.metadata["decode_steps"] == 1
    assert scheduler.status()["decoding"] == 0
    assert scheduler.status()["failed_requests"] == 1
    assert scheduler.status()["memory_usage"]["running_requests"] == 0
    assert scheduler.status()["kv_health"]["kv_allocator_leak_detected"] is False
    assert scheduler.metrics()["kv_allocator_running_requests"] == 0
    assert scheduler.metrics()["kv_health_kv_allocator_leak_detected"] == 0


def test_continuous_scheduler_times_out_waiting_request_by_ttl() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(max_running_requests=1)
    )
    scheduler.submit(make_request(0, decode_steps=4))
    scheduler.tick()
    scheduler.submit(make_request(1, decode_steps=3, metadata={"timeout_ticks": 1}))

    first = scheduler.tick()
    second = scheduler.tick()

    assert first.finished_request_ids == ()
    assert second.finished_request_ids == ("req-1",)
    response = scheduler.finished["req-1"]
    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "request_timeout"
    assert response.metadata["timeout_ticks"] == 1
    assert response.metadata["submitted_tick"] == 1
    assert response.metadata["finished_tick"] == 3


def test_continuous_scheduler_times_out_decoding_request_by_ttl() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=4, metadata={"timeout_ticks": 1}))

    first = scheduler.tick()
    second = scheduler.tick()

    assert first.finished_request_ids == ()
    assert second.finished_request_ids == ("req-0",)
    response = scheduler.finished["req-0"]
    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "request_timeout"
    assert response.metadata["decode_steps"] == 1
    assert scheduler.status()["memory_usage"]["running_requests"] == 0


def test_continuous_scheduler_releases_kv_lease_after_finish() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_context_tokens=8,
            max_beam_slots=4,
        )
    )
    scheduler.submit(make_request(0, decode_steps=2, beam_width=2, context_len=8))

    first = scheduler.tick()
    assert first.memory_usage == {
        "running_requests": 1,
        "context_tokens": 8,
        "beam_slots": 4,
    }

    second = scheduler.tick()

    assert second.finished_request_ids == ("req-0",)
    assert second.memory_usage == {
        "running_requests": 0,
        "context_tokens": 0,
        "beam_slots": 0,
    }
    assert scheduler.status()["memory_usage"]["running_requests"] == 0


def test_continuous_scheduler_releases_kv_lease_after_decode_failure() -> None:
    scheduler = GRContinuousScheduler()
    scheduler.submit(make_request(0, decode_steps=2))

    def fail_decode(_decode_batches):
        raise RuntimeError("decode boom")

    tick = scheduler.tick(decode_executor=fail_decode)

    assert tick.finished_request_ids == ("req-0",)
    assert tick.memory_usage["running_requests"] == 0
    assert scheduler.metrics()["kv_running_requests"] == 0


def test_continuous_scheduler_uses_paged_kv_allocator() -> None:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=2,
            max_decode_batch_size=2,
        ),
        kv_allocator=GRPagedKVLeaseAllocator(
            context_page_size=4,
            beam_page_size=2,
            max_context_pages=2,
            max_beam_pages=2,
        ),
    )
    scheduler.submit(make_request(0, decode_steps=2, beam_width=2, context_len=8))
    scheduler.submit(make_request(1, decode_steps=1, beam_width=2, context_len=4))

    first = scheduler.tick()

    assert first.prefill_request_ids == ("req-0",)
    assert first.memory_usage["context_pages"] == 2
    assert scheduler.status()["kv_allocator"]["free_context_pages"] == 0
    assert scheduler.status()["waiting_prefill"] == 1

    second = scheduler.tick()
    assert second.finished_request_ids == ("req-0",)
    assert second.memory_usage["context_pages"] == 0

    third = scheduler.tick()
    assert third.prefill_request_ids == ("req-1",)
    assert third.finished_request_ids == ("req-1",)
    assert third.memory_usage["context_pages"] == 0
    assert scheduler.status()["kv_allocator"]["free_context_pages"] == 2


def test_continuous_serving_executor_runs_model_steps() -> None:
    torch = torch_or_skip()

    from gr_inference.gr_kernels.attention import GRDecodeAttention
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import (
        GRDecodeEngine,
        TimingRecorder,
        TokenTrie,
        TrieItemMaskProvider,
    )
    from gr_inference.gr_serving import (
        GRContinuousServingExecutor,
        GRServingConfig,
        GRServingEngine,
    )

    config = Qwen3GRConfig(
        model_name="tiny-continuous-qwen3-gr",
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=8,
        max_seq_len=10,
        max_decode_steps=1,
        max_beam_width=2,
        intermediate_size=64,
        vocab_size=32,
    )
    engine = GRServingEngine(
        model=Qwen3GRModel(
            config,
            prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        ),
        decode_engine=GRDecodeEngine(
            attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
            fixed_beam_width=2,
        ),
        config=GRServingConfig(
            max_decode_steps=1,
            max_beam_width=2,
            return_beam_details=True,
        ),
    )
    provider = TrieItemMaskProvider(
        TokenTrie.from_items([("item-a", (1, 10)), ("item-b", (2, 11))]),
        vocab_size=config.vocab_size,
    )
    executor = GRContinuousServingExecutor(
        engine=engine,
        timing_recorder=TimingRecorder(sync_module=torch, detail="fine"),
        scheduler=GRContinuousScheduler(
            policy=GRContinuousBatchingPolicy(
                max_prefill_batch_size=2,
                max_decode_batch_size=2,
            )
        ),
        beam_kv_pool=GRDenseBeamKVPool(
            key=torch.empty(
                config.num_layers,
                2,
                config.max_decode_steps,
                config.max_beam_width,
                config.num_kv_heads,
                config.head_dim,
            ),
            value=torch.empty(
                config.num_layers,
                2,
                config.max_decode_steps,
                config.max_beam_width,
                config.num_kv_heads,
                config.head_dim,
            ),
        ),
    )
    for idx in range(2):
        executor.submit(
            GRServingRequest(
                request_id=f"req-{idx}",
                input_ids=torch.randint(0, config.vocab_size, (1, 8)),
                max_decode_steps=1,
                beam_width=2,
                item_mask_provider=provider,
            )
        )

    responses = executor.run_until_empty()

    assert len(responses) == 2
    assert all(len(response.token_ids) == 2 for response in responses)
    assert all(
        response.metadata["continuous_execution"] == "model_step"
        for response in responses
    )
    assert all(response.metadata["prefill_ms"] >= 0.0 for response in responses)
    assert all(response.metadata["decode_ms"] >= 0.0 for response in responses)
    assert all(response.metadata["total_ms"] >= 0.0 for response in responses)
    assert executor.metrics()["prefill_ms"] >= 0.0
    assert executor.metrics()["decode_ms"] >= 0.0
    assert executor.beam_kv_pool is not None
    assert executor.beam_kv_pool.usage()["beam_kv_pool_used"] == 0
    assert executor.beam_kv_pool.usage()["beam_kv_pool_free"] == 2
    assert all(
        state.generation is None and state.token_logprob_steps is None
        for state in executor.scheduler.states.values()
    )
    assert all(
        state.request.input_ids is None for state in executor.scheduler.states.values()
    )
    assert all(
        state.beam_kv_pool_lease is None for state in executor.scheduler.states.values()
    )
    assert executor.status()["beam_kv_pool"]["beam_kv_pool_max_used"] == 2
    assert (
        executor.status()["beam_kv_pool_health"]["beam_kv_pool_leak_detected"] is False
    )
    assert executor.metrics()["beam_kv_pool_max_used"] == 2
    assert executor.metrics()["beam_kv_pool_health_beam_kv_pool_leak_detected"] == 0
    profile = executor.timing_recorder.summary()
    assert "continuous.decode_microbatch_total" in profile
    assert "model.forward_decode_step" in profile
    assert "layer0.decode_attention" in profile
    assert responses[0].metadata["decode_batch"]["size"] == 2
    assert len(responses[0].metadata["beam_details"]) == 2
    assert len(responses[0].metadata["beam_details"][0]["token_ids"]) == 2
    assert len(responses[0].metadata["beam_details"][0]["token_logprobs"]) == 2
    assert (
        responses[0].metadata["beam_details"][0]["logprob_type"] == "token_logsoftmax"
    )
    assert responses[0].metadata["beam_details"][0]["score_type"] == (
        "beam_score_logprob_cumulative"
    )
    assert len(responses[0].metadata["item_results"]) == 2
    assert all(
        result["is_complete"] for result in responses[0].metadata["item_results"]
    )
    assert {result["item_id"] for result in responses[0].metadata["item_results"]} == {
        "item-a",
        "item-b",
    }


@pytest.mark.parametrize(
    ("kind", "env_var", "runner_attr", "status_key"),
    [
        (
            "decode",
            "GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH",
            "decode_cuda_graph_runner",
            "decode_cuda_graph",
        ),
        (
            "prefill",
            "GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH",
            "prefill_cuda_graph_runner",
            "prefill_cuda_graph",
        ),
    ],
)
def test_continuous_executor_cuda_graph_default_and_reuse(
    monkeypatch,
    kind,
    env_var,
    runner_attr,
    status_key,
) -> None:
    monkeypatch.delenv(env_var, raising=False)
    engine = cuda_graph_engine(kind)

    executor = GRContinuousServingExecutor(engine=engine)
    second_executor = GRContinuousServingExecutor(engine=engine)

    assert getattr(executor, runner_attr) is not None
    assert getattr(second_executor, runner_attr) is getattr(executor, runner_attr)
    assert executor.status()[status_key][f"{status_key}_entries"] == 0
    assert executor.metrics()[f"{status_key}_entries"] == 0


@pytest.mark.parametrize(
    ("kind", "env_var", "runner_attr", "status_key"),
    [
        (
            "decode",
            "GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH",
            "decode_cuda_graph_runner",
            "decode_cuda_graph",
        ),
        (
            "prefill",
            "GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH",
            "prefill_cuda_graph_runner",
            "prefill_cuda_graph",
        ),
    ],
)
def test_continuous_executor_cuda_graph_can_be_disabled(
    monkeypatch,
    kind,
    env_var,
    runner_attr,
    status_key,
) -> None:
    monkeypatch.setenv(env_var, "1")
    executor = GRContinuousServingExecutor(engine=cuda_graph_engine(kind))

    assert getattr(executor, runner_attr) is None
    assert status_key not in executor.status()


def test_no_sync_profile_keeps_cuda_graph_enabled() -> None:
    from gr_inference.gr_runtime import TimingRecorder
    from gr_inference.gr_serving.continuous import (
        _profile_allows_decode_cuda_graph,
        _profile_allows_prefill_cuda_graph,
    )

    assert _profile_allows_decode_cuda_graph(None)
    assert _profile_allows_decode_cuda_graph(TimingRecorder(sync_timing=False))
    assert not _profile_allows_decode_cuda_graph(TimingRecorder(sync_timing=True))
    assert _profile_allows_prefill_cuda_graph(None)
    assert _profile_allows_prefill_cuda_graph(TimingRecorder(sync_timing=False))
    assert not _profile_allows_prefill_cuda_graph(TimingRecorder(sync_timing=True))


def test_continuous_executor_sync_timing_flag_controls_timer_syncs() -> None:
    calls = 0

    def synchronize() -> None:
        nonlocal calls
        calls += 1

    executor = GRContinuousServingExecutor(
        engine=SimpleNamespace(model=SimpleNamespace()),
        synchronize=synchronize,
        sync_timing=False,
    )
    start = executor._start_timer()
    executor._elapsed_ms(start)

    assert calls == 0
    assert executor.status()["sync_timing"] is False
    assert executor.metrics()["sync_timing_enabled"] == 0

    executor.sync_timing = True
    start = executor._start_timer()
    executor._elapsed_ms(start)

    assert calls == 2


def test_continuous_executor_decode_cuda_graph_pads_to_bucket_with_pool_views() -> None:
    torch = torch_or_skip()

    from gr_inference.gr_kv import BeamKV, ContextKV
    from gr_inference.gr_runtime import GRGenerationState, PrefillResult
    from gr_inference.gr_serving import GRDenseBeamKVPool, GRDenseContextKVPool

    context_pool = GRDenseContextKVPool(
        key=torch.empty(1, 4, 6, 1, 4),
        value=torch.empty(1, 4, 6, 1, 4),
    )
    beam_pool = GRDenseBeamKVPool(
        key=torch.empty(1, 4, 2, 2, 1, 4),
        value=torch.empty(1, 4, 2, 2, 1, 4),
    )
    request_ids = ("req-0", "req-1", "req-2")
    context_leases = context_pool.allocate_batch(request_ids, context_len=6)
    beam_leases = tuple(beam_pool.allocate(request_id) for request_id in request_ids)
    states = tuple(
        GRContinuousRequestState(
            request=make_request(idx, context_len=6),
            context_kv_pool_lease=context_leases[idx],
            beam_kv_pool_lease=beam_leases[idx],
        )
        for idx in range(3)
    )
    generation = GRGenerationState.from_prefill(
        request_id="batched",
        prefill=PrefillResult(
            logits=torch.zeros(3, 1, 8),
            context_kv=ContextKV(
                context_pool.key[:, :3, :6],
                context_pool.value[:, :3, :6],
            ),
        ),
        max_decode_steps=2,
        max_beam_width=2,
        fixed_beam_width=2,
        beam_kv=BeamKV(beam_pool.key[:, :3], beam_pool.value[:, :3]),
    )
    executor = GRContinuousServingExecutor(
        engine=SimpleNamespace(),
        context_kv_pool=context_pool,
        beam_kv_pool=beam_pool,
        decode_cuda_graph_batch_buckets=(1, 2, 4),
    )

    beam_token_ids = torch.ones(3, 2, dtype=torch.long)
    topk_indices = torch.arange(6, dtype=torch.long).view(3, 2)
    graph_inputs = executor._decode_cuda_graph_inputs(
        beam_token_ids,
        generation,
        topk_indices=topk_indices,
        states=states,
    )

    assert graph_inputs.actual_batch_size == 3
    assert graph_inputs.graph_batch_size == 4
    assert graph_inputs.generation.prefill.context_kv.batch_size == 4
    assert graph_inputs.generation.beam_kv.batch_size == 4
    assert torch.equal(graph_inputs.beam_token_ids[:3], beam_token_ids)
    assert torch.equal(
        graph_inputs.beam_token_ids[3:], torch.zeros(1, 2, dtype=torch.long)
    )
    assert torch.equal(graph_inputs.topk_indices[:3], topk_indices)
    assert executor.decode_cuda_graph_padding_applied == 1
    assert executor.decode_cuda_graph_padding_buffer_misses == 2

    second = executor._decode_cuda_graph_inputs(
        beam_token_ids + 1,
        generation,
        topk_indices=topk_indices + 1,
        states=states,
    )

    assert second.beam_token_ids.data_ptr() == graph_inputs.beam_token_ids.data_ptr()
    assert second.topk_indices.data_ptr() == graph_inputs.topk_indices.data_ptr()
    assert executor.decode_cuda_graph_padding_buffer_hits == 2


def test_continuous_executor_decode_cuda_graph_skips_dynamic_pool_gaps() -> None:
    torch = torch_or_skip()

    from gr_inference.gr_kv import BeamKV, ContextKV
    from gr_inference.gr_runtime import GRGenerationState, PrefillResult
    from gr_inference.gr_serving import GRDenseBeamKVPool, GRDenseContextKVPool

    context_pool = GRDenseContextKVPool(
        key=torch.empty(1, 4, 6, 1, 4),
        value=torch.empty(1, 4, 6, 1, 4),
    )
    beam_pool = GRDenseBeamKVPool(
        key=torch.empty(1, 4, 2, 2, 1, 4),
        value=torch.empty(1, 4, 2, 2, 1, 4),
    )
    request_ids = ("req-0", "hole", "req-2")
    context_leases = context_pool.allocate_batch(request_ids, context_len=6)
    beam_leases = tuple(beam_pool.allocate(request_id) for request_id in request_ids)
    states = (
        GRContinuousRequestState(
            request=make_request(0, context_len=6),
            context_kv_pool_lease=context_leases[0],
            beam_kv_pool_lease=beam_leases[0],
        ),
        GRContinuousRequestState(
            request=make_request(2, context_len=6),
            context_kv_pool_lease=context_leases[2],
            beam_kv_pool_lease=beam_leases[2],
        ),
    )
    generation = GRGenerationState.from_prefill(
        request_id="batched",
        prefill=PrefillResult(
            logits=torch.zeros(2, 1, 8),
            context_kv=ContextKV(
                torch.cat(
                    (
                        context_leases[0].context_kv.key,
                        context_leases[2].context_kv.key,
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        context_leases[0].context_kv.value,
                        context_leases[2].context_kv.value,
                    ),
                    dim=1,
                ),
            ),
        ),
        max_decode_steps=2,
        max_beam_width=2,
        fixed_beam_width=2,
        beam_kv=BeamKV(
            torch.cat((beam_leases[0].beam_kv.key, beam_leases[2].beam_kv.key), dim=1),
            torch.cat(
                (beam_leases[0].beam_kv.value, beam_leases[2].beam_kv.value),
                dim=1,
            ),
        ),
    )
    executor = GRContinuousServingExecutor(
        engine=SimpleNamespace(),
        context_kv_pool=context_pool,
        beam_kv_pool=beam_pool,
        decode_cuda_graph_batch_buckets=(1, 2, 4),
    )

    graph_inputs = executor._decode_cuda_graph_inputs(
        torch.ones(2, 2, dtype=torch.long),
        generation,
        topk_indices=torch.zeros(2, 2, dtype=torch.long),
        states=states,
    )

    assert graph_inputs.use_cuda_graph is False
    assert executor.decode_cuda_graph_dynamic_skips == 1
    assert executor.decode_cuda_graph_dynamic_skip_reasons == {"pool_window": 1}


@pytest.mark.parametrize(
    ("max_entries", "expected_graphs", "expected_status"),
    [
        ("1", ["second"], {"decode_cuda_graph_evictions": 1}),
        ("0", [], {"decode_cuda_graph_entries": 0}),
    ],
)
def test_decode_cuda_graph_runner_store_limits(
    monkeypatch,
    max_entries,
    expected_graphs,
    expected_status,
) -> None:
    from gr_inference.gr_serving.decode_cuda_graph import GRDecodeCudaGraphRunner

    monkeypatch.setenv("GR_INFERENCE_DECODE_CUDA_GRAPH_MAX_ENTRIES", max_entries)
    runner = GRDecodeCudaGraphRunner(
        model=SimpleNamespace(), decode_engine=SimpleNamespace()
    )

    runner._store_graph(("first",), SimpleNamespace())
    runner._store_graph(("second",), SimpleNamespace())

    assert list(runner._graphs) == [(key,) for key in expected_graphs]
    for key, value in expected_status.items():
        assert runner.status()[key] == value


def test_prefill_cuda_graph_key_distinguishes_pool_views() -> None:
    torch = torch_or_skip()
    from gr_inference.gr_kv import ContextKV
    from gr_inference.gr_serving.prefill_cuda_graph import GRPrefillCudaGraphRunner

    key_storage = torch.zeros(1, 4, 8, 1, 2)
    value_storage = torch.zeros_like(key_storage)
    input_ids = torch.zeros(2, 8, dtype=torch.long)

    first_view = ContextKV(key_storage[:, 0:2], value_storage[:, 0:2])
    second_view = ContextKV(key_storage[:, 1:3], value_storage[:, 1:3])

    first_key = GRPrefillCudaGraphRunner._key(
        input_ids=input_ids,
        context_kv=first_view,
        last_token_logits_only=True,
    )
    second_key = GRPrefillCudaGraphRunner._key(
        input_ids=input_ids,
        context_kv=second_view,
        last_token_logits_only=True,
    )

    assert first_key != second_key


def test_decode_cuda_graph_key_distinguishes_pool_views() -> None:
    torch = torch_or_skip()
    from gr_inference.gr_kv import BeamKV, ContextKV
    from gr_inference.gr_serving.decode_cuda_graph import GRDecodeCudaGraphRunner

    context_key_storage = torch.zeros(1, 4, 8, 1, 2)
    context_value_storage = torch.zeros_like(context_key_storage)
    beam_key_storage = torch.zeros(1, 4, 3, 1, 2, 2)
    beam_value_storage = torch.zeros_like(beam_key_storage)
    beam_token_ids = torch.zeros(2, 2, dtype=torch.long)
    topk_indices = torch.zeros(2, 2, dtype=torch.long)

    def generation_for_slots(start: int):
        return SimpleNamespace(
            prefill=SimpleNamespace(
                context_len=8,
                context_kv=ContextKV(
                    context_key_storage[:, start : start + 2],
                    context_value_storage[:, start : start + 2],
                ),
            ),
            beam_kv=BeamKV(
                beam_key_storage[:, start : start + 2],
                beam_value_storage[:, start : start + 2],
            ),
        )

    first_key = GRDecodeCudaGraphRunner._key(
        beam_token_ids=beam_token_ids,
        generation=generation_for_slots(0),
        topk_indices=topk_indices,
        step=0,
        active_beam_width=2,
        decode_nums=1,
    )
    second_key = GRDecodeCudaGraphRunner._key(
        beam_token_ids=beam_token_ids,
        generation=generation_for_slots(1),
        topk_indices=topk_indices,
        step=0,
        active_beam_width=2,
        decode_nums=1,
    )

    assert first_key != second_key


def test_decode_cuda_graph_runner_records_fallback_reason() -> None:
    from gr_inference.gr_serving.decode_cuda_graph import GRDecodeCudaGraphRunner

    runner = GRDecodeCudaGraphRunner(
        model=SimpleNamespace(), decode_engine=SimpleNamespace()
    )
    generation = SimpleNamespace()

    logits = runner.forward_decode_step(
        SimpleNamespace(),
        generation,
        step=0,
        active_beam_width=1,
        topk_indices=None,
        decode_nums=1,
    )

    status = runner.status()
    assert logits is None
    assert status["decode_cuda_graph_requests"] == 1
    assert status["decode_cuda_graph_miss_count"] == 1
    assert status["decode_cuda_graph_fallback_eager_count"] == 1
    assert status["decode_cuda_graph_miss_reason_missing_topk_indices"] == 1


def test_prefill_cuda_graph_runner_records_fallback_reason() -> None:
    from gr_inference.gr_serving.prefill_cuda_graph import GRPrefillCudaGraphRunner

    runner = GRPrefillCudaGraphRunner(model=SimpleNamespace())

    prefill = runner.forward_prefill(
        SimpleNamespace(),
        context_kv=None,
        last_token_logits_only=True,
    )

    status = runner.status()
    assert prefill is None
    assert status["prefill_cuda_graph_requests"] == 1
    assert status["prefill_cuda_graph_miss_count"] == 1
    assert status["prefill_cuda_graph_fallback_eager_count"] == 1
    assert status["prefill_cuda_graph_miss_reason_missing_context_kv"] == 1


@pytest.mark.parametrize(
    ("request_id", "beam_width", "kwargs", "message"),
    [
        ("bad-stop", 1, {"stop_token_ids": (-1,)}, "stop_token_ids"),
        (
            "bad-policy",
            2,
            {"beam_width_policy": ScheduledBeamPolicy({0: 4})},
            "beam_width_policy",
        ),
        ("bad-policy-contract", 2, {"beam_width_policy": object()}, "width_for_step"),
    ],
)
def test_serving_request_validation_errors(
    request_id,
    beam_width,
    kwargs,
    message,
) -> None:
    request = GRServingRequest(
        request_id=request_id,
        input_ids=FakeInputIds((1, 8)),
        max_decode_steps=1,
        beam_width=beam_width,
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        request.validate()


def scheduler_state(request: GRServingRequest):
    from gr_inference.gr_serving import GRContinuousRequestState

    return GRContinuousRequestState(request=request, stage="decoding")
