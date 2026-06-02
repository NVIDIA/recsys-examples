import json

import pytest
from gr_inference.gr_runtime import (
    TokenTrie,
    TrieItemMaskProvider,
    TrieItemMaskProviderStore,
)
from gr_inference.gr_serving import (
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRInProcessServingFacade,
    GRServingRequest,
)


class FakeInputIds:
    def __init__(self, shape):
        self.shape = shape


def make_request(idx: int, *, decode_steps: int = 1) -> GRServingRequest:
    return GRServingRequest(
        request_id=f"req-{idx}",
        input_ids=FakeInputIds((1, 8)),
        max_decode_steps=decode_steps,
        beam_width=2,
    )


def test_in_process_facade_submit_poll_and_run_until_idle() -> None:
    api = GRInProcessServingFacade(
        GRContinuousScheduler(
            policy=GRContinuousBatchingPolicy(
                max_prefill_batch_size=2,
                max_decode_batch_size=2,
            )
        )
    )

    request_ids = api.submit_many((make_request(0), make_request(1)))

    assert request_ids == ("req-0", "req-1")
    assert api.poll("req-0") is None
    assert api.status()["waiting_prefill"] == 2

    responses = api.run_until_idle()

    assert len(responses) == 2
    assert api.poll("req-0") is responses[0]
    assert api.require_result("req-1").request_id == "req-1"
    assert api.metrics()["finished_requests"] == 2


def test_in_process_facade_submit_many_rejects_duplicate_batch_atomically() -> None:
    api = GRInProcessServingFacade(GRContinuousScheduler())

    with pytest.raises(RuntimeError, match="duplicate request_id"):
        api.submit_many((make_request(0), make_request(0)))

    assert api.status()["waiting_prefill"] == 0
    assert api.request_statuses() == ()


def test_in_process_facade_submit_many_rejects_existing_id_atomically() -> None:
    api = GRInProcessServingFacade(GRContinuousScheduler())
    api.submit(make_request(0))

    with pytest.raises(RuntimeError, match="duplicate request_id"):
        api.submit_many((make_request(1), make_request(0)))

    assert api.status()["waiting_prefill"] == 1
    assert [request["request_id"] for request in api.request_statuses()] == ["req-0"]


def test_in_process_facade_cancel_waiting_request() -> None:
    api = GRInProcessServingFacade(GRContinuousScheduler())
    api.submit(make_request(0))

    response = api.cancel("req-0", reason="client_cancelled")

    assert response.metadata["cancelled"] is True
    assert response.metadata["stop_reason"] == "client_cancelled"
    assert api.poll("req-0") is response
    assert api.status()["cancelled_requests"] == 1


def test_in_process_facade_timeout_unfinished_requests() -> None:
    api = GRInProcessServingFacade(GRContinuousScheduler())
    api.submit(make_request(0, decode_steps=3))

    (response,) = api.run_until_idle(max_ticks=1, timeout_unfinished=True)

    assert response.metadata["failed"] is True
    assert response.metadata["stop_reason"] == "timeout"
    assert api.status()["failed_requests"] == 1


def test_in_process_facade_require_result_rejects_unfinished_request() -> None:
    api = GRInProcessServingFacade(GRContinuousScheduler())
    api.submit(make_request(0))

    with pytest.raises(KeyError, match="has not finished"):
        api.require_result("req-0")


def test_in_process_facade_applies_default_item_constraints() -> None:
    provider = TrieItemMaskProvider(
        TokenTrie.from_items([("item-a", (1, 10))]),
        vocab_size=32,
    )
    store = TrieItemMaskProviderStore(provider, metadata={"source": "test"})
    scheduler = GRContinuousScheduler()
    api = GRInProcessServingFacade(scheduler, item_mask_provider_store=store)

    api.submit(make_request(0))

    assert scheduler.states["req-0"].request.item_mask_provider is provider
    assert api.status()["item_constraints"]["source"] == "test"
    assert api.catalog_status()["version"] == 1


def test_in_process_facade_keeps_explicit_item_constraints() -> None:
    default_provider = TrieItemMaskProvider(
        TokenTrie.from_items([("item-a", (1, 10))]),
        vocab_size=32,
    )
    explicit_provider = TrieItemMaskProvider(
        TokenTrie.from_items([("item-b", (2, 11))]),
        vocab_size=32,
    )
    scheduler = GRContinuousScheduler()
    api = GRInProcessServingFacade(
        scheduler,
        item_mask_provider_store=TrieItemMaskProviderStore(default_provider),
    )

    api.submit(
        GRServingRequest(
            request_id="req-explicit",
            input_ids=FakeInputIds((1, 8)),
            max_decode_steps=1,
            beam_width=2,
            item_mask_provider=explicit_provider,
        )
    )

    assert (
        scheduler.states["req-explicit"].request.item_mask_provider is explicit_provider
    )


def test_in_process_facade_reloads_item_catalog_jsonl(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"item_id": "item-a", "token_ids": [1, 10]}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"item_id": "item-b", "token_ids": [2, 11]}),
        encoding="utf-8",
    )
    api = GRInProcessServingFacade(
        GRContinuousScheduler(),
        item_mask_provider_store=TrieItemMaskProviderStore.from_jsonl(
            first,
            vocab_size=32,
        ),
    )
    old_provider = api.item_mask_provider_store.snapshot()

    version = api.reload_item_catalog_jsonl(second, vocab_size=32)
    new_provider = api.item_mask_provider_store.snapshot()

    assert version == 2
    assert api.catalog_status()["item_count"] == 1
    assert old_provider.resolve_item_ids((1, 10)) == ("item-a",)
    assert new_provider.resolve_item_ids((2, 11)) == ("item-b",)


def test_in_process_facade_requires_store_for_catalog_reload(tmp_path) -> None:
    api = GRInProcessServingFacade(GRContinuousScheduler())

    with pytest.raises(RuntimeError, match="item_mask_provider_store"):
        api.reload_item_catalog_jsonl(tmp_path / "missing.jsonl", vocab_size=32)
