# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json

import pytest
from gr_inference.gr_kv import ContextKV, TensorSpec
from gr_inference.gr_runtime import (
    GRGenerationState,
    PrefillResult,
    SemanticItem,
    SemanticItemCatalog,
    TokenTrie,
    TrieItemMaskProviderStore,
)


def test_token_trie_allowed_next() -> None:
    trie = TokenTrie.from_sequences([[1, 10], [1, 11], [2, 20]])

    assert trie.allowed_next(()) == {1, 2}
    assert trie.allowed_next((1,)) == {10, 11}
    assert trie.allowed_next((2,)) == {20}
    assert trie.allowed_next((3,)) == set()


def test_token_trie_tracks_terminal_item_ids() -> None:
    trie = TokenTrie.from_items(
        [
            SemanticItem(item_id="item-a", token_ids=(1, 10)),
            ("item-b", (1, 11)),
            ("item-c", (2, 20)),
        ]
    )

    assert trie.allowed_next(()) == {1, 2}
    assert trie.is_terminal((1, 10))
    assert not trie.is_terminal((1,))
    assert trie.item_ids((1, 10)) == ("item-a",)
    assert trie.item_ids((1, 11)) == ("item-b",)


def test_trie_item_mask_provider_initial_and_step_masks() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import TrieItemMaskProvider

    trie = TokenTrie.from_sequences([[1, 10], [1, 11], [2, 20]])
    provider = TrieItemMaskProvider(trie, vocab_size=32)

    initial = provider.initial_mask(torch.zeros(1, 4, 32))

    assert initial[1]
    assert initial[2]
    assert not initial[3]

    prefill = PrefillResult(
        logits=torch.tensor([[[0.0, 5.0, 4.0] + [0.0] * 29]]),
        context_kv=ContextKV(
            TensorSpec("context_k", (1, 1, 4, 1, 8)),
            TensorSpec("context_v", (1, 1, 4, 1, 8)),
        ),
    )
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=2,
        max_beam_width=2,
    )
    generation.initialize_beams(item_mask=initial)

    step_mask = provider.step_mask(generation, torch.zeros(1, 2, 32))

    assert step_mask[0, 10] or step_mask[0, 20]
    assert step_mask[1, 10] or step_mask[1, 20]


def test_trie_item_mask_provider_allows_eos_for_terminal_items() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import TrieItemMaskProvider

    trie = TokenTrie.from_items([("item-a", (1, 10)), ("item-b", (2, 11))])
    provider = TrieItemMaskProvider(trie, vocab_size=32, eos_token_id=0)

    prefill = PrefillResult(
        logits=torch.tensor([[[0.0, 5.0, 4.0] + [0.0] * 29]]),
        context_kv=ContextKV(
            TensorSpec("context_k", (1, 1, 4, 1, 8)),
            TensorSpec("context_v", (1, 1, 4, 1, 8)),
        ),
    )
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=3,
        max_beam_width=2,
    )
    generation.initialize_beams_with_width(
        beam_width=2, item_mask=provider.initial_mask(prefill.logits)
    )
    generation.beam_path.append(
        parent_beams=(0, 1), token_ids=(10, 11), scores=(1.0, 0.9)
    )

    step_mask = provider.step_mask(generation, torch.zeros(1, 2, 32))

    assert step_mask[:, 0].tolist() == [True, True]
    assert not step_mask[:, 12].any()
    assert provider.resolve_item_ids((1, 10, 0)) == ("item-a",)
    assert provider.beam_item_results(generation.beam_path, beam_width=2)[0][
        "item_id"
    ] in {
        "item-a",
        "item-b",
    }


def test_semantic_item_catalog_loads_jsonl(tmp_path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"item_id": "item-a", "token_ids": [1, 10], "metadata": {"score": 0.9}},
                {"item_id": "item-b", "token_ids": [2, 11]},
            ]
        ),
        encoding="utf-8",
    )

    catalog = SemanticItemCatalog.from_jsonl(path, vocab_size=32)
    provider = catalog.provider(vocab_size=32, eos_token_id=0)

    assert catalog.status()["item_count"] == 2
    assert catalog.items[0].metadata["score"] == 0.9
    assert provider.allowed_next(()) == {1, 2}
    assert provider.resolve_item_ids((1, 10, 0)) == ("item-a",)


def test_semantic_item_catalog_rejects_duplicate_paths(tmp_path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"item_id": "item-a", "token_ids": [1, 10]},
                {"item_id": "item-b", "token_ids": [1, 10]},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate semantic token path"):
        SemanticItemCatalog.from_jsonl(path, vocab_size=32)


def test_semantic_item_catalog_rejects_tokens_outside_vocab(tmp_path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        json.dumps({"item_id": "item-a", "token_ids": [1, 40]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exceeds vocab_size"):
        SemanticItemCatalog.from_jsonl(path, vocab_size=32)


def test_trie_item_mask_provider_store_reload_jsonl(tmp_path) -> None:
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

    store = TrieItemMaskProviderStore.from_jsonl(first, vocab_size=32, eos_token_id=0)
    old_snapshot = store.snapshot()

    version = store.reload_jsonl(second, vocab_size=32, eos_token_id=0)
    new_snapshot = store.snapshot()

    assert version == 2
    assert store.status()["item_count"] == 1
    assert old_snapshot.resolve_item_ids((1, 10, 0)) == ("item-a",)
    assert old_snapshot.resolve_item_ids((2, 11, 0)) == ()
    assert new_snapshot.resolve_item_ids((2, 11, 0)) == ("item-b",)
