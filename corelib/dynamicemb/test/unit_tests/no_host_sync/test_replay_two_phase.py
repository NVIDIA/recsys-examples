# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A replay places every key first, reads once that they all landed, and only then writes.

``replay_increment`` used to place a batch of keys, read how many were not placed, and
write the batch's rows and values, batch after batch: one wait per batch for the count,
two more for the masks of the write (``keep.any()``, ``fresh.any()``) and three
boolean-mask indexes that compact the rows. It now has two phases (see
``key_value_table._replay_tiers``): every batch places its keys, one read says whether
all of them did, and every batch writes its rows and values with masked stores. A key that
did not land raises before the first row or value is written -- where the old call had
already written every batch before the failing one.

What is checked: the keys that were in their slot keep their optimizer state and the
others start from the optimizer's initial state (the masked stores), a delta whose key
cannot be placed raises and leaves the rows and values of the table as they were, with
the failing key in the second batch of two (so that the first batch's writes would have
been visible), the waits of a replay of 1 and of 16 batches (one, for the placement), and
the waits and the result of ``erase_keys``: one, the count it returns, with the retained
keys and slots as before.
"""

import importlib
import random

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

DIM = 8
CAPACITY = 1 << 16


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _storage(optimizer="adam"):
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    opt = _need("dynamicemb.optimizer")
    kvt = _need("dynamicemb.key_value_table")
    optimizer_cls = {
        "sgd": opt.SGDDynamicEmbeddingOptimizer,
        "adam": opt.AdamDynamicEmbeddingOptimizer,
    }[optimizer]
    options = [
        config.DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=DIM,
            max_capacity=CAPACITY,
            init_capacity=CAPACITY,
            bucket_capacity=128,
            safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=dynamicemb.DynamicEmbScoreStrategy.CUSTOMIZED,
            evict_strategy=config.DynamicEmbEvictStrategy.CUSTOMIZED,
            evicted_item_mode=dynamicemb.EvictedItemMode.RETAIN_KEY,
        )
    ]
    return kvt.DynamicEmbStorage(options, optimizer_cls(opt.OptimizerArgs()))


def _fill(storage, num_keys, seed):
    """``num_keys`` random keys with random values (the optimizer state included)."""
    device = _device()
    rng = random.Random(seed)
    keys = torch.tensor(
        rng.sample(range(1, 1 << 40), num_keys), dtype=torch.int64, device=device
    )
    values = torch.randn(num_keys, storage._state.value_dim, device=device)
    scores = torch.as_tensor(
        list(range(1, num_keys + 1)), dtype=torch.uint64, device=device
    )
    storage.insert(keys, torch.zeros_like(keys), values, scores=scores)
    torch.cuda.synchronize()


def _delta(storage):
    """What an incremental dump of the whole table gives a replica: host tensors."""
    keys, values, slot_index, opt, scores = storage.incremental_dump(0, 0, None)
    return keys, values, opt, scores, slot_index


def _replay(storage, delta, content, timestamp=0):
    keys, values, opt, scores, slot_index = delta
    config = _need("dynamicemb.dynamicemb_config")
    return storage.replay_increment(
        0,
        keys,
        values,
        opt,
        scores,
        slot_index,
        1,
        getattr(config.ReplayContent, content),
        timestamp,
    )


def _contents(storage):
    """``{key: (embedding, optimizer state or None)}``, read the slow way."""
    out = {}
    for keys, emb, opt, _ in storage.export_keys_values(_device(), 8192, 0):
        keys, emb = keys.cpu().tolist(), emb.cpu()
        opt = None if opt is None else opt.cpu()
        for i, key in enumerate(keys):
            out[key] = (emb[i].clone(), None if opt is None else opt[i].clone())
    return out


def test_the_keys_in_their_slot_keep_their_state_and_the_others_start_over(gpu):
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    source, target = _storage(), _storage()
    _fill(source, 3000, 1)
    keys, values, opt, scores, slot_index = delta = _delta(source)
    n = keys.numel()
    half = n // 2
    first = tuple(t[:half] for t in delta)  # the keys that are in the target already
    _replay(target, first, "ALL")
    # their optimizer state in the target is made to be something the replay must keep
    state = target._state
    rows = slot_index[:half, 1].to(device)
    sentinel = torch.full((half, state.value_dim - DIM), 7.0, device=device)
    kvt.store_to_flat_single_table(
        state, rows, 0, torch.cat([values[:half].to(device), sentinel], dim=-1)
    )
    torch.cuda.synchronize()
    initial = torch.empty(1, state.value_dim - DIM, device=device)
    state.optimizer.reset_optimizer_states(initial, emb_dims=DIM)
    initial = initial[0].cpu()
    # embeddings that differ from what the target holds, so that a write is visible
    fresh_values = values + 1.0

    _replay(target, (keys, fresh_values, opt, scores, slot_index), "EMBEDDING_ONLY")
    torch.cuda.synchronize()

    got = _contents(target)
    assert len(got) == n
    for i, key in enumerate(keys.tolist()):
        emb, state_row = got[key]
        assert torch.equal(
            emb, fresh_values[i]
        ), "every key gets the embedding of the delta"
        if i < half:
            assert bool(
                (state_row == 7.0).all()
            ), "a key that was in its slot keeps its state"
        else:
            assert torch.equal(
                state_row, initial
            ), "a key that took the slot starts over"


def test_a_key_that_cannot_be_placed_raises_before_anything_is_written(gpu):
    """The failing key is in the second of two batches: the first batch's rows and values
    must not have been written either."""
    _need("dynamicemb.key_value_table")
    source, target = _storage(), _storage()
    _fill(source, 400, 2)
    keys, values, opt, scores, slot_index = _delta(source)
    n = keys.numel()
    state = target._state
    state.threads_in_wave = n // 2  # two batches
    bucket = state.key_index_map.bucket_capacity_
    bad = slot_index.clone()
    victim = n - 3  # in the second batch
    bad[victim, 0] = (bad[victim, 0] + bucket) % CAPACITY  # another bucket: not a home

    kim = state.key_index_map
    rows_before = kim.rows_.clone()
    counters_before = kim.row_counters_.clone()
    values_before = state.tables[0].tensor().clone()
    torch.cuda.synchronize()

    with pytest.raises(RuntimeError, match=r"1 of %d keys could not be written" % n):
        _replay(target, (keys, values, opt, scores, bad), "ALL")
    torch.cuda.synchronize()

    assert torch.equal(kim.rows_, rows_before), "no key was given a row"
    assert torch.equal(kim.row_counters_, counters_before)
    assert torch.equal(state.tables[0].tensor(), values_before), "no value was written"
    # and the same delta without the divergence goes through on the same target
    stats = _replay(target, (keys, values, opt, scores, slot_index), "ALL")
    assert stats.upserted == n


@pytest.mark.parametrize("batches", [1, 16])
def test_a_replay_waits_once_however_many_batches_it_has(gpu, count_syncs, batches):
    source, target = _storage(), _storage()
    _fill(source, 3200, 3)
    delta = _delta(source)
    n = delta[0].numel()
    target._state.threads_in_wave = max(1, -(-n // batches))  # ceil: `batches` batches
    count = count_syncs(
        lambda i: _replay(target, delta, "EMBEDDING_ONLY", timestamp=1),
        iters=3,
        warmup=1,
    )
    # the one read of the placement, and the staging rings' back-pressure: a wait for a
    # buffer when the device is still busy with the transfer two batches back, which a
    # device that keeps up never makes (so a quarter of the ring turns is a generous bound)
    assert count.blocking <= 1.0 + batches // 4, count.describe()


def _erase_fixture(count):
    storage = _storage("sgd")
    _fill(storage, count, 4)
    keys, _, slot_index, _, _ = storage.incremental_dump(0, 0, None)
    return storage, keys, slot_index


def test_erase_keys_returns_the_count_and_retains_what_was_there(gpu):
    config = _need("dynamicemb.dynamicemb_config")
    storage, keys, slot_index = _erase_fixture(600)
    present = keys[:200].tolist()
    asked = torch.tensor(
        present + [(1 << 41) + i for i in range(50)], dtype=torch.int64
    )

    removed = storage.erase_keys(0, asked, mode=config.EvictedItemMode.RETAIN_KEY)

    assert removed == 200
    assert storage.pop_erased_keys(0).tolist() == sorted(present)
    by_key = {k: tuple(s) for k, s in zip(keys.tolist(), slot_index.tolist())}
    got = storage.pop_erased_slot_index(0).tolist()
    assert sorted(map(tuple, got)) == sorted(
        by_key[k] for k in present
    ), "(slot, row) of each"
    # what is gone is not there any more, what was not asked for is
    assert set(_contents(storage)) == set(keys.tolist()) - set(present)
    assert storage.erase_keys(0, asked) == 0, "erased once"


@pytest.mark.parametrize("retain", [False, True])
def test_erase_keys_waits_once(gpu, count_syncs, retain):
    """The count it returns is the one wait. Each measured call puts the keys back first
    (an insert, which waits for nothing), so that each call really removes them."""
    config = _need("dynamicemb.dynamicemb_config")
    storage, keys, _ = _erase_fixture(600)
    device = _device()
    mode = (
        config.EvictedItemMode.RETAIN_KEY if retain else config.EvictedItemMode.DISCARD
    )
    asked = keys[:300].clone()  # a host tensor: the upload is a pinned copy, not a wait
    again = asked.to(device)
    tids = torch.zeros_like(again)
    values = torch.randn(300, storage._state.value_dim, device=device)
    scores = torch.as_tensor(list(range(1, 301)), dtype=torch.uint64, device=device)
    torch.cuda.synchronize()

    def put_back_and_erase(i):
        storage.insert(again, tids, values, scores=scores)
        assert storage.erase_keys(0, asked, mode=mode) == 300

    count = count_syncs(put_back_and_erase, iters=3, warmup=1)
    assert count.blocking <= 1.0, count.describe()
