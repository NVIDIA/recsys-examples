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

"""What the cache evicted goes back to storage without counting anything.

Without a tier below storage, ``_write_back_evicted`` takes arrays of any length with
inert entries (the reserved all-ones key, table id ``-1``) and writes the value of
every key storage still has to the row it holds there, which is what
``storage.insert(present, preserve_existing=True)`` does for the keys it was given. The
keys storage no longer has are dropped, or retained when storage retains evicted keys.
An entry whose ``slots`` index is negative is an insert the cache refused: its value
row was never loaded, so it is neither written back nor retained.

The reference of every test is that insert, given exactly the keys that are present and
live.
"""

import random

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy, EvictedItemMode
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import DynamicEmbStorage, _write_back_evicted
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

DIM = 8
POPULATION = 200  # keys 1..POPULATION are in storage, in one table
CAPACITY = 1024  # more than the keys of any test: nothing is evicted by them
REFUSED_AS_PRESENT = 0.25  # share of the entries that are refused inserts


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _storage(retain, device):
    """A storage holding the keys ``1..POPULATION`` with known values and scores."""
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=DIM,
            max_capacity=CAPACITY,
            init_capacity=CAPACITY,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
            evict_strategy=DynamicEmbEvictStrategy.CUSTOMIZED,
            evicted_item_mode=(
                EvictedItemMode.RETAIN_KEY if retain else EvictedItemMode.DISCARD
            ),
        )
    ]
    storage = DynamicEmbStorage(options, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))
    keys = torch.arange(1, POPULATION + 1, dtype=torch.int64, device=device)
    scores = torch.as_tensor(
        [10 * k for k in range(1, POPULATION + 1)], dtype=torch.uint64, device=device
    )
    storage.insert(
        keys, torch.zeros_like(keys), _original(keys.tolist(), device), scores=scores
    )
    torch.cuda.synchronize()
    return storage


def _original(keys, device):
    """The value storage starts with for each key: ``key + column / 16``."""
    base = torch.tensor(keys, dtype=torch.float32, device=device).unsqueeze(1)
    return base + torch.arange(DIM, dtype=torch.float32, device=device) / 16


def _new_values(n, device):
    """The values of an evicted batch: ``1000 + row + column / 16``."""
    base = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1) + 1000
    return base + torch.arange(DIM, dtype=torch.float32, device=device) / 16


def _snapshot(storage, device):
    """``(keys, embeddings, scores)`` of everything the storage holds, by key."""
    parts = list(storage.export_keys_values(device, batch_size=4096, table_id=0))
    keys = torch.cat([p[0] for p in parts])
    embs = torch.cat([p[1] for p in parts])
    scores = torch.cat([p[3] for p in parts])
    order = torch.argsort(keys)
    return keys[order], embs[order], scores[order]


def _assert_same_storage(a, b, device):
    keys_a, embs_a, scores_a = _snapshot(a, device)
    keys_b, embs_b, scores_b = _snapshot(b, device)
    assert torch.equal(keys_a, keys_b)
    assert torch.equal(embs_a, embs_b)
    assert torch.equal(scores_a, scores_b), "a write-back must keep every score"


def _batch(rng, n, with_pads, device):
    """Evicted arrays as an insert leaves them: keys storage has, keys it has not,
    and (with ``with_pads``) inert entries. Returns the arrays and, for the host
    side, which entries are present and live."""
    present_keys = iter(rng.sample(range(1, POPULATION + 1), n))
    keys, slots, present, live = [], [], [], []
    for i in range(n):
        kind = rng.choice(["present", "present", "absent"] + ["pad"] * with_pads)
        if kind == "pad":
            keys.append(-1)
            slots.append(-1)
            present.append(False)
            live.append(False)
            continue
        refused = rng.random() < REFUSED_AS_PRESENT
        keys.append(next(present_keys) if kind == "present" else 1000 + i)
        slots.append(-(i + 1) if refused else i)  # a refused insert: -(i + 1)
        present.append(kind == "present")
        live.append(not refused)
    tids = [0 if k != -1 else -1 for k in keys]
    return (
        torch.tensor(keys, dtype=torch.int64, device=device),
        torch.tensor(tids, dtype=torch.int64, device=device),
        torch.tensor(slots, dtype=torch.int64, device=device),
        present,
        live,
    )


def _reference_insert(storage, keys, tids, values, present, live):
    """What ``_write_back_evicted`` used to do: ``insert(present, preserve_existing)``,
    here of the entries that are present and not refused."""
    take = torch.tensor(
        [p and l for p, l in zip(present, live)], dtype=torch.bool, device=keys.device
    )
    if bool(take.any()):
        storage.insert(keys[take], tids[take], values[take], preserve_existing=True)


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_masked_write_back_equals_the_preserving_insert(device, retain, seed):
    rng = random.Random(seed)
    masked, reference = _storage(retain, device), _storage(retain, device)
    keys, tids, slots, present, live = _batch(rng, 48, with_pads=True, device=device)
    values = _new_values(keys.numel(), device)

    _write_back_evicted(masked, None, keys, tids, values, slots)
    _reference_insert(reference, keys, tids, values, present, live)
    torch.cuda.synchronize()

    _assert_same_storage(masked, reference, device)
    # the keys that were written really changed, and the refused ones did not
    snap_keys, snap_embs, _ = _snapshot(masked, device)
    by_key = dict(zip(snap_keys.tolist(), snap_embs))
    written = 0
    for i, key in enumerate(keys.tolist()):
        if key < 1 or key > POPULATION:
            continue
        expected = values[i] if live[i] else _original([key], device)[0]
        assert torch.equal(by_key[key], expected), (key, live[i])
        written += live[i]
    assert written > 0, "the batch must write something back"

    if retain:
        gone = sorted(
            key
            for key, p, l in zip(keys.tolist(), present, live)
            if key != -1 and not p and l
        )
        assert gone, "the batch must have keys that storage no longer has"
        assert masked.pop_evicted_keys(0).tolist() == gone
        # the reference insert of present keys evicts nothing
        assert reference.pop_evicted_keys(0).numel() == 0
    else:
        assert masked.pop_evicted_keys(0).numel() == 0


@pytest.mark.parametrize("retain", [False, True])
def test_a_write_back_without_slots_is_a_flush(device, retain):
    """A flush gives exact arrays and no ``slots``: nothing is refused, every key that
    is present is written."""
    rng = random.Random(5)
    masked, reference = _storage(retain, device), _storage(retain, device)
    keys, tids, _, present, _ = _batch(rng, 40, with_pads=False, device=device)
    values = _new_values(keys.numel(), device)

    _write_back_evicted(masked, None, keys, tids, values)
    _reference_insert(reference, keys, tids, values, present, [True] * len(present))
    torch.cuda.synchronize()

    _assert_same_storage(masked, reference, device)
    if retain:
        gone = sorted(k for k, p in zip(keys.tolist(), present) if not p)
        assert masked.pop_evicted_keys(0).tolist() == gone


def test_nothing_to_write_back_is_nothing(device):
    storage = _storage(True, device)
    before = _snapshot(storage, device)
    empty = torch.empty(0, dtype=torch.int64, device=device)
    _write_back_evicted(storage, None, empty, empty, _new_values(0, device), empty)
    # arrays of nothing but inert entries: not a key written, not a key retained
    keys = torch.full((6,), -1, dtype=torch.int64, device=device)
    _write_back_evicted(
        storage, None, keys, keys.clone(), _new_values(6, device), keys.clone()
    )
    after = _snapshot(storage, device)
    assert all(torch.equal(x, y) for x, y in zip(before, after))
    assert storage.pop_evicted_keys(0).numel() == 0


class _Tier:
    """A store below storage: it takes host-sized tensors, like an external one."""

    def __init__(self):
        self.calls = []

    def insert(self, keys, table_ids, values, scores=None, preserve_existing=False):
        self.calls.append((keys.tolist(), table_ids.tolist(), values.clone()))


def test_a_next_tier_still_gets_the_keys_storage_no_longer_has(device):
    """With a tier below, the keys are compacted as before: the ones storage has are
    written to it, the others go to the tier, in the order they came."""
    rng = random.Random(9)
    storage, reference = _storage(False, device), _storage(False, device)
    keys, tids, _, present, _ = _batch(rng, 40, with_pads=False, device=device)
    values = _new_values(keys.numel(), device)
    tier = _Tier()

    _write_back_evicted(storage, tier, keys, tids, values)
    _reference_insert(reference, keys, tids, values, present, [True] * len(present))
    torch.cuda.synchronize()

    _assert_same_storage(storage, reference, device)
    gone = [i for i, p in enumerate(present) if not p]
    assert len(tier.calls) == 1
    gone_keys, gone_tids, gone_values = tier.calls[0]
    assert gone_keys == [keys[i].item() for i in gone]
    assert gone_tids == [0] * len(gone)
    assert torch.equal(gone_values, values[gone])
