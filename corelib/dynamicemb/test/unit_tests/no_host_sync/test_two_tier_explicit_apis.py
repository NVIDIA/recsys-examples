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

"""The explicit APIs of a two-tier storage, and of a group of ranks.

``HybridStorage`` (an HBM tier over a host tier) has a replay of its own (one call for both
tiers, one read of the device whatever the batches), a ``pop_many`` (the evicted keys of the
host tier, the erased keys of both tiers merged on the host) and an ``erase_keys`` (one read
per tier), and its dump writes the host tier and then appends the HBM tier, which may hold
nothing. The group gather of the pops (``incremental_dump._all_gather_evicted_many``) puts the
keys of every table of every rank through two waits. None of them is reached by a test that
builds a single-tier storage or one rank.

What is checked: a two-tier replay leaves the target as the source is (the dump of the
target is the dump of the source, tier bits included) after one read, for SGD and Adam and for
int64 and uint64 tables; ``erase_keys`` removes from both tiers and the pops return the union
of what they removed, host tensors in key order, with a read per tier that held something; the
evicted keys of the host tier are popped as before; a dump of a storage with an empty tier (or
two) writes the other tier's keys and values; the gather of 3 tables over 3 ranks, with tables
that are empty on some of the ranks and a uint64 table, returns for each table the distinct keys
of the group in the order of its key type; the module's pop falls back to the per-table pop for
a storage that has no ``pop_many``.
"""

import importlib
import random
import types

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

DIM = 8
BUCKET = 128
HBM_KEYS = list(range(1, 41))
HOST_KEYS = list(range(101, 141))
BIG = (1 << 63) + 12345  # a uint64 key above 2**63


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _keys(values, key_type, device=None):
    """A key tensor of ``key_type`` for the given numbers (a uint64 key above 2**63 is its
    bit pattern as an int64)."""
    bits = [v - (1 << 64) if v >= (1 << 63) else v for v in values]
    return torch.tensor(bits, dtype=torch.int64, device=device).view(key_type)


def _numbers(t, key_type):
    """The numbers the keys of ``t`` stand for, in the key type's order."""
    bits = t.detach().cpu()
    bits = bits.view(torch.int64) if bits.dtype == torch.uint64 else bits
    mask = (1 << 64) - 1
    return [v & mask if key_type == torch.uint64 else v for v in bits.tolist()]


def _options(capacity, host, key_type, retain=False):
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    return config.DynamicEmbTableOptions(
        index_type=key_type,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=DIM,
        max_capacity=capacity,
        init_capacity=capacity,
        bucket_capacity=BUCKET,
        safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=0 if host else 1024**3,
        score_strategy=dynamicemb.DynamicEmbScoreStrategy.LFU,
        evict_strategy=config.DynamicEmbEvictStrategy.LFU,
        evicted_item_mode=(
            dynamicemb.EvictedItemMode.RETAIN_KEY
            if retain and host
            else dynamicemb.EvictedItemMode.DISCARD
        ),
    )


def _hybrid(optimizer="sgd", key_type=torch.int64, retain=False):
    """An HBM tier of one bucket over a host tier of eight: 128 and 1024 slots."""
    opt = _need("dynamicemb.optimizer")
    kvt = _need("dynamicemb.key_value_table")
    optimizer_cls = {
        "sgd": opt.SGDDynamicEmbeddingOptimizer,
        "adam": opt.AdamDynamicEmbeddingOptimizer,
    }[optimizer]
    return kvt.HybridStorage(
        [_options(BUCKET, False, key_type)],
        [_options(8 * BUCKET, True, key_type, retain)],
        optimizer_cls(opt.OptimizerArgs()),
    )


def _fill(state, key_values, key_type, seed):
    """Put keys into one tier directly, with random values (optimizer state included) and
    LFU counts. Returns ``{key: value row}``."""
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    keys = _keys(key_values, key_type, device)
    n = keys.numel()
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(n, state.value_dim, generator=generator)
    scores = torch.randint(1, 50, (n,), generator=generator, dtype=torch.int64)
    kvt._insert_key_values(
        state,
        keys,
        torch.zeros(n, dtype=torch.int64, device=device),
        values.to(device),
        scores.to(device),
    )
    torch.cuda.synchronize()
    return {k: row for k, row in zip(key_values, values)}


def _populated(
    optimizer="sgd",
    key_type=torch.int64,
    retain=False,
    hbm=HBM_KEYS,
    host=HOST_KEYS,
    seed=0,
):
    storage = _hybrid(optimizer, key_type, retain)
    rows = {}
    if hbm:
        rows.update(_fill(storage._hbm, hbm, key_type, seed))
    if host:
        rows.update(_fill(storage._host, host, key_type, seed + 1))
    return storage, rows


def _dump_by_key(storage):
    """The incremental dump of the whole table, columns in the order of the keys."""
    keys, values, slot_index, opt, scores = storage.incremental_dump(0, 0, None, 1)
    order = torch.argsort(keys)
    columns = {
        "keys": keys[order],
        "values": values[order],
        "slots": slot_index[order],
        "scores": scores[order],
    }
    if opt is not None:
        columns["opt"] = opt[order]
    return columns


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_a_two_tier_replay_leaves_the_target_as_the_source_is(gpu, optimizer, key_type):
    sf = _need("dynamicemb.sync_free")
    config = _need("dynamicemb.dynamicemb_config")
    source, _ = _populated(optimizer, key_type)
    target, _ = _populated(optimizer, key_type, hbm=[], host=[])
    keys, values, slot_index, opt, scores = source.incremental_dump(0, 0, None, 1)
    n = keys.numel()
    assert n == len(HBM_KEYS) + len(HOST_KEYS)
    assert int((slot_index < 0).sum()) == len(HOST_KEYS), "the host tier's tier bit"
    # several batches in each tier
    target._hbm.threads_in_wave = 16
    target._host.threads_in_wave = 16
    reads = sf.READ_NOW_CALLS

    stats = target.replay_increment(
        0, keys, values, opt, scores, slot_index, 1, config.ReplayContent.ALL, 1
    )
    torch.cuda.synchronize()

    assert stats.upserted == n
    assert sf.READ_NOW_CALLS - reads == 1, "one read for both tiers and every batch"
    got, expected = _dump_by_key(target), _dump_by_key(source)
    assert sorted(got) == sorted(expected)
    for column in got:
        assert torch.equal(got[column], expected[column]), column


def test_a_two_tier_replay_of_one_tier_only(gpu):
    """A delta whose keys are all in one tier replays that tier and nothing else."""
    config = _need("dynamicemb.dynamicemb_config")
    source, _ = _populated(hbm=[], host=HOST_KEYS)
    target, _ = _populated(hbm=[], host=[])
    keys, values, slot_index, opt, scores = source.incremental_dump(0, 0, None, 1)
    stats = target.replay_increment(
        0, keys, values, opt, scores, slot_index, 1, config.ReplayContent.ALL, 1
    )
    torch.cuda.synchronize()
    assert stats.upserted == len(HOST_KEYS)
    got, expected = _dump_by_key(target), _dump_by_key(source)
    for column in got:
        assert torch.equal(got[column], expected[column]), column


def test_a_two_tier_replay_of_a_key_that_cannot_be_placed_writes_no_value(gpu):
    """The failing key is in the host tier, the batches of the HBM tier are placed before it:
    the two tiers share one read, and no value of either tier is written."""
    config = _need("dynamicemb.dynamicemb_config")
    source, _ = _populated()
    target, _ = _populated(hbm=[], host=[])
    keys, values, slot_index, opt, scores = source.incremental_dump(0, 0, None, 1)
    victim = int((slot_index < 0).nonzero().view(-1)[-1])  # a key of the host tier
    bad = slot_index.clone()
    # the slot of the next bucket of the host tier (with the tier bit): not a home of the key
    slot = int(slot_index[victim]) & 0x7FFFFFFFFFFFFFFF
    bad[victim] = ((slot + BUCKET) % (8 * BUCKET)) | (-(1 << 63))
    target._hbm.threads_in_wave = 16
    target._host.threads_in_wave = 16
    hbm_before = target._hbm.tables[0].tensor().clone()
    host_before = target._host.tables[0].tensor().clone()
    torch.cuda.synchronize()

    with pytest.raises(RuntimeError, match="could not be written"):
        target.replay_increment(
            0, keys, values, opt, scores, bad, 1, config.ReplayContent.ALL, 1
        )
    torch.cuda.synchronize()

    assert torch.equal(target._hbm.tables[0].tensor(), hbm_before)
    assert torch.equal(target._host.tables[0].tensor(), host_before)


# ---------------------------------------------------------------------------
# erase and pop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
def test_a_two_tier_storage_erases_from_both_tiers_and_pops_their_union(gpu, key_type):
    sf = _need("dynamicemb.sync_free")
    config = _need("dynamicemb.dynamicemb_config")
    one, _ = _populated(key_type=key_type)
    other, _ = _populated(key_type=key_type)
    erased = [5, 6, 103, 104, 9999]  # 9999 is in neither tier
    asked = _keys(erased, key_type)  # a host tensor, as a delta's keys are
    mode = config.EvictedItemMode.RETAIN_KEY

    reads = sf.READ_NOW_CALLS
    assert one.erase_keys(0, asked, mode=mode) == 4, "the keys that were there"
    assert other.erase_keys(0, asked, mode=mode) == 4
    assert sf.READ_NOW_CALLS - reads == 4, "one read for each tier of each call"
    expected = [5, 6, 103, 104]

    reads = sf.READ_NOW_CALLS
    got = other.pop_many("pop_erased_keys", [0])
    torch.cuda.synchronize()
    assert sf.READ_NOW_CALLS - reads == 2, "one pass a tier that held something"
    assert got[0].device.type == "cpu" and got[0].dtype == key_type
    assert _numbers(got[0], key_type) == expected, "the union, in the order of the type"
    if key_type == torch.int64:
        assert (
            one.pop_erased_keys(0).tolist() == expected
        ), "what the per-table pop gives"
    # nothing is left, and an empty pop does not read
    reads = sf.READ_NOW_CALLS
    again = other.pop_many("pop_erased_keys", [0])
    assert again[0].numel() == 0 and again[0].dtype == key_type
    assert sf.READ_NOW_CALLS == reads
    # the erased keys are gone from both tiers, the others are not
    stay = _dump_by_key(other)["keys"].tolist()
    assert sorted(stay) == sorted(set(HBM_KEYS + HOST_KEYS) - set(expected))


def test_a_two_tier_storage_that_erased_from_one_tier_pops_that_tier(gpu):
    sf = _need("dynamicemb.sync_free")
    config = _need("dynamicemb.dynamicemb_config")
    storage, _ = _populated()
    assert (
        storage.erase_keys(
            0, _keys([101, 102], torch.int64), mode=config.EvictedItemMode.RETAIN_KEY
        )
        == 2
    )
    reads = sf.READ_NOW_CALLS
    assert storage.pop_many("pop_erased_keys", [0])[0].tolist() == [101, 102]
    assert sf.READ_NOW_CALLS - reads == 1, "the HBM tier held nothing erased"


def test_pop_many_of_a_two_tier_storage_pops_the_evicted_keys_of_the_host_tier(gpu):
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    one, _ = _populated(retain=True, hbm=[], host=[])
    other, _ = _populated(retain=True, hbm=[], host=[])
    members = [(0, 11), (0, 12), (0, 11), (0, 13)]
    for storage in (one, other):
        rng = random.Random(3)
        keys, tids = [-1] * 8, [-1] * 8
        for pos, (t, k) in zip(rng.sample(range(8), len(members)), members):
            keys[pos], tids[pos] = k, t
        kvt._append_evicted_masked(
            storage._host,
            torch.tensor(keys, dtype=torch.int64, device=device),
            torch.tensor(tids, dtype=torch.int64, device=device),
        )
    got = other.pop_many("pop_evicted_keys", [0])
    assert got[0].device.type == "cpu" and got[0].tolist() == [11, 12, 13]
    assert one.pop_evicted_keys(0).tolist() == [11, 12, 13]
    assert other._hbm.evicted_key_chunks == [], "the HBM tier retains nothing"
    assert other.pop_many("pop_evicted_keys", [0])[0].numel() == 0


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------
def _read(path, dtype, width=1):
    """A file of ``dtype`` values as ``[n, width]`` (an empty file is ``[0, width]``)."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return torch.empty(0, width, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).view(-1, width)


@pytest.mark.parametrize("which", ["no_hbm", "no_host", "both", "neither"])
def test_a_dump_of_a_two_tier_storage_with_an_empty_tier(gpu, tmp_path, which):
    """The host tier is written first and the HBM tier appended: either may hold no key, and
    the tier that is empty has batches with none (it is a table with fewer keys than
    batches)."""
    hbm = HBM_KEYS if which in ("no_host", "both") else []
    host = HOST_KEYS if which in ("no_hbm", "both") else []
    storage, rows = _populated(hbm=hbm, host=host)
    paths = [str(tmp_path / n) for n in ("keys", "emb", "scores", "opt")]

    storage.dump(
        0,
        str(tmp_path / "meta.json"),
        *paths,
        include_optim=False,
        include_meta=False,
        timestamp=0,
    )

    keys = _read(paths[0], torch.int64).view(-1).tolist()
    emb = _read(paths[1], torch.float32, DIM)
    assert sorted(keys) == sorted(hbm + host)
    assert emb.size(0) == len(keys) == _read(paths[2], torch.int64).size(0)
    for i, key in enumerate(keys):
        assert torch.equal(emb[i], rows[key]), key


# ---------------------------------------------------------------------------
# the group gather of the pops
# ---------------------------------------------------------------------------
def _fake_group(monkeypatch, world, dtypes):
    """``world[r][t]``: the keys (python numbers) rank ``r`` holds for table ``t``. Rank 0 is
    the caller; the collectives answer for the others from ``world``."""
    inc = _need("dynamicemb.incremental_dump")
    num_ranks, num_tables = len(world), len(dtypes)
    calls = []

    def as_bits(values, dtype):
        return _keys(values, dtype).view(torch.int64)

    def all_gather(out, inp, group=None):
        calls.append(inp.numel())
        if len(calls) == 1:  # the sizes: [T] on every rank
            for r in range(num_ranks):
                sizes = [len(world[r][t]) for t in range(num_tables)]
                out[r].copy_(torch.tensor(sizes, dtype=torch.int64))
        else:  # the keys, every table after the other, padded to the longest rank
            for r in range(num_ranks):
                flat = torch.cat(
                    [as_bits(world[r][t], dtypes[t]) for t in range(num_tables)]
                )
                padded = torch.zeros(out[r].numel(), dtype=torch.int64)
                padded[: flat.numel()] = flat
                out[r].copy_(padded)

    monkeypatch.setattr(inc.dist, "get_world_size", lambda group=None: num_ranks)
    monkeypatch.setattr(inc.dist, "all_gather", all_gather)
    return inc, calls, [_keys(world[0][t], dtypes[t]) for t in range(num_tables)]


@pytest.mark.parametrize(
    "world, dtypes",
    [
        # 3 ranks, 3 tables: table 1 is empty on rank 0 and table 0 on rank 2, a key of
        # table 0 is on two ranks, table 2 is empty everywhere but on rank 2
        (
            [
                [[3, 1, 2], [], []],
                [[2, 7], [4, 4], []],
                [[], [9], [5, 6, 5]],
            ],
            [torch.int64, torch.int64, torch.int64],
        ),
        # a uint64 table with keys above 2**63, which sort after the others
        (
            [
                [[BIG, 5], [-3, 8], []],
                [[7, BIG + 1], [], [1]],
            ],
            [torch.uint64, torch.int64, torch.int64],
        ),
        # every table empty everywhere
        ([[[], []], [[], []]], [torch.int64, torch.int64]),
    ],
)
def test_the_group_gather_returns_the_union_of_every_table_after_two_waits(
    gpu, monkeypatch, world, dtypes
):
    sf = _need("dynamicemb.sync_free")
    inc, calls, local = _fake_group(monkeypatch, world, dtypes)
    reads = sf.READ_NOW_CALLS

    got = inc._all_gather_evicted_many(local, object())

    assert len(calls) == 2, "the sizes of every table, then the keys of every table"
    assert sf.READ_NOW_CALLS - reads == 1, "one read: the sizes of every rank and table"
    assert len(got) == len(dtypes)
    for t, dtype in enumerate(dtypes):
        union = sorted({v for rank in world for v in rank[t]})
        assert got[t].device.type == "cpu" and got[t].dtype == dtype
        assert _numbers(got[t], dtype) == union, t


def test_the_group_gather_of_one_rank_returns_what_it_was_given(gpu, monkeypatch):
    sf = _need("dynamicemb.sync_free")
    inc = _need("dynamicemb.incremental_dump")
    monkeypatch.setattr(inc.dist, "get_world_size", lambda group=None: 1)
    host = [_keys([4, 2], torch.int64), _keys([BIG], torch.uint64)]
    reads = sf.READ_NOW_CALLS
    got = inc._all_gather_evicted_many(host, None)
    assert got[0] is host[0] and got[1] is host[1], "host keys are not copied"
    assert sf.READ_NOW_CALLS == reads
    # keys that are on the device (a storage without pop_many) are copied together
    device_keys = [host[0].to(_device()), host[1].to(_device())]
    on_host = inc._all_gather_evicted_many(device_keys, None)
    assert all(t.device.type == "cpu" for t in on_host)
    assert _numbers(on_host[0], torch.int64) == [4, 2]
    assert _numbers(on_host[1], torch.uint64) == [BIG]
    assert inc._all_gather_evicted_many([], None) == []


# ---------------------------------------------------------------------------
# the module's pop, with a storage that has no pop_many
# ---------------------------------------------------------------------------
def test_the_modules_pop_falls_back_to_the_per_table_pop_of_a_user_storage(gpu):
    tables = _need("dynamicemb.batched_dynamicemb_tables")
    config = _need("dynamicemb.dynamicemb_config")
    retain, discard = (
        config.EvictedItemMode.RETAIN_KEY,
        config.EvictedItemMode.DISCARD,
    )

    class UserStorage:
        """A storage of the user's: it pops one table at a time and has no pop_many."""

        def __init__(self):
            self.popped = []

        def pop_evicted_keys(self, table_id):
            self.popped.append(table_id)
            return torch.tensor(
                [10 * table_id + 1, 10 * table_id + 2], device=_device()
            )

    module = types.SimpleNamespace(
        _storage=UserStorage(),
        _table_names=["a", "b", "c"],
        _dynamicemb_options=[
            types.SimpleNamespace(evicted_item_mode=mode)
            for mode in (retain, discard, retain)
        ],
    )

    pop = tables.BatchedDynamicEmbeddingTablesV2._pop_retained
    wanted = lambda mode: retain in mode  # noqa: E731
    got = pop(module, "pop_evicted_keys", wanted, None)

    assert sorted(got) == ["a", "c"], "the tables that retain"
    assert module._storage.popped == [0, 2]
    assert got["a"].device.type == "cpu" and got["a"].tolist() == [1, 2]
    assert got["c"].tolist() == [21, 22]
    only = pop(module, "pop_evicted_keys", wanted, ["c"])
    assert sorted(only) == ["c"]
    # a storage without the method at all pops nothing
    assert pop(module, "pop_erased_keys", wanted, None) == {}
