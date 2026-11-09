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

"""A dump waits for the device once per batch, and moves only the rows the table holds.

A batch of the export is 65,536 slots at full length with its count on the device; the
rows of the live entries are copied to pinned memory, the count of the next batch rides
that copy (it is queued behind the next batch's launch), and one more batch is launched
before the rows in hand are yielded, so the device works while the files are written. That
is one wait per batch (and one for the first batch's count) where the dump used to wait for
the count, the keys, the values, the scores and the optimizer state of every batch.

What is checked: the files of a dump are the files the exact dump writes (the export it
replaced, kept below as ``reference_dump``), as sets of records, for several optimizers and
score layouts and with and without the optimizer state; the waits of a dump of 1 and of 8
batches (``nb + 1`` at most); the elements copied to the host (the live rows, not the slots
scanned); the order of the launches and the yields of the pipeline; the admission counter's
dump.
"""

import importlib
import random

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

BATCH = 65536
DIM = 8


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _storage(batches, optimizer="sgd", strategy="timestamp", key_type=torch.int64):
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    opt = _need("dynamicemb.optimizer")
    kvt = _need("dynamicemb.key_value_table")
    score = dynamicemb.DynamicEmbScoreStrategy
    strategies = {
        "timestamp": (score.TIMESTAMP, config.DynamicEmbEvictStrategy.LRU),
        "customized": (score.CUSTOMIZED, config.DynamicEmbEvictStrategy.CUSTOMIZED),
        "lfu_timestamp": (
            (score.LFU, score.TIMESTAMP),
            config.DynamicEmbEvictStrategy.LFU,
        ),
    }
    chosen, evict = strategies[strategy]
    args = opt.OptimizerArgs()
    optimizer_cls = {
        "sgd": opt.SGDDynamicEmbeddingOptimizer,
        "adam": opt.AdamDynamicEmbeddingOptimizer,
    }[optimizer]
    options = [
        config.DynamicEmbTableOptions(
            index_type=key_type,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=DIM,
            max_capacity=batches * BATCH,
            init_capacity=batches * BATCH,
            bucket_capacity=128,
            safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=chosen,
            evict_strategy=evict,
        )
    ]
    return kvt.DynamicEmbStorage(options, optimizer_cls(args))


def _key_numbers(num_keys, seed, key_type):
    """Random keys; with a uint64 table half of them are above ``2**63``, whose int64 bit
    pattern is negative: they come after the others as unsigned numbers."""
    numbers = random.Random(seed).sample(range(1, 1 << 40), num_keys)
    if key_type == torch.uint64:
        numbers = [v | (1 << 63) if v % 2 else v for v in numbers]
    return [v - (1 << 64) if v >= (1 << 63) else v for v in numbers]


def _fill(storage, num_keys, seed, strategy, key_type=torch.int64):
    """``num_keys`` random keys with random values (the optimizer state included)."""
    device = _device()
    rng = random.Random(seed)
    keys = torch.tensor(
        _key_numbers(num_keys, seed, key_type), dtype=torch.int64, device=device
    ).view(key_type)
    state = storage._state
    values = torch.randn(num_keys, state.value_dim, device=device)
    scores = None
    if strategy == "customized":
        scores = torch.as_tensor(
            list(range(1, num_keys + 1)), dtype=torch.uint64, device=device
        )
    elif strategy == "lfu_timestamp":
        scores = torch.as_tensor(
            [rng.randrange(1, 100) for _ in range(num_keys)],
            dtype=torch.uint64,
            device=device,
        )
    table_ids = torch.zeros(num_keys, dtype=torch.int64, device=device)
    storage.insert(keys, table_ids, values, scores=scores)
    torch.cuda.synchronize()
    return keys


def reference_dump(state, table_id, paths, include_optim, timestamp):
    """The dump before it was pipelined: the exact export, a copy of every column per batch."""
    kvt = _need("dynamicemb.key_value_table")
    ext = _need("dynamicemb_extensions")
    key_path, emb_path, score_path, opt_path = paths
    with open(key_path, "wb") as fkey, open(emb_path, "wb") as fembedding, open(
        score_path, "wb"
    ) as fscore, open(opt_path, "wb") as fopt:
        for keys, embeddings, opt_states, scores in kvt.export_keys_values_iter(
            state, device=_device(), table_id=table_id
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            fembedding.write(kvt._raw_bytes(embeddings))
            if state.evict_strategy == ext.EvictStrategy.KLru:
                scores = timestamp - scores
            fscore.write(scores.cpu().numpy().tobytes())
            if include_optim and opt_states is not None:
                to_write = state.optimizer.states_for_checkpoint(
                    opt_states, state.table_emb_dims_cpu[table_id]
                )
                fopt.write(kvt._raw_bytes(to_write))


def _read(path, dtype, width):
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return torch.empty(0, width, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).view(-1, width)


def _records(paths, num_scores, opt_width):
    """A dump's files as columns sorted by key."""
    key_path, emb_path, score_path, opt_path = paths
    keys = _read(key_path, torch.int64, 1).view(-1)
    order = torch.argsort(keys)
    out = {
        "keys": keys[order],
        "values": _read(emb_path, torch.float32, DIM)[order],
        "scores": _read(score_path, torch.int64, num_scores)[order],
    }
    if opt_width:
        out["opt"] = _read(opt_path, torch.float32, opt_width)[order]
    return out


CASES = [
    # (optimizer, score strategy, include the optimizer state)
    ("sgd", "timestamp", False),
    ("adam", "timestamp", True),
    ("adam", "customized", False),
    ("sgd", "lfu_timestamp", False),
]


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("optimizer, strategy, include_optim", CASES)
def test_the_files_are_the_files_of_the_exact_dump(
    gpu, tmp_path, optimizer, strategy, include_optim, key_type
):
    kvt = _need("dynamicemb.key_value_table")
    ext = _need("dynamicemb_extensions")
    storage = _storage(4, optimizer, strategy, key_type)  # 4 batches
    num_keys = 30000
    _fill(storage, num_keys, 3, strategy, key_type)
    state = storage._state
    timestamp = ext.device_timestamp()
    new = [str(tmp_path / ("new_" + n)) for n in ("keys", "emb", "scores", "opt")]
    old = [str(tmp_path / ("old_" + n)) for n in ("keys", "emb", "scores", "opt")]

    kvt._dump_table(
        state,
        0,
        str(tmp_path / "meta.json"),
        new[0],
        new[1],
        new[2],
        new[3],
        include_optim=include_optim,
        include_meta=False,
        timestamp=timestamp,
    )
    reference_dump(state, 0, old, include_optim, timestamp)

    num_scores = state.key_index_map.num_scores_
    opt_width = 0
    if include_optim and state.value_dim > DIM:
        opt_width = state.optimizer.get_ckpt_state_dim(DIM)
    got, expected = _records(new, num_scores, opt_width), _records(
        old, num_scores, opt_width
    )
    assert got["keys"].numel() == num_keys
    assert sorted(got) == sorted(expected)
    for column in got:
        assert torch.equal(got[column], expected[column]), column


def test_a_dump_of_an_empty_table_writes_empty_files(gpu, tmp_path):
    kvt = _need("dynamicemb.key_value_table")
    storage = _storage(2)
    paths = [str(tmp_path / n) for n in ("keys", "emb", "scores", "opt")]
    kvt._dump_table(
        storage._state,
        0,
        str(tmp_path / "meta.json"),
        *paths,
        include_optim=True,
        include_meta=False,
        timestamp=0,
    )
    for path in paths:
        with open(path, "rb") as f:
            assert f.read() == b""


def _counts(table):
    """How many keys each batch of the export of table 0 holds, in the order of the scan."""
    return [int(count) for count, *_ in table._export_batches_masked(0)]


def _has_a_gap(counts):
    """A batch that holds no key between two that do."""
    live = [i for i, n in enumerate(counts) if n]
    return bool(live) and 0 in counts[live[0] : live[-1]]


@pytest.mark.parametrize("optimizer, include_optim", [("sgd", False), ("adam", True)])
def test_a_dump_goes_through_batches_that_hold_no_key(
    gpu, tmp_path, optimizer, include_optim
):
    """A table of 8 batches that holds 6 keys has batches with none: in front of, after and
    (the layout is looked for among the keys of a few seeds) between the ones that hold a
    key. The dump writes what the exact dump writes. A table with fewer keys than batches,
    which is every small table and every new one, is this case."""
    kvt = _need("dynamicemb.key_value_table")
    ext = _need("dynamicemb_extensions")
    for seed in range(1, 60):
        storage = _storage(8, optimizer)
        _fill(storage, 6, seed, "timestamp")
        counts = _counts(storage._state.key_index_map)
        if _has_a_gap(counts):
            break
    else:
        pytest.fail(
            "no layout of 6 keys in 8 batches has a batch of none in the middle"
        )
    assert sum(counts) == 6
    state = storage._state
    timestamp = ext.device_timestamp()
    new = [str(tmp_path / ("new_" + n)) for n in ("keys", "emb", "scores", "opt")]
    old = [str(tmp_path / ("old_" + n)) for n in ("keys", "emb", "scores", "opt")]

    kvt._dump_table(
        state,
        0,
        str(tmp_path / "meta.json"),
        *new,
        include_optim=include_optim,
        include_meta=False,
        timestamp=timestamp,
    )
    reference_dump(state, 0, old, include_optim, timestamp)

    num_scores = state.key_index_map.num_scores_
    opt_width = state.optimizer.get_ckpt_state_dim(DIM) if include_optim else 0
    got, expected = _records(new, num_scores, opt_width), _records(
        old, num_scores, opt_width
    )
    assert got["keys"].numel() == 6
    assert sorted(got) == sorted(expected)
    for column in got:
        assert torch.equal(got[column], expected[column]), column


def test_the_files_of_a_dump_are_closed_when_it_fails(gpu, tmp_path, monkeypatch):
    """A dump that fails half way does not leave its files open."""
    kvt = _need("dynamicemb.key_value_table")
    storage = _storage(2)
    _fill(storage, 100, 11, "timestamp")
    opened = []
    real_open = open

    def spying_open(path, *args, **kwargs):
        f = real_open(path, *args, **kwargs)
        opened.append(f)
        return f

    def broken(*args, **kwargs):
        raise RuntimeError("the device failed")
        yield  # a generator, like the batches it replaces

    monkeypatch.setattr(kvt, "open", spying_open, raising=False)
    monkeypatch.setattr(kvt, "_dump_batches", broken)
    paths = [str(tmp_path / n) for n in ("keys", "emb", "scores", "opt")]
    with pytest.raises(RuntimeError, match="the device failed"):
        kvt._dump_table(
            storage._state,
            0,
            str(tmp_path / "meta.json"),
            *paths,
            include_optim=True,
            include_meta=False,
            timestamp=0,
        )
    assert len(opened) == 4 and all(f.closed for f in opened)


def _dump_once(kvt, state, tmp_path, include_optim=False):
    paths = [str(tmp_path / n) for n in ("keys", "emb", "scores", "opt")]

    def call(i):
        kvt._dump_table(
            state,
            0,
            str(tmp_path / "meta.json"),
            *paths,
            include_optim=include_optim,
            include_meta=False,
            timestamp=0,
        )

    return call


@pytest.mark.parametrize("batches", [1, 8])
def test_a_dump_waits_once_per_batch(
    gpu, count_syncs, instruments_ok, tmp_path, batches
):
    """``batches`` batches: the count of the first, then one copy per batch. The exact dump
    waited for the count, the keys, the values and the scores of each (4 per batch)."""
    kvt = _need("dynamicemb.key_value_table")
    storage = _storage(batches)
    _fill(storage, 3000, 5, "timestamp")
    count = count_syncs(_dump_once(kvt, storage._state, tmp_path), iters=3, warmup=1)
    assert count.blocking <= batches + 1, count.describe()


def test_a_dump_copies_the_rows_the_table_holds_not_the_slots_it_scans(
    gpu, tmp_path, monkeypatch
):
    sf = _need("dynamicemb.sync_free")
    kvt = _need("dynamicemb.key_value_table")
    batches, num_keys = 8, 3000
    storage = _storage(batches, "adam")
    _fill(storage, num_keys, 7, "timestamp")
    copied = []
    real = sf.pull_to_host

    def spy(*tensors):
        copied.append(sum(t.numel() for t in tensors))
        return real(*tensors)

    monkeypatch.setattr(sf, "pull_to_host", spy)
    _dump_once(kvt, storage._state, tmp_path, include_optim=False)(0)
    row = 1 + DIM + 1  # key, embedding, score: the optimizer state stays on the device
    assert sum(copied) <= num_keys * row + batches + 1, sum(copied)
    # the exact dump's batches at full length would have been
    assert sum(copied) < batches * BATCH * row // 10


def test_the_batches_of_a_dump_stay_inside_their_byte_budget(
    gpu, tmp_path, monkeypatch
):
    """The batches are full length and the pipeline keeps several alive, so a wide row has to
    make the batch shorter: the device holds the values of ``DUMP_BATCH_BYTES``, not of
    3 x 65,536 rows. What the files hold does not depend on the length of a batch."""
    kvt = _need("dynamicemb.key_value_table")
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")
    storage = _storage(1, "adam")
    _fill(storage, 3000, 7, "timestamp")
    state = storage._state
    row_bytes = state.table_value_dims_cpu[0] * state.emb_dtype.itemsize
    in_flight = scored.LIVE_PREFIX_BATCHES_IN_FLIGHT + 1
    budget = in_flight * row_bytes * 1024  # 1024 rows a batch
    monkeypatch.setattr(kvt, "DUMP_BATCH_BYTES", budget)
    rows = kvt._dump_batch_rows(state, 0)
    assert rows == 1024 and rows % 32 == 0
    # a narrow row does not make a batch longer than the 65,536 slots it was asked for
    monkeypatch.setattr(kvt, "DUMP_BATCH_BYTES", 1 << 40)
    assert kvt._dump_batch_rows(state, 0) == 65536
    assert kvt._dump_batch_rows(state, 0, 4096) == 4096
    # and a row too wide for any budget is still a tile of slots
    monkeypatch.setattr(kvt, "DUMP_BATCH_BYTES", 1)
    assert kvt._dump_batch_rows(state, 0) == 32

    monkeypatch.setattr(kvt, "DUMP_BATCH_BYTES", budget)
    timestamp = ext.device_timestamp()
    new = [str(tmp_path / ("new_" + n)) for n in ("keys", "emb", "scores", "opt")]
    old = [str(tmp_path / ("old_" + n)) for n in ("keys", "emb", "scores", "opt")]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    kvt._dump_table(
        state,
        0,
        str(tmp_path / "meta.json"),
        *new,
        include_optim=True,
        include_meta=False,
        timestamp=timestamp,
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    # the keys, scores and slots of a batch ride along, 8 bytes each a slot: a little more
    # than the values, and far from the 3 x 65,536 rows of an unbounded dump
    assert peak <= 3 * budget, "the dump held %d bytes of transients" % peak

    reference_dump(state, 0, old, True, timestamp)
    opt_width = state.optimizer.get_ckpt_state_dim(DIM)
    got = _records(new, state.key_index_map.num_scores_, opt_width)
    expected = _records(old, state.key_index_map.num_scores_, opt_width)
    assert got["keys"].numel() == 3000
    for column in got:
        assert torch.equal(got[column], expected[column]), column


def test_the_pipeline_yields_the_live_rows_and_launches_ahead(gpu):
    scored = _need("dynamicemb.scored_hashtable")
    device = _device()
    counts = [3, 0, 5, 0, 2]
    log = []

    def batches():
        for b, n in enumerate(counts):
            log.append(("launch", b))
            column = torch.arange(8, device=device) + 100 * b
            block = column.float().unsqueeze(1).expand(8, 3).contiguous()
            count = torch.tensor([n], dtype=torch.int64, device=device)
            yield count, [column, block]

    seen = []
    for rows, (keys, block) in scored.pull_live_prefixes(batches()):
        b = len(seen)
        log.append(("yield", b))
        seen.append(rows)
        assert rows == counts[b]
        # a batch that holds nothing has its columns all the same, empty: the columns of
        # every batch can be unpacked without asking for the count first
        assert keys.dtype == torch.int64 and keys.shape == (rows,)
        assert block.dtype == torch.float32 and block.shape == (rows, 3)
        assert keys.tolist() == [100 * b + i for i in range(rows)]
        assert block[:, 0].tolist() == [float(100 * b + i) for i in range(rows)]
    assert seen == counts
    # the batch after the next is launched before a batch is handed over, so the device
    # is busy while the caller works on it
    for b in range(len(counts)):
        launched = sum(
            1 for kind, _ in log[: log.index(("yield", b))] if kind == "launch"
        )
        assert launched == min(b + 3, len(counts)), (b, log)


def test_the_pipeline_of_nothing_and_of_one_batch(gpu):
    scored = _need("dynamicemb.scored_hashtable")
    device = _device()
    assert list(scored.pull_live_prefixes(iter([]))) == []
    column = torch.arange(4, device=device)
    count = torch.tensor([4], dtype=torch.int64, device=device)
    (only,) = list(scored.pull_live_prefixes(iter([(count, [column])])))
    assert only[0] == 4 and only[1][0].tolist() == [0, 1, 2, 3]


def test_the_pipeline_waits_once_per_batch_and_once_for_the_first_count(
    gpu, count_syncs, instruments_ok
):
    scored = _need("dynamicemb.scored_hashtable")
    device = _device()
    counts = [3, 0, 5, 0, 2]
    batches = []
    for b, n in enumerate(counts):
        column = torch.arange(8, device=device) + 100 * b
        batches.append(
            (torch.tensor([n], dtype=torch.int64, device=device), [column, column])
        )
    torch.cuda.synchronize()

    def drain(i):
        assert [r for r, _ in scored.pull_live_prefixes(iter(batches))] == counts

    count = count_syncs(drain)
    assert count.blocking <= len(counts) + 1, count.describe()


def _counter_table(capacity, policy_name):
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")
    policy = {
        "frequency": ext.ScorePolicy.ACCUMULATE,
        "timestamp": ext.ScorePolicy.GLOBAL_TIMER,
    }[policy_name]
    return scored.LinearBucketTable(
        [capacity],
        [scored.ScoreSpec(name=policy_name, policy=policy)],
        key_type=torch.int64,
        bucket_capacity=128,
        device=_device(),
    )


def test_the_admission_counters_dump_is_the_exact_export(gpu, tmp_path):
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")
    table = _counter_table(2 * BATCH, "frequency")
    rng = random.Random(1)
    keys = rng.sample(range(1, 1 << 40), 20000)
    freq = [rng.randrange(1, 1000) for _ in keys]
    k = torch.tensor(keys, dtype=torch.int64, device=_device())
    arg = scored.ScoreArg(
        name="frequency",
        value=torch.tensor(freq, dtype=torch.uint64, device=_device()),
        policy=ext.ScorePolicy.ASSIGN,
    )
    table.insert(k, torch.zeros_like(k), arg)
    torch.cuda.synchronize()
    key_path, score_path = str(tmp_path / "keys"), str(tmp_path / "freq")

    table.dump(key_path, {"frequency": score_path}, table_id=0)

    got_keys = _read(key_path, torch.int64, 1).view(-1)
    got_freq = _read(score_path, torch.int64, 1).view(-1)
    assert got_keys.numel() == len(keys)
    assert dict(zip(got_keys.tolist(), got_freq.tolist())) == dict(zip(keys, freq))


def test_a_counters_dump_goes_through_batches_that_hold_no_key(gpu, tmp_path):
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")
    for seed in range(1, 60):
        table = _counter_table(8 * BATCH, "frequency")
        rng = random.Random(seed)
        keys = rng.sample(range(1, 1 << 40), 6)
        freq = [rng.randrange(1, 1000) for _ in keys]
        arg = scored.ScoreArg(
            name="frequency",
            value=torch.tensor(freq, dtype=torch.uint64, device=_device()),
            policy=ext.ScorePolicy.ASSIGN,
        )
        k = torch.tensor(keys, dtype=torch.int64, device=_device())
        table.insert(k, torch.zeros_like(k), arg)
        torch.cuda.synchronize()
        counts = _counts(table)
        if _has_a_gap(counts):
            break
    else:
        pytest.fail(
            "no layout of 6 keys in 8 batches has a batch of none in the middle"
        )
    key_path, score_path = str(tmp_path / "keys"), str(tmp_path / "freq")

    table.dump(key_path, {"frequency": score_path}, table_id=0)

    got_keys = _read(key_path, torch.int64, 1).view(-1)
    got_freq = _read(score_path, torch.int64, 1).view(-1)
    assert dict(zip(got_keys.tolist(), got_freq.tolist())) == dict(zip(keys, freq))


def test_a_counter_with_a_timer_score_dumps_ages(gpu, tmp_path):
    """A recency score is dumped as an age against one timestamp read for the whole dump:
    age + score is the same number for every key."""
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")
    table = _counter_table(BATCH, "timestamp")
    k = torch.arange(1, 2001, dtype=torch.int64, device=_device())
    arg = scored.ScoreArg(
        name="timestamp", value=None, policy=ext.ScorePolicy.GLOBAL_TIMER
    )
    table.insert(k, torch.zeros_like(k), arg)
    torch.cuda.synchronize()
    exact = {}
    for keys, named, _ in table._batched_export_keys_scores(
        ["timestamp"], _device(), 0
    ):
        exact.update(zip(keys.cpu().tolist(), named["timestamp"].cpu().tolist()))
    key_path, score_path = str(tmp_path / "keys"), str(tmp_path / "ages")

    table.dump(key_path, {"timestamp": score_path}, table_id=0)

    got_keys = _read(key_path, torch.int64, 1).view(-1).tolist()
    ages = _read(score_path, torch.int64, 1).view(-1).tolist()
    assert sorted(got_keys) == sorted(exact)
    totals = {a + exact[key] for key, a in zip(got_keys, ages)}
    assert len(totals) == 1 and totals.pop() >= max(exact.values())


@pytest.mark.parametrize("batches", [1, 8])
def test_a_counter_dump_waits_once_per_batch(
    gpu, count_syncs, instruments_ok, tmp_path, batches
):
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")
    table = _counter_table(batches * BATCH, "frequency")
    k = torch.arange(1, 3001, dtype=torch.int64, device=_device())
    arg = scored.ScoreArg(
        name="frequency",
        value=torch.ones(3000, dtype=torch.uint64, device=_device()),
        policy=ext.ScorePolicy.ASSIGN,
    )
    table.insert(k, torch.zeros_like(k), arg)
    torch.cuda.synchronize()
    key_path, score_path = str(tmp_path / "keys"), str(tmp_path / "freq")

    count = count_syncs(
        lambda i: table.dump(key_path, {"frequency": score_path}, table_id=0),
        iters=3,
        warmup=1,
    )
    assert count.blocking <= batches + 1, count.describe()
