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

"""A checkpoint load does not wait for the device per batch.

The keys a rank does not own are dropped on the host, before anything goes to the device
(the ownership rule is plain tensor arithmetic, which a CPU tensor runs as a CUDA one
does), the owned keys are counted on the host, and every column of a batch crosses
through a ring of pinned staging buffers with a non-blocking copy. The load used to copy
each column from pageable memory (three to four waits per batch), to filter a sharded
checkpoint with four boolean-mask indexes per batch, and to copy every key batch to the
device and read the count of its owned keys back, only to size the table.

What is checked: the host rule is the device rule (and the textbook rule), the host count
is the count of owned keys, the batches are the owned rows of the files, a sharded load
puts into each shard the keys it owns with their values, scores and optimizer states, the
counter table's load does the same, and the waits of a load do not grow with the batches
(the difference between a load of 4 batches and of 1 is at most the ring's two
back-pressure waits, whatever else a load costs: the exact size confirmation is the
growth path's).
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
MASK64 = (1 << 64) - 1


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _owner(key, num_shards, dist_type="roundrobin"):
    """The textbook rule, in plain Python: the shard a key belongs to."""
    if dist_type == "roundrobin":
        return (key & MASK64) % num_shards
    ko = _need("dynamicemb.key_ownership")
    return ko.murmur3_hash_64bits(key) % num_shards


def _random_keys(n, seed):
    rng = random.Random(seed)
    # negative keys included: the modulo is unsigned
    return [rng.randrange(-(1 << 62), 1 << 62) for _ in range(n)]


@pytest.mark.parametrize("dist_type", ["roundrobin", "hash_roundrobin"])
@pytest.mark.parametrize("num_shards", [2, 3, 4, 8])
def test_the_owner_of_a_key_is_the_same_on_the_host_the_device_and_in_the_book(
    gpu, dist_type, num_shards
):
    ko = _need("dynamicemb.key_ownership")
    keys = _random_keys(20000, 1) + [0, -1, 1, -(1 << 63), (1 << 63) - 1, -2]
    tensor = torch.tensor(keys, dtype=torch.int64)
    host = ko.owning_shard(tensor, num_shards, dist_type)
    device = ko.owning_shard(tensor.to(_device()), num_shards, dist_type).cpu()
    assert torch.equal(host, device), "a CPU tensor runs the rule as a CUDA tensor does"
    book = [_owner(k, num_shards, dist_type) for k in keys]
    assert host.tolist() == book


def _write(path, tensor):
    with open(path, "wb") as f:
        f.write(tensor.numpy().tobytes())


def _checkpoint(tmp_path, keys, num_scores=1, opt_dim=0, seed=0):
    """Files of ``keys`` with random values, scores and optimizer states, and the
    columns they hold, as host tensors."""
    g = torch.Generator().manual_seed(seed)
    n = len(keys)
    columns = {
        "keys": torch.tensor(keys, dtype=torch.int64),
        "emb": torch.randn(n, DIM, generator=g),
        "scores": torch.randint(
            1, 1 << 40, (n, num_scores), generator=g, dtype=torch.int64
        ),
    }
    paths = {
        "keys": str(tmp_path / "keys"),
        "emb": str(tmp_path / "emb"),
        "scores": str(tmp_path / "scores"),
        "opt": str(tmp_path / "opt"),
    }
    _write(paths["keys"], columns["keys"])
    _write(paths["emb"], columns["emb"])
    _write(paths["scores"], columns["scores"])
    if opt_dim:
        columns["opt"] = torch.randn(n, opt_dim, generator=g)
        _write(paths["opt"], columns["opt"])
    return paths, columns


@pytest.mark.parametrize("shard, num_shards", [(0, 1), (0, 2), (1, 2), (2, 3)])
def test_the_owned_keys_of_a_file_are_counted_on_the_host(
    gpu, count_syncs, tmp_path, shard, num_shards
):
    kvt = _need("dynamicemb.key_value_table")
    keys = _random_keys(3 * BATCH // 2, 2)
    paths, _ = _checkpoint(tmp_path, keys)
    expected = sum(1 for k in keys if _owner(k, num_shards) == shard)

    def count(dist_type):
        return kvt._count_owned_keys(
            paths["keys"],
            _device(),
            dist_type,
            shard_index=shard,
            num_shards=num_shards,
        )

    assert count("roundrobin") == expected
    if num_shards > 1:
        hashed = sum(
            1 for k in keys if _owner(k, num_shards, "hash_roundrobin") == shard
        )
        assert count("hash_roundrobin") == hashed
    result = count_syncs(lambda i: count("roundrobin"), iters=2, warmup=1)
    assert result.blocking == 0, result.describe()


@pytest.mark.parametrize("shard, num_shards", [(0, 1), (1, 3)])
@pytest.mark.parametrize("num_scores, opt_dim", [(1, 0), (2, 16)])
def test_the_batches_of_a_checkpoint_are_the_rows_this_shard_owns(
    gpu, tmp_path, shard, num_shards, num_scores, opt_dim
):
    kvt = _need("dynamicemb.key_value_table")
    n = 2 * BATCH + 4321  # three batches, the last short
    keys = _random_keys(n, 3)
    paths, columns = _checkpoint(tmp_path, keys, num_scores, opt_dim)
    owned = torch.tensor([_owner(k, num_shards) == shard for k in keys])

    batches = list(
        kvt._iter_batches_from_files(
            paths["keys"],
            paths["emb"],
            paths["scores"],
            paths["opt"] if opt_dim else None,
            DIM,
            opt_dim,
            _device(),
            "roundrobin",
            num_scores=num_scores,
            shard_index=shard,
            num_shards=num_shards,
        )
    )
    assert len(batches) == 3
    assert all(t.is_cuda for b in batches for t in b if t is not None)
    got_keys = torch.cat([b[0] for b in batches]).cpu()
    got_emb = torch.cat([b[1] for b in batches]).cpu()
    got_scores = torch.cat([b[2] for b in batches]).cpu()
    assert torch.equal(got_keys, columns["keys"][owned]), "file order, owned keys only"
    assert torch.equal(got_emb, columns["emb"][owned])
    expected_scores = columns["scores"][owned]
    assert torch.equal(got_scores.view(-1, num_scores), expected_scores)
    assert got_scores.dim() == (2 if num_scores > 1 else 1)
    if opt_dim:
        got_opt = torch.cat([b[3] for b in batches]).cpu()
        assert torch.equal(got_opt, columns["opt"][owned])
    else:
        assert all(b[3] is None for b in batches)


def test_reading_the_batches_waits_for_the_device_at_most_for_a_buffer(
    gpu, count_syncs, tmp_path
):
    """Nothing is waited for but a staging buffer, and only when the device is still busy
    with the transfer of the batch two before: at most once a ring turn."""
    kvt = _need("dynamicemb.key_value_table")
    keys = _random_keys(3 * BATCH, 4)
    paths, _ = _checkpoint(tmp_path, keys)

    def read(i):
        for _batch in kvt._iter_batches_from_files(
            paths["keys"],
            paths["emb"],
            paths["scores"],
            None,
            DIM,
            0,
            _device(),
            "roundrobin",
            shard_index=1,
            num_shards=4,
        ):
            pass

    count = count_syncs(read, iters=2, warmup=1)
    assert count.blocking <= 1.0, count.describe()


def _storage(optimizer="sgd", capacity=1 << 20):
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
            max_capacity=capacity,
            init_capacity=capacity,
            bucket_capacity=128,
            safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=dynamicemb.DynamicEmbScoreStrategy.CUSTOMIZED,
            evict_strategy=config.DynamicEmbEvictStrategy.CUSTOMIZED,
        )
    ]
    return kvt.DynamicEmbStorage(options, optimizer_cls(opt.OptimizerArgs()))


def _fill(storage, keys, seed):
    device = _device()
    g = torch.Generator().manual_seed(seed)
    k = torch.tensor(keys, dtype=torch.int64, device=device)
    values = torch.randn(len(keys), storage._state.value_dim, generator=g).to(device)
    scores = torch.as_tensor(
        [(7 * i) % 1000 + 1 for i in range(len(keys))],
        dtype=torch.uint64,
        device=device,
    )
    storage.insert(k, torch.zeros_like(k), values, scores=scores)
    torch.cuda.synchronize()


def _contents(storage):
    """``{key: (embedding, optimizer state or None, score)}``, read the slow way."""
    out = {}
    for keys, emb, opt, scores in storage.export_keys_values(_device(), 8192, 0):
        keys, emb, scores = keys.cpu().tolist(), emb.cpu(), scores.cpu()
        opt = None if opt is None else opt.cpu()
        for i, key in enumerate(keys):
            out[key] = (
                emb[i].clone(),
                None if opt is None else opt[i].clone(),
                int(scores[i]),
            )
    return out


def _dump(storage, tmp_path, include_optim):
    names = ("meta.json", "keys", "emb", "scores", "opt")
    paths = [str(tmp_path / n) for n in names]
    storage.dump(0, *paths, include_optim=include_optim, include_meta=True, timestamp=0)
    return paths


@pytest.mark.parametrize("optimizer, include_optim", [("sgd", False), ("adam", True)])
@pytest.mark.parametrize("num_shards", [1, 3])
def test_a_sharded_load_puts_each_shard_the_keys_it_owns_with_their_state(
    gpu, tmp_path, optimizer, include_optim, num_shards
):
    keys = _random_keys(20000, 5)
    source = _storage(optimizer)
    _fill(source, keys, 1)
    expected = _contents(source)
    assert len(expected) == len(keys)
    paths = _dump(source, tmp_path, include_optim)
    seen = {}
    for shard in range(num_shards):
        target = _storage(optimizer)
        target.load(
            0,
            *paths,
            include_optim=include_optim,
            timestamp=0,
            shard_index=shard,
            num_shards=num_shards,
        )
        torch.cuda.synchronize()
        got = _contents(target)
        assert sorted(got) == sorted(
            k for k in expected if _owner(k, num_shards) == shard
        )
        for key, (emb, opt, score) in got.items():
            want_emb, want_opt, want_score = expected[key]
            assert torch.equal(emb, want_emb) and score == want_score
            if include_optim:
                assert opt is not None and torch.equal(opt, want_opt)
        seen.update(got)
    assert sorted(seen) == sorted(expected), "every key went to exactly one shard"


@pytest.mark.parametrize("num_shards", [1, 3])
def test_the_counter_table_loads_what_it_dumped(gpu, tmp_path, num_shards):
    scored = _need("dynamicemb.scored_hashtable")
    ext = _need("dynamicemb_extensions")

    def table():
        return scored.LinearBucketTable(
            [BATCH],
            [scored.ScoreSpec(name="frequency", policy=ext.ScorePolicy.ACCUMULATE)],
            key_type=torch.int64,
            bucket_capacity=128,
            device=_device(),
        )

    keys = _random_keys(5000, 6)
    freq = {k: 1 + i % 50 for i, k in enumerate(keys)}
    source = table()
    k = torch.tensor(keys, dtype=torch.int64, device=_device())
    arg = scored.ScoreArg(
        name="frequency",
        value=torch.as_tensor(
            [freq[x] for x in keys], dtype=torch.uint64, device=_device()
        ),
        policy=ext.ScorePolicy.ASSIGN,
    )
    source.insert(k, torch.zeros_like(k), arg)
    torch.cuda.synchronize()
    key_path, score_path = str(tmp_path / "keys"), str(tmp_path / "freq")
    source.dump(key_path, {"frequency": score_path}, table_id=0)

    seen = {}
    for shard in range(num_shards):
        target = table()
        target.load(
            key_path,
            {"frequency": score_path},
            table_id=0,
            dist_type="roundrobin",
            shard_index=shard,
            num_shards=num_shards,
        )
        torch.cuda.synchronize()
        got = {}
        for ks, named, _ in target._batched_export_keys_scores(
            ["frequency"], _device(), 0
        ):
            got.update(zip(ks.cpu().tolist(), named["frequency"].cpu().tolist()))
        assert sorted(got) == sorted(x for x in keys if _owner(x, num_shards) == shard)
        assert all(got[x] == freq[x] for x in got)
        seen.update(got)
    assert seen == freq


def _load_cost(directory, num_keys, count_syncs):
    """The waits of one load of a checkpoint of ``num_keys`` keys into a storage."""
    directory.mkdir(parents=True, exist_ok=True)
    source = _storage(capacity=1 << 20)
    _fill(source, _random_keys(num_keys, 8), 2)
    paths = _dump(source, directory, False)
    target = _storage(capacity=1 << 20)
    # the first load makes whatever the target needs, the measured ones load the same keys
    # again, half of them (the shard it owns)
    return count_syncs(
        lambda i: target.load(
            0, *paths, include_optim=False, timestamp=0, shard_index=0, num_shards=2
        ),
        iters=2,
        warmup=1,
    )


def test_the_waits_of_a_load_do_not_grow_with_its_batches(gpu, count_syncs, tmp_path):
    """4 batches of keys against 1: what a load costs besides its batches (the exact size
    confirmation of the growth path, once) is in both numbers, and the batches add at most
    the ring's back-pressure waits. The load used to add about nine a batch."""
    one = _load_cost(tmp_path / "one", 1000, count_syncs)
    many = _load_cost(tmp_path / "many", 3 * BATCH + 1000, count_syncs)
    assert many.blocking - one.blocking <= 2.0, (one.describe(), many.describe())
