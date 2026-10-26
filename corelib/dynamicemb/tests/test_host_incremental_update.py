# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from types import SimpleNamespace

import pytest
import torch
from pynve.torch.nve_export import save_nve

from dynamicemb.exportable_embedding import (
    EmbeddingCollectionUpdateSubscriber,
    dump_embedding_collection_indexers,
    embedding_collection_indexers_from_model,
)
from dynamicemb.exportable_embedding.incremental_update import (
    EmbeddingCollectionUpdateCoordinator,
)
from dynamicemb.exportable_embedding.indexer_snapshot import linear_hash_entries
from test_host_embedding import Indexer, collection, table


def delta(names, keys, values, evicted=None, erased=None):
    # The inference consumer needs only these DeltaDumpResult fields; importing
    # the training dataclass would unnecessarily require the CUDA training stack.
    return SimpleNamespace(
        table_names=names,
        keys=[torch.tensor(k, dtype=torch.int64) for k in keys],
        values=[torch.as_tensor(v, dtype=torch.float32) for v in values],
        evicted_keys=[None if k is None else torch.tensor(k, dtype=torch.int64)
                      for k in (evicted or [None] * len(names))],
        erased_keys=[None if k is None else torch.tensor(k, dtype=torch.int64)
                     for k in (erased or [None] * len(names))],
    )


def runtime(path, kind, dtype=torch.float32, sidecar=True, pooling=-1, rows=4, dim=8):
    names = ["a"] if kind == Indexer.IDENTITY else ["a", "b"]
    model = collection(indexer=kind, sidecar=sidecar, dtype=dtype, pooling=pooling,
                       tables=[table(name, rows, dim) for name in names]).eval()
    model.load_from_embedding_table(torch.zeros((rows * len(names), dim), dtype=dtype))
    path.mkdir(parents=True, exist_ok=True)
    save_nve(model, str(path))
    indexers = embedding_collection_indexers_from_model(model)
    dump_embedding_collection_indexers(indexers, path)
    coordinator = EmbeddingCollectionUpdateCoordinator.open(
        package_dir=path, shared_update_dir=path / "updates", device=torch.device("cpu")
    )
    subscriber = EmbeddingCollectionUpdateSubscriber(
        str(path), indexers, {"nve_embedding_": model.nve_embedding_.emb_layer}, -1
    )
    return SimpleNamespace(model=model, indexers=indexers, coordinator=coordinator,
                           subscriber=subscriber, id=next(iter(indexers.bindings)), names=names)


def lookup(rt, local_keys, pooling=-1):
    keys = torch.tensor(local_keys * len(rt.names), dtype=torch.int64)
    offsets = torch.arange(len(rt.names) + 1) * len(local_keys)
    args = (keys, offsets)
    if pooling != -1:
        args += (torch.arange(0, keys.numel() + 1, 2),)
    return rt.model(*args)


@pytest.mark.parametrize("kind", [Indexer.IDENTITY, Indexer.FUSED_IDENTITY])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("pooling", [-1, 2])
def test_fixed_updates_and_removals(tmp_path, kind, dtype, sidecar, pooling):
    rt = runtime(tmp_path, kind, dtype, sidecar, pooling)
    updates = delta(rt.names, [[0, 0, 1, 3]] * len(rt.names),
                    [[[1.] * 8, [2.] * 8, [4.] * 8, [8.] * 8]] * len(rt.names))
    update = rt.coordinator.apply_delta(rt.id, updates)
    assert update.indexer_snapshot_path is None
    rt.subscriber.apply_incremental_load(update.to_json())
    expected = torch.tensor([[x] * 8 for x in [2, 4, 0, 8]] * len(rt.names), dtype=dtype)
    if pooling == 2:
        expected = expected.reshape(-1, 2, 8).mean(1)
    torch.testing.assert_close(lookup(rt, [0, 1, 2, 3], pooling), expected)
    # An upsert wins over an earlier removal; explicit erases also zero fixed rows.
    removal = delta(rt.names, [[0]] * len(rt.names), [[[9.] * 8]] * len(rt.names),
                    evicted=[[0, 1]] * len(rt.names), erased=[[3]] * len(rt.names))
    update = rt.coordinator.apply_delta(rt.id, removal)
    rt.subscriber.apply_incremental_load(update.to_json())
    expected = torch.tensor([[x] * 8 for x in [9, 0, 0, 0]] * len(rt.names), dtype=dtype)
    if pooling == 2:
        expected = expected.reshape(-1, 2, 8).mean(1)
    torch.testing.assert_close(lookup(rt, [0, 1, 2, 3], pooling), expected)
    assert json.loads(rt.subscriber.wait_for_retirement(rt.id, update.snapshot_id))["snapshot_id"] == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_hash_paused_reuse_and_capacity(tmp_path, dtype):
    rt = runtime(tmp_path, Indexer.LINEAR_HASH_MAP, dtype)
    old = linear_hash_entries(rt.coordinator.snapshots[rt.id])
    changes = delta(["a", "b"], [[0, 1 << 40], [-9]],
                    [[[3.] * 8, [5.] * 8], [[7.] * 8]], evicted=[[1], [2]])
    update = rt.coordinator.apply_delta(rt.id, changes)
    with pytest.raises(RuntimeError, match="paused"):
        rt.subscriber.apply_incremental_load(update.to_json())
    rt.subscriber.apply_incremental_load(update.to_json(), inference_paused=True)
    rt.subscriber.wait_for_retirement(rt.id, update.snapshot_id)  # No lookup needed to retire on CPU.
    entries = linear_hash_entries(rt.coordinator.snapshots[rt.id])
    assert entries[0][0] == old[0][0]
    assert 1 not in entries[0] and 2 not in entries[1]
    assert {entries[0][1 << 40], entries[1][-9]} == {old[0][1], old[1][2]}
    torch.testing.assert_close(lookup(rt, [0, 1, 1 << 40, -9]),
        torch.tensor([[x] * 8 for x in [3, 0, 5, 0, 0, 0, 0, 7]], dtype=dtype))
    with pytest.raises(ValueError, match="capacity"):
        rt.coordinator.apply_delta(rt.id, delta(["a"], [[99]], [[[1.] * 8]]))
    # Repeated replacements never exhaust a monotonically increasing row counter.
    previous = 1 << 40
    for key in range(100, 120):
        update = rt.coordinator.apply_delta(rt.id, delta(["a"], [[key]], [[[2.] * 8]], evicted=[[previous]]))
        rt.subscriber.apply_incremental_load(update.to_json(), inference_paused=True)
        previous = key
    assert rt.coordinator.snapshots[rt.id].next_fused_key <= 10
    torch.testing.assert_close(lookup(rt, [119]), torch.tensor([[2.] * 8, [0.] * 8], dtype=dtype))


@pytest.mark.parametrize("kind", [Indexer.IDENTITY, Indexer.FUSED_IDENTITY, Indexer.LINEAR_HASH_MAP])
def test_reject_invalid_delta_without_advancing_state(tmp_path, kind):
    rt = runtime(tmp_path, kind)
    bad = [-1, 4] if kind != Indexer.LINEAR_HASH_MAP else [-1, -2, -3, -4]
    for key in bad:
        with pytest.raises(ValueError, match="range|reserved"):
            rt.coordinator.apply_delta(rt.id, delta(["a"], [[0, key]], [[[8.] * 8] * 2]))
    with pytest.raises(ValueError, match="values"):
        rt.coordinator.apply_delta(rt.id, delta(["a"], [[0]], [[[1.] * 7]]))
    update = rt.coordinator.apply_delta(rt.id, delta(["a"], [[0]], [[[2.] * 8]]))
    assert update.snapshot_id == 1
    payload = json.loads(update.to_json())
    payload["cache_update_keys"] = [-1]
    with pytest.raises(RuntimeError, match="outside storage"):
        rt.subscriber.apply_incremental_load(json.dumps(payload), inference_paused=True)
    rt.subscriber.apply_incremental_load(update.to_json(), inference_paused=True)
    with pytest.raises(RuntimeError, match="snapshot order"):
        rt.subscriber.apply_incremental_load(update.to_json(), inference_paused=True)


def test_hash_requires_sidecar(tmp_path):
    rt = runtime(tmp_path, Indexer.LINEAR_HASH_MAP, sidecar=False)
    with pytest.raises(ValueError, match="sidecar"):
        rt.coordinator.apply_delta(rt.id, delta(["a"], [[0]], [[[2.] * 8]]))


@pytest.mark.parametrize("kind", [Indexer.IDENTITY, Indexer.FUSED_IDENTITY])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pooling", [-1, 2])
def test_nonstop_cpu_lookup_during_updates(tmp_path, kind, dtype, pooling):
    rt = runtime(tmp_path, kind, dtype, pooling=pooling, rows=2048, dim=32)
    ready, stop = threading.Event(), threading.Event()
    errors, count = [], [0]
    def reader():
        try:
            while not stop.is_set():
                result = lookup(rt, list(range(0, 2048, 8)), pooling)
                assert torch.isfinite(result).all()
                count[0] += 1
                ready.set()
        except BaseException as error:
            errors.append(error)
            ready.set()
    worker = threading.Thread(target=reader)
    worker.start()
    assert ready.wait(30), "CPU reader did not start"
    progress = 0
    try:
        for value in range(1, 9):
            changes = delta(rt.names, [list(range(2048))] * len(rt.names),
                            [torch.full((2048, 32), float(value))] * len(rt.names))
            update = rt.coordinator.apply_delta(rt.id, changes)
            payload = update.to_json()
            before = count[0]
            rt.subscriber.apply_incremental_load(payload)
            progress += count[0] - before
            # Mixed/stale values are accepted during the write; completion must
            # expose the final value without a reader pause or publication call.
            torch.testing.assert_close(lookup(rt, [0, 2047], pooling),
                torch.full((len(rt.names) * (1 if pooling == 2 else 2), 32),
                           float(value), dtype=dtype))
    finally:
        stop.set()
        worker.join(30)
    assert not worker.is_alive()
    assert not errors, errors
    assert progress > 0, "Inference made no progress inside update calls"
