# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redis deltas must respect table ranges and update every cache tier."""

import itertools
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pynve import nve
from pynve.torch import nve_layers
from pynve.torch.nve_export import save_nve
from pynve.torch.nve_ps import NVEParameterServer

from dynamicemb.exportable_embedding import (
    EmbeddingCollectionUpdateAck,
    EmbeddingCollectionUpdateCoordinator,
    EmbeddingCollectionUpdateSubscriber,
    dump_embedding_collection_indexers,
    embedding_collection_indexers_from_model,
)
from dynamicemb.exportable_embedding.config import BitConcatConfig
from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType as Indexer,
    InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA Redis layers"
)
INDEXERS = (
    Indexer.IDENTITY,
    Indexer.FUSED_IDENTITY,
    Indexer.LINEAR_HASH_MAP,
    Indexer.BIT_CONCAT,
)
DIM, ROWS, CACHE_BYTES = 64, 4, 1 << 20
NAMESPACES = itertools.count(1)


@pytest.fixture(scope="module")
def redis_address(tmp_path_factory):
    executable = shutil.which("redis-server")
    assert (
        executable
    ), "redis-server is required, as it is for the core incremental demo"
    directory = tmp_path_factory.mktemp("redis-removals")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (directory / "redis.log").open("w") as log:
        process = subprocess.Popen(
            [
                executable,
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(directory),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                assert process.poll() is None, "Redis exited during startup"
                try:
                    with socket.create_connection(
                        ("127.0.0.1", port), timeout=0.2
                    ) as client:
                        client.sendall(b"*1\r\n$4\r\nPING\r\n")
                        if client.recv(64) == b"+PONG\r\n":
                            break
                except OSError:
                    pass
                assert time.monotonic() < deadline, "Redis startup timed out"
                time.sleep(0.05)
            yield f"127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def runtime(path, address, kind, dtype, host_cache):
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    names = ["a"] if kind == Indexer.IDENTITY else ["a", "b"]
    ps = NVEParameterServer(
        0,
        DIM,
        dtype,
        ps_type=nve.Redis,
        extra_params={
            "plugin": {"address": address, "single_node": True},
            "table": {"num_partitions": 0, "string_namespace_id": next(NAMESPACES)},
        },
    )
    model = create_inference_embedding_collection(
        [
            SimpleNamespace(
                name=name, num_embeddings=ROWS, embedding_dim=DIM, feature_names=[name]
            )
            for name in names
        ],
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=kind,
            nve_layer_type="hierarchical",
            parameter_server=ps,
            gpu_cache_size=CACHE_BYTES,
            host_cache_size=host_cache,
            bit_concat=(
                BitConcatConfig(table_id_bits=8, feature_id_bits=55)
                if kind == Indexer.BIT_CONCAT
                else None
            ),
        ),
        pooling_mode=-1,
        output_dtype=dtype,
        device=device,
    ).eval()
    weights = (
        torch.arange(1, ROWS * len(names) + 1, dtype=dtype).view(-1, 1).repeat(1, DIM)
    )
    model.load_from_embedding_table(weights)
    keys = torch.tensor(list(range(ROWS)) * len(names), device=device)
    offsets = torch.arange(len(names) + 1, device=device) * ROWS
    table_ids = torch.arange(len(names), device=device).repeat_interleave(ROWS)
    storage_keys = model.indexer_(keys, table_ids).cpu()
    save_nve(model, str(path))
    indexers = embedding_collection_indexers_from_model(model)
    dump_embedding_collection_indexers(indexers, path)
    coordinator = EmbeddingCollectionUpdateCoordinator.open(
        package_dir=path,
        shared_update_dir=path / "updates",
        device=device,
        subscriber_ids=["reader"],
    )
    subscriber = EmbeddingCollectionUpdateSubscriber(
        str(path), indexers, {"nve_embedding_": model.nve_embedding_.emb_layer}, 0
    )
    rt = SimpleNamespace(
        model=model,
        ps=ps,
        kind=kind,
        dtype=dtype,
        names=names,
        keys=keys,
        offsets=offsets,
        storage_keys=storage_keys,
        weights=weights,
        coordinator=coordinator,
        subscriber=subscriber,
        indexers=indexers,
        id=next(iter(indexers.bindings)),
        binding=model.nve_embedding_.emb_layer,
        host_cache=host_cache,
    )
    # Explicit admission makes warm-tier assertions deterministic; lookup-driven
    # admission is asynchronous and heuristic. This changes only this fixture.
    for tier in ([1, 0] if host_cache else [0]):
        rt.binding.insert(
            storage_keys.numel(),
            storage_keys.data_ptr(),
            weights.data_ptr(),
            tier,
            torch.cuda.current_stream().cuda_stream,
        )
    torch.testing.assert_close(lookup(rt), weights)
    return rt


def lookup(rt):
    return rt.model(rt.keys, rt.offsets).cpu()


def cold_ps_rows(rt):
    # A fresh layer has no warm cache: observe the actual backing Redis rows.
    layer = nve_layers.NVEmbedding(
        rt.model.indexer_.nve_num_embeddings,
        DIM,
        rt.dtype,
        nve_layers.LayerType.Hierarchical,
        storage=rt.ps,
        gpu_cache_size=CACHE_BYTES,
        optimize_for_training=False,
        device=torch.device("cuda:0"),
    )
    return layer(rt.storage_keys.cuda()).cpu()


def check_removal(rt, delta, expected, ps_before_ack):
    update = rt.coordinator.apply_delta(rt.id, delta)
    torch.testing.assert_close(cold_ps_rows(rt), ps_before_ack)
    # Exercise the unchanged JSON payload and native C++ subscriber.
    rt.subscriber.apply_incremental_load(
        update.to_json(), torch.cuda.current_stream().cuda_stream
    )
    torch.testing.assert_close(lookup(rt), expected)
    ack = EmbeddingCollectionUpdateAck.from_json(
        rt.subscriber.wait_for_retirement(rt.id, update.snapshot_id)
    )
    rt.coordinator.acknowledge("reader", ack)
    torch.testing.assert_close(cold_ps_rows(rt), expected)
    if rt.host_cache:
        # Remove GPU entries so an uncleared intermediate host cache cannot hide.
        rt.binding.erase(
            rt.storage_keys.numel(),
            rt.storage_keys.data_ptr(),
            0,
            torch.cuda.current_stream().cuda_stream,
        )
        torch.testing.assert_close(lookup(rt), expected)


@pytest.mark.parametrize("kind", INDEXERS, ids=lambda kind: kind.value)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("host_cache", [0, CACHE_BYTES], ids=["gpu_only", "host_gpu"])
@pytest.mark.parametrize(
    "field", ["erased_keys", "evicted_keys"], ids=["erase", "evict"]
)
def test_removal_only(tmp_path, redis_address, kind, dtype, host_cache, field):
    rt = runtime(tmp_path, redis_address, kind, dtype, host_cache)
    columns = len(rt.names)
    delta = SimpleNamespace(
        table_names=rt.names,
        keys=[torch.empty(0, dtype=torch.int64) for _ in rt.names],
        values=[torch.empty((0, DIM), dtype=dtype) for _ in rt.names],
        evicted_keys=[None] * columns,
        erased_keys=[None] * columns,
    )
    setattr(delta, field, [torch.tensor([1, 1]) for _ in rt.names])
    expected = rt.weights.clone()
    expected[1::ROWS] = 0
    before_ack = rt.weights if kind == Indexer.LINEAR_HASH_MAP else expected
    check_removal(rt, delta, expected, before_ack)
    # An already removed key is harmless, including an erase-only empty upsert.
    check_removal(rt, delta, expected, expected)


@pytest.mark.parametrize("kind", INDEXERS, ids=lambda kind: kind.value)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("host_cache", [0, CACHE_BYTES], ids=["gpu_only", "host_gpu"])
def test_final_upserts_win_over_removals(
    tmp_path, redis_address, kind, dtype, host_cache
):
    rt = runtime(tmp_path, redis_address, kind, dtype, host_cache)
    values = [
        torch.tensor([9 + 10 * i, 8 + 10 * i], dtype=dtype).view(-1, 1).repeat(1, DIM)
        for i in range(len(rt.names))
    ]
    delta = SimpleNamespace(
        table_names=rt.names,
        keys=[torch.tensor([1, 2]) for _ in rt.names],
        values=values,
        evicted_keys=[torch.tensor([1, 3, 3]) for _ in rt.names],
        erased_keys=[torch.tensor([0, 1, 2, 3]) for _ in rt.names],
    )
    expected = torch.zeros_like(rt.weights)
    for i, rows in enumerate(values):
        expected[ROWS * i + 1 : ROWS * i + 3] = rows
    before_ack = expected.clone()
    if kind == Indexer.LINEAR_HASH_MAP:
        before_ack[0::ROWS] = rt.weights[0::ROWS]
        before_ack[3::ROWS] = rt.weights[3::ROWS]
    check_removal(rt, delta, expected, before_ack)


@pytest.mark.parametrize(
    "kind,table",
    [(Indexer.IDENTITY, 0), (Indexer.FUSED_IDENTITY, 0), (Indexer.FUSED_IDENTITY, 1)],
    ids=["identity", "fused_first", "fused_second"],
)
@pytest.mark.parametrize(
    "field,with_upserts",
    [
        ("keys", True),
        ("erased_keys", True),
        ("evicted_keys", True),
        ("erased_keys", False),
        ("evicted_keys", False),
    ],
    ids=["upsert", "erase_mixed", "evict_mixed", "erase_only", "evict_only"],
)
@pytest.mark.parametrize("bad_id", [-2, -1, ROWS, ROWS + 1])
@pytest.mark.parametrize("host_cache", [0, CACHE_BYTES], ids=["gpu_only", "host_gpu"])
def test_dense_delta_rejects_invalid_ids_before_mutation(
    tmp_path, redis_address, kind, table, field, with_upserts, bad_id, host_cache
):
    rt = runtime(tmp_path, redis_address, kind, torch.float32, host_cache)
    # Put the invalid table last, after valid work for another table when fused.
    names = [name for i, name in enumerate(rt.names) if i != table] + [rt.names[table]]
    delta = SimpleNamespace(
        table_names=names,
        keys=[torch.tensor([0]) for _ in names],
        values=[torch.full((1, DIM), 99.0) for _ in names],
        erased_keys=[torch.tensor([ROWS - 1]) for _ in names],
        evicted_keys=[None] * len(names),
    )
    if not with_upserts:
        delta.keys = [torch.empty(0, dtype=torch.int64) for _ in names]
        delta.values = [torch.empty((0, DIM)) for _ in names]
    if field == "keys":
        delta.keys[-1] = torch.tensor([0, bad_id])
        delta.values[-1] = torch.full((2, DIM), 99.0)
    else:
        getattr(delta, field)[-1] = torch.tensor([ROWS - 1, bad_id])

    coordinator = rt.coordinator
    ps = Mock(wraps=coordinator.parameter_servers[rt.id])
    coordinator.parameter_servers[rt.id] = ps
    snapshot_ids = coordinator._snapshot_ids.copy()
    snapshots = coordinator.snapshots.copy()
    pending_deletes = coordinator._pending_deletes.copy()
    update_files = set(coordinator.shared_update_dir.iterdir())
    with pytest.raises(ValueError, match="reserved range"):
        coordinator.apply_delta(rt.id, delta)
    ps.insert.assert_not_called()
    ps.erase.assert_not_called()
    assert coordinator._snapshot_ids == snapshot_ids
    assert coordinator._pending_deletes == pending_deletes
    assert coordinator.snapshots.keys() == snapshots.keys()
    assert all(
        coordinator.snapshots[name] is snapshot for name, snapshot in snapshots.items()
    )
    assert set(coordinator.shared_update_dir.iterdir()) == update_files
    torch.testing.assert_close(cold_ps_rows(rt), rt.weights)
    torch.testing.assert_close(lookup(rt), rt.weights)


@pytest.mark.parametrize("kind", [Indexer.IDENTITY, Indexer.FUSED_IDENTITY])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("host_cache", [0, CACHE_BYTES], ids=["gpu_only", "host_gpu"])
def test_dense_delta_accepts_boundary_upserts(
    tmp_path, redis_address, kind, dtype, host_cache
):
    rt = runtime(tmp_path, redis_address, kind, dtype, host_cache)
    # Both ends of each reserved range are valid; final upserts still win.
    keys = [torch.tensor([0, ROWS - 1]) for _ in rt.names]
    values = [
        torch.full((2, DIM), 20.0 + i, dtype=dtype) for i, _ in enumerate(rt.names)
    ]
    delta = SimpleNamespace(
        table_names=rt.names,
        keys=keys,
        values=values,
        erased_keys=[torch.tensor([0]) for _ in rt.names],
        evicted_keys=[torch.tensor([ROWS - 1]) for _ in rt.names],
    )
    expected = rt.weights.clone()
    for table, rows in enumerate(values):
        expected[table * ROWS] = rows[0]
        expected[(table + 1) * ROWS - 1] = rows[1]
    check_removal(rt, delta, expected, expected)
