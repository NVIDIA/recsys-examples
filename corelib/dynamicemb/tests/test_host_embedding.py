# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType as Indexer,
    InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)
from dynamicemb.exportable_embedding.indexer import LinearHashMapIndexer
from dynamicemb.key_ownership import murmur3_hash_64bits


def table(name, rows=4, dim=8):
    # Same configuration fields as TorchRec, without importing its training runtime.
    return SimpleNamespace(
        name=name, num_embeddings=rows, embedding_dim=dim, feature_names=[name]
    )


def collection(
    device="cpu",
    indexer=Indexer.FUSED_IDENTITY,
    sidecar=True,
    dtype=torch.float32,
    pooling=-1,
    tables=None,
):
    return create_inference_embedding_collection(
        tables or [table("a"), table("b")],
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=indexer,
            nve_layer_type="host",
            indexer_state_sidecar=sidecar,
            bucket_capacity=16,
        ),
        pooling_mode=pooling,
        output_dtype=dtype,
        device=torch.device(device),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "indexer", list((Indexer.FUSED_IDENTITY, Indexer.LINEAR_HASH_MAP, Indexer.IDENTITY))
)
@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pooling", [-1, 1, 2])
def test_host_lookup(device, indexer, sidecar, dtype, pooling):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA comparison")
    tables = [table("a")] if indexer == Indexer.IDENTITY else [table("a"), table("b")]
    model = collection(device, indexer, sidecar, dtype, pooling, tables)
    weights = (torch.arange(4 * len(tables) * 8).reshape(-1, 8) / 8).to(dtype)
    model.load_from_embedding_table(weights)
    assert model.nve_embedding_.weight.device.type == "cpu"
    keys = torch.tensor([0, 3, 7, -8] * len(tables), device=device)
    offsets = torch.arange(len(tables) + 1, device=device) * 4
    expected = torch.cat(
        [
            torch.cat(
                (
                    weights[i * 4 : i * 4 + 1],
                    weights[i * 4 + 3 : i * 4 + 4],
                    torch.zeros((2, 8), dtype=dtype),
                )
            )
            for i in range(len(tables))
        ]
    )
    if pooling == -1:
        actual = model(keys, offsets)
    else:
        bags = torch.arange(0, keys.numel() + 1, 2, device=device)
        actual = model(keys, offsets, bags)
        expected = expected.reshape(-1, 2, 8).sum(1)
        if pooling == 2:
            expected /= 2
    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.device.type == torch.device(device).type
    if pooling == -1:
        empty = model(keys[:0], torch.zeros_like(offsets))
        assert empty.shape == (0, 8)
    with pytest.raises(RuntimeError, match="loads once"):
        model.load_from_embedding_table(weights)


def dump_table(path, name, keys, values, rank=0, world=1):
    (path / f"{name}_emb_keys.rank_{rank}.world_size_{world}").write_bytes(
        torch.tensor(keys, dtype=torch.int64).numpy().tobytes()
    )
    (path / f"{name}_emb_values.rank_{rank}.world_size_{world}").write_bytes(
        values.contiguous().view(torch.uint8).numpy().tobytes()
    )
    (path / f"{name}_opt_args.json").write_text(
        json.dumps(
            {
                "embedding_dtype": str(values.dtype).split(".")[-1],
                "embedding_dim": values.shape[1],
                "dist_type": "hash_roundrobin",
            }
        )
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sparse_checkpoint_duplicates_and_shards(tmp_path, device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA comparison")
    dump_table(
        tmp_path,
        "a",
        [-9, 1 << 40, -9],
        torch.tensor([[1.0] * 8, [2.0] * 8, [3.0] * 8], dtype=dtype),
        0,
        2,
    )
    dump_table(
        tmp_path,
        "a",
        [123, -9],
        torch.tensor([[4.0] * 8, [5.0] * 8], dtype=dtype),
        1,
        2,
    )
    model = collection(
        device, Indexer.LINEAR_HASH_MAP, dtype=dtype, tables=[table("a", 3)]
    )
    model.load_from_dynamicemb_file(str(tmp_path))
    keys = torch.tensor([-9, 1 << 40, 123, 99, -1, -2, -3, -4], device=device)
    actual = model(keys, torch.tensor([0, 8], device=device))
    expected = torch.tensor(
        [[v] * 8 for v in [5.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype
    )
    torch.testing.assert_close(actual.cpu(), expected)
    assert model.indexer_._snapshot.next_fused_key == 4


def test_source_errors(tmp_path):
    for indexer in (Indexer.FUSED_IDENTITY, Indexer.IDENTITY):
        dump_table(tmp_path, "a", [4], torch.ones((1, 8)))
        with pytest.raises(ValueError, match="table-local"):
            collection(indexer=indexer, tables=[table("a")]).load_from_dynamicemb_file(
                str(tmp_path)
            )
    dump_table(tmp_path, "a", [10, 20], torch.ones((2, 8)))
    with pytest.raises(RuntimeError, match="capacity"):
        collection(
            indexer=Indexer.LINEAR_HASH_MAP, tables=[table("a", 1)]
        ).load_from_dynamicemb_file(str(tmp_path))
    dump_table(tmp_path, "a", [-1], torch.ones((1, 8)))
    with pytest.raises(RuntimeError, match="reserved"):
        collection(
            indexer=Indexer.LINEAR_HASH_MAP, tables=[table("a")]
        ).load_from_dynamicemb_file(str(tmp_path))


def test_cpu_hash_grows_and_preserves_rows():
    ids = []
    key = 0
    while len(ids) < 32:
        if ((murmur3_hash_64bits(key) & ((1 << 63) - 1)) % 64) // 16 == 0:
            ids.append(key)
        key += 1
    model = LinearHashMapIndexer([32], bucket_capacity=16, device=torch.device("cpu"))
    keys = torch.tensor(ids)
    tables = torch.zeros_like(keys)
    expected = model.build_index(keys, tables)
    assert model._snapshot.table_bucket_offsets[-1] > 4
    model.finish_build()
    torch.testing.assert_close(model(keys, tables), expected)
    if torch.cuda.is_available():
        # CPU-built bytes must be readable by the existing CUDA kernel.
        snap = model._snapshot.to(torch.device("cuda", 0))
        values, found, _ = torch.ops.INFERENCE_EMB.table_lookup(
            snap.table_storage,
            snap.table_bucket_offsets,
            16,
            keys.cuda(),
            tables.cuda(),
            None,
            0,
            None,
            0,
            None,
        )
        assert found.all()
        torch.testing.assert_close(values.cpu(), expected)


def test_cuda_snapshot_readable_on_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA producer")
    model = LinearHashMapIndexer([4], device=torch.device("cuda", 0))
    keys = torch.tensor([9, 11, -9, 1 << 40], device="cuda")
    tables = torch.zeros_like(keys)
    expected = model.build_index(keys, tables)
    model.finish_build()
    snap = model._snapshot.to(torch.device("cpu"))
    actual = torch.ops.INFERENCE_EMB.linear_hash_lookup(
        snap.table_storage,
        snap.table_bucket_offsets,
        snap.bucket_capacity,
        snap.miss_storage_indices,
        keys.cpu(),
        tables.cpu(),
    )
    torch.testing.assert_close(actual, expected.cpu())


def test_empty_feature_boundaries():
    offsets = torch.tensor([0, 0, 0, 2, 2, 4])
    result = torch.ops.INFERENCE_EMB.expand_table_ids(
        offsets, torch.arange(4), torch.tensor([0, 2, 4, 5]), 3, 1
    )
    torch.testing.assert_close(result, torch.tensor([1, 1, 2, 2]))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_empty_table_and_weighted_bags(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA comparison")
    model = collection(device, pooling=1, tables=[table("empty", 0), table("b", 2)])
    values = torch.tensor([[1.0] * 8, [3.0] * 8])
    model.load_from_embedding_table(values)
    result = model(
        torch.tensor([0, 1], device=device),
        torch.tensor([0, 0, 2], device=device),
        torch.tensor([0, 0, 2], device=device),
        torch.tensor([2.0, 4.0], device=device),
    )
    torch.testing.assert_close(result.cpu(), torch.tensor([[0.0] * 8, [14.0] * 8]))


def test_host_config_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="BitConcat"):
        collection(indexer=Indexer.BIT_CONCAT)
    with pytest.raises(ValueError, match="exactly one"):
        collection(indexer=Indexer.IDENTITY)
    for kwargs in (
        {"gpu_cache_size": 1024},
        {"host_cache_size": 1024},
        {"parameter_server": object()},
    ):
        with pytest.raises(ValueError):
            create_inference_embedding_collection(
                [table("a")],
                pooling_mode=-1,
                device=torch.device("cpu"),
                config=InferenceEmbeddingCollectionConfig(
                    indexer_type=Indexer.FUSED_IDENTITY, nve_layer_type="host", **kwargs
                ),
            )


def test_cpu_process_needs_no_training_imports_or_cuda_initialization():
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        import torch
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU collection initialized CUDA")
        torch.cuda._lazy_init = forbidden
        from types import SimpleNamespace
        from dynamicemb.exportable_tables import (
            EmbeddingCollectionIndexerType, InferenceEmbeddingCollectionConfig,
            create_inference_embedding_collection,
        )
        model = create_inference_embedding_collection(
            [SimpleNamespace(name="a", num_embeddings=2, embedding_dim=8, feature_names=["a"])],
            config=InferenceEmbeddingCollectionConfig(
                indexer_type=EmbeddingCollectionIndexerType.LINEAR_HASH_MAP, nve_layer_type="host"),
            pooling_mode=-1, device=torch.device("cpu"))
        weights = torch.arange(16).reshape(2, 8).float()
        model.load_from_embedding_table(weights)
        torch.testing.assert_close(model(torch.tensor([1]), torch.tensor([0, 1])), weights[1:])
        assert "dynamicemb_extensions" not in sys.modules
        assert "torchrec" not in sys.modules
        assert not torch.cuda.is_initialized()
    """)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
