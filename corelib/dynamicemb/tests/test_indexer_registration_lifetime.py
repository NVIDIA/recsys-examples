# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discarded sidecar indexers must release storage after their last owner."""

import gc
import uuid

import pytest
import torch
from dynamicemb.exportable_embedding import (
    EmbeddingCollectionIndexerDirectory,
    dump_embedding_collection_indexers,
    embedding_collection_indexers_from_model,
    load_embedding_collection_indexers,
)
from dynamicemb.exportable_embedding.indexer import (
    FusedIdentityIndexer,
    LinearHashMapIndexer,
)
from torch.multiprocessing.reductions import StorageWeakRef

STORAGE_FIELDS = (
    "table_storage",
    "table_bucket_offsets",
    "miss_storage_indices",
    "valid_bases",
    "reserved_sizes",
)


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA indexer lifetime")
    return torch.device(request.param)


@pytest.fixture(params=["linear_hash_map", "fused_identity"])
def kind(request):
    return request.param


class Model(torch.nn.Module):
    def __init__(self, kind, device):
        super().__init__()
        self.device = device
        self.collection_id_ = "emb"
        self.table_names_ = ["a", "b"]
        self.table_capacities_ = [2, 2]
        if kind == "linear_hash_map":
            self.indexer_ = LinearHashMapIndexer(
                [2, 2], bucket_capacity=16, build_on_cpu=True, device=device
            )
        else:
            self.indexer_ = FusedIdentityIndexer([2, 2], device=device)
        self.indexer_.build_index(
            torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
        )
        self.indexer_.finish_build()


def storage_refs(snapshot):
    return [
        StorageWeakRef(tensor.untyped_storage())
        for name in STORAGE_FIELDS
        if (tensor := getattr(snapshot, name)).numel()
    ]


def assert_lookup(marker, marker_value, kind, device):
    rows = torch.ops.INFERENCE_EMB.embedding_collection_index(
        marker,
        marker_value,
        torch.tensor([0, 1, 99, 0, 1, 99], device=device),
        torch.tensor([0, 0, 0, 1, 1, 1], device=device),
    )
    expected = [2, 3, 0, 4, 5, 1] if kind == "linear_hash_map" else [1, 2, 0, 4, 5, 3]
    torch.testing.assert_close(rows.cpu(), torch.tensor(expected))


def assert_released(refs):
    gc.collect()
    assert refs and all(
        ref.expired() for ref in refs
    ), "Native registry retains indexer storage"


def test_eager_storage_released(kind, device):
    model = Model(kind, device)
    refs = storage_refs(model.indexer_.snapshot)
    assert_lookup(
        model.indexer_.marker_tensor, model.indexer_.marker_value_, kind, device
    )
    del model
    assert_released(refs)


@pytest.mark.parametrize("first", ["model", "directory"])
def test_shared_registration_survives_either_owner(kind, device, first):
    model = Model(kind, device)
    refs = storage_refs(model.indexer_.snapshot)
    marker, marker_value = model.indexer_.marker_tensor, model.indexer_.marker_value_
    # Keep this owner alive until the explicit del below.
    directory = embedding_collection_indexers_from_model(model)  # noqa: F841
    second = embedding_collection_indexers_from_model(model)
    if first == "model":
        del model
    else:
        del directory
    gc.collect()
    assert all(not ref.expired() for ref in refs)
    assert_lookup(marker, marker_value, kind, device)
    del second
    # One owner still lives after two of the three are discarded.
    assert_lookup(marker, marker_value, kind, device)
    if first == "model":
        del directory
    else:
        del model
    assert_released(refs)
    with pytest.raises(RuntimeError, match="No embedding-collection indexer is bound"):
        assert_lookup(marker, marker_value, kind, device)


def test_create_owns_registration_without_eager_owner(kind, device):
    model = Model(kind, device)
    eager = embedding_collection_indexers_from_model(model)
    refs = storage_refs(model.indexer_.snapshot)
    marker_value = uuid.uuid4().int & ((1 << 63) - 1)
    marker = torch.tensor([marker_value], device=device)
    directory = EmbeddingCollectionIndexerDirectory.create(
        eager.bindings,
        {"emb": marker},
        {"emb": eager.snapshot("emb")},
        -1 if device.type == "cpu" else 0,
    )
    del eager, model
    assert_lookup(marker, marker_value, kind, device)
    del directory
    assert_released(refs)


def test_loaded_directory_owns_registration(tmp_path, kind, device):
    model = Model(kind, device)
    directory = embedding_collection_indexers_from_model(model)
    dump_embedding_collection_indexers(directory, tmp_path)
    refs = storage_refs(model.indexer_.snapshot)
    del model, directory
    assert_released(refs)
    loaded = load_embedding_collection_indexers(tmp_path, device)
    refs = storage_refs(loaded.snapshot("emb"))
    marker_value = loaded.bindings["emb"].marker_value
    marker = torch.tensor([marker_value], device=device)
    assert_lookup(marker, marker_value, kind, device)
    del loaded
    assert_released(refs)


def test_repeated_create_discard_releases_storage(kind, device):
    refs = []
    for _ in range(8):
        model = Model(kind, device)
        directory = embedding_collection_indexers_from_model(model)
        refs.extend(storage_refs(model.indexer_.snapshot))
        del model, directory
        assert_released(refs)
