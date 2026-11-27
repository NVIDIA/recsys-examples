# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An inference collection loads a ``hash_bucket`` checkpoint, however many shards wrote it."""

import json
from types import SimpleNamespace

import pytest
import torch
from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)
from dynamicemb.key_ownership import owning_shard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA loading")

DIM = 8
BUCKET_CAPACITY = 128
NUM_KEYS = 3000
# The table the checkpoint was trained with: its hash space, independent of the
# shard count that wrote it.
GLOBAL_CAPACITY = 8192


def write_checkpoint(path, name, keys, values, num_shards, with_capacities=True):
    """What a ``num_shards``-way ``hash_bucket`` dump holds: each shard its own keys."""
    owner = owning_shard(
        keys,
        num_shards,
        "hash_bucket",
        global_capacity=GLOBAL_CAPACITY,
        bucket_capacity=BUCKET_CAPACITY,
    )
    for shard in range(num_shards):
        mine = owner == shard
        suffix = f"rank_{shard}.world_size_{num_shards}"
        (path / f"{name}_emb_keys.{suffix}").write_bytes(
            keys[mine].numpy().tobytes()
        )
        (path / f"{name}_emb_values.{suffix}").write_bytes(
            values[mine].numpy().tobytes()
        )
    meta = {
        "embedding_dtype": "float32",
        "embedding_dim": DIM,
        "dist_type": "hash_bucket",
    }
    if with_capacities:
        meta["global_capacity"] = GLOBAL_CAPACITY
        meta["bucket_capacity"] = BUCKET_CAPACITY
    (path / f"{name}_opt_args.json").write_text(json.dumps(meta))
    return [int((owner == s).sum()) for s in range(num_shards)]


def collection():
    return create_inference_embedding_collection(
        [
            SimpleNamespace(
                name="t",
                num_embeddings=2 * NUM_KEYS,
                embedding_dim=DIM,
                feature_names=["t"],
            )
        ],
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
            nve_layer_type="gpu",
        ),
        pooling_mode=-1,
        output_dtype=torch.float32,
        device=torch.device("cuda:0"),
    ).eval()


@pytest.mark.parametrize("num_shards", [1, 2, 8])
def test_hash_bucket_checkpoint_loads_whole_whatever_the_shard_count(
    tmp_path, num_shards
):
    gen = torch.Generator().manual_seed(7)
    keys = torch.randint(0, 1 << 40, (2 * NUM_KEYS,), generator=gen).unique()[:NUM_KEYS]
    keys = keys[torch.randperm(keys.numel(), generator=gen)]
    assert keys.numel() == NUM_KEYS
    values = torch.randn(NUM_KEYS, DIM, generator=gen)
    held = write_checkpoint(tmp_path, "t", keys, values, num_shards)
    if num_shards > 1:
        # The files really are split: no one shard holds everything.
        assert max(held) < NUM_KEYS and min(held) > 0

    model = collection()
    model.load_from_dynamicemb_file(str(tmp_path))

    # One table, one sample: the offsets name its single bag of all the keys, and a
    # sequence collection returns one row per key.
    offsets = torch.tensor([0, NUM_KEYS], device="cuda")
    out = model(keys.cuda(), offsets)
    torch.testing.assert_close(out.cpu(), values)
