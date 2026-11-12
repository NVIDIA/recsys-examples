# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from dynamicemb.exportable_embedding.indexer import LinearHashMapIndexer
from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA loading")
def test_checkpoint_overflow_does_not_poison_cuda_context(tmp_path):
    model = create_inference_embedding_collection(
        [
            SimpleNamespace(
                name="a", num_embeddings=4, embedding_dim=8, feature_names=["a"]
            )
        ],
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
            nve_layer_type="gpu",
        ),
        pooling_mode=-1,
        device=torch.device("cuda:0"),
    )
    for suffix, values in (
        ("keys", torch.arange(5, dtype=torch.int64)),
        ("values", torch.ones(5, 8)),
    ):
        (tmp_path / f"a_emb_{suffix}.rank_0.world_size_1").write_bytes(
            values.numpy().tobytes()
        )
    with pytest.raises(ValueError, match="capacity exceeded"):
        model.load_from_dynamicemb_file(str(tmp_path))
    # Previously the write indexed beyond NVE storage and poisoned the context.
    torch.testing.assert_close(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA loading")
@pytest.mark.parametrize(
    "batches",
    [
        [([10, 10, 20, 10], [0, 0, 0, 1])],
        [([10, 10], [0, 0]), ([20, 10], [0, 1])],
    ],
    ids=["one_batch", "later_batch"],
)
def test_duplicate_keys_use_one_storage_row(batches):
    indexer = LinearHashMapIndexer(
        [2, 1], bucket_capacity=16, device=torch.device("cuda:0")
    )
    for keys, tables in batches:
        indexer.build_index(
            torch.tensor(keys, device="cuda"), torch.tensor(tables, device="cuda")
        )
    indexer.finish_build()
    rows = indexer(
        torch.tensor([10, 10, 20, 10], device="cuda"),
        torch.tensor([0, 0, 0, 1], device="cuda"),
    ).cpu()
    assert rows[0] == rows[1]
    assert rows.unique().numel() == 3
    assert indexer.snapshot.next_fused_key == indexer.nve_num_embeddings
