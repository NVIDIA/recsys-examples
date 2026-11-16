# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
)
from modules.exportable_embedding import checkpoint_table_capacities


@pytest.mark.parametrize("name", ["user_id", "scope/user_id", "a/b/user_id"])
def test_capacity_counts_scoped_checkpoint_shards(tmp_path, name):
    directory = (
        tmp_path
        / "dynamicemb_module"
        / "model._embedding_collection._model_parallel_embedding_collection"
    )
    for rank, count in enumerate([3, 2]):
        path = directory / f"{name}_emb_keys.rank_{rank}.world_size_2"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(torch.arange(count, dtype=torch.int64).numpy().tobytes())
    config = InferenceEmbeddingCollectionConfig(
        indexer_type=EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
        nve_layer_type="linear_uvm",
    )
    capacities = checkpoint_table_capacities(str(tmp_path), {name: config}, {name: 1})
    assert capacities == {name: 5}
