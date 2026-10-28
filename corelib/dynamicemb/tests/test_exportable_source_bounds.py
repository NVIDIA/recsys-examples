# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from pynve import nve
from pynve.torch.nve_ps import NVEParameterServer

from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType as Indexer,
    InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA source loading"
)
LAYERS = ("gpu", "linear_uvm", "hierarchical")
DENSE_INDEXERS = (Indexer.IDENTITY, Indexer.FUSED_IDENTITY)
DIM = 64


def collection(layer, indexer, dtype=torch.float32, capacities=None):
    torch.cuda.set_device(0)
    capacities = capacities or ([2] if indexer == Indexer.IDENTITY else [2, 2])
    tables = [
        SimpleNamespace(
            name=f"t{i}",
            num_embeddings=size,
            embedding_dim=DIM,
            feature_names=[f"t{i}"],
        )
        for i, size in enumerate(capacities)
    ]
    ps = (
        NVEParameterServer(0, DIM, dtype, ps_type=nve.NVHashMap)
        if layer == "hierarchical"
        else None
    )
    return create_inference_embedding_collection(
        tables,
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=indexer,
            nve_layer_type=layer,
            indexer_state_sidecar=False,
            gpu_cache_size=(1 << 20) if layer != "gpu" else None,
            parameter_server=ps,
        ),
        pooling_mode=-1,
        output_dtype=dtype,
        device=torch.device("cuda:0"),
    ).eval()


def dump_table(path, name, keys, values):
    (path / f"{name}_emb_keys.rank_0.world_size_1").write_bytes(
        torch.tensor(keys, dtype=torch.int64).numpy().tobytes()
    )
    (path / f"{name}_emb_values.rank_0.world_size_1").write_bytes(
        values.contiguous().view(torch.uint8).numpy().tobytes()
    )
    (path / f"{name}_opt_args.json").write_text(
        json.dumps(
            {
                "embedding_dtype": str(values.dtype).split(".")[-1],
                "embedding_dim": DIM,
                "dist_type": "hash_roundrobin",
            }
        )
    )


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("indexer", DENSE_INDEXERS)
@pytest.mark.parametrize("bad_id", [-1, 2])
def test_dense_checkpoint_rejects_invalid_source_ids(tmp_path, layer, indexer, bad_id):
    model = collection(layer, indexer)
    # For fused capacities [2, 2], t0 ID 2 would overwrite t1's miss row.
    dump_table(tmp_path, "t0", [0, bad_id], torch.full((2, DIM), 9.0))
    with pytest.raises(ValueError, match="table-local ID range"):
        model.load_from_dynamicemb_file(str(tmp_path))
    if layer != "hierarchical":
        # The offending batch must not partially write any valid or miss rows.
        torch.testing.assert_close(
            model.nve_embedding_.weight.cpu(),
            torch.zeros_like(model.nve_embedding_.weight, device="cpu"),
        )
    # Rejection must not leave a device assertion or poison the CUDA context.
    torch.testing.assert_close(torch.ones(1, device="cuda").cpu(), torch.ones(1))


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("indexer", DENSE_INDEXERS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("source", ["checkpoint", "cpu_weights", "cuda_weights"])
def test_dense_sources_preserve_rows(tmp_path, layer, indexer, dtype, source):
    model = collection(layer, indexer, dtype)
    count = len(model.table_names_)
    weights = (
        torch.arange(count * 2 * DIM, dtype=torch.float32).reshape(-1, DIM).to(dtype)
    )
    if source == "checkpoint":
        for i, name in enumerate(model.table_names_):
            # File keys are table-local and may arrive out of order.
            dump_table(tmp_path, name, [1, 0], weights[2 * i : 2 * i + 2].flip(0))
        model.load_from_dynamicemb_file(str(tmp_path))
    else:
        model.load_from_embedding_table(
            weights.cuda() if source == "cuda_weights" else weights
        )
    keys = torch.tensor([0, 1] * count, device="cuda")
    offsets = torch.arange(count + 1, device="cuda") * 2
    torch.testing.assert_close(model(keys, offsets).cpu(), weights)
    if indexer == Indexer.FUSED_IDENTITY:
        missing = torch.tensor([-1, 2] * count, device="cuda")
        torch.testing.assert_close(
            model(missing, offsets).cpu(), torch.zeros_like(weights)
        )


@pytest.mark.parametrize("layer", LAYERS)
def test_dense_source_allows_empty_table(layer):
    model = collection(layer, Indexer.FUSED_IDENTITY, capacities=[0, 2])
    values = torch.arange(2 * DIM, dtype=torch.float32).reshape(2, DIM)
    model.load_from_embedding_table(values)
    output = model(
        torch.tensor([0, 0, 1], device="cuda"), torch.tensor([0, 1, 3], device="cuda")
    )
    torch.testing.assert_close(output.cpu(), torch.cat((torch.zeros(1, DIM), values)))
