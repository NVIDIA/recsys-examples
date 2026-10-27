# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local tier bindings must work against unmodified NVE 26.10."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from pynve import nve
from pynve.torch import nve_layers
from pynve.torch.nve_ps import NVEParameterServer

from dynamicemb.exportable_embedding import _C, export_embedding_collection_aot
from dynamicemb.exportable_embedding.config import (
    EmbeddingCollectionIndexerType as Indexer,
    InferenceEmbeddingCollectionConfig,
)
from dynamicemb.exportable_tables import create_inference_embedding_collection

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA tier bindings"
)


def collection(kind, dtype=torch.float32, pooling=-1, ps=None, host_cache=0):
    torch.cuda.set_device(0)
    cfg = InferenceEmbeddingCollectionConfig(
        indexer_type=Indexer.IDENTITY,
        nve_layer_type=kind,
        gpu_cache_size=(1 << 20) if kind != "gpu" else None,
        host_cache_size=host_cache,
        parameter_server=ps,
    )
    return create_inference_embedding_collection(
        [
            SimpleNamespace(
                name="a", num_embeddings=4, embedding_dim=64, feature_names=["a"]
            )
        ],
        config=cfg,
        pooling_mode=pooling,
        output_dtype=dtype,
        device=torch.device("cuda:0"),
    ).eval()


@pytest.mark.parametrize("kind", ["gpu", "linear_uvm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pooling", [-1, 1, 2])
def test_dense_tier_binding(kind, dtype, pooling):
    model = collection(kind, dtype, pooling)
    model.load_from_embedding_table(torch.ones((4, 64), dtype=dtype))
    keys = torch.tensor([0, 1], device="cuda")
    args = (keys, torch.tensor([0, 2], device="cuda"))
    if pooling != -1:
        args += (torch.tensor([0, 2], device="cuda"),)
    model(*args).cpu()  # Populate the cache before updating the resident key.
    binding = model.nve_embedding_.emb_layer
    assert _C.has_nve_tier_update(binding)
    cpu_key = torch.tensor([1], dtype=torch.int64)
    value = torch.full((1, 64), 7, dtype=dtype)
    binding.update_tier(
        1,
        cpu_key.data_ptr(),
        value.data_ptr(),
        0,
        torch.cuda.current_stream().cuda_stream,
    )
    expected = torch.tensor([1, 7], dtype=dtype).view(2, 1).expand(2, 64)
    if pooling != -1:
        expected = expected.sum(0, keepdim=True) / (2 if pooling == 2 else 1)
    torch.testing.assert_close(model(*args).cpu(), expected)
    with pytest.raises(Exception, match="Invalid update tier"):
        binding.update_tier(1, cpu_key.data_ptr(), value.data_ptr(), 99, 0)


@pytest.mark.parametrize("host_cache", [0, 1 << 20])
def test_hierarchical_updates_only_selected_cache(host_cache):
    ps = NVEParameterServer(4, 64, torch.float32, ps_type=nve.NVHashMap)
    model = collection("hierarchical", ps=ps, host_cache=host_cache)
    model.load_from_embedding_table(torch.ones(4, 64))
    keys = torch.tensor([1], device="cuda")
    args = (keys, torch.tensor([0, 1], device="cuda"))
    binding = model.nve_embedding_.emb_layer
    cpu_key = torch.tensor([1], dtype=torch.int64)
    seed = torch.full((1, 64), 3.0)
    stream = torch.cuda.current_stream().cuda_stream
    # Populate resident keys explicitly: one lookup does not guarantee that
    # asynchronous/heuristic cache admission has finished. Distinct PS/cache
    # values make the tier source observable.
    if host_cache:
        binding.insert(1, cpu_key.data_ptr(), seed.data_ptr(), 1, stream)
    binding.insert(1, cpu_key.data_ptr(), seed.data_ptr(), 0, stream)
    torch.testing.assert_close(model(*args).cpu(), seed)
    gpu_value = torch.full((1, 64), 9.0)
    binding.update_tier(1, cpu_key.data_ptr(), gpu_value.data_ptr(), 0, stream)
    torch.testing.assert_close(model(*args).cpu(), gpu_value)
    if host_cache:
        host_value = torch.full((1, 64), 5.0)
        binding.update_tier(1, cpu_key.data_ptr(), host_value.data_ptr(), 1, stream)
        torch.testing.assert_close(model(*args).cpu(), gpu_value)
        binding.erase(1, cpu_key.data_ptr(), 0, stream)
        torch.testing.assert_close(model(*args).cpu(), host_value)
    # A fresh binding has cold caches and reads the same backing PS. Updating a
    # selected tier must not have written through to that PS.
    independent = nve_layers.NVEmbedding(
        4,
        64,
        torch.float32,
        nve_layers.LayerType.Hierarchical,
        storage=ps,
        gpu_cache_size=1 << 20,
        optimize_for_training=False,
        device=torch.device("cuda:0"),
    )
    assert not _C.has_nve_tier_update(independent.emb_layer)
    torch.testing.assert_close(independent(keys).cpu(), torch.ones(1, 64))


@pytest.mark.parametrize("kind", ["gpu", "linear_uvm"])
def test_fresh_process_aoti_retains_tier_binding(tmp_path, kind):
    model = collection(kind)
    model.load_from_embedding_table(torch.ones(4, 64))
    inputs = (torch.tensor([0, 1], device="cuda"), torch.tensor([0, 2], device="cuda"))
    package = tmp_path / kind
    export_embedding_collection_aot(model, inputs, package)
    # A separate interpreter catches registrations/owners accidentally retained
    # only by the exporter process. The subscriber must share the lookup binding.
    program = r"""
import json,sys,torch
from dynamicemb.exportable_embedding import _C,load_embedding_collection_aot,EmbeddingCollectionUpdateSubscriber
torch.cuda.set_device(0)
path=sys.argv[1]
loader,layers,indexers=load_embedding_collection_aot(path,torch.device("cuda:0"))
assert all(_C.has_nve_tier_update(layer.emb_layer) for layer in layers)
keys=torch.tensor([0,1],device="cuda")
offsets=torch.tensor([0,2],device="cuda")
torch.testing.assert_close(loader.run([keys,offsets])[0].cpu(),torch.ones(2,64))
manifest=json.load(open(path+"/embedding_collection_indexers/manifest.json"))
collection_id=next(iter(manifest["collections"]))
subscriber=EmbeddingCollectionUpdateSubscriber(path,indexers,{x._export_module_path:x.emb_layer for x in layers},0)
message=dict(collection_id=collection_id,snapshot_id=1,cache_update_keys=[1],cache_update_values=[7.]*64,indexer_snapshot_path=None)
subscriber.apply_incremental_load(json.dumps(message),torch.cuda.current_stream().cuda_stream)
expected=torch.tensor([1.,7.]).view(2,1).expand(2,64)
torch.testing.assert_close(loader.run([keys,offsets])[0].cpu(),expected)
print("fresh-process tier update passed")
"""
    subprocess.run(
        [sys.executable, "-c", program, str(package)], check=True, env=os.environ.copy()
    )


def test_inference_without_update_extension(monkeypatch):
    monkeypatch.delattr(_C, "register_nve_tier_bindings")
    model = collection("gpu")
    model.load_from_embedding_table(torch.ones(4, 64))
    assert not _C.has_nve_tier_update(model.nve_embedding_.emb_layer)
    keys = torch.tensor([1], device="cuda")
    torch.testing.assert_close(
        model(keys, torch.tensor([0, 1], device="cuda")).cpu(), torch.ones(1, 64)
    )
