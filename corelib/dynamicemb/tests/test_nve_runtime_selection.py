# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dual-NVE application image must keep each process on one native ABI."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA lookup")
@pytest.mark.parametrize("version", ["default", "26.05"])
@pytest.mark.parametrize("import_order", ["pynve-first", "dynamicemb-first"])
def test_installed_nve_runtime_selection(tmp_path, version, import_order):
    prefix = Path("/opt/nve") / version / "python"
    if not (prefix / "pynve").is_dir():
        pytest.skip("Requires the dual-NVE application image")
    env = os.environ.copy()
    # No source tree or alternate DynamicEmb prefix: exercise the installed package.
    env["PYTHONPATH"] = str(prefix)
    env.pop("NVE_VERSION", None)
    if version == "26.05":
        env["NVE_VERSION"] = version
    program = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
if sys.argv[2] == "pynve-first":
    import pynve.torch
import torch
from dynamicemb.exportable_tables import (
    create_inference_embedding_collection, InferenceEmbeddingCollectionConfig,
    EmbeddingCollectionIndexerType,
)
torch.cuda.set_device(0)
model = create_inference_embedding_collection(
    [SimpleNamespace(name="a", num_embeddings=4, embedding_dim=64, feature_names=["a"])],
    config=InferenceEmbeddingCollectionConfig(
        indexer_type=EmbeddingCollectionIndexerType.FUSED_IDENTITY,
        nve_layer_type="gpu", indexer_state_sidecar=False,
    ), pooling_mode=-1, device=torch.device("cuda:0"),
)
model.load_from_embedding_table(torch.ones(4, 64))
torch.testing.assert_close(
    model(torch.tensor([0, 1], device="cuda"), torch.tensor([0, 2], device="cuda")).cpu(),
    torch.ones(2, 64),
)
maps = Path("/proc/self/maps").read_text()
other = "default" if sys.argv[1] == "26.05" else "26.05"
assert f"/opt/nve/{other}/" not in maps, f"Loaded the other NVE ABI: {other}"
assert f"/opt/nve/{sys.argv[1]}/python/pynve/" in maps
"""
    subprocess.run(
        [sys.executable, "-c", program, version, import_order],
        cwd=tmp_path,
        env=env,
        check=True,
    )


@pytest.mark.parametrize(
    "generation,kind",
    [
        ((26, 6), "gpu"),
        ((27, 1), "gpu"),
        ((26, 6), "host"),
        ((26, 8), "host"),
        ((26, 9), "host"),
        ((27, 1), "host"),
    ],
)
@pytest.mark.parametrize("tier_bindings", [False, True])
def test_collection_version_dispatch(
    tmp_path, monkeypatch, generation, kind, tier_bindings
):
    # Exercise real construction/export/replay with another reported generation.
    # This checks dispatch, not ABI compatibility with an unreleased NVE binary.
    from dynamicemb.exportable_embedding import _C, nve_runtime
    from dynamicemb.exportable_tables import (
        EmbeddingCollectionIndexerType,
        InferenceEmbeddingCollectionConfig,
        create_inference_embedding_collection,
    )

    if nve_runtime.imported_nve_generation() < (26, 6):
        pytest.skip("Modern NVE API required")
    if kind == "gpu" and not torch.cuda.is_available():
        pytest.skip("CUDA lookup")
    monkeypatch.setattr(nve_runtime, "imported_nve_generation", lambda: generation)
    if not tier_bindings:
        monkeypatch.delattr(_C, "register_nve_tier_bindings")
    device = torch.device("cuda:0" if kind == "gpu" else "cpu")
    if kind == "host" and generation < (26, 9):
        with pytest.raises(ValueError, match="HostLayer requires NVE 26.09"):
            nve_runtime._nve_layer_constructor_kwargs("host")
        return
    model = create_inference_embedding_collection(
        [
            SimpleNamespace(
                name="a", num_embeddings=4, embedding_dim=64, feature_names=["a"]
            )
        ],
        config=InferenceEmbeddingCollectionConfig(
            indexer_type=EmbeddingCollectionIndexerType.IDENTITY,
            nve_layer_type=kind,
        ),
        pooling_mode=-1,
        device=device,
    ).eval()
    expected = torch.arange(256, dtype=torch.float32).reshape(4, 64)
    model.load_from_embedding_table(expected)
    inputs = (torch.tensor([0, 3], device=device), torch.tensor([0, 2], device=device))
    torch.testing.assert_close(model(*inputs).cpu(), expected[[0, 3]])
    assert _C.has_nve_tier_update(model.nve_embedding_.emb_layer) == (
        tier_bindings and kind == "gpu"
    )
    nve_runtime.export_embedding_collection_aot(model, inputs, tmp_path)
    loader, layers, indexers = nve_runtime.load_embedding_collection_aot(
        tmp_path, device
    )
    assert indexers.bindings  # Keep the sidecar registrations alive through replay.
    assert all(_C.has_nve_tier_update(layer.emb_layer) for layer in layers) == (
        tier_bindings and kind == "gpu"
    )
    torch.testing.assert_close(loader.run(list(inputs))[0].cpu(), expected[[0, 3]])


@pytest.mark.parametrize("host", [False, True])
def test_2606_rejects_incomplete_layer_apis(tmp_path, monkeypatch, host):
    from dynamicemb.exportable_embedding import nve_runtime
    from dynamicemb.exportable_embedding.config import (
        EmbeddingCollectionIndexerType,
        InferenceEmbeddingCollectionConfig,
    )

    monkeypatch.setattr(nve_runtime, "imported_nve_generation", lambda: (26, 6))
    message = (
        "HostLayer requires NVE 26.09"
        if host
        else "Pooled collections require NVE 26.07"
    )
    with pytest.raises(ValueError, match=message):
        nve_runtime.create_nve_layer(
            num_embeddings=4,
            embedding_dim=64,
            dtype=torch.float32,
            pooling_mode=-1 if host else 1,
            config=InferenceEmbeddingCollectionConfig(
                indexer_type=EmbeddingCollectionIndexerType.IDENTITY,
                nve_layer_type="host" if host else "gpu",
            ),
            device=torch.device("cpu" if host else "cuda:0"),
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer_type": "HostLayer" if host else "GPULayer",
                        "is_bag": not host,
                        "mode": None if host else "sum",
                    }
                ]
            }
        )
    )
    # Reject before trying to execute an unsupported package, even without weights/PT2.
    with pytest.raises(ValueError, match=message):
        nve_runtime.load_embedding_collection_aot(tmp_path, torch.device("cpu"))
