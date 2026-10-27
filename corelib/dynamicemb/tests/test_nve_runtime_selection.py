# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dual-NVE application image must keep each process on one native ABI."""

import os
import subprocess
import sys
from pathlib import Path

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
