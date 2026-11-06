# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed on a mismatched GB300 build or runtime; print observed versions."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
from pathlib import Path


def version_tuple(value):
    return tuple(
        int(v) for v in re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", value).groups(default="0")
    )


def module_path(name):
    spec = importlib.util.find_spec(name)
    assert (
        spec and spec.origin and Path(spec.origin).is_file()
    ), f"Missing artifact: {name}"
    return spec.origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Check metadata, artifacts and compiler without loading GPU-dependent modules",
    )
    args = parser.parse_args()

    # These TorchRec dependencies are CPU-safe; importing them catches missing
    # Python dependencies before launching a GPU job. GPU libraries load below.
    import pyre_extensions
    import tensordict
    import torch
    import torchmetrics
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = f"{CUDA_HOME or '/usr/local/cuda'}/bin/nvcc"
    compiler = subprocess.check_output([nvcc, "--version"], text=True)
    compiler_version = re.search(r"V(\d+\.\d+\.\d+)", compiler).group(1)
    gpu_codes = subprocess.check_output([nvcc, "--list-gpu-code"], text=True).split()
    paths = {
        name: module_path(name)
        for name in ("fbgemm_gpu", "torchrec", "dynamicemb", "dynamicemb_extensions")
    }
    report = {
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": compiler_version,
        "cudnn": torch.backends.cudnn.version(),
        "fbgemm_gpu": importlib.metadata.version("fbgemm-gpu"),
        "fbgemm_gpu_module": paths["fbgemm_gpu"],
        "torchrec": importlib.metadata.version("torchrec"),
        "tensordict": importlib.metadata.version("tensordict"),
        "tensordict_module": tensordict.__file__,
        "pyvers": importlib.metadata.version("pyvers"),
        "cloudpickle": importlib.metadata.version("cloudpickle"),
        "pyre_extensions": importlib.metadata.version("pyre-extensions"),
        "pyre_extensions_module": pyre_extensions.__file__,
        "torchmetrics": importlib.metadata.version("torchmetrics"),
        "torchmetrics_module": torchmetrics.__file__,
        "dynamicemb_version": importlib.metadata.version("dynamicemb"),
        "dynamicemb": paths["dynamicemb"],
        "extension": paths["dynamicemb_extensions"],
        "scope": (
            "build/metadata/artifacts"
            if args.build_only
            else "GB300 single-device smoke"
        ),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    assert version_tuple(compiler_version) >= (
        13,
        0,
        2,
    ), "CUDA Toolkit >= 13.0.2 required"
    assert "sm_103" in gpu_codes, "The compiler must support GB300 (SM103)"
    assert torch.version.cuda and version_tuple(torch.version.cuda) >= (13, 0, 0)
    assert version_tuple(torch.__version__) >= (2, 9, 0)
    assert report["cudnn"] and report["cudnn"] >= 91801, "cuDNN >= 9.18.1 required"
    if args.build_only:
        return

    assert platform.machine() == "aarch64", "GB300 NVL72 uses the Grace ARM CPU"
    assert torch.cuda.is_available(), "A live CUDA device is required"
    assert torch.cuda.get_device_capability() == (10, 3), "GB300 is SM103, not SM100"
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()[0]
    assert version_tuple(driver) >= (
        580,
        167,
        8,
    ), "Driver is older than the documented GB300 stack"

    # Real library loading belongs to runtime, where the container receives
    # the host driver's NVML/CUDA libraries through the NVIDIA runtime.
    import dynamicemb
    import dynamicemb_extensions
    import fbgemm_gpu
    import torchrec

    for name, module in (
        ("fbgemm_gpu", fbgemm_gpu),
        ("torchrec", torchrec),
        ("dynamicemb", dynamicemb),
        ("dynamicemb_extensions", dynamicemb_extensions),
    ):
        assert module.__file__ == paths[name], f"Unexpected runtime module: {name}"

    # Run a real FBGEMM CUDA operator, not just a shared-library import.
    lengths = torch.tensor([1, 2, 0], dtype=torch.int64, device="cuda")
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    torch.testing.assert_close(offsets, lengths.new_tensor([0, 1, 3, 3]))

    from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
    from dynamicemb_extensions import InsertResult, ScorePolicy

    table = get_scored_table(
        capacity=[128],
        bucket_capacity=128,
        score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
    )
    keys = torch.arange(1, 9, dtype=torch.int64, device="cuda")
    tids = torch.zeros_like(keys)
    results = torch.empty(keys.numel(), dtype=table.result_type, device="cuda")
    scores = torch.ones(keys.numel(), dtype=torch.uint64, device="cuda")
    slots = table.insert(keys, tids, ScoreArg(name="score", value=scores), results)
    observed, found, restored_slots = table.lookup(
        keys, tids, ScoreArg(name="score", policy=ScorePolicy.CONST)
    )
    assert (results == InsertResult.INSERT.value).all().item()
    assert found.all().item()
    # Lookup exposes the uint64 score bit patterns through an int64 tensor.
    torch.testing.assert_close(observed, scores.view(torch.int64))
    torch.testing.assert_close(restored_slots, slots)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "cc": [10, 3],
                "driver": driver,
                "runtime_imports": "passed",
                "fbgemm_cuda": "passed",
                "dynamicemb_insert_lookup": "passed",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
