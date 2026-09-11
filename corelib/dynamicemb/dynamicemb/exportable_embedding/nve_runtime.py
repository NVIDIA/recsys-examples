# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from .config import InferenceEmbeddingCollectionConfig
from .indexer_state import (
    EmbeddingCollectionIndexerDirectory,
    dump_embedding_collection_indexers,
    load_embedding_collection_indexers,
)


def imported_nve_generation() -> tuple[int, int]:
    import pynve

    return tuple(int(value) for value in pynve.__version__.split(".")[:2])


def register_nve_export_compat() -> None:
    """Install the only export shim needed by the supported 26.05 package."""
    if imported_nve_generation() != (26, 5):
        return

    import pynve.torch  # noqa: F401

    @torch.library.register_fake(
        "nve_ops::embedding_lookup", allow_override=True
    )
    def _embedding_lookup_fake(
        keys: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        del layer_id
        context = torch.library.get_ctx()
        return keys.new_empty(
            (keys.size(0), context.new_dynamic_size()), dtype=torch.float32
        )


def _nve_layer_constructor_kwargs(
    nve_layer_type: str, *, storage: Optional[Any] = None
) -> dict[str, object]:
    import pynve.torch.nve_layers as nve_layers

    if imported_nve_generation() == (26, 5):
        layer_type = {
            "gpu": nve_layers.CacheType.NoCache,
            "linear_uvm": nve_layers.CacheType.LinearUVM,
            "hierarchical": nve_layers.CacheType.Hierarchical,
        }[nve_layer_type]
        result: dict[str, object] = {"cache_type": layer_type}
        if nve_layer_type == "hierarchical":
            result["remote_interface"] = storage
        return result

    layer_type = {
        "gpu": nve_layers.LayerType.GPULayer,
        "linear_uvm": nve_layers.LayerType.LinearUVM,
        "hierarchical": nve_layers.LayerType.Hierarchical,
    }[nve_layer_type]
    result = {"layer_type": layer_type}
    if nve_layer_type == "hierarchical":
        result["storage"] = storage
    return result


def create_nve_layer(
    *,
    num_embeddings: int,
    embedding_dim: int,
    dtype: torch.dtype,
    pooling_mode: int,
    config: InferenceEmbeddingCollectionConfig,
    device: torch.device,
) -> torch.nn.Module:
    import pynve.torch.nve_layers as nve_layers

    if pooling_mode == -1:
        layer_class = nve_layers.NVEmbedding
        pooling_args: dict[str, Any] = {}
    else:
        layer_class = nve_layers.NVEmbeddingBag
        pooling_args = {"mode": "sum" if pooling_mode == 1 else "mean"}

    kwargs: dict[str, Any] = {
        "num_embeddings": num_embeddings,
        "embedding_size": embedding_dim,
        "data_type": dtype,
        "optimize_for_training": False,
        "device": device,
        **pooling_args,
        **_nve_layer_constructor_kwargs(
            config.nve_layer_type, storage=config.parameter_server
        ),
    }
    if config.nve_layer_type != "gpu":
        kwargs["gpu_cache_size"] = config.gpu_cache_size
    if config.nve_layer_type == "hierarchical":
        kwargs["host_cache_size"] = config.host_cache_size
    return layer_class(**kwargs)


@torch.no_grad()
def insert_parameter_server(
    parameter_server: Any,
    storage_keys: torch.Tensor,
    embeddings: torch.Tensor,
    value_dtype: torch.dtype,
) -> None:
    if storage_keys.numel() == 0:
        return
    parameter_server.insert(
        storage_keys.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        embeddings.detach().to(device="cpu", dtype=value_dtype).contiguous(),
    )


def parameter_server_backend(parameter_server: Any) -> str:
    config = parameter_server.export_config()
    text = " ".join(
        (
            str(config.get("plugin_name", "")),
            str(config.get("factory_config", "")),
        )
    ).lower()
    return "redis" if "redis" in text else "nvhashmap"


def dump_nvhashmap_storage_sidecars(
    model: torch.nn.Module, package_dir: str | os.PathLike[str]
) -> dict[str, tuple[str, str]]:
    root = Path(package_dir).resolve() / "nve_ps_data"
    paths: dict[str, tuple[str, str]] = {}
    sidecar_number = 0
    for module_path, module in model.named_modules():
        parameter_server = getattr(module, "parameter_server_", None)
        batches = getattr(module, "_hierarchical_population_batches", None)
        if (
            parameter_server is None
            or batches is None
            or parameter_server_backend(parameter_server) != "nvhashmap"
        ):
            continue
        root.mkdir(parents=True, exist_ok=True)
        keys_path = root / f"storage_{sidecar_number}_keys.dyn"
        values_path = root / f"storage_{sidecar_number}_values.dyn"
        with keys_path.open("wb") as key_stream, values_path.open("wb") as value_stream:
            for keys, values in batches:
                key_stream.write(keys.contiguous().numpy().tobytes())
                value_stream.write(values.contiguous().numpy().tobytes())
        nve_path = f"{module_path}.nve_embedding_" if module_path else "nve_embedding_"
        paths[nve_path] = (str(keys_path), str(values_path))
        batches.clear()
        sidecar_number += 1
    return paths


def export_embedding_collection_aot(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    package_dir: str | os.PathLike[str],
    *,
    dynamic_shapes: Any = None,
    inductor_configs: Optional[dict[str, Any]] = None,
) -> EmbeddingCollectionIndexerDirectory:
    from pynve.torch.nve_export import export_aot

    package_dir = os.fspath(Path(package_dir).resolve())
    indexers = EmbeddingCollectionIndexerDirectory.from_model(model)
    dump_embedding_collection_indexers(indexers, package_dir)
    ps_data_paths = dump_nvhashmap_storage_sidecars(model, package_dir)
    configs = {"aot_inductor.use_runtime_constant_folding": True}
    if inductor_configs:
        configs.update(inductor_configs)
    export_aot(
        model,
        example_inputs,
        package_dir,
        dynamic_shapes=dynamic_shapes,
        inductor_configs=configs,
        ps_data_paths=ps_data_paths,
    )
    return indexers


def load_embedding_collection_aot(
    package_dir: str | os.PathLike[str], device: torch.device
) -> tuple[Any, list[Any], EmbeddingCollectionIndexerDirectory]:
    package_dir = os.fspath(Path(package_dir).resolve())
    if imported_nve_generation() == (26, 5):
        from pynve.torch.nve_export import load_nve_layers
        from torch._C._aoti import AOTIModelPackageLoader

        with torch.cuda.device(device):
            layers = load_nve_layers(package_dir)
        loader = AOTIModelPackageLoader(
            os.path.join(package_dir, "model.pt2"),
            "model",
            False,
            1,
            device.index if device.index is not None else torch.cuda.current_device(),
        )
    else:
        from pynve.torch.nve_export import load_aot

        loader, layers = load_aot(package_dir, device=device)

    indexers = load_embedding_collection_indexers(package_dir, device)
    indexers.bind_aoti(loader)
    return loader, layers, indexers
