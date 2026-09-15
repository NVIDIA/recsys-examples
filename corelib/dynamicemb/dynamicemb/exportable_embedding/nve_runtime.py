# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from .config import InferenceEmbeddingCollectionConfig
from .indexer_directory import (
    dump_embedding_collection_indexers,
    embedding_collection_indexers_from_model,
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


def export_embedding_collection_aot(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    package_dir: str | os.PathLike[str],
    *,
    dynamic_shapes: Any = None,
    inductor_configs: Optional[dict[str, Any]] = None,
) -> Any:
    from pynve.torch.nve_export import export_aot

    package_dir = os.fspath(Path(package_dir).resolve())
    indexers = embedding_collection_indexers_from_model(model)
    dump_embedding_collection_indexers(indexers, package_dir)
    configs = {"aot_inductor.use_runtime_constant_folding": True}
    if inductor_configs:
        configs.update(inductor_configs)
    export_aot(
        model,
        example_inputs,
        package_dir,
        dynamic_shapes=dynamic_shapes,
        inductor_configs=configs,
    )
    return indexers


def load_embedding_collection_aot(
    package_dir: str | os.PathLike[str], device: torch.device
) -> tuple[Any, list[Any], Any]:
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
