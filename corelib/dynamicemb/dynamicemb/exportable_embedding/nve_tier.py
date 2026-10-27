# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference constructors for tier-capable bindings on NVE 26.10.

The stock NVE constructors/loaders hardcode their binding types. Keep the small
constructor adaptation here; forward, export discovery and serialization remain
NVE's. Host uses its stock constructor and its separate concurrency patch.
"""

import builtins

import torch
from pynve import nve
from pynve.torch import nve_layers, nve_ps
from pynve.torch.nve_tensors import CachedTable

from . import _C

_C.register_nve_tier_bindings()


def _init_layer(
    self,
    num_embeddings,
    embedding_size,
    data_type,
    layer_type,
    *,
    device,
    gpu_cache_size=0,
    host_cache_size=0,
    storage=None,
    optimize_for_training=False,
):
    torch.nn.Module.__init__(self)
    if optimize_for_training or device.type != "cuda":
        raise ValueError("Tier bindings require CUDA inference")
    if data_type not in (torch.float16, torch.float32):
        raise ValueError("Tier bindings require FP16 or FP32 values")
    self.device = device
    self.device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    self.id = builtins.id(self)
    self.num_embeddings, self.embedding_size = num_embeddings, embedding_size
    self.data_type, self.layer_type = data_type, layer_type
    self.layer_data_type = nve_layers.torch_type_to_nve_type(data_type)
    self.dtype_tag = int(self.layer_data_type)
    self.gpu_cache_size, self.host_cache_size = gpu_cache_size, host_cache_size
    self.optimize_for_training = False
    self.config = nve.EmbedLayerConfig()
    common = (embedding_size, num_embeddings, self.layer_data_type)
    if layer_type == nve_layers.LayerType.GPULayer:
        values = torch.empty(
            num_embeddings, embedding_size, dtype=data_type, device=device
        )
        self.storage = nve.UserMemBlock(values.data_ptr(), values.nbytes)
        self.memblock_type = self.storage.get_type()
        self.emb_layer = _C.TierGPUEmbedding(
            *common, self.storage, self.device_index, self.config
        )
        self.table_ids = {0: "gpu"}
    elif layer_type == nve_layers.LayerType.LinearUVM:
        if not gpu_cache_size:
            raise ValueError("LinearUVM requires a GPU cache")
        self.storage = nve.ManagedMemBlock(*common, [self.device_index])
        self.memblock_type = self.storage.get_type()
        self.emb_layer = _C.TierUVMEmbedding(
            *common, self.storage, gpu_cache_size, False, self.device_index, self.config
        )
        values = torch.utils.dlpack.from_dlpack(
            nve.get_dl_tensor(self.emb_layer, self.layer_data_type)
        )
        self.table_ids = {0: "gpu_cache"}
    elif layer_type == nve_layers.LayerType.Hierarchical:
        if not gpu_cache_size or not isinstance(
            storage, (nve.Table, nve_ps.NVEParameterServer)
        ):
            raise ValueError("Hierarchical requires a GPU cache and parameter server")
        self.storage = storage
        remote = storage
        if isinstance(storage, nve_ps.NVEParameterServer):
            if storage.data_type != data_type:
                raise ValueError("Parameter-server dtype mismatch")
            remote = storage.parameter_server
        self.config.default_embedding = [0] * (
            embedding_size * torch.empty(0, dtype=data_type).element_size()
        )
        self.emb_layer = _C.TierHierarchicalEmbedding(
            embedding_size,
            self.layer_data_type,
            gpu_cache_size,
            host_cache_size,
            remote,
            num_embeddings,
            False,
            self.device_index,
            self.config,
        )
        values = torch.sparse_coo_tensor(
            size=(num_embeddings, embedding_size), dtype=data_type, device=device
        )
        self.table_ids = {
            0: "gpu_cache",
            **(
                {1: "host_cache", 2: "remote_ps"}
                if host_cache_size
                else {1: "remote_ps"}
            ),
        }
    else:
        raise ValueError(f"Unsupported tier binding: {layer_type}")
    self.weight = CachedTable(values, cache=self.emb_layer)
    self.register_buffer(
        "marker_tensor",
        torch.tensor([self.id], dtype=torch.int64, device=device),
        persistent=False,
    )
    nve.register_for_torch(self.marker_tensor.data_ptr(), self.emb_layer)


class TierNVEmbedding(nve_layers.NVEmbedding):
    def __init__(self, **kwargs):
        _init_layer(self, **kwargs)


class TierNVEmbeddingBag(nve_layers.NVEmbeddingBag):
    def __init__(self, *, mode, **kwargs):
        if mode not in ("sum", "mean"):
            raise ValueError("Collection pooling requires sum or mean")
        _init_layer(self, **kwargs)
        self.mode = mode


class _RuntimeLayers(list):
    def __init__(self, directory):
        super().__init__(directory.layers())
        # Entries contain bindings/markers; directory owns their registry entries.
        self.directory = directory


def load_tier_aot(package_dir, device):
    from torch._C._aoti import AOTIModelPackageLoader

    index = (
        -1
        if device.type == "cpu"
        else (device.index if device.index is not None else torch.cuda.current_device())
    )
    directory = _C.NveRuntimeDirectory(package_dir, index)
    layers = _RuntimeLayers(directory)
    loader = AOTIModelPackageLoader(
        package_dir + "/model.pt2", "model", False, 1, index
    )
    markers = directory.markers()
    missing = markers.keys() - set(loader.get_constant_fqns())
    if missing:
        raise RuntimeError(
            f"NVE markers are not updatable AOTI constants: {sorted(missing)}"
        )
    for inactive in (False, True):
        loader.load_constants(markers, inactive, False, True)
    return loader, layers
