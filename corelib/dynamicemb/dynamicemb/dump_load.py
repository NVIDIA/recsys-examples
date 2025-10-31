# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from torch import nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.model_parallel import get_unwrapped_module


def find_sharded_modules(
    module: nn.Module, path: str = ""
) -> List[Tuple[str, str, nn.Module]]:
    sharded_modules: List[Tuple[str, str, nn.Module]] = []
    stack: deque[Tuple[nn.Module, str, str]] = deque(
        [(module, path + "model", "model")]
    )

    while stack:
        current_module, current_path, current_name = stack.pop()
        current_module = get_unwrapped_module(current_module)
        if isinstance(
            current_module, (ShardedEmbeddingCollection, ShardedEmbeddingBagCollection)
        ):
            sharded_modules.append((current_path, current_name, current_module))
        else:
            for name, child in current_module.named_children():
                child_path = current_path + ("." if current_path else "") + name
                stack.append((child, child_path, name))
    return sharded_modules


def check_emb_collection_modules(module: nn.Module, ret_list: List[nn.Module]):
    if isinstance(module, BatchedDynamicEmbeddingTablesV2):
        ret_list.append(module)
        return ret_list

    if isinstance(module, nn.Module):
        if hasattr(module, "_emb_module"):
            check_emb_collection_modules(module._emb_module, ret_list)

        if hasattr(module, "_emb_modules"):
            check_emb_collection_modules(module._emb_modules, ret_list)

        if hasattr(module, "_lookups"):
            tmp_module_list = module._lookups
            for tmp_emb_module in tmp_module_list:
                check_emb_collection_modules(tmp_emb_module, ret_list)

    if isinstance(module, nn.ModuleList):
        for i in range(len(module)):
            tmp_emb_module = module[i]

            if isinstance(tmp_emb_module, nn.Module):
                check_emb_collection_modules(tmp_emb_module, ret_list)
            else:
                continue


def get_dynamic_emb_module(model: nn.Module) -> List[nn.Module]:
    dynamic_emb_module_list: List[nn.Module] = []
    check_emb_collection_modules(model, dynamic_emb_module_list)
    return dynamic_emb_module_list


# TODO: Now only support Row-Wise sharding, will support TW/TWRW in future.
# TODO: Currently, the dump and load functions enforce dumping and loading of all parameters of the optimizer.
#      This mechanism prevents users from controlling certain parameters of the optimizer.
#      In order to allow users to set optimizer parameters more flexibly in the future,
#      we need to add functionality for dumping and loading specific args,
#      allowing for more flexible configuration of optimizer arguments
def DynamicEmbDump(
    path: str,
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: Optional[bool] = False,
    pg: dist.ProcessGroup = dist.group.WORLD,
    allow_overwrite: bool = False,
) -> None:
    """
    Dump the distributed weights and corresponding optimizer states of dynamic embedding tables from the model to the filesystem.
    The weights of the dynamic embedding table will be stored in each EmbeddingCollection or EmbeddingBagCollection folder.
    The name of the collection is the path of the torch module within the model, with the input module defined as str of model.

    Each dynamic embedding table will be stored as a key binary file and a value binary file, where the dtype of the key is int64_t,
    and the dtype of the value is float. Each optimizer state is also treated as a dynamic embedding table.

    Parameters
    ----------
    path : str
        The main folder for weight files.
    model : nn.Module
        The model containing dynamic embedding tables.
    table_names : Optional[Dict[str, List[str]]], optional
        A dictionary specifying which embedding collection and which table to dump. The key is the name of the embedding collection,
        and the value is a list of dynamic embedding table names within that collection. Defaults to None.
    optim : Optional[bool], optional
        Whether to dump the optimizer states. Defaults to False.
    pg : Optional[dist.ProcessGroup], optional
        The process group used to control the communication scope in the dump. Defaults to None.

    Returns
    -------
    None
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # create path
    created_dirs = []
    if not os.path.exists(path):
        created_dirs.append(path)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise Exception("can't build path:", path) from e
    else:
        if os.path.isdir(path):
            if not os.listdir(path):
                pass
            elif not allow_overwrite:
                raise Exception(
                    f"DynamicEmb Cannot dump to {path} because it already contains files, "
                    "as it may cause overwriting of existing files with the same name."
                )
            else:
                logging.warning(f"DynamicEmb Overwriting existing files in {path}")
        else:
            raise Exception(f"The path '{path}' exists and is not a directory.")
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, will not dump any embedding tables to filesystem!",
            UserWarning,
        )
        return

    for collection_path, _, _ in collections_list:
        full_collection_path = os.path.join(path, collection_path)
        if not os.path.exists(full_collection_path):
            os.makedirs(full_collection_path, exist_ok=True)
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    for collection_path, _, current_collection_module in collections_list:
        full_collection_path = os.path.join(path, collection_path)
        current_dynamic_emb_module_list = get_dynamic_emb_module(
            current_collection_module
        )
        table_names_to_dump = (
            table_names.get(collection_path, None) if table_names else None
        )
        for dynamic_emb_module in current_dynamic_emb_module_list:
            dynamic_emb_module.dump(
                full_collection_path,
                optim=optim,
                table_names=table_names_to_dump,
                pg=pg,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    return


# TODO: Now only support Row-Wise sharding, will support TW/TWRW in future.
# TODO: Currently, the dump and load functions enforce dumping and loading of all parameters of the optimizer.
#      This mechanism prevents users from controlling certain parameters of the optimizer.
#      In order to allow users to set optimizer parameters more flexibly in the future,
#      we need to add functionality for dumping and loading specific args,
#      allowing for more flexible configuration of optimizer arguments
def DynamicEmbLoad(
    path: str,
    model: nn.Module,
    table_names: Optional[List[str]] = None,
    optim: bool = False,
    pg: dist.ProcessGroup = dist.group.WORLD,
):
    """
    Load the distributed weights and corresponding optimizer states of dynamic embedding tables from the filesystem into the model.

    Each dynamic embedding table will be stored as a key binary file and a value binary file, where the dtype of the key is int64_t,
    and the dtype of the value is float. Each optimizer state is also treated as a dynamic embedding table.

    Parameters
    ----------
    path : str
        The main folder for weight files.
    model : nn.Module
        The model containing dynamic embedding tables.
    table_names : Optional[Dict[str, List[str]]], optional
        A dictionary specifying which embedding collection and which table to load. The key is the name of the embedding collection,
        and the value is a list of dynamic embedding table names within that collection. Defaults to None.
    optim : bool, optional
        Whether to load the optimizer states. Defaults to False.
    pg : Optional[dist.ProcessGroup], optional
        The process group used to control the communication scope in the load. Defaults to None.

    Returns
    -------
    None
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if not os.path.exists(path):
        raise Exception("can't find path to load, path:", path)

    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, can't load any embedding tables from filesystem!",
            UserWarning,
        )
        return

    for _, current_collection in enumerate(collections_list):
        (
            collection_path,
            _,
            current_collection_module,
        ) = current_collection
        full_collection_path = os.path.join(path, collection_path)
        current_dynamic_emb_module_list = get_dynamic_emb_module(
            current_collection_module
        )

        for dynamic_emb_module in current_dynamic_emb_module_list:
            table_names_to_load = (
                table_names.get(collection_path, None) if table_names else None
            )

            dynamic_emb_module.load(
                full_collection_path,
                optim=optim,
                table_names=table_names_to_load,
                pg=pg,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return
