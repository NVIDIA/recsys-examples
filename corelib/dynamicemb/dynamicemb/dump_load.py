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

import json
import os
import warnings
from collections import deque
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables
from dynamicemb.dynamicemb_config import dyn_emb_to_torch
from dynamicemb_extensions import (
    DynamicEmbTable,
    EvictStrategy,
    dyn_emb_capacity,
    dyn_emb_cols,
    export_batch,
    insert_or_assign,
)
from torch import nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.model_parallel import get_unwrapped_module

torch_dtype_to_np_dtype = {
    torch.uint64: np.uint64,
    torch.int64: np.int64,
    torch.float32: np.float32,
}

KEY_TYPE = torch.int64
EMBEDDING_TYPE = torch.float32
SCORE_TYPE = torch.int64
OPT_STATE_TYPE = torch.float32


def encode_key_file_path(
    root_path: str, table_name: str, rank: int, world_size: int
) -> str:
    return os.path.join(
        root_path, f"{table_name}_emb_keys.rank_{rank}.world_size_{world_size}"
    )


def encode_value_file_path(
    root_path: str, table_name: str, rank: int, world_size: int
) -> str:
    return os.path.join(
        root_path, f"{table_name}_emb_values.rank_{rank}.world_size_{world_size}"
    )


def encode_score_file_path(
    root_path: str, table_name: str, rank: int, world_size: int
) -> str:
    return os.path.join(
        root_path, f"{table_name}_emb_scores.rank_{rank}.world_size_{world_size}"
    )


def encode_opt_file_path(
    root_path: str, table_name: str, rank: int, world_size: int
) -> str:
    return os.path.join(
        root_path, f"{table_name}_opt_values.rank_{rank}.world_size_{world_size}"
    )


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving data to JSON file: {e}")


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


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
    if isinstance(module, BatchedDynamicEmbeddingTables):
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


def export_keys_values(
    dynamic_table: DynamicEmbTable,
    device: torch.device,
    batch_size: int = 65536,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    export keys, embeddings, opt_states, scores
    """

    search_capacity = dyn_emb_capacity(dynamic_table)

    offset = 0

    while offset < search_capacity:
        key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
        value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
        dim = dyn_emb_cols(dynamic_table)
        optstate_dim = dynamic_table.optstate_dim()
        total_dim = dim + optstate_dim

        keys = torch.empty(batch_size, dtype=key_dtype, device=device)
        values = torch.empty(batch_size * total_dim, dtype=value_dtype, device=device)
        scores = torch.zeros(batch_size, dtype=SCORE_TYPE, device=device)
        d_counter = torch.zeros(1, dtype=torch.uint64, device=device)

        export_batch(dynamic_table, batch_size, offset, d_counter, keys, values, scores)

        values = values.reshape(batch_size, total_dim)

        embeddings = values[:, :dim].contiguous()
        opt_states = values[:, dim:].contiguous()

        d_counter = d_counter.to(dtype=torch.int64)
        actual_length = d_counter.item()
        if actual_length > 0:
            yield (
                keys[:actual_length].to(KEY_TYPE),
                embeddings[:actual_length, :].to(EMBEDDING_TYPE),
                opt_states[:actual_length, :].to(OPT_STATE_TYPE),
                scores[:actual_length].to(SCORE_TYPE),
            )
        offset += batch_size


def local_export(
    dynamic_table: DynamicEmbTable,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str] = None,
    opt_file_path: Optional[str] = None,
    batch_size: int = 65536,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if dynamic_table.evict_strategy() == EvictStrategy.KLru:
        if score_file_path is not None:
            raise RuntimeError("Scores are not supported for KLru evict strategy")
    else:
        if score_file_path is None:
            raise RuntimeError("Scores are required for non-KLru evict strategy")

    fkey = open(emb_key_path, "wb")
    fembedding = open(embedding_file_path, "wb")
    fscore = open(score_file_path, "wb") if score_file_path else None
    fopt_states = open(opt_file_path, "wb") if opt_file_path else None

    for keys, embeddings, opt_states, scores in export_keys_values(
        dynamic_table, device, batch_size
    ):
        fkey.write(keys.cpu().numpy().tobytes())
        fembedding.write(embeddings.cpu().numpy().tobytes())
        if fopt_states:
            fopt_states.write(opt_states.cpu().numpy().tobytes())
        if fscore:
            fscore.write(scores.cpu().numpy().tobytes())

    fkey.close()
    fembedding.close()

    if fscore:
        fscore.close()

    if fopt_states:
        fopt_states.close()

    return


def distributed_export(
    dynamic_table: DynamicEmbTable,
    root_path: str,
    name: str,
    batch_size: int = 65536,
    pg: Optional[dist.ProcessGroup] = None,
    optim: bool = False,
):
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    emb_key_path = encode_key_file_path(root_path, name, rank, world_size)
    emb_value_path = encode_value_file_path(root_path, name, rank, world_size)

    emb_score_path = None
    if dynamic_table.evict_strategy() != EvictStrategy.KLru:
        emb_score_path = encode_score_file_path(root_path, name, rank, world_size)

    opt_value_path = None
    if optim and dynamic_table.optstate_dim() > 0:
        opt_value_path = encode_opt_file_path(root_path, name, rank, world_size)

    local_export(
        dynamic_table,
        emb_key_path,
        emb_value_path,
        emb_score_path,
        opt_value_path,
        batch_size,
        device,
    )

    return


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
    pg: Optional[dist.ProcessGroup] = None,
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
            else:
                raise Exception(
                    f"DynamicEmb Cannot dump to {path} because it already contains files, "
                    "as it may cause overwriting of existing files with the same name."
                )
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

    # Get the rank of the current process
    rank = dist.get_rank(group=pg)

    for _, current_collection in enumerate(collections_list):
        (
            collection_path,
            current_collection_name,
            current_collection_module,
        ) = current_collection
        full_collection_path = os.path.join(path, collection_path)
        current_dynamic_emb_module_list = get_dynamic_emb_module(
            current_collection_module
        )

        if not os.path.exists(full_collection_path):
            os.makedirs(full_collection_path, exist_ok=True)

        for _, dynamic_emb_module in enumerate(current_dynamic_emb_module_list):
            current_table_names = dynamic_emb_module.table_names
            current_tables = dynamic_emb_module.tables

            for dynamic_table_name, dynamic_table in zip(
                current_table_names, current_tables
            ):
                if table_names is not None and dynamic_table_name not in set(
                    table_names[current_collection_name]
                ):
                    continue

                distributed_export(
                    dynamic_table,
                    full_collection_path,
                    dynamic_table_name,
                    pg=pg,
                    optim=optim,
                )

                if optim and rank == 0:
                    optimizer = dynamic_emb_module.optimizer
                    opt_args = optimizer.get_opt_args()
                    args_filename = dynamic_table_name + "_opt_args.json"
                    args_path = os.path.join(full_collection_path, args_filename)
                    save_to_json(opt_args, args_path)
                print(
                    f"Rank {rank}DynamicEmb dump table {dynamic_table_name} from module {current_collection_name} success!"
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    return


def load_key_values(
    dynamic_table: DynamicEmbTable,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
):
    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    if opt_states is None and optstate_dim > 0:
        opt_states = (
            torch.ones(
                keys.numel(),
                optstate_dim,
                dtype=dyn_emb_to_torch(dynamic_table.value_type()),
                device=embeddings.device,
            )
            * dynamic_table.get_initial_optstate()
        )

    values = (
        torch.cat([embeddings.view(-1, dim), opt_states.view(-1, optstate_dim)], dim=-1)
        if opt_states is not None
        else embeddings
    )

    if dynamic_table.evict_strategy() == EvictStrategy.KLru:
        if scores is not None:
            raise RuntimeError("Scores are not supported for KLru evict strategy")
    else:
        if scores is None:
            raise RuntimeError("Scores are required for non-KLru evict strategy")

    key_type = dyn_emb_to_torch(dynamic_table.key_type())
    value_type = dyn_emb_to_torch(dynamic_table.value_type())
    if scores is not None:
        insert_or_assign(
            dynamic_table,
            keys.numel(),
            keys.to(key_type),
            values.to(value_type),
            scores.to(SCORE_TYPE),
        )
    else:
        insert_or_assign(
            dynamic_table, keys.numel(), keys.to(key_type), values.to(value_type)
        )


def local_load(
    dynamic_table: DynamicEmbTable,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str] = None,
    opt_file_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
    checkpoint_version: int = 2,
):
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    fkey = open(emb_key_path, "rb")
    fembedding = open(embedding_file_path, "rb")
    fscore = open(score_file_path, "rb") if score_file_path else None
    fopt_states = open(opt_file_path, "rb") if opt_file_path else None

    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()

    batch_size = 65536

    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

    if checkpoint_version == 1:
        num_embeddings = (
            os.path.getsize(embedding_file_path)
            // EMBEDDING_TYPE.itemsize
            // (dim + optstate_dim)
        )
    else:
        num_embeddings = (
            os.path.getsize(embedding_file_path) // EMBEDDING_TYPE.itemsize // dim
        )

    if num_keys != num_embeddings:
        raise ValueError(
            f"The number of keys in {emb_key_path} does not match with number of embeddings in {embedding_file_path}."
        )

    if score_file_path:
        num_scores = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
        if num_keys != num_scores:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
            )

    if opt_file_path and checkpoint_version == 2:
        num_opt_states = (
            os.path.getsize(opt_file_path) // OPT_STATE_TYPE.itemsize // optstate_dim
        )
        if num_keys != num_opt_states:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of opt_states in {opt_file_path}."
            )

    for start in range(0, num_keys, batch_size):
        num_keys_to_read = min(num_keys - start, batch_size)
        keys_bytes = fkey.read(KEY_TYPE.itemsize * num_keys_to_read)
        if checkpoint_version == 1:
            value_bytes = fembedding.read(
                EMBEDDING_TYPE.itemsize * (dim + optstate_dim) * num_keys_to_read
            )
            values = torch.tensor(
                np.frombuffer(
                    value_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]
                ),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim + optstate_dim)
            embeddings = values[:, :dim]
            opt_states = None
            if fopt_states:
                opt_states = values[:, dim:]
        elif checkpoint_version == 2:
            embedding_bytes = fembedding.read(
                EMBEDDING_TYPE.itemsize * dim * num_keys_to_read
            )
            embeddings = torch.tensor(
                np.frombuffer(
                    embedding_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]
                ),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states:
                opt_state_bytes = fopt_states.read(
                    OPT_STATE_TYPE.itemsize * optstate_dim * num_keys_to_read
                )
                opt_states = torch.tensor(
                    np.frombuffer(
                        opt_state_bytes, dtype=torch_dtype_to_np_dtype[OPT_STATE_TYPE]
                    ),
                    dtype=OPT_STATE_TYPE,
                    device=device,
                ).view(-1, optstate_dim)
        else:
            raise ValueError(f"Invalid checkpoint version: {checkpoint_version}")

        keys = torch.tensor(
            np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
            dtype=KEY_TYPE,
            device=device,
        )

        scores = None
        if fscore:
            score_bytes = fscore.read(SCORE_TYPE.itemsize * num_keys_to_read)
            scores = torch.tensor(
                np.frombuffer(score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]),
                dtype=SCORE_TYPE,
                device=device,
            )

        if world_size > 1:
            masks = keys % world_size == rank
            keys = keys[masks]
            embeddings = embeddings[masks, :]
            if scores is not None:
                scores = scores[masks]
            if opt_states is not None:
                opt_states = opt_states[masks, :]
        load_key_values(dynamic_table, keys, embeddings, scores, opt_states)

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


def find_files(
    root_path: str, table_name: str, suffix: str
) -> Tuple[List[str], int, int]:
    suffix_to_encode_file_path_func = {
        "emb_keys": encode_key_file_path,
        "emb_values": encode_value_file_path,
        "emb_scores": encode_score_file_path,
        "opt_values": encode_opt_file_path,
    }
    if suffix not in suffix_to_encode_file_path_func:
        raise RuntimeError(f"Invalid suffix: {suffix}")
    encode_file_path_func = suffix_to_encode_file_path_func[suffix]

    import glob

    # v2 version
    files = glob.glob(
        os.path.join(root_path, f"{table_name}_{suffix}.rank_*.world_size_*")
    )
    if len(files) == 0:
        # v1 version
        checkpoint_version = 1
        suffix_to_v1_path = {
            "emb_keys": os.path.join(root_path, table_name + "_keys"),
            "emb_values": os.path.join(root_path, table_name + "_values"),
            "emb_scores": os.path.join(root_path, table_name + "_scores"),
            "opt_values": os.path.join(root_path, table_name + "_values"),
        }
        file = suffix_to_v1_path[suffix]
        if not os.path.exists(file):
            return [], 0, checkpoint_version
        return [file], 1, checkpoint_version
    files = sorted(files)
    world_size = int(files[0].split(".")[-1].split("_")[-1])
    if len(files) != world_size:
        raise RuntimeError(
            f"Checkpoints is corrupted. Found {len(files)} under path {root_path} for table {table_name}, but the number of checkpointed world size is {world_size}."
        )

    for i in range(world_size):
        expected_file_path = encode_file_path_func(root_path, table_name, i, world_size)
        if expected_file_path not in set(files):
            raise RuntimeError(
                f"Checkpoints is corrupted. Expected file path {expected_file_path} for table {table_name}, but it is not found."
            )

    checkpoint_version = 2
    return files, len(files), checkpoint_version


def get_loading_files(
    root_path: str,
    name: str,
    pg: Optional[dist.ProcessGroup] = None,
    need_dump_score: bool = False,
    optim: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], int, int, int]:
    checkpoint_version = 2
    world_size = dist.get_world_size(group=pg)

    if not os.path.exists(root_path):
        raise RuntimeError(f"can't find path to load, path:", root_path)

    key_files, num_key_files, checkpoint_version = find_files(
        root_path, name, "emb_keys"
    )
    value_files, num_value_files, _ = find_files(root_path, name, "emb_values")
    score_files, num_score_files, _ = (
        find_files(root_path, name, "emb_scores") if need_dump_score else ([], 0, None)
    )
    opt_files, num_opt_files, _ = (
        find_files(root_path, name, "opt_values") if optim else ([], 0, None)
    )

    if num_key_files != num_value_files:
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of value files."
        )

    if need_dump_score and num_key_files != num_score_files:
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of score files."
        )

    if optim and num_key_files != num_opt_files:
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of opt files."
        )

    rank = dist.get_rank(group=pg)
    if world_size == num_key_files and checkpoint_version == 2:
        return (
            [encode_key_file_path(root_path, name, rank, world_size)],
            [encode_value_file_path(root_path, name, rank, world_size)],
            [encode_score_file_path(root_path, name, rank, world_size)]
            if need_dump_score
            else [None],
            [encode_opt_file_path(root_path, name, rank, world_size)]
            if num_opt_files > 0
            else [None],
            0,
            1,
            checkpoint_version,
        )
    # TODO: support skipping files.
    return (
        key_files,
        value_files,
        score_files if need_dump_score else [None for _ in range(num_key_files)],
        opt_files if optim else [None for _ in range(num_key_files)],
        rank,
        world_size,
        checkpoint_version,
    )


def distributed_load(
    dynamic_table: DynamicEmbTable,
    root_path: str,
    name: str,
    optim: bool = False,
    pg: Optional[dist.ProcessGroup] = None,
):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    need_dump_score = dynamic_table.evict_strategy() != EvictStrategy.KLru

    (
        emb_key_files,
        emb_value_files,
        emb_score_files,
        opt_value_files,
        rank,
        world_size,
        checkpoint_version,
    ) = get_loading_files(
        root_path, name, pg, need_dump_score, optim and dynamic_table.optstate_dim() > 0
    )

    for emb_key_file, emb_value_file, score_file, opt_file in zip(
        emb_key_files, emb_value_files, emb_score_files, opt_value_files
    ):
        local_load(
            dynamic_table,
            emb_key_file,
            emb_value_file,
            score_file,
            opt_file,
            device,
            rank,
            world_size,
            checkpoint_version,
        )


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
    pg: Optional[dist.ProcessGroup] = None,
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
            current_collection_name,
            current_collection_module,
        ) = current_collection
        full_collection_path = os.path.join(path, collection_path)
        current_dynamic_emb_module_list = get_dynamic_emb_module(
            current_collection_module
        )

        for _, dynamic_emb_module in enumerate(current_dynamic_emb_module_list):
            current_table_names = dynamic_emb_module.table_names
            current_tables = dynamic_emb_module.tables

            for dynamic_table_name, dynamic_table in zip(
                current_table_names, current_tables
            ):
                if table_names is not None and dynamic_table_name not in set(
                    table_names[current_collection_name]
                ):
                    continue

                if optim:
                    args_filename = dynamic_table_name + "_opt_args.json"
                    args_path = os.path.join(full_collection_path, args_filename)
                    opt_args = load_from_json(args_path)
                    dynamic_emb_module.optimizer.set_opt_args(opt_args)

                distributed_load(
                    dynamic_table,
                    full_collection_path,
                    dynamic_table_name,
                    optim=optim,
                    pg=pg,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return
